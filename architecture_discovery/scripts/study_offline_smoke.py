"""Run the common C0-C3 engine end to end without Torch or provider calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.contracts import StudySpec
from artifacts import (
    ArtifactContext,
    ArtifactEmittingStudyEngine,
    ImmutableStudyEventSink,
    RunArtifactStore,
)
from study.budget import BudgetLedger
from study.contracts import ConditionId, ConditionSpec
from study.fakes import DeterministicFakeEvaluator, DeterministicFakeGenerator
from study.interfaces import ProposalContext
from study.randomization import load_or_create_plan
from study.scheduling import NoPendingRuns, SequentialRunScheduler
from study.serialization import (
    content_hash,
    create_json_exclusive,
    read_json,
)
from baselines.no_search import (
    DeterministicFakeBackend,
    NoSearchProposalGenerator,
    NoSearchSpec,
)


def _offline_no_search(spec: StudySpec) -> dict:
    backend = DeterministicFakeBackend()
    generator = NoSearchProposalGenerator(
        spec=NoSearchSpec(
            system_prompt="Offline no-search system fixture.",
            task_prompt="Propose one independent synthetic candidate.",
            max_completion_tokens=100,
        ),
        backend=backend,
        scientific=False,
    )
    evaluator = DeterministicFakeEvaluator()
    ledger = BudgetLedger(spec.budget)
    seed = evaluator.evaluate_seed(spec.initial_candidate_id, spec.study_seed)
    ledger.record_seed_evaluation(
        training_attempts=seed.training_attempts,
        training_steps=seed.training_steps,
        training_examples=seed.training_examples,
        mps_seconds=seed.mps_seconds,
        evaluation_cases=seed.evaluation_cases,
    )
    for opportunity in range(1, spec.budget.proposal_opportunities + 1):
        ledger.begin_opportunity(opportunity)
        attempt = ledger.start_provider_attempt()
        proposal = generator.generate(
            ProposalContext(
                study_id=spec.study_id,
                block_id="offline-no-search-block",
                run_id="offline-no-search-run",
                run_seed=spec.study_seed,
                condition=ConditionSpec.for_id(ConditionId.C0),
                opportunity_index=opportunity,
                provider_attempt=attempt,
                # Deliberately populated with changing fake state. The no-search
                # projection must discard it before constructing model input.
                parent_ids=(f"feedback-parent-{opportunity}",),
                transition_active=opportunity % 2 == 0,
                repair=True,
            )
        )
        ledger.record_provider_usage(
            prompt_tokens=proposal.prompt_tokens,
            completion_tokens=proposal.completion_tokens,
        )
        candidate_id = content_hash(proposal.candidate_source or "")
        ledger.record_candidate_source(candidate_id)
        evaluation = evaluator.evaluate_candidate(
            proposal.candidate_source or "",
            candidate_id=candidate_id,
            opportunity_index=opportunity,
            run_seed=spec.study_seed,
        )
        ledger.record_training(
            attempts=evaluation.training_attempts,
            steps=evaluation.training_steps,
            examples=evaluation.training_examples,
            mps_seconds=evaluation.mps_seconds,
        )
        ledger.record_evaluation(evaluation.evaluation_cases)
        ledger.finish_opportunity(evaluation.outcome)
    provider_inputs = [request.model_input for request in backend.requests]
    return {
        "condition_id": "NO_SEARCH",
        "scientific": False,
        "adaptive_feedback_visible_to_backend": False,
        "provider_input_constant": all(
            value == provider_inputs[0] for value in provider_inputs
        ),
        "request_count": len(backend.requests),
        "ledger": ledger.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provider-free infrastructure smoke for the common causal engine."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--study-id", default="offline-causal-smoke")
    parser.add_argument("--study-seed", type=int, default=7)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument("--opportunities", type=int, default=3)
    arguments = parser.parse_args()

    output_root = Path(arguments.output_dir).resolve()
    spec = StudySpec.toy(
        study_id=arguments.study_id,
        study_seed=arguments.study_seed,
        block_count=arguments.blocks,
        proposal_opportunities=arguments.opportunities,
    )
    study_directory = output_root / spec.study_id
    plan = load_or_create_plan(
        spec,
        output_root=output_root,
        plan_path=study_directory / "randomization_plan.json",
    )
    scheduler = SequentialRunScheduler(
        plan,
        state_path=study_directory / "schedule_state.json",
        lease_path=output_root / ".study_mps.lock",
    )

    while True:
        try:
            claim = scheduler.claim_next()
        except NoPendingRuns:
            break
        with claim as run:
            artifact_store = RunArtifactStore(
                Path(run.run_directory) / "artifact_ledger",
                ArtifactContext(
                    study_id=spec.study_id,
                    block_id=run.block_id,
                    run_id=run.run_id,
                    condition_id=run.condition.condition_id.value,
                    writer_component="scripts.study_offline_smoke",
                    code_sha256=spec.code_hash,
                    config_sha256=spec.common_config_hash,
                    environment_sha256=spec.environment_hash,
                    run_seed=run.run_seed,
                    assignment_sha256=run.assignment_hash,
                ),
            )
            engine = ArtifactEmittingStudyEngine(
                study=spec,
                run=run,
                generator=DeterministicFakeGenerator(),
                evaluator=DeterministicFakeEvaluator(),
                artifact_sink=ImmutableStudyEventSink(artifact_store),
                # The scheduler holds the global lease for the complete run.
                evaluation_lease_path=None,
            )
            engine.execute()

    run_summaries = []
    frozen_indexes = []
    for run in plan.runs:
        state = read_json(Path(run.run_directory) / "run_state.json")
        store = RunArtifactStore.open(Path(run.run_directory) / "artifact_ledger")
        frozen_index, _ = store.load_frozen_index("search_completion")
        frozen_indexes.append(frozen_index.to_dict())
        run_summaries.append(
            {
                "run_id": run.run_id,
                "condition_id": run.condition.condition_id.value,
                "status": state["status"],
                "seed_evaluations": state["ledger"]["seed_evaluations"],
                "proposal_opportunities": state["ledger"][
                    "proposal_opportunities"
                ],
            }
        )
    index_manifest = {
        "schema_name": "StudyArtifactIndexManifest",
        "schema_version": "1.0",
        "study_id": spec.study_id,
        "assignment_hash": plan.assignment_hash,
        "run_indexes": frozen_indexes,
    }
    index_manifest_path = study_directory / "artifact_index_manifest.json"
    if index_manifest_path.exists():
        if read_json(index_manifest_path) != index_manifest:
            raise ValueError("study artifact-index manifest changed across resume")
    else:
        create_json_exclusive(index_manifest_path, index_manifest)
    print(
        json.dumps(
            {
                "mode": "offline_synthetic_only",
                "provider_calls": 0,
                "torch_training_runs": 0,
                "scientific": False,
                "study_id": spec.study_id,
                "assignment_hash": plan.assignment_hash,
                "scheduler": scheduler.summary(),
                "runs": run_summaries,
                "artifact_index_manifest": str(index_manifest_path),
                "no_search": _offline_no_search(spec),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
