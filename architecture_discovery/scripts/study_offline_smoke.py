"""Run the common C0-C3 engine end to end without Torch or provider calls."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from study.contracts import ConditionId, StudySpec
from artifacts import (
    ArtifactContext,
    ArtifactEmittingStudyEngine,
    ImmutableStudyEventSink,
    RunArtifactStore,
)
from study.fakes import DeterministicFakeEvaluator, DeterministicFakeGenerator
from study.randomization import load_or_create_plan
from study.scheduling import NoPendingRuns, SequentialAcceleratorScheduler
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


def _offline_no_search(spec: StudySpec, *, output_root: Path) -> dict:
    backend = DeterministicFakeBackend()
    generator = NoSearchProposalGenerator(
        spec=NoSearchSpec(
            system_prompt="Offline no-search system fixture.",
            task_prompt=(
                "Return one independent complete architecture_tensor_graph v1.0 "
                "JSON candidate with a mechanism_hypothesis."
            ),
            max_completion_tokens=100,
        ),
        backend=backend,
        scientific=False,
    )
    no_search_study = replace(
        spec,
        study_id=f"{spec.study_id}-no-search-smoke",
        block_count=1,
        initial_candidate_id=content_hash(backend.candidate_source),
        scientific=False,
    )
    no_search_root = output_root / no_search_study.study_id
    plan = load_or_create_plan(
        no_search_study,
        output_root=output_root,
        plan_path=no_search_root / "randomization_plan.json",
    )
    run = next(
        assigned
        for assigned in plan.runs
        if assigned.condition.condition_id is ConditionId.C0
    )
    artifact_store = RunArtifactStore(
        run.execution_directory / "artifact_ledger",
        ArtifactContext(
            study_id=no_search_study.study_id,
            block_id=run.block_id,
            run_id=run.run_id,
            condition_id=run.condition.condition_id.value,
            writer_component="scripts.study_offline_smoke.no_search",
            code_sha256=no_search_study.code_hash,
            config_sha256=no_search_study.common_config_hash,
            environment_sha256=no_search_study.environment_hash,
            run_seed=run.run_seed,
            assignment_sha256=run.assignment_hash,
        ),
    )
    state = ArtifactEmittingStudyEngine(
        study=no_search_study,
        run=run,
        generator=generator,
        evaluator=DeterministicFakeEvaluator(),
        artifact_sink=ImmutableStudyEventSink(
            artifact_store,
            initial_candidate_source=backend.candidate_source,
        ),
        evaluation_lease_path=None,
    ).execute()
    provider_inputs = [request.model_input for request in backend.requests]
    return {
        "condition_id": "NO_SEARCH",
        "scientific": False,
        "adaptive_feedback_visible_to_backend": False,
        "provider_input_constant": (
            not provider_inputs
            or all(value == provider_inputs[0] for value in provider_inputs)
        ),
        "request_count": int(state.ledger["provider_attempts"]),
        "ledger": dict(state.ledger),
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
    scheduler = SequentialAcceleratorScheduler(
        plan,
        state_path=study_directory / "schedule_state.json",
        lease_path=output_root / ".study_accelerator.lock",
        accelerator_kind="cpu",
    )

    while True:
        try:
            claim = scheduler.claim_next()
        except NoPendingRuns:
            break
        with claim as run:
            artifact_store = RunArtifactStore(
                run.execution_directory / "artifact_ledger",
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
        state = read_json(run.execution_directory / "run_state.json")
        store = RunArtifactStore.open(run.execution_directory / "artifact_ledger")
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
    summary = {
        "schema_name": "OfflineStudySmokeSummary",
        "schema_version": "1.0",
        "mode": "offline_synthetic_only",
        "provider_calls": 0,
        "torch_training_runs": 0,
        "scientific": False,
        "study_id": spec.study_id,
        "assignment_hash": plan.assignment_hash,
        "scheduler": scheduler.summary(),
        "runs": run_summaries,
        "artifact_index_manifest": "artifact_index_manifest.json",
        "no_search": _offline_no_search(spec, output_root=output_root),
    }
    summary_path = study_directory / "offline_smoke_summary.json"
    if summary_path.exists():
        if read_json(summary_path) != summary:
            raise ValueError("offline smoke summary changed across resume")
    else:
        create_json_exclusive(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
