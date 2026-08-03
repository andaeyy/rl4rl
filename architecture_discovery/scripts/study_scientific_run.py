"""Run a frozen C0-C3 study plan only after every readiness gate passes.

The readiness audit runs before provider credentials are read or a client is
constructed. This entrypoint is intentionally unusable while the checked-in PI
decision ledger and external evidence gates remain unresolved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from artifacts import (
    ArtifactContext,
    ArtifactEmittingStudyEngine,
    ImmutableStudyEventSink,
    RunArtifactStore,
)
from common.gpt56_sol import GPT56SolProfile
from scripts.audit_scientific_readiness import audit_readiness
from study.contracts import StudySpec
from study.randomization import load_or_create_plan
from study.runtime_adapters import (
    CandidateSourceStore,
    LayerACandidateEvaluator,
    MatchedCausalProposalGenerator,
)
from study.scheduling import NoPendingRuns, SequentialRunScheduler
from study.serialization import content_hash, create_json_exclusive, read_json


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing provider configuration: {name}")
    return value


def _validated_runtime_contract(
    spec: StudySpec,
    *,
    initial_source: str,
    readiness: dict[str, object],
) -> dict[str, object]:
    """Bind runtime values to the checked-in manifest and frozen PI ledger."""

    manifest = yaml.safe_load((ROOT / "experiment_manifest.yaml").read_text())
    decisions = yaml.safe_load((ROOT / "scientific_decisions.yaml").read_text())
    layer_a = manifest["evaluation"]["layer_a"]
    expected = {
        "portfolio_size": (
            int(decisions["treatment"]["portfolio_size_k"]),
            spec.portfolio_size,
        ),
        "transition_schedule": (
            tuple(int(value) for value in decisions["treatment"]["transition_schedule"]),
            spec.transition_opportunities,
        ),
        "proposal_opportunities": (
            int(decisions["budgets"]["proposal_opportunity_ceiling"]),
            spec.budget.proposal_opportunities,
        ),
        "provider_attempts_per_opportunity": (
            int(decisions["budgets"]["provider_attempt_limit_per_opportunity"]),
            spec.budget.provider_attempts_per_opportunity,
        ),
        "repair_attempts_per_opportunity": (
            int(decisions["budgets"]["repair_limit_per_opportunity"]),
            spec.budget.repair_attempts_per_opportunity,
        ),
        "layer_a_case_count": (
            int(decisions["evaluation"]["layer_a_case_count"]),
            int(layer_a["case_count"]),
        ),
    }
    mismatches = {
        name: {"frozen": frozen, "runtime": runtime}
        for name, (frozen, runtime) in expected.items()
        if frozen != runtime
    }
    manifest_portfolio = manifest["primary_causal_design"]["portfolio_size_k"]
    if manifest_portfolio != spec.portfolio_size:
        mismatches["manifest_portfolio_size"] = {
            "frozen": manifest_portfolio,
            "runtime": spec.portfolio_size,
        }
    manifest_schedule = tuple(
        manifest["primary_causal_design"]["transition_schedule"]
    )
    if manifest_schedule != spec.transition_opportunities:
        mismatches["manifest_transition_schedule"] = {
            "frozen": manifest_schedule,
            "runtime": spec.transition_opportunities,
        }
    expected_initial_id = content_hash(initial_source)
    if spec.initial_candidate_id != expected_initial_id:
        mismatches["initial_candidate_id"] = {
            "frozen": expected_initial_id,
            "runtime": spec.initial_candidate_id,
        }
    manifest_hash = hashlib.sha256(
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    if spec.common_config_hash != manifest_hash:
        mismatches["manifest_hash"] = {
            "frozen": manifest_hash,
            "runtime": spec.common_config_hash,
        }
    if mismatches:
        raise SystemExit(
            "scientific runtime differs from the frozen contract: "
            + json.dumps(mismatches, sort_keys=True)
        )
    threshold = layer_a.get("eligibility_threshold")
    if threshold is None:
        raise SystemExit("Layer A eligibility threshold remains unresolved")
    return {
        "case_count": int(layer_a["case_count"]),
        "eligibility_threshold": float(threshold),
        "decision_record_id": str(readiness["decision_ledger_sha256"]),
        "generation": manifest["shared_generation"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-spec", type=Path, required=True)
    parser.add_argument("--phase", choices=("pilot", "main"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initial-candidate", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--corpus-seal", type=Path)
    parser.add_argument("--analysis-plan", type=Path)
    parser.add_argument("--mechanism-plan", type=Path)
    parser.add_argument("--replication-policy", type=Path)
    parser.add_argument("--research-protocol", type=Path)
    parser.add_argument("--mps-evidence", type=Path)
    arguments = parser.parse_args()

    readiness = audit_readiness(
        corpus_manifest=arguments.corpus_manifest,
        corpus_seal=arguments.corpus_seal,
        analysis_plan=arguments.analysis_plan,
        mechanism_plan=arguments.mechanism_plan,
        replication_policy=arguments.replication_policy,
        research_protocol=arguments.research_protocol,
        mps_evidence=arguments.mps_evidence,
        study_spec=arguments.study_spec,
    )
    readiness_key = (
        "pilot_ready" if arguments.phase == "pilot" else "main_study_ready"
    )
    if not readiness[readiness_key]:
        print(json.dumps(readiness, indent=2, sort_keys=True))
        print(
            f"Scientific {arguments.phase} blocked before provider initialization.",
            file=sys.stderr,
        )
        return 2

    spec = StudySpec.from_dict(
        json.loads(arguments.study_spec.read_text(encoding="utf-8"))
    )
    if not spec.scientific:
        raise SystemExit("study_scientific_run refuses a toy/non-scientific StudySpec")
    initial_source = arguments.initial_candidate.resolve().read_text(encoding="utf-8")
    runtime_contract = _validated_runtime_contract(
        spec,
        initial_source=initial_source,
        readiness=readiness,
    )
    output_root = arguments.output_root.resolve()
    study_root = output_root / spec.study_id
    plan = load_or_create_plan(
        spec,
        output_root=output_root,
        plan_path=study_root / "randomization_plan.json",
    )
    scheduler = SequentialRunScheduler(
        plan,
        state_path=study_root / "schedule_state.json",
        lease_path=output_root / ".study_mps.lock",
    )

    api_key = _required_environment("DISCOVERY_API_KEY")
    api_base = _required_environment("DISCOVERY_API_BASE")
    model = _required_environment("DISCOVERY_MODEL")
    generation_contract = runtime_contract["generation"]
    if model != generation_contract["target_model"]:
        raise SystemExit(
            "DISCOVERY_MODEL differs from the frozen generation contract: "
            f"{model!r} != {generation_contract['target_model']!r}"
        )
    client = OpenAI(api_key=api_key, base_url=api_base, max_retries=0)
    shared_system = "\n\n".join(
        (
            (ROOT / "common" / "prompts" / "shared_system.md").read_text(),
            (ROOT / "common" / "prompts" / "shared_task.md").read_text(),
        )
    )
    while True:
        try:
            claim = scheduler.claim_next()
        except NoPendingRuns:
            break
        with claim as run:
            run_root = Path(run.run_directory)
            source_store = CandidateSourceStore(run_root / "candidate_sources")
            source_store.register(spec.initial_candidate_id, initial_source)
            generation = GPT56SolProfile.resolve(
                model=model,
                seed=run.run_seed,
                default_reasoning_effort=str(
                    generation_contract["reasoning_effort"]
                ),
                default_max_completion_tokens=int(
                    generation_contract["max_completion_tokens"]
                ),
                default_timeout_seconds=int(
                    generation_contract["request_timeout_seconds"]
                ),
                default_retries=int(generation_contract["retries"]),
                default_retry_delay_seconds=int(
                    generation_contract["retry_delay_seconds"]
                ),
            )
            generator = MatchedCausalProposalGenerator(
                client=client,
                generation=generation,
                source_store=source_store,
                portfolio_size=spec.portfolio_size,
                system_prompt=shared_system,
                request_log_root=run_root / "provider_records",
            )
            evaluator = LayerACandidateEvaluator(
                study_id=spec.study_id,
                block_id=run.block_id,
                run_id=run.run_id,
                condition_id=run.condition.condition_id.value,
                initial_candidate_id=spec.initial_candidate_id,
                source_store=source_store,
                output_root=run_root,
                training_profile="full_train_v1",
                device="mps",
                allow_cpu_for_tests=False,
                evaluation_profile="scientific_layer_a_v1",
                evaluation_case_count=int(runtime_contract["case_count"]),
                pi_decision_record_id=str(runtime_contract["decision_record_id"]),
                eligibility_threshold=float(
                    runtime_contract["eligibility_threshold"]
                ),
            )
            artifact_store = RunArtifactStore(
                run_root / "artifact_ledger",
                ArtifactContext(
                    study_id=spec.study_id,
                    block_id=run.block_id,
                    run_id=run.run_id,
                    condition_id=run.condition.condition_id.value,
                    writer_component="scripts.study_scientific_run",
                    code_sha256=spec.code_hash,
                    config_sha256=spec.common_config_hash,
                    environment_sha256=spec.environment_hash,
                    run_seed=run.run_seed,
                    assignment_sha256=run.assignment_hash,
                ),
            )
            ArtifactEmittingStudyEngine(
                study=spec,
                run=run,
                generator=generator,
                evaluator=evaluator,
                artifact_sink=ImmutableStudyEventSink(artifact_store),
                # The scheduler holds the study-wide MPS lease for this run.
                evaluation_lease_path=None,
            ).execute()

    frozen_indexes = []
    for run in plan.runs:
        store = RunArtifactStore.open(Path(run.run_directory) / "artifact_ledger")
        frozen_index, _ = store.load_frozen_index("search_completion")
        frozen_indexes.append(frozen_index.to_dict())
    index_manifest = {
        "schema_name": "StudyArtifactIndexManifest",
        "schema_version": "1.0",
        "study_id": spec.study_id,
        "study_phase": arguments.phase,
        "assignment_hash": plan.assignment_hash,
        "run_indexes": frozen_indexes,
    }
    index_manifest_path = study_root / "artifact_index_manifest.json"
    if index_manifest_path.exists():
        if read_json(index_manifest_path) != index_manifest:
            raise ValueError("study artifact-index manifest changed across resume")
    else:
        create_json_exclusive(index_manifest_path, index_manifest)

    print(json.dumps(scheduler.summary(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
