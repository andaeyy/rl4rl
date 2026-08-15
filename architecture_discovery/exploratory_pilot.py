"""Bounded exploratory C0-C3 pilot shared by local tests and Modal.

The module has two deliberately separate execution modes:

* ``provider-free`` uses deterministic fakes for local end-to-end validation.
* ``provider`` uses one OpenAI-compatible request per opportunity and the
  trusted CUDA evaluator.  It is only called from the approval-gated Modal
  function; importing this module never reads a credential or contacts a
  service.

Every output is marked ``exploratory_non_scientific``.  This module never edits
the scientific decision or readiness files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts import (  # noqa: E402
    ArtifactContext,
    ArtifactEmittingStudyEngine,
    ImmutableStudyEventSink,
    RunArtifactStore,
)
from common.evaluator import SearchEvaluationContext, evaluate_candidate  # noqa: E402
from common.gpt56_sol import GPT56SolProfile, resolve_provider_endpoint  # noqa: E402
from common.training_config import get_training_profile  # noqa: E402
from modal_boundary import (  # noqa: E402
    ArtifactIntegrityError,
    build_image_source_manifest,
    build_artifact_manifest,
    load_artifact_manifest,
    verify_artifact_manifest,
    write_artifact_manifest,
)
from study.budget import BudgetSpec, OpportunityOutcome  # noqa: E402
from study.contracts import StudySpec  # noqa: E402
from study.fakes import DeterministicFakeEvaluator, DeterministicFakeGenerator  # noqa: E402
from study.interfaces import EvaluationResult, ProposalContext, ProposalResult  # noqa: E402
from study.randomization import load_or_create_plan  # noqa: E402
from study.serialization import content_hash, create_json_exclusive, read_json  # noqa: E402
from study.scheduling import NoPendingRuns, SequentialAcceleratorScheduler  # noqa: E402

PRESET_PATH = ROOT / "configs" / "exploratory_modal_pilot.yaml"
MODE = "exploratory_non_scientific"
IMAGE_SOURCE_MANIFEST = build_image_source_manifest(ROOT)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(?:sk|gh[pousr]|hf)_[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:tinker|tml)[_-][A-Za-z0-9_-]{20,}\b", re.I),
)


@dataclass(frozen=True)
class ExploratoryPilotConfig:
    schema_name: str
    schema_version: str
    mode: str
    scientific: bool
    blocks: int
    proposal_opportunities: int
    provider_attempts_per_opportunity: int
    repair_attempts_per_opportunity: int
    provider_retries: int
    modal_retries: int
    sequential_accelerator_leases: int
    training_profile: str
    evaluation_profile: str
    evaluation_cases: int
    device: str
    maximum_wall_seconds: int
    modal_cost_cap_usd: str
    provider_cost_cap_usd: str
    maximum_artifact_bytes: int
    notes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ExploratoryPilotConfig":
        expected = {
            "schema_name", "schema_version", "mode", "scientific", "blocks",
            "proposal_opportunities", "provider_attempts_per_opportunity",
            "repair_attempts_per_opportunity", "provider_retries", "modal_retries",
            "sequential_accelerator_leases", "training_profile", "evaluation_profile",
            "evaluation_cases", "device", "maximum_wall_seconds", "modal_cost_cap_usd",
            "provider_cost_cap_usd", "maximum_artifact_bytes", "notes",
        }
        if set(payload) != expected:
            raise ValueError("exploratory preset has an invalid exact schema")
        config = cls(
            schema_name=str(payload["schema_name"]),
            schema_version=str(payload["schema_version"]),
            mode=str(payload["mode"]),
            scientific=payload["scientific"] is True,
            blocks=int(payload["blocks"]),
            proposal_opportunities=int(payload["proposal_opportunities"]),
            provider_attempts_per_opportunity=int(payload["provider_attempts_per_opportunity"]),
            repair_attempts_per_opportunity=int(payload["repair_attempts_per_opportunity"]),
            provider_retries=int(payload["provider_retries"]),
            modal_retries=int(payload["modal_retries"]),
            sequential_accelerator_leases=int(payload["sequential_accelerator_leases"]),
            training_profile=str(payload["training_profile"]),
            evaluation_profile=str(payload["evaluation_profile"]),
            evaluation_cases=int(payload["evaluation_cases"]),
            device=str(payload["device"]),
            maximum_wall_seconds=int(payload["maximum_wall_seconds"]),
            modal_cost_cap_usd=str(payload["modal_cost_cap_usd"]),
            provider_cost_cap_usd=str(payload["provider_cost_cap_usd"]),
            maximum_artifact_bytes=int(payload["maximum_artifact_bytes"]),
            notes=tuple(str(item) for item in payload["notes"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema_name != "exploratory_modal_pilot" or self.schema_version != "1":
            raise ValueError("unsupported exploratory preset")
        if self.mode != MODE or self.scientific:
            raise ValueError("exploratory preset must remain non-scientific")
        if self.blocks != 1 or self.proposal_opportunities != 1:
            raise ValueError("exploratory preset is intentionally one-block/one-opportunity")
        if self.provider_attempts_per_opportunity != 1 or self.provider_retries != 0:
            raise ValueError("exploratory provider budget must be one single-shot attempt")
        if self.modal_retries != 0 or self.sequential_accelerator_leases != 1:
            raise ValueError("exploratory Modal execution must be sequential and retry-free")
        if self.repair_attempts_per_opportunity < 0:
            raise ValueError("repair ceiling cannot be negative")
        if self.device != "cuda":
            raise ValueError("exploratory experiments are CUDA-only")
        if self.maximum_wall_seconds <= 0 or self.evaluation_cases <= 0:
            raise ValueError("exploratory ceilings must be positive")
        if not self.modal_cost_cap_usd or not self.provider_cost_cap_usd:
            raise ValueError("exploratory cost caps are required")
        profile = get_training_profile(self.training_profile)
        if profile.device_requirement != "cuda" or profile.scientific:
            raise ValueError("exploratory training profile must be non-scientific CUDA")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "mode": self.mode,
            "scientific": self.scientific,
            "blocks": self.blocks,
            "proposal_opportunities": self.proposal_opportunities,
            "provider_attempts_per_opportunity": self.provider_attempts_per_opportunity,
            "repair_attempts_per_opportunity": self.repair_attempts_per_opportunity,
            "provider_retries": self.provider_retries,
            "modal_retries": self.modal_retries,
            "sequential_accelerator_leases": self.sequential_accelerator_leases,
            "training_profile": self.training_profile,
            "evaluation_profile": self.evaluation_profile,
            "evaluation_cases": self.evaluation_cases,
            "device": self.device,
            "maximum_wall_seconds": self.maximum_wall_seconds,
            "modal_cost_cap_usd": self.modal_cost_cap_usd,
            "provider_cost_cap_usd": self.provider_cost_cap_usd,
            "maximum_artifact_bytes": self.maximum_artifact_bytes,
            "notes": list(self.notes),
        }


def load_config(path: str | Path = PRESET_PATH) -> ExploratoryPilotConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("exploratory preset must be a YAML object")
    return ExploratoryPilotConfig.from_mapping(payload)


def _safe_text_scan(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
            raise ValueError(f"credential-like material found in exploratory artifact: {path.name}")


def _build_spec(config: ExploratoryPilotConfig, study_id: str) -> StudySpec:
    budget = BudgetSpec(
        proposal_opportunities=config.proposal_opportunities,
        provider_attempts_per_opportunity=config.provider_attempts_per_opportunity,
        prompt_tokens=32_768,
        completion_tokens=16_384,
        repairs=config.repair_attempts_per_opportunity,
        candidate_training_attempts=8,
        training_steps=500,
        training_examples=8_000,
        accelerator_kind="cuda",
        accelerator_seconds=float(config.maximum_wall_seconds),
        evaluation_cases=config.evaluation_cases * 8,
        infrastructure_retries=0,
        repair_attempts_per_opportunity=config.repair_attempts_per_opportunity,
        seed_evaluations=1,
    )
    return StudySpec(
        study_id=study_id,
        study_seed=7,
        block_count=config.blocks,
        budget=budget,
        portfolio_size=2,
        transition_opportunities=(1,),
        initial_candidate_id=content_hash(
            (ROOT / "common" / "initial_candidate.ir.json").read_text(encoding="utf-8")
        ),
        common_config_hash=content_hash(config.to_dict()),
        code_hash=content_hash({"module": "exploratory_pilot", "version": "1"}),
        environment_hash=content_hash({"training_profile": config.training_profile}),
        scientific=False,
    )


class _CudaExploratoryEvaluator:
    def __init__(self, *, spec: StudySpec, run_directory: Path, config: ExploratoryPilotConfig):
        self.spec = spec
        self.run_directory = run_directory
        self.config = config
        self.profile = get_training_profile(config.training_profile)

    def _evaluate(self, candidate_source: str, *, candidate_id: str, run_seed: int, stage: str) -> Any:
        candidate_dir = self.run_directory / "candidate_sources"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = candidate_dir / f"{candidate_id}.ir.json"
        if not candidate_path.exists():
            candidate_path.write_text(candidate_source, encoding="utf-8")
        return evaluate_candidate(
            candidate_path,
            training_profile=self.config.training_profile,
            training_seed=run_seed,
            training_output_dir=self.run_directory / "training" / stage,
            device="cuda",
            evaluation_profile=self.config.evaluation_profile,
            evaluation_case_count=self.config.evaluation_cases,
            eligibility_threshold=0.0,
            context=SearchEvaluationContext(
                study_id=self.spec.study_id,
                block_id=self.run_directory.name,
                run_id=self.run_directory.name,
                condition_id=stage,
            ),
        )

    def evaluate_seed(self, initial_candidate_id: str, run_seed: int) -> EvaluationResult:
        candidate = (ROOT / "common" / "initial_candidate.ir.json").read_text(encoding="utf-8")
        record = self._evaluate(candidate, candidate_id=initial_candidate_id, run_seed=run_seed, stage="seed")
        return _record_to_evaluation(record, profile=self.profile, cases=self.config.evaluation_cases)

    def evaluate_candidate(self, candidate_source: str, *, candidate_id: str, opportunity_index: int, run_seed: int) -> EvaluationResult:
        record = self._evaluate(candidate_source, candidate_id=candidate_id, run_seed=run_seed, stage=f"candidate_{opportunity_index}")
        return _record_to_evaluation(record, profile=self.profile, cases=self.config.evaluation_cases)


class _ExploratoryFakeEvaluator:
    """Provider-free fake with CUDA-shaped accounting, never real training."""

    def __init__(self) -> None:
        self._delegate = DeterministicFakeEvaluator()

    @staticmethod
    def _cuda(result: EvaluationResult) -> EvaluationResult:
        return EvaluationResult(
            outcome=result.outcome,
            score=result.score,
            training_attempts=result.training_attempts,
            training_steps=result.training_steps,
            training_examples=result.training_examples,
            accelerator_kind="cuda",
            accelerator_seconds=result.accelerator_seconds,
            evaluation_cases=result.evaluation_cases,
            infrastructure_retries=result.infrastructure_retries,
            failure_stage=result.failure_stage,
        )

    def evaluate_seed(self, initial_candidate_id: str, run_seed: int) -> EvaluationResult:
        return self._cuda(self._delegate.evaluate_seed(initial_candidate_id, run_seed))

    def evaluate_candidate(self, candidate_source: str, *, candidate_id: str, opportunity_index: int, run_seed: int) -> EvaluationResult:
        return self._cuda(
            self._delegate.evaluate_candidate(
                candidate_source,
                candidate_id=candidate_id,
                opportunity_index=opportunity_index,
                run_seed=run_seed,
            )
        )

def _record_to_evaluation(record: Any, *, profile: Any, cases: int) -> EvaluationResult:
    outcome = OpportunityOutcome.ACCEPTED if record.eligible_for_parent else OpportunityOutcome.REJECTED
    return EvaluationResult(
        outcome=outcome,
        score=float(record.search_score),
        training_attempts=1,
        training_steps=profile.max_steps,
        training_examples=profile.max_steps * profile.global_batch_size,
        accelerator_kind="cuda",
        accelerator_seconds=float(profile.maximum_wall_seconds),
        evaluation_cases=cases,
        failure_stage=str(record.failure_stage),
    )


class _OpenAIExploratoryGenerator:
    def __init__(self, *, spec: StudySpec, output_root: Path, config: ExploratoryPilotConfig):
        from openai import OpenAI

        api_key = os.environ.get("DISCOVERY_API_KEY")
        api_base = os.environ.get("DISCOVERY_API_BASE", "https://api.openai.com/v1")
        model = os.environ.get("DISCOVERY_MODEL", "gpt-5.6-sol")
        if not api_key:
            raise RuntimeError("DISCOVERY_API_KEY is required only for provider mode")
        endpoint = resolve_provider_endpoint(api_base, scientific=False)
        generation = GPT56SolProfile.resolve(
            model=model,
            seed=7,
            default_reasoning_effort="high",
            default_max_completion_tokens=16_384,
            default_timeout_seconds=180,
            default_retries=0,
            default_retry_delay_seconds=0,
            environ={},
            allow_environment_overrides=False,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=endpoint.base_url,
            timeout=generation.timeout_seconds,
            max_retries=0,
        )
        self.generation = generation
        self.spec = spec
        self.config = config
        self.output_root = output_root

    def generate(self, context: ProposalContext) -> ProposalResult:
        prompt = {
            "mode": MODE,
            "instruction": "Return only a complete Architecture IR JSON candidate; no Python and no markdown.",
            "condition": context.condition.to_dict(),
            "opportunity": context.opportunity_index,
            "parents": list(context.parent_ids),
            "transition_active": context.transition_active,
        }
        response = self.client.chat.completions.create(
            model=self.generation.model,
            messages=[
                {"role": "system", "content": "You are an exploratory architecture proposer."},
                {"role": "user", "content": json.dumps(prompt, sort_keys=True)},
            ],
            reasoning_effort=self.generation.reasoning_effort,
            max_completion_tokens=self.generation.max_completion_tokens,
            seed=self.generation.seed,
        )
        text = response.choices[0].message.content or ""
        candidate = None
        try:
            from study.runtime_adapters import canonicalize_architecture_ir

            candidate = canonicalize_architecture_ir(text, require_hypothesis=True)
        except Exception:
            candidate = None
        usage = response.usage
        return ProposalResult(
            response_text=text,
            candidate_source=candidate,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


def run_pilot(output_dir: str | Path, *, run_id: str, provider: bool = False, config_path: str | Path = PRESET_PATH) -> dict[str, Any]:
    config = load_config(config_path)
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    study_id = f"exploratory-{run_id}"
    spec = _build_spec(config, study_id)
    study_directory = output_root / spec.study_id
    plan = load_or_create_plan(spec, output_root=output_root, plan_path=study_directory / "randomization_plan.json")
    scheduler = SequentialAcceleratorScheduler(
        plan,
        state_path=study_directory / "schedule_state.json",
        lease_path=output_root / ".study_accelerator.lock",
        accelerator_kind="cuda",
    )
    while True:
        try:
            claim = scheduler.claim_next()
        except NoPendingRuns:
            break
        with claim as run:
            store = RunArtifactStore(
                run.execution_directory / "artifact_ledger",
                ArtifactContext(
                    study_id=spec.study_id,
                    block_id=run.block_id,
                    run_id=run.run_id,
                    condition_id=run.condition.condition_id.value,
                    writer_component="exploratory_pilot",
                    code_sha256=spec.code_hash,
                    config_sha256=spec.common_config_hash,
                    environment_sha256=spec.environment_hash,
                    run_seed=run.run_seed,
                    assignment_sha256=run.assignment_hash,
                ),
            )
            generator = _OpenAIExploratoryGenerator(spec=spec, output_root=output_root, config=config) if provider else DeterministicFakeGenerator()
            evaluator = _CudaExploratoryEvaluator(spec=spec, run_directory=run.execution_directory, config=config) if provider else _ExploratoryFakeEvaluator()
            ArtifactEmittingStudyEngine(
                study=spec,
                run=run,
                generator=generator,
                evaluator=evaluator,
                artifact_sink=ImmutableStudyEventSink(store),
                evaluation_lease_path=None,
                remote_call_id=run_id if provider else None,
                artifact_location=f"volume://rl4rl-architecture-artifacts/runs/{run.run_id}" if provider else None,
            ).execute()

    summaries = []
    for block in plan.blocks:
        for run in block.runs:
            state = read_json(run.execution_directory / "run_state.json")
            summaries.append({
                "run_id": run.run_id,
                "condition_id": run.condition.condition_id.value,
                "status": state["status"],
                "ledger": state["ledger"],
            })
    summary = {
        "schema_name": "ExploratoryModalPilotSummary",
        "schema_version": "1.0",
        "mode": MODE,
        "scientific": False,
        "provider_access": provider,
        "study_id": spec.study_id,
        "assignment_hash": plan.assignment_hash,
        "training_profile": config.training_profile,
        "evaluation_profile": config.evaluation_profile,
        "runs": summaries,
        "cost_ceiling_usd": {
            "modal": config.modal_cost_cap_usd,
            "provider": config.provider_cost_cap_usd,
        },
    }
    create_json_exclusive(study_directory / "run_summary.json", summary)
    create_json_exclusive(study_directory / "assignment_randomization_plan.json", plan.to_dict())
    create_json_exclusive(study_directory / "provider_attempt_ledger.json", {
        "schema_name": "ExploratoryProviderAttemptLedger",
        "schema_version": "1.0",
        "mode": MODE,
        "provider_access": provider,
        "attempts": sum(int(item["ledger"]["provider_attempts"]) for item in summaries),
        "responses_retained": False,
    })
    create_json_exclusive(study_directory / "candidate_ir_source_records.json", {
        "schema_name": "ExploratoryCandidateSourceIndex",
        "schema_version": "1.0",
        "mode": MODE,
        "source_hashes_only": True,
    })
    create_json_exclusive(study_directory / "training_evaluation_summaries.json", {
        "schema_name": "ExploratoryTrainingEvaluationSummary",
        "schema_version": "1.0",
        "mode": MODE,
        "training_profile": config.training_profile,
        "evaluation_profile": config.evaluation_profile,
        "cuda_only": True,
        "scientific": False,
    })
    create_json_exclusive(study_directory / "modal_terminal_receipt.json", {
        "schema_name": "ExploratoryModalTerminalReceipt",
        "schema_version": "1.0",
        "mode": MODE,
        "run_id": run_id,
        "status": "success",
        "provider_access": provider,
    })
    _safe_text_scan(output_root)
    artifact_run_id = spec.study_id
    manifest = build_artifact_manifest(study_directory, run_id=artifact_run_id, image_source_sha256=IMAGE_SOURCE_MANIFEST.manifest_sha256)
    write_artifact_manifest(study_directory, manifest)
    return {**summary, "artifact_manifest_sha256": manifest.manifest_sha256, "artifact_file_count": len(manifest.files)}


def verify_pilot_artifacts(run_directory: str | Path, *, run_id: str) -> dict[str, Any]:
    root = Path(run_directory).resolve()
    stored = root / "artifact_manifest.json"
    if not stored.is_file():
        raise ArtifactIntegrityError("exploratory artifact manifest is missing")
    stored_manifest = load_artifact_manifest(stored)
    if stored_manifest.run_id != run_id:
        raise ArtifactIntegrityError("exploratory artifact manifest run identity changed")
    return verify_artifact_manifest(root, stored_manifest)


def verify_exploratory_staging(root: str | Path, *, run_id: str) -> dict[str, Any]:
    """Validate the provider action's private staging tree before publication."""

    staging = Path(root).resolve()
    study_directory = staging / f"exploratory-{run_id}"
    summary_path = study_directory / "run_summary.json"
    if not summary_path.is_file():
        raise ArtifactIntegrityError("exploratory run summary is missing")
    summary = read_json(summary_path)
    if summary.get("mode") != MODE or summary.get("scientific") is not False:
        raise ArtifactIntegrityError("exploratory staging lost its non-scientific marker")
    return verify_pilot_artifacts(study_directory, run_id=study_directory.name)


def approval_text(*, action: str = "exploratory_c0c3_pilot", run_id: str, modal_cap: str = "0.25", provider_cap: str = "0.25") -> str:
    return (
        f"I approve exactly one provider-backed Modal {action} for run/cohort {run_id}, "
        f"with a ${modal_cap} Modal cap and ${provider_cap} provider cap, zero retries, "
        "provider access, and stop after first success or failure."
    )


def build_provider_approval_plan(
    *,
    source_tree_sha256: str,
    image_source_sha256: str,
    cohort_id: str,
) -> dict[str, Any]:
    """Build the small provider-cost plan used by the exploratory launcher."""

    unsigned = {
        "schema_name": "ExploratoryModalProviderApprovalPlan",
        "schema_version": "1",
        "action": "exploratory_c0c3_pilot",
        "source_tree_sha256": source_tree_sha256,
        "image_source_sha256": image_source_sha256,
        "cohort_id": cohort_id,
        "training_profile": "exploratory_train_cuda_v2",
        "provider_attempts": 4,
        "maximum_completion_tokens": 16_384,
        "retries": 0,
        "repair_attempts_per_opportunity": 1,
        "approval_plan_sha256_scope": "canonical_json_sha256_of_complete_payload_excluding_approval_plan_sha256",
    }
    return {**unsigned, "approval_plan_sha256": content_hash(unsigned)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exploratory Modal pilot tooling")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--run-id", required=True)
    preflight.add_argument("--config", default=str(PRESET_PATH))
    preflight.add_argument("--print-approval", action="store_true")
    plan = sub.add_parser("approval-plan")
    plan.add_argument("--source-tree-sha256", required=True)
    plan.add_argument("--image-source-sha256", required=True)
    plan.add_argument("--cohort-id", required=True)
    plan.add_argument("--output")
    fake = sub.add_parser("fake-run")
    fake.add_argument("--output-dir", required=True)
    fake.add_argument("--run-id", default="exploratory-local-fake-1")
    fake.add_argument("--config", default=str(PRESET_PATH))
    fake.add_argument("--provider", action="store_true")
    verify = sub.add_parser("verify")
    verify.add_argument("--run-directory", required=True)
    verify.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        config = load_config(args.config)
        result = {
            "mode": MODE,
            "run_id": args.run_id,
            "config": config.to_dict(),
            "modal_profile": os.environ.get("MODAL_PROFILE", "scalingintelligence"),
            "modal_environment": os.environ.get("MODAL_ENVIRONMENT", "main"),
            "provider_secret_read": False,
            "network_calls": 0,
            "approval": approval_text(run_id=args.run_id, modal_cap=config.modal_cost_cap_usd, provider_cap=config.provider_cost_cap_usd) if args.print_approval else None,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "approval-plan":
        payload = build_provider_approval_plan(
            source_tree_sha256=args.source_tree_sha256,
            image_source_sha256=args.image_source_sha256,
            cohort_id=args.cohort_id,
        )
        if args.output:
            create_json_exclusive(Path(args.output), payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.command == "fake-run":
        print(json.dumps(run_pilot(args.output_dir, run_id=args.run_id, provider=args.provider, config_path=args.config), indent=2, sort_keys=True))
        return 0
    print(json.dumps(verify_pilot_artifacts(args.run_directory, run_id=args.run_id), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
