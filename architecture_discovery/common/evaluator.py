"""Controller-visible Layer A evaluation with evaluator-owned training.

This module must remain free of imports from ``private_eval`` and
``sealed_eval``. Sealed qualification is a post-run operation over a frozen
snapshot and lives behind the separate evaluation firewall.
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from common.candidate_contract import inspect_candidate_path, validate_candidate
from common.candidate_loader import load_candidate
from common.descriptor_extractor import extract_descriptors
from common.device import cleanup_accelerator, resolve_training_device
from common.evaluation_profiles import (
    EvaluationLayer,
    EvaluationPlan,
    resolve_evaluation_plan,
)
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
    public_search_cases,
)
from common.task_adapter import DEFAULT_TASK
from common.training_client import WorkerError, run_worker_job
from common.training_config import TrainingResult, TrainingSeedBundle, get_training_profile
from evaluation.records import (
    RecordEnvelope,
    SearchEvaluationRecord,
    content_sha256,
    search_evaluation_from_dict,
)


ROOT = Path(__file__).resolve().parents[1]
INFRASTRUCTURE_FAILURE_STAGES = {
    "checkpoint_write",
    "checkpoint_resume_mismatch",
    "containment_unproven",
    "cuda_driver_runtime_failure",
    "cuda_unavailable",
    "device_unavailable",
    "training_oom",
    "training_timeout",
    "unsupported_operation",
    "worker_infrastructure",
}


@dataclass(frozen=True)
class SearchEvaluationContext:
    """Stable IDs needed by the versioned Layer A record envelope."""

    study_id: str
    block_id: str
    run_id: str
    condition_id: str

    @classmethod
    def development(cls) -> "SearchEvaluationContext":
        return cls(
            study_id="development-only",
            block_id="development-block",
            run_id=f"development-{uuid.uuid4().hex}",
            condition_id="development",
        )


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _record_envelope(
    *,
    context: SearchEvaluationContext,
    training: TrainingResult,
    plan: EvaluationPlan,
    requested_device: str,
) -> RecordEnvelope:
    return RecordEnvelope.create(
        schema_name="search_evaluation",
        study_id=context.study_id,
        block_id=context.block_id,
        run_id=context.run_id,
        condition_id=context.condition_id,
        writer_component="common.evaluator",
        code_sha256=file_hash(__file__),
        config_sha256=content_sha256(
            {
                "training_profile_hash": training.profile_hash,
                "evaluation_plan_hash": plan.plan_hash,
            }
        ),
        environment_sha256=content_sha256(
            {
                "requested_device": requested_device,
                "selected_device": training.device,
                "torch": torch.__version__,
            }
        ),
    )


def _training_record_id(training: TrainingResult) -> str:
    identity = content_sha256(
        {
            "candidate": training.candidate_source_hash,
            "profile": training.profile_hash,
            "initialization_seed": training.initialization_seed,
            "data_seed": training.data_seed,
            "checkpoint": training.checkpoint_sha256,
        }
    )
    return f"training-{identity}"


def _candidate_id(training: TrainingResult) -> str:
    source_hash = training.candidate_source_hash or ("0" * 64)
    return f"candidate-{source_hash}"


def _training_failure(
    *,
    training: TrainingResult,
    context: SearchEvaluationContext,
    plan: EvaluationPlan,
    requested_device: str,
) -> SearchEvaluationRecord:
    return SearchEvaluationRecord(
        envelope=_record_envelope(
            context=context,
            training=training,
            plan=plan,
            requested_device=requested_device,
        ),
        candidate_id=_candidate_id(training),
        training_record_id=_training_record_id(training),
        execution_ok=False,
        transformer_valid=training.failure_stage != "candidate_contract",
        public_accuracy=0.0,
        search_score=0.0,
        eligible_for_parent=False,
        failure_stage=training.failure_stage,
        infrastructure_failure=(
            training.failure_stage in INFRASTRUCTURE_FAILURE_STAGES
        ),
        parameter_count_metadata=max(0, training.parameter_count_metadata),
    )


def evaluate_trained_candidate_in_process(
    *,
    candidate_path: str | Path,
    training: TrainingResult,
    seeds: TrainingSeedBundle,
    requested_device: str,
    allow_cpu_for_tests: bool,
    evaluation_plan: EvaluationPlan,
    context: SearchEvaluationContext,
    eligibility_threshold: float,
) -> SearchEvaluationRecord:
    """Reload the public-development winner and evaluate public Layer A only."""

    evaluation_plan.validate()
    if evaluation_plan.layer is not EvaluationLayer.SEARCH:
        raise ValueError("online candidate evaluation requires a Layer A plan")
    if evaluation_plan.sealed or not evaluation_plan.controller_visible:
        raise ValueError("online candidate evaluation plan has invalid visibility")
    if not 0.0 <= eligibility_threshold <= 1.0:
        raise ValueError("eligibility threshold must be in [0, 1]")
    if not training.success:
        return _training_failure(
            training=training,
            context=context,
            plan=evaluation_plan,
            requested_device=requested_device,
        )

    execution_ok = False
    transformer_valid = False
    public_accuracy = 0.0
    failure_stage = ""
    descriptor_codes: tuple[tuple[str, float], ...] = ()
    verification_started = time.perf_counter()
    model: torch.nn.Module | None = None
    evaluation_device: torch.device | None = None
    try:
        profile = get_training_profile(training.profile_name)
        selection = resolve_training_device(
            profile,
            requested_device,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        device = selection.device
        evaluation_device = device
        module = load_candidate(candidate_path)
        built = module.build_untrained_model(seeds.model_initialization_seed)
        if not isinstance(built, tuple) or len(built) != 2:
            raise TypeError(
                "build_untrained_model(seed) must return (torch.nn.Module, metadata)"
            )
        model, _metadata = built
        contract = validate_candidate(module, model)
        transformer_valid = contract.valid
        if not contract.valid:
            failure_stage = "candidate_contract"
        else:
            checkpoint_path = Path(training.checkpoint_path).resolve()
            event_parent = Path(training.event_log_path).resolve().parent
            if checkpoint_path.parent != event_parent:
                raise ValueError("best checkpoint escaped its training output directory")
            if file_hash(checkpoint_path) != training.checkpoint_sha256:
                raise ValueError("best checkpoint SHA-256 mismatch")
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict):
                raise TypeError("best checkpoint must contain a mapping")
            if checkpoint.get("candidate_source_hash") != training.candidate_source_hash:
                raise ValueError("best checkpoint candidate hash mismatch")
            if checkpoint.get("profile_hash") != training.profile_hash:
                raise ValueError("best checkpoint profile hash mismatch")
            model_state = checkpoint.get("model_state")
            if not isinstance(model_state, dict):
                raise TypeError("best checkpoint is missing model_state")
            model.load_state_dict(model_state, strict=True)
            model = model.to(device=device, dtype=torch.float32)
            model.eval()
            execution_ok = True
            public_accuracy, _ = DEFAULT_TASK.exact_match(
                model,
                public_search_cases(evaluation_plan.case_count),
                device=device,
                batch_size=min(
                    profile.global_batch_size,
                    evaluation_plan.case_count,
                ),
                failure_limit=0,
            )
            descriptors = extract_descriptors(module, model)
            descriptor_codes = tuple(sorted(descriptors.codes.items()))
            if public_accuracy < eligibility_threshold:
                failure_stage = "public_accuracy"
    except Exception:
        execution_ok = False
        failure_stage = "post_training_evaluation"
    finally:
        if model is not None:
            del model
        gc.collect()
        if evaluation_device is not None:
            try:
                cleanup_accelerator(evaluation_device)
            except (AssertionError, RuntimeError):
                pass
        _ = time.perf_counter() - verification_started

    eligible = (
        execution_ok
        and transformer_valid
        and public_accuracy >= eligibility_threshold
    )
    return SearchEvaluationRecord(
        envelope=_record_envelope(
            context=context,
            training=training,
            plan=evaluation_plan,
            requested_device=requested_device,
        ),
        candidate_id=_candidate_id(training),
        training_record_id=_training_record_id(training),
        execution_ok=execution_ok,
        transformer_valid=transformer_valid,
        public_accuracy=public_accuracy,
        search_score=public_accuracy if transformer_valid else 0.0,
        eligible_for_parent=eligible,
        failure_stage=failure_stage,
        infrastructure_failure=(failure_stage in INFRASTRUCTURE_FAILURE_STAGES),
        parameter_count_metadata=max(0, training.parameter_count_metadata),
        online_descriptor_codes=descriptor_codes,
    )


def _resolve_layer_a_plan(
    *,
    training_profile_name: str,
    evaluation_profile: str | None,
    evaluation_case_count: int | None,
    pi_decision_record_id: str | None,
) -> EvaluationPlan:
    training_profile = get_training_profile(training_profile_name)
    profile_name = evaluation_profile or os.environ.get(
        "DISCOVERY_LAYER_A_PROFILE",
        "scientific_layer_a_v1" if training_profile.scientific else "smoke_eval_v1",
    )
    count = evaluation_case_count
    if count is None and os.environ.get("DISCOVERY_LAYER_A_CASES"):
        count = int(os.environ["DISCOVERY_LAYER_A_CASES"])
    decision = pi_decision_record_id or os.environ.get(
        "DISCOVERY_SCIENTIFIC_DECISION_RECORD"
    )
    return resolve_evaluation_plan(
        profile_name,
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=count,
        pi_decision_record_id=decision,
    )


def evaluate_candidate(
    candidate_path: str | Path,
    *,
    training_profile: str | None = None,
    training_seed: int | None = None,
    training_output_dir: str | Path | None = None,
    device: str | None = None,
    allow_cpu_for_tests: bool = False,
    resume: str | Path | None = None,
    evaluation_profile: str | None = None,
    evaluation_case_count: int | None = None,
    pi_decision_record_id: str | None = None,
    eligibility_threshold: float = 0.99,
    context: SearchEvaluationContext | None = None,
) -> SearchEvaluationRecord:
    """Train from scratch and return one strictly typed Layer A record."""

    candidate = Path(candidate_path).resolve()
    profile = get_training_profile(
        training_profile
        or os.environ.get("DISCOVERY_TRAINING_PROFILE", "full_train_v1")
    )
    plan = _resolve_layer_a_plan(
        training_profile_name=profile.name,
        evaluation_profile=evaluation_profile,
        evaluation_case_count=evaluation_case_count,
        pi_decision_record_id=pi_decision_record_id,
    )
    resolved_context = context or SearchEvaluationContext.development()
    source_contract = inspect_candidate_path(candidate)
    if not source_contract.valid:
        failed = TrainingResult(
            success=False,
            failure_stage="candidate_contract",
            error="; ".join(source_contract.reasons),
            profile_name=profile.name,
            profile_version=profile.version,
            profile_hash=profile.profile_hash,
            candidate_source_hash=file_hash(candidate) if candidate.is_file() else "",
            initialization_seed=0,
            data_seed=0,
            development_seed=0,
            dataloader_seed=0,
            device=device or "",
            dtype=profile.dtype,
            steps_completed=0,
            examples_processed=0,
            best_development_step=-1,
            best_development_exact_match_accuracy=0.0,
            best_development_loss=0.0,
            final_training_loss=0.0,
            train_seconds=0.0,
            peak_mps_allocated_bytes=None,
            current_mps_allocated_bytes=None,
            driver_mps_allocated_bytes=None,
            recommended_mps_memory_bytes=None,
            parameter_count_metadata=0,
            checkpoint_path="",
            checkpoint_sha256="",
            event_log_path="",
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=False,
            cleanup_completed=True,
        )
        return _training_failure(
            training=failed,
            context=resolved_context,
            plan=plan,
            requested_device=device or "mps",
        )

    run_seed = (
        int(training_seed)
        if training_seed is not None
        else int(os.environ.get("DISCOVERY_TRAINING_SEED", "1"))
    )
    seeds = TrainingSeedBundle.from_run_seed(run_seed)
    requested_device = device or os.environ.get("DISCOVERY_TRAIN_DEVICE", "mps")
    if training_output_dir is None:
        identifier = f"{file_hash(candidate)[:12]}_{uuid.uuid4().hex[:8]}"
        output_dir = ROOT / "outputs" / "candidate_training" / identifier
    else:
        output_dir = Path(training_output_dir).resolve()
    try:
        response = run_worker_job(
            mode="evaluate",
            candidate_path=candidate,
            output_dir=output_dir,
            profile=profile,
            seeds=seeds,
            requested_device=requested_device,
            allow_cpu_for_tests=allow_cpu_for_tests,
            resume=resume,
            evaluation_plan=evaluation_plan_to_dict(plan),
            evaluation_context=asdict(resolved_context),
            eligibility_threshold=eligibility_threshold,
        )
    except (WorkerError, OSError, ValueError) as error:
        failed = TrainingResult(
            success=False,
            failure_stage="worker_infrastructure",
            error=f"{type(error).__name__}: {error}"[:2_000],
            profile_name=profile.name,
            profile_version=profile.version,
            profile_hash=profile.profile_hash,
            candidate_source_hash=file_hash(candidate),
            initialization_seed=seeds.model_initialization_seed,
            data_seed=seeds.training_data_seed,
            development_seed=seeds.development_set_seed,
            dataloader_seed=seeds.dataloader_seed,
            device=requested_device,
            dtype=profile.dtype,
            steps_completed=0,
            examples_processed=0,
            best_development_step=-1,
            best_development_exact_match_accuracy=0.0,
            best_development_loss=0.0,
            final_training_loss=0.0,
            train_seconds=0.0,
            peak_mps_allocated_bytes=None,
            current_mps_allocated_bytes=None,
            driver_mps_allocated_bytes=None,
            recommended_mps_memory_bytes=None,
            parameter_count_metadata=0,
            checkpoint_path="",
            checkpoint_sha256="",
            event_log_path="",
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=False,
            cleanup_completed=True,
        )
        return _training_failure(
            training=failed,
            context=resolved_context,
            plan=plan,
            requested_device=requested_device,
        )
    if response.get("kind") == "search_evaluation":
        return search_evaluation_from_dict(response["evaluation"])
    if response.get("kind") == "training_result":
        training = TrainingResult(**response["training"])
        return _training_failure(
            training=training,
            context=resolved_context,
            plan=plan,
            requested_device=requested_device,
        )
    raise WorkerError(
        str(response.get("error", "candidate worker returned an invalid response"))
    )


def evaluation_plan_to_dict(plan: EvaluationPlan) -> dict[str, object]:
    payload = asdict(plan)
    payload["layer"] = plan.layer.value
    return payload
