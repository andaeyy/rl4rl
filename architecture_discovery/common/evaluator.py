"""Controller-visible Layer A evaluation with evaluator-owned training.

This module must remain free of imports from ``private_eval`` and
``sealed_eval``. Sealed qualification is a post-run operation over a frozen
snapshot and lives behind the separate evaluation firewall.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from architecture_ir import probe_runtime_validity
from containment.audit import audit_runtime
from containment.policy import (
    CandidateFormat,
    GatePhase,
    ScientificExecutionRequest,
    assess_scientific_execution,
)
from evaluation.records import (
    SCHEMA_VERSION,
    ArtifactReference,
    ControllerSearchView,
    RecordEnvelope,
    SearchEvaluationRecord,
    content_sha256,
    search_evaluation_from_dict,
)

from common.candidate_artifact import (
    build_candidate_artifact,
    inspect_candidate_artifact,
)
from common.candidate_loader import load_candidate  # noqa: F401 - public compatibility
from common.descriptor_extractor import extract_descriptors, extract_ir_descriptors
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
from common.trainer import (
    _dependency_lock_hash,
    trusted_component_hashes,
    trusted_component_set_sha256,
    validate_training_request,
)
from common.training_client import WorkerError, run_worker_job
from common.training_config import (
    TrainingResult,
    TrainingSeedBundle,
    get_training_profile,
)

ROOT = Path(__file__).resolve().parents[1]
INFRASTRUCTURE_FAILURE_STAGES = {
    "checkpoint_write",
    "checkpoint_resume_mismatch",
    "containment_unproven",
    "device_unavailable",
    "accelerator_cleanup_failure",
    "cuda_unavailable",
    "cuda_driver_failure",
    "cuda_deterministic_kernel_unavailable",
    "modal_infrastructure_failure",
    "training_oom",
    "training_timeout",
    "unsupported_operation",
    "reproducibility_binding",
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
    def development(cls) -> SearchEvaluationContext:
        return cls(
            study_id="development-only",
            block_id="development-block",
            run_id=f"development-{uuid.uuid4().hex}",
            condition_id="development",
        )


class EvaluationBindingError(ValueError):
    """Raised when an evaluator result is stale or belongs to another request."""


def _atomic_json_artifact(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_training_artifact_path(
    relative_or_absolute: str,
    *,
    artifact_root: str | Path | None,
    expected_name: str,
) -> Path | None:
    if not relative_or_absolute:
        return None
    raw = Path(relative_or_absolute).expanduser()
    if raw.is_absolute():
        candidate = raw
    else:
        if artifact_root is None or len(raw.parts) != 1:
            return None
        candidate = Path(artifact_root).resolve() / raw
    if candidate.is_symlink():
        return None
    resolved = candidate.resolve()
    if resolved.name != expected_name:
        return None
    if artifact_root is not None:
        root = Path(artifact_root).resolve()
        if resolved.parent != root:
            return None
    return resolved


def _training_manifest_hash(
    training: TrainingResult,
    *,
    artifact_root: str | Path | None = None,
) -> str | None:
    """Return the colocated manifest hash without trusting arbitrary paths."""

    if not training.event_log_path:
        return None
    event_path = _resolve_training_artifact_path(
        training.event_log_path,
        artifact_root=artifact_root,
        expected_name="training_events.jsonl",
    )
    if event_path is None:
        return None
    manifest_path = event_path.parent / "training_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return None
    return file_hash(manifest_path)


def _validated_immutable_candidate(
    *,
    requested_candidate_path: str | Path,
    training: TrainingResult,
    seeds: TrainingSeedBundle,
    artifact_root: str | Path | None = None,
) -> tuple[Path, CandidateFormat, dict[str, str], str]:
    """Resolve and authenticate the exact artifact that produced a checkpoint.

    The caller-provided path is only an identity assertion.  Model construction
    always consumes the read-only copy created inside the training output.
    """

    requested_candidate = Path(requested_candidate_path).resolve()
    if not requested_candidate.is_file():
        raise EvaluationBindingError("requested candidate artifact is missing")
    requested_hash = file_hash(requested_candidate)
    if requested_hash != training.candidate_source_hash:
        raise EvaluationBindingError(
            "requested candidate hash does not match the training result"
        )

    if not training.event_log_path:
        raise EvaluationBindingError("training result is missing its event log path")
    event_path = _resolve_training_artifact_path(
        training.event_log_path,
        artifact_root=artifact_root,
        expected_name="training_events.jsonl",
    )
    if event_path is None or not event_path.is_file():
        raise EvaluationBindingError("training event log is missing or misnamed")
    output_dir = event_path.parent

    manifest_path = output_dir / "training_manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise EvaluationBindingError("training manifest is missing or is a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationBindingError("training manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise EvaluationBindingError("training manifest must contain an object")
    if manifest.get("candidate_source_hash") != training.candidate_source_hash:
        raise EvaluationBindingError("training manifest candidate hash mismatch")
    if manifest.get("candidate_artifact_hash") != training.candidate_source_hash:
        raise EvaluationBindingError("training manifest artifact hash mismatch")
    if manifest.get("profile_hash") != training.profile_hash:
        raise EvaluationBindingError("training manifest profile hash mismatch")
    if manifest.get("selected_device") != training.device:
        raise EvaluationBindingError("training manifest device mismatch")
    if manifest.get("task_adapter_version") != DEFAULT_TASK.version:
        raise EvaluationBindingError("training manifest task-adapter version mismatch")
    if manifest.get("task_adapter_hash") != DEFAULT_TASK.config_hash:
        raise EvaluationBindingError("training manifest task-adapter hash mismatch")
    if manifest.get("seed_bundle") != asdict(seeds):
        raise EvaluationBindingError("training manifest seed bundle mismatch")
    if manifest.get("seed_bundle_hash") != seeds.bundle_hash:
        raise EvaluationBindingError("training manifest seed-bundle hash mismatch")
    if (
        training.profile_version == "2"
        and manifest.get("dependency_lock_hash") != _dependency_lock_hash()
    ):
        raise EvaluationBindingError("training manifest dependency lock mismatch")

    try:
        candidate_format = CandidateFormat(str(manifest["candidate_format"]))
    except (KeyError, ValueError) as error:
        raise EvaluationBindingError(
            "training manifest candidate format is invalid"
        ) from error
    expected_copy_name = (
        "candidate_graph.json"
        if candidate_format is CandidateFormat.ARCHITECTURE_IR
        else "candidate_source.py"
    )
    if training.profile_version == "2":
        if manifest.get("schema_name") != "TrainingManifest":
            raise EvaluationBindingError("training manifest schema name mismatch")
        if manifest.get("schema_version") != "2.0":
            raise EvaluationBindingError("training manifest schema version mismatch")
        if manifest.get("candidate_path") != expected_copy_name:
            raise EvaluationBindingError(
                "training manifest candidate path is not portable"
            )
    if manifest.get("immutable_candidate_relative_path") != expected_copy_name:
        raise EvaluationBindingError(
            "training manifest immutable candidate path mismatch"
        )
    immutable_candidate = output_dir / expected_copy_name
    if immutable_candidate.is_symlink() or not immutable_candidate.is_file():
        raise EvaluationBindingError(
            "immutable training candidate is missing or is a symlink"
        )
    if immutable_candidate.resolve().parent != output_dir:
        raise EvaluationBindingError("immutable candidate escaped the training output")
    if file_hash(immutable_candidate) != training.candidate_source_hash:
        raise EvaluationBindingError("immutable training candidate hash mismatch")
    immutable_inspection = inspect_candidate_artifact(immutable_candidate)
    if not immutable_inspection.valid:
        raise EvaluationBindingError("immutable training candidate is no longer valid")
    if immutable_inspection.candidate_format is not candidate_format:
        raise EvaluationBindingError("immutable training candidate format mismatch")
    manifest_graph_hash = manifest.get("candidate_graph_hash")
    if immutable_inspection.graph_hash != manifest_graph_hash:
        raise EvaluationBindingError("immutable candidate graph hash mismatch")

    recorded_hashes = manifest.get("trusted_executable_component_hashes")
    if not isinstance(recorded_hashes, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in recorded_hashes.items()
    ):
        raise EvaluationBindingError(
            "training manifest trusted component hashes are invalid"
        )
    current_hashes = trusted_component_hashes()
    if recorded_hashes != current_hashes:
        raise EvaluationBindingError(
            "trusted executable components differ from the training run"
        )
    component_set_hash = trusted_component_set_sha256(current_hashes)
    if manifest.get("trusted_component_set_sha256") != component_set_hash:
        raise EvaluationBindingError("training manifest component-set hash mismatch")
    if manifest.get("controller_source_hash") != component_set_hash:
        raise EvaluationBindingError("legacy controller source hash is not bound")

    return immutable_candidate, candidate_format, current_hashes, component_set_hash


def validate_controller_view_binding(
    view: ControllerSearchView,
    *,
    candidate_source_hash: str,
    context: SearchEvaluationContext,
) -> None:
    """Validate the typed evaluator/controller trust boundary.

    Controllers must not make retention or parent-selection decisions from an
    evaluation produced for another source file, run, or condition.
    """

    if not isinstance(view, ControllerSearchView):
        raise EvaluationBindingError(
            "evaluator did not return a typed ControllerSearchView"
        )
    view.as_dict()
    expected_candidate_id = f"candidate-{candidate_source_hash}"
    mismatches: list[str] = []
    if view.schema_name != "search_evaluation" or view.schema_version != SCHEMA_VERSION:
        mismatches.append("search-evaluation schema")
    if view.candidate_id != expected_candidate_id:
        mismatches.append("candidate source hash")
    if view.run_id != context.run_id:
        mismatches.append("run_id")
    if view.condition_id != context.condition_id:
        mismatches.append("condition_id")
    if not view.record_id.strip():
        mismatches.append("record_id")
    for field_name in (
        "execution_ok",
        "transformer_valid",
        "eligible_for_parent",
        "infrastructure_failure",
    ):
        if type(getattr(view, field_name)) is not bool:
            mismatches.append(f"{field_name} type")
    for field_name in ("public_accuracy", "search_score"):
        value = getattr(view, field_name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            mismatches.append(f"{field_name} type")
        elif not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            mismatches.append(f"{field_name} range")
    if view.eligible_for_parent and not (
        view.execution_ok and view.transformer_valid
    ):
        mismatches.append("eligibility invariant")
    if mismatches:
        raise EvaluationBindingError(
            "evaluator result is not bound to this request: "
            + ", ".join(dict.fromkeys(mismatches))
        )


def _record_envelope(
    *,
    context: SearchEvaluationContext,
    training: TrainingResult,
    plan: EvaluationPlan,
    requested_device: str,
    artifact_root: str | Path | None = None,
) -> RecordEnvelope:
    component_hashes = trusted_component_hashes()
    component_set_hash = trusted_component_set_sha256(component_hashes)
    return RecordEnvelope.create(
        schema_name="search_evaluation",
        study_id=context.study_id,
        block_id=context.block_id,
        run_id=context.run_id,
        condition_id=context.condition_id,
        writer_component="common.evaluator",
        # This is the deterministic identity of every named trusted executable
        # component, not merely the evaluator source file.
        code_sha256=component_set_hash,
        config_sha256=content_sha256(
            {
                "candidate_source_hash": training.candidate_source_hash,
                "checkpoint_sha256": training.checkpoint_sha256,
                "training_profile_hash": training.profile_hash,
                "evaluation_plan_hash": plan.plan_hash,
                "training_manifest_sha256": _training_manifest_hash(
                    training, artifact_root=artifact_root
                ),
                "trusted_component_set_sha256": component_set_hash,
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
    artifact_root: str | Path | None = None,
) -> SearchEvaluationRecord:
    return SearchEvaluationRecord(
        envelope=_record_envelope(
            context=context,
            training=training,
            plan=plan,
            requested_device=requested_device,
            artifact_root=artifact_root,
        ),
        candidate_id=_candidate_id(training),
        training_record_id=_training_record_id(training),
        execution_ok=False,
        # A failed training job never reaches the evaluator-owned runtime
        # transformer-validity probe.  Static contract acceptance is not
        # runtime evidence and must not be reported as such.
        transformer_valid=False,
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
    artifact_root: str | Path | None = None,
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
            artifact_root=artifact_root,
        )

    execution_ok = False
    transformer_valid = False
    public_accuracy = 0.0
    failure_stage = ""
    descriptor_codes: tuple[tuple[str, float], ...] = ()
    runtime_validity_artifact: ArtifactReference | None = None
    verification_started = time.perf_counter()
    model: torch.nn.Module | None = None
    try:
        (
            immutable_candidate,
            recorded_candidate_format,
            _component_hashes,
            component_set_hash,
        ) = _validated_immutable_candidate(
            requested_candidate_path=candidate_path,
            training=training,
            seeds=seeds,
            artifact_root=artifact_root,
        )
        if (
            seeds.model_initialization_seed != training.initialization_seed
            or seeds.training_data_seed != training.data_seed
            or seeds.development_set_seed != training.development_seed
            or seeds.dataloader_seed != training.dataloader_seed
        ):
            raise EvaluationBindingError(
                "evaluation seed bundle does not match the training result"
            )
        profile = get_training_profile(training.profile_name)
        if profile.version != training.profile_version:
            raise EvaluationBindingError("training profile version mismatch")
        if profile.profile_hash != training.profile_hash:
            raise EvaluationBindingError("training profile hash mismatch")
        selection = resolve_training_device(
            profile,
            requested_device,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        device = selection.device
        event_parent = immutable_candidate.parent
        checkpoint_path = _resolve_training_artifact_path(
            training.checkpoint_path,
            artifact_root=artifact_root,
            expected_name="best_checkpoint.pt",
        )
        if (
            checkpoint_path is None
            or checkpoint_path != event_parent / "best_checkpoint.pt"
            or not checkpoint_path.is_file()
        ):
            raise EvaluationBindingError(
                "best checkpoint is missing, misnamed, symlinked, or outside its run"
            )
        if file_hash(checkpoint_path) != training.checkpoint_sha256:
            raise EvaluationBindingError("best checkpoint SHA-256 mismatch")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("best checkpoint must contain a mapping")
        expected_checkpoint_identity = {
            "checkpoint_kind": (
                "best_evaluation_weights_v1"
                if training.profile_version == "1"
                else "best_evaluation_weights_v2"
            ),
            "candidate_source_hash": training.candidate_source_hash,
            "profile_hash": training.profile_hash,
            "task_adapter_version": DEFAULT_TASK.version,
            "task_adapter_hash": DEFAULT_TASK.config_hash,
            "seed_bundle_hash": seeds.bundle_hash,
            "seed_bundle": asdict(seeds),
            "trusted_component_set_sha256": component_set_hash,
        }
        if training.profile_version == "2":
            expected_checkpoint_identity["dependency_lock_hash"] = (
                _dependency_lock_hash()
            )
        checkpoint_mismatches = {
            key: {"expected": expected, "observed": checkpoint.get(key)}
            for key, expected in expected_checkpoint_identity.items()
            if checkpoint.get(key) != expected
        }
        if checkpoint_mismatches:
            raise EvaluationBindingError(
                "best checkpoint provenance mismatch: "
                + json.dumps(checkpoint_mismatches, sort_keys=True)
            )

        # Only after all caller, run-directory, manifest, component, and
        # checkpoint identities agree may trusted construction consume bytes.
        built = build_candidate_artifact(
            immutable_candidate,
            seed=seeds.model_initialization_seed,
        )
        if built.candidate_format is not recorded_candidate_format:
            raise EvaluationBindingError("built candidate format differs from manifest")
        model = built.model
        transformer_valid = True
        if model is None:  # pragma: no cover - typed builder contract
            failure_stage = "candidate_contract"
            transformer_valid = False
        else:
            model_state = checkpoint.get("model_state")
            if not isinstance(model_state, dict):
                raise TypeError("best checkpoint is missing model_state")
            model.load_state_dict(model_state, strict=True)
            model = model.to(device=device, dtype=torch.float32)
            model.eval()
            execution_ok = True
            if built.candidate_format is CandidateFormat.ARCHITECTURE_IR:
                if built.runtime_bindings is None or built.graph is None:
                    raise ValueError(
                        "trusted IR interpreter omitted graph/runtime bindings"
                    )
                runtime_evidence = probe_runtime_validity(
                    model,
                    bindings=built.runtime_bindings,
                    token_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                    expected_device=device.type,
                )
                post_decision = assess_scientific_execution(
                    audit_runtime(),
                    ScientificExecutionRequest(
                        candidate_format=CandidateFormat.ARCHITECTURE_IR,
                        requested_device=requested_device,
                        required_accelerator=profile.device_requirement,
                        phase=GatePhase.POST_EXECUTION,
                        scientific=profile.scientific,
                        ir_validated=True,
                        trusted_ir_interpreter=True,
                        runtime_validity_passed=runtime_evidence.passed,
                        candidate_artifact_hash=training.candidate_source_hash,
                    ),
                )
                transformer_valid = runtime_evidence.passed and post_decision.allowed
                runtime_path = event_parent / "runtime_validity.json"
                runtime_payload = {
                    "candidate_artifact_hash": training.candidate_source_hash,
                    "candidate_graph_hash": built.graph.graph_hash,
                    "candidate_format": CandidateFormat.ARCHITECTURE_IR.value,
                    "checkpoint_sha256": training.checkpoint_sha256,
                    "training_profile_name": training.profile_name,
                    "training_profile_hash": training.profile_hash,
                    "seed_bundle": asdict(seeds),
                    "seed_bundle_hash": seeds.bundle_hash,
                    "requested_device": requested_device,
                    "selected_device": str(device),
                    "evaluation_plan_hash": evaluation_plan.plan_hash,
                    "training_manifest_sha256": _training_manifest_hash(
                        training, artifact_root=artifact_root
                    ),
                    "trusted_component_set_sha256": component_set_hash,
                    "runtime_evidence": runtime_evidence.to_dict(),
                    "post_execution_decision": post_decision.to_dict(),
                }
                _atomic_json_artifact(runtime_path, runtime_payload)
                runtime_validity_artifact = ArtifactReference(
                    layer=EvaluationLayer.SEARCH,
                    relative_path=runtime_path.name,
                    sha256=file_hash(runtime_path),
                )
                descriptors = extract_ir_descriptors(built.graph)
                if not transformer_valid:
                    failure_stage = "runtime_transformer_validity"
            else:
                if built.module is None:
                    raise ValueError("legacy Python candidate omitted loaded module")
                descriptors = extract_descriptors(built.module, model)
            descriptor_codes = tuple(sorted(descriptors.codes.items()))
            if transformer_valid:
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
                if public_accuracy < eligibility_threshold:
                    failure_stage = "public_accuracy"
    except EvaluationBindingError:
        execution_ok = False
        transformer_valid = False
        failure_stage = "reproducibility_binding"
    except Exception:
        execution_ok = False
        failure_stage = "post_training_evaluation"
    finally:
        if model is not None:
            del model
        gc.collect()
        try:
            cleanup_accelerator(torch.device(requested_device))
        except (RuntimeError, ValueError):
            execution_ok = False
            transformer_valid = False
            failure_stage = "accelerator_cleanup_failure"
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
        runtime_validity_artifact=runtime_validity_artifact,
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
    plan = resolve_evaluation_plan(
        profile_name,
        layer=EvaluationLayer.SEARCH,
        case_source_id=PUBLIC_LAYER_A_SOURCE_ID,
        case_source_sha256=PUBLIC_LAYER_A_SOURCE_SHA256,
        case_count=count,
        pi_decision_record_id=decision,
    )
    if plan.scientific != training_profile.scientific:
        raise ValueError(
            "training and evaluation profiles must both be scientific or both "
            "be engineering-only"
        )
    return plan


def preflight_candidate_evaluation(
    candidate_path: str | Path,
    *,
    training_profile: str,
    training_seed: int,
    training_output_dir: str | Path,
    device: str,
    allow_cpu_for_tests: bool,
    evaluation_profile: str | None = None,
    evaluation_case_count: int | None = None,
    pi_decision_record_id: str | None = None,
) -> dict[str, object]:
    """Validate a native-controller evaluation before it can spend an API call.

    This performs the same profile, Layer-A, candidate-contract, device, and
    containment checks used by training, but it does not create an output
    directory or execute candidate code.  Scientific arbitrary-Python runs
    therefore fail closed before proposal generation when the required
    containment evidence is absent.
    """

    profile = get_training_profile(training_profile)
    plan = _resolve_layer_a_plan(
        training_profile_name=profile.name,
        evaluation_profile=evaluation_profile,
        evaluation_case_count=evaluation_case_count,
        pi_decision_record_id=pi_decision_record_id,
    )
    seeds = TrainingSeedBundle.from_run_seed(int(training_seed))
    training_validation = validate_training_request(
        candidate_path=candidate_path,
        profile=profile,
        seeds=seeds,
        requested_device=device,
        allow_cpu_for_tests=allow_cpu_for_tests,
        output_dir=training_output_dir,
    )
    return {
        "training": training_validation,
        "evaluation": {
            "profile": plan.profile_name,
            "profile_version": plan.profile_version,
            "profile_hash": plan.profile_hash,
            "plan_hash": plan.plan_hash,
            "case_count": plan.case_count,
            "case_source_id": plan.case_source_id,
            "case_source_sha256": plan.case_source_sha256,
            "scientific": plan.scientific,
            "synthetic": plan.synthetic,
            "controller_visible": plan.controller_visible,
            "sealed": plan.sealed,
            "pi_decision_record_id": plan.pi_decision_record_id,
        },
    }


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
        or os.environ.get("DISCOVERY_TRAINING_PROFILE", "full_train_cuda_v2")
    )
    plan = _resolve_layer_a_plan(
        training_profile_name=profile.name,
        evaluation_profile=evaluation_profile,
        evaluation_case_count=evaluation_case_count,
        pi_decision_record_id=pi_decision_record_id,
    )
    resolved_context = context or SearchEvaluationContext.development()
    inspection = inspect_candidate_artifact(candidate)
    if not inspection.valid:
        failed = TrainingResult(
            success=False,
            failure_stage="candidate_contract",
            error="; ".join(inspection.reasons),
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
            accelerator_kind=device or profile.device_requirement,
            peak_accelerator_allocated_bytes=None,
            current_accelerator_allocated_bytes=None,
            reserved_accelerator_allocated_bytes=None,
            accelerator_total_memory_bytes=None,
            accelerator_fingerprint={},
            parameter_count_metadata=0,
            checkpoint_path="",
            checkpoint_sha256="",
            event_log_path="",
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=False,
            cleanup_completed=True,
            schema_version="1.0" if profile.version == "1" else "2.0",
        )
        return _training_failure(
            training=failed,
            context=resolved_context,
            plan=plan,
            requested_device=device or profile.device_requirement,
        )

    run_seed = (
        int(training_seed)
        if training_seed is not None
        else int(os.environ.get("DISCOVERY_TRAINING_SEED", "1"))
    )
    seeds = TrainingSeedBundle.from_run_seed(run_seed)
    requested_device = device or os.environ.get(
        "DISCOVERY_TRAIN_DEVICE", profile.device_requirement
    )
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
        error_text = (
            f"{type(error).__name__}: worker failed; details suppressed"
            if profile.version == "2"
            else f"{type(error).__name__}: {error}"[:2_000]
        )
        failed = TrainingResult(
            success=False,
            failure_stage="worker_infrastructure",
            error=error_text,
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
            accelerator_kind=requested_device,
            peak_accelerator_allocated_bytes=None,
            current_accelerator_allocated_bytes=None,
            reserved_accelerator_allocated_bytes=None,
            accelerator_total_memory_bytes=None,
            accelerator_fingerprint={},
            parameter_count_metadata=0,
            checkpoint_path="",
            checkpoint_sha256="",
            event_log_path="",
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=False,
            cleanup_completed=True,
            schema_version="1.0" if profile.version == "1" else "2.0",
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
        training = TrainingResult.from_dict(response["training"])
        return _training_failure(
            training=training,
            context=resolved_context,
            plan=plan,
            requested_device=requested_device,
            artifact_root=output_dir,
        )
    raise WorkerError(
        str(response.get("error", "candidate worker returned an invalid response"))
    )


def evaluation_plan_to_dict(plan: EvaluationPlan) -> dict[str, object]:
    payload = asdict(plan)
    payload["layer"] = plan.layer.value
    return payload
