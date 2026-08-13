"""Provider-free verification that a v2 resume advanced to completion.

The verifier reads one retained nonterminal checkpoint and the final artifacts
from the same training directory.  It does not train, import Modal, contact a
provider, or mutate any input.  Checkpoints are loaded on CPU with
``weights_only=True`` and the resulting evidence contains only artifact-root
relative paths and content hashes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from architecture_ir import validate_ir_candidate_json  # noqa: E402
from common.device import AcceleratorFingerprint  # noqa: E402
from common.runtime_context import ExecutionContextV1  # noqa: E402
from common.task_adapter import DEFAULT_TASK  # noqa: E402
from common.trainer import (  # noqa: E402
    ResumeMismatchError,
    _dependency_lock_hash,
    _validate_resume,
    sha256_file,
    trusted_component_set_sha256,
)
from common.training_config import (  # noqa: E402
    TrainingResult,
    TrainingSeedBundle,
    get_training_profile,
)
from scripts.validate_engineering_canaries import (  # noqa: E402
    _load_training_events,
    _strict_json_loads,
    _validate_cuda_checkpoints,
    _validate_cuda_manifest,
    _validate_cuda_rng_state,
    _validate_cuda_summary,
    _validate_json_security,
)
from study.serialization import create_json_exclusive  # noqa: E402

SCHEMA_NAME = "ResumeProgressionEvidence"
SCHEMA_VERSION = "1.0"
VERIFICATION_MODE = "provider_free_v2_artifact_progression"
MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
MAX_SUMMARY_BYTES = 2 * 1024 * 1024
MAX_EVENT_LOG_BYTES = 64 * 1024 * 1024

NONTERMINAL_CHECKPOINT_NAME = "partial_resume_checkpoint.pt"
RESUMED_CHECKPOINT_NAME = "latest_resume_checkpoint.pt"
SUMMARY_NAME = "training_summary.json"
EVENT_LOG_NAME = "training_events.jsonl"
CANDIDATE_NAME = "candidate_graph.json"
BEST_CHECKPOINT_NAME = "best_checkpoint.pt"
MANIFEST_NAME = "training_manifest.json"
RNG_ATTESTATION_NAME = "rng_restore_attestation.json"

_RESUME_TRAINING_FILE_POLICY = {
    BEST_CHECKPOINT_NAME: ("checkpoint", MAX_CHECKPOINT_BYTES),
    CANDIDATE_NAME: ("json", 16 * 1024 * 1024),
    RESUMED_CHECKPOINT_NAME: ("checkpoint", MAX_CHECKPOINT_BYTES),
    NONTERMINAL_CHECKPOINT_NAME: ("checkpoint", MAX_CHECKPOINT_BYTES),
    RNG_ATTESTATION_NAME: ("json", MAX_SUMMARY_BYTES),
    EVENT_LOG_NAME: ("jsonl", MAX_EVENT_LOG_BYTES),
    MANIFEST_NAME: ("json", MAX_SUMMARY_BYTES),
    SUMMARY_NAME: ("json", MAX_SUMMARY_BYTES),
}

_RNG_ATTESTATION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_checkpoint_sha256",
        "source_rng_state_sha256",
        "observed_post_restore_rng_state_sha256",
        "restored_exactly",
        "source_optimizer_step",
        "source_examples_processed",
        "final_checkpoint_sha256",
        "final_rng_state_sha256",
        "final_optimizer_step",
        "final_examples_processed",
        "rng_progressed",
        "execution_context",
    }
)

IDENTITY_FIELDS = (
    "checkpoint_kind",
    "candidate_source_hash",
    "profile_hash",
    "task_adapter_version",
    "task_adapter_hash",
    "seed_bundle",
    "seed_bundle_hash",
    "trusted_component_set_sha256",
    "dependency_lock_hash",
)

EVIDENCE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "verification_mode",
        "evidence_relative_path",
        "training_output_relative_path",
        "profile",
        "identity",
        "artifacts",
        "progression",
        "checks",
    }
)


class ResumeProgressionVerificationError(RuntimeError):
    """The persisted artifacts do not prove a valid resume progression."""


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ResumeProgressionVerificationError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _sha256_text(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ResumeProgressionVerificationError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _load_json_object(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise ResumeProgressionVerificationError(
            f"{path.name} exceeds its bounded verification size"
        )
    try:
        payload = _strict_json_loads(
            path.read_text(encoding="utf-8"),
            label=path.name,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ResumeProgressionVerificationError(
            f"{path.name} is not canonical JSON evidence"
        ) from error
    if not isinstance(payload, dict):
        raise ResumeProgressionVerificationError(
            f"{path.name} must contain one JSON object"
        )
    return payload


def _validate_training_roster(training: Path) -> None:
    entries = tuple(training.iterdir())
    observed = {entry.name for entry in entries}
    expected = set(_RESUME_TRAINING_FILE_POLICY)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ResumeProgressionVerificationError(
            "resume training artifact roster differs from the exact action policy "
            f"(missing={missing}, extra={extra})"
        )
    for entry in entries:
        kind, maximum_bytes = _RESUME_TRAINING_FILE_POLICY[entry.name]
        if entry.is_symlink() or not entry.is_file():
            raise ResumeProgressionVerificationError(
                f"resume artifact must be a regular file: {entry.name}"
            )
        size = entry.stat().st_size
        if size < 1 or size > maximum_bytes:
            raise ResumeProgressionVerificationError(
                f"resume artifact size violates policy: {entry.name}"
            )
        suffix = entry.suffix
        expected_suffix = {
            "checkpoint": ".pt",
            "json": ".json",
            "jsonl": ".jsonl",
        }[kind]
        if suffix != expected_suffix:
            raise ResumeProgressionVerificationError(
                f"resume artifact has an invalid file type: {entry.name}"
            )


def _portable_relative(root: Path, path: Path, field: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ResumeProgressionVerificationError(
            f"{field} must be contained by artifact_root"
        ) from error
    value = relative.as_posix() or "."
    logical = PurePosixPath(value)
    if logical.is_absolute() or ".." in logical.parts or "\\" in value:
        raise ResumeProgressionVerificationError(f"{field} is not portable")
    if PureWindowsPath(value).is_absolute():
        raise ResumeProgressionVerificationError(f"{field} is not portable")
    return value


def _regular_file(path: Path, *, maximum_bytes: int) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ResumeProgressionVerificationError(
            f"{path.name} must be a regular, non-symlink file"
        )
    size = path.stat().st_size
    if size < 1 or size > maximum_bytes:
        raise ResumeProgressionVerificationError(
            f"{path.name} is outside its bounded verification size"
        )
    return path.resolve()


def _artifact_record(root: Path, path: Path, digest: str) -> dict[str, Any]:
    return {
        "relative_path": _portable_relative(root, path, path.name),
        "sha256": digest,
        "size_bytes": path.stat().st_size,
    }


def _checkpoint_optimizer_step(
    checkpoint: dict[str, Any],
    *,
    label: str,
) -> tuple[int, int]:
    global_step = _exact_int(checkpoint.get("global_step"), f"{label}.global_step")
    optimizer = checkpoint.get("optimizer_state")
    if not isinstance(optimizer, dict):
        raise ResumeProgressionVerificationError(
            f"{label} optimizer_state must be an object"
        )
    states = optimizer.get("state")
    if not isinstance(states, dict) or not states:
        raise ResumeProgressionVerificationError(
            f"{label} optimizer_state contains no parameter steps"
        )
    observed_steps: list[int] = []
    for state in states.values():
        if not isinstance(state, dict) or "step" not in state:
            raise ResumeProgressionVerificationError(
                f"{label} optimizer parameter state lacks a step"
            )
        raw_step = state["step"]
        if isinstance(raw_step, torch.Tensor):
            if raw_step.numel() != 1:
                raise ResumeProgressionVerificationError(
                    f"{label} optimizer parameter step is not scalar"
                )
            raw_step = raw_step.item()
        if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
            raise ResumeProgressionVerificationError(
                f"{label} optimizer parameter step is not numeric"
            )
        numeric_step = float(raw_step)
        if not math.isfinite(numeric_step) or not numeric_step.is_integer():
            raise ResumeProgressionVerificationError(
                f"{label} optimizer parameter step is not a finite integer"
            )
        observed_steps.append(int(numeric_step))
    if any(step != global_step for step in observed_steps):
        raise ResumeProgressionVerificationError(
            f"{label} optimizer state does not match global_step"
        )
    scheduler = checkpoint.get("scheduler_state")
    if not isinstance(scheduler, dict):
        raise ResumeProgressionVerificationError(
            f"{label} scheduler_state must be an object"
        )
    scheduler_step = _exact_int(
        scheduler.get("last_epoch"),
        f"{label}.scheduler_state.last_epoch",
    )
    if scheduler_step != global_step:
        raise ResumeProgressionVerificationError(
            f"{label} scheduler state does not match global_step"
        )
    return global_step, len(observed_steps)


def _load_checkpoint(path: Path, *, label: str) -> dict[str, Any]:
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ResumeProgressionVerificationError(
            f"{label} could not be loaded as a weights-only checkpoint"
        ) from error
    if not isinstance(loaded, dict):
        raise ResumeProgressionVerificationError(f"{label} must be a mapping")
    return loaded


def _validate_resumed_manifest(
    *,
    artifact_root: Path,
    training: Path,
    manifest_payload: dict[str, Any],
    summary_payload: dict[str, Any],
    profile: Any,
    candidate_path: Path,
    best_checkpoint_path: Path,
    event_path: Path,
) -> tuple[TrainingSeedBundle, str, ExecutionContextV1]:
    try:
        candidate_validation = validate_ir_candidate_json(
            candidate_path.read_text(encoding="utf-8")
        )
        if not candidate_validation.valid or candidate_validation.graph_hash is None:
            raise ValueError("resumed candidate is not valid Architecture IR")
        _validate_cuda_summary(
            summary_payload,
            profile=profile,
            candidate_hash=sha256_file(candidate_path),
            checkpoint_path=best_checkpoint_path,
            event_path=event_path,
        )
        seeds, dependency_lock_hash = _validate_cuda_manifest(
            manifest_payload,
            output=training,
            profile=profile,
            candidate_hash=sha256_file(candidate_path),
            candidate_graph_hash=candidate_validation.graph_hash,
            summary=summary_payload,
            require_modal_context=True,
            expected_function="checkpoint_resume",
            outer_context_name="resume_execution_context.json",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ResumeProgressionVerificationError(
            f"resumed TrainingManifest/TrainingResult v2 is invalid: {error}"
        ) from error
    context_payload = manifest_payload["execution_context"]
    try:
        context = ExecutionContextV1.from_dict(context_payload)
    except ValueError as error:
        raise ResumeProgressionVerificationError(
            "resumed TrainingManifest execution context is invalid"
        ) from error
    if context.run_id != artifact_root.name:
        raise ResumeProgressionVerificationError(
            "resumed TrainingManifest run ID differs from artifact_root"
        )
    return seeds, dependency_lock_hash, context


def _validate_rng_attestation(
    payload: dict[str, Any],
    *,
    initial: dict[str, Any],
    latest: dict[str, Any],
    initial_checkpoint_sha256: str,
    latest_checkpoint_sha256: str,
    execution_context: ExecutionContextV1,
) -> dict[str, str]:
    try:
        _validate_json_security(payload, label="RNG restore attestation")
    except ValueError as error:
        raise ResumeProgressionVerificationError(str(error)) from error
    if set(payload) != _RNG_ATTESTATION_FIELDS:
        raise ResumeProgressionVerificationError(
            "RNG restore attestation fields differ from the exact v1 schema"
        )
    if (
        payload["schema_name"] != "RNGRestoreAttestation"
        or payload["schema_version"] != "1.0"
    ):
        raise ResumeProgressionVerificationError(
            "RNG restore attestation schema identity is invalid"
        )
    for field in (
        "source_checkpoint_sha256",
        "source_rng_state_sha256",
        "observed_post_restore_rng_state_sha256",
        "final_checkpoint_sha256",
        "final_rng_state_sha256",
    ):
        _sha256_text(payload[field], f"rng_restore_attestation.{field}")
    for field in ("restored_exactly", "rng_progressed"):
        if type(payload[field]) is not bool or payload[field] is not True:
            raise ResumeProgressionVerificationError(
                f"RNG restore attestation {field} must be exactly True"
            )
    initial_step = _exact_int(initial["global_step"], "initial.global_step")
    initial_examples = _exact_int(
        initial["examples_processed"],
        "initial.examples_processed",
    )
    latest_step = _exact_int(latest["global_step"], "latest.global_step")
    latest_examples = _exact_int(
        latest["examples_processed"],
        "latest.examples_processed",
    )
    expected_integers = {
        "source_optimizer_step": initial_step,
        "source_examples_processed": initial_examples,
        "final_optimizer_step": latest_step,
        "final_examples_processed": latest_examples,
    }
    for field, expected in expected_integers.items():
        if _exact_int(payload[field], f"rng_restore_attestation.{field}") != expected:
            raise ResumeProgressionVerificationError(
                f"RNG restore attestation {field} differs from its checkpoint"
            )
    try:
        source_rng_sha256 = _validate_cuda_rng_state(
            initial.get("rng_state"),
            field="nonterminal_checkpoint.rng_state",
        )
        final_rng_sha256 = _validate_cuda_rng_state(
            latest.get("rng_state"),
            field="resumed_latest_checkpoint.rng_state",
        )
    except ValueError as error:
        raise ResumeProgressionVerificationError(str(error)) from error
    expected_hashes = {
        "source_checkpoint_sha256": initial_checkpoint_sha256,
        "source_rng_state_sha256": source_rng_sha256,
        "observed_post_restore_rng_state_sha256": source_rng_sha256,
        "final_checkpoint_sha256": latest_checkpoint_sha256,
        "final_rng_state_sha256": final_rng_sha256,
    }
    for field, expected in expected_hashes.items():
        if payload[field] != expected:
            raise ResumeProgressionVerificationError(
                f"RNG restore attestation {field} differs from persisted evidence"
            )
    if source_rng_sha256 == final_rng_sha256:
        raise ResumeProgressionVerificationError(
            "resumed CUDA RNG state did not progress after restoration"
        )
    if payload["execution_context"] != execution_context.to_dict():
        raise ResumeProgressionVerificationError(
            "RNG restore attestation execution context differs from training"
        )
    return {
        "source_rng_state_sha256": source_rng_sha256,
        "observed_post_restore_rng_state_sha256": source_rng_sha256,
        "final_rng_state_sha256": final_rng_sha256,
    }


def _validate_summary(
    payload: dict[str, Any],
    *,
    profile: Any,
    candidate_hash: str,
    seeds: TrainingSeedBundle,
    latest: dict[str, Any],
    event_path: Path,
    best_checkpoint_path: Path,
) -> TrainingResult:
    expected_fields = set(TrainingResult.__dataclass_fields__)
    if set(payload) != expected_fields:
        raise ResumeProgressionVerificationError(
            "training summary fields differ from the exact v2 schema"
        )
    try:
        summary = TrainingResult.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise ResumeProgressionVerificationError(
            "training summary does not parse as TrainingResult v2"
        ) from error
    if summary.to_dict() != payload:
        raise ResumeProgressionVerificationError(
            "training summary uses coerced or noncanonical field values"
        )
    exact_booleans = {
        "success": True,
        "unsupported_operation_fallback": False,
        "scientific": profile.scientific,
        "hardware_matched": True,
        "cleanup_completed": True,
    }
    for field, expected in exact_booleans.items():
        if type(payload.get(field)) is not bool or payload[field] is not expected:
            raise ResumeProgressionVerificationError(
                f"training summary {field} must be exactly {expected}"
            )
    if payload.get("schema_name") != "TrainingResult":
        raise ResumeProgressionVerificationError(
            "training summary schema_name is invalid"
        )
    if payload.get("schema_version") != "2.0":
        raise ResumeProgressionVerificationError(
            "training summary is not portable schema v2"
        )
    if payload.get("failure_stage") != "" or payload.get("error") != "":
        raise ResumeProgressionVerificationError(
            "successful training summary contains failure information"
        )
    expected_text = {
        "profile_name": profile.name,
        "profile_version": profile.version,
        "profile_hash": profile.profile_hash,
        "candidate_source_hash": candidate_hash,
        "dtype": profile.dtype,
        "accelerator_kind": profile.device_requirement,
        "checkpoint_path": best_checkpoint_path.name,
        "event_log_path": event_path.name,
    }
    for field, expected in expected_text.items():
        if payload.get(field) != expected:
            raise ResumeProgressionVerificationError(
                f"training summary {field} differs from the resumed identity"
            )
    device = payload.get("device")
    if not isinstance(device, str) or device.split(":", 1)[0] != (
        profile.device_requirement
    ):
        raise ResumeProgressionVerificationError(
            "training summary selected an unexpected accelerator"
        )
    expected_integers = {
        "initialization_seed": seeds.model_initialization_seed,
        "data_seed": seeds.training_data_seed,
        "development_seed": seeds.development_set_seed,
        "dataloader_seed": seeds.dataloader_seed,
        "steps_completed": profile.max_steps,
        "examples_processed": profile.max_steps * profile.global_batch_size,
    }
    for field, expected in expected_integers.items():
        if _exact_int(payload.get(field), f"training_summary.{field}") != expected:
            raise ResumeProgressionVerificationError(
                f"training summary {field} differs from the profile maximum"
            )
    if payload["steps_completed"] != latest.get("global_step"):
        raise ResumeProgressionVerificationError(
            "training summary step differs from the resumed checkpoint"
        )
    if payload["examples_processed"] != latest.get("examples_processed"):
        raise ResumeProgressionVerificationError(
            "training summary examples differ from the resumed checkpoint"
        )
    best_step = _exact_int(
        payload.get("best_development_step"),
        "training_summary.best_development_step",
    )
    if best_step > profile.max_steps:
        raise ResumeProgressionVerificationError(
            "training summary best step exceeds the profile maximum"
        )
    _exact_int(
        payload.get("parameter_count_metadata"),
        "training_summary.parameter_count_metadata",
        minimum=1,
    )
    _sha256_text(payload.get("checkpoint_sha256"), "checkpoint_sha256")
    if sha256_file(best_checkpoint_path) != payload["checkpoint_sha256"]:
        raise ResumeProgressionVerificationError(
            "best checkpoint hash differs from the training summary"
        )
    try:
        fingerprint = AcceleratorFingerprint.from_dict(
            payload.get("accelerator_fingerprint")
        ).validate_cuda(exact_gpu_count=1, require_driver=True)
    except ValueError as error:
        raise ResumeProgressionVerificationError(
            f"training summary accelerator fingerprint is invalid: {error}"
        ) from error
    if fingerprint.selected_device != device:
        raise ResumeProgressionVerificationError(
            "training summary device differs from its accelerator fingerprint"
        )
    return summary


def _validate_events(
    path: Path,
    *,
    max_steps: int,
    global_batch_size: int,
    initial_step: int,
) -> dict[str, int]:
    profile = get_training_profile("smoke_train_cuda_v2")
    if (
        profile.max_steps != max_steps
        or profile.global_batch_size != global_batch_size
    ):
        raise ResumeProgressionVerificationError(
            "event verification arguments differ from the frozen CUDA smoke profile"
        )
    try:
        events = _load_training_events(path, profile=profile)
    except (OSError, TypeError, ValueError) as error:
        message = str(error).replace(
            "smoke optimizer-step sequence",
            "training optimizer-step event chain",
        )
        raise ResumeProgressionVerificationError(message) from error
    event_count = len(events)
    return {
        "event_count": event_count,
        "events_after_nonterminal_checkpoint": event_count - initial_step,
        "first_event_optimizer_step": 1,
        "last_event_optimizer_step": event_count,
    }


def _assert_portable_evidence(value: object) -> None:
    if isinstance(value, dict):
        for field, item in value.items():
            if field.endswith("path") or field.endswith("relative_path"):
                if not isinstance(item, str):
                    raise ResumeProgressionVerificationError(
                        f"evidence path field {field} is not text"
                    )
                if Path(item).is_absolute() or PureWindowsPath(item).is_absolute():
                    raise ResumeProgressionVerificationError(
                        f"evidence path field {field} is absolute"
                    )
            _assert_portable_evidence(item)
    elif isinstance(value, list):
        for item in value:
            _assert_portable_evidence(item)


def verify_resume_progression(
    *,
    artifact_root: str | Path,
    training_output_dir: str | Path,
    profile_name: str,
    run_seed: int,
    output_path: str | Path,
) -> dict[str, Any]:
    """Verify one retained nonterminal checkpoint advanced to the v2 maximum."""

    raw_root = Path(artifact_root).expanduser()
    raw_training = Path(training_output_dir).expanduser()
    raw_output = Path(output_path).expanduser()
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ResumeProgressionVerificationError(
            "artifact_root must be a regular, non-symlink directory"
        )
    if raw_training.is_symlink() or not raw_training.is_dir():
        raise ResumeProgressionVerificationError(
            "training_output_dir must be a regular, non-symlink directory"
        )
    root = raw_root.resolve()
    training = raw_training.resolve()
    _portable_relative(root, training, "training_output_dir")
    _validate_training_roster(training)

    paths = {
        "nonterminal_checkpoint": _regular_file(
            training / NONTERMINAL_CHECKPOINT_NAME,
            maximum_bytes=MAX_CHECKPOINT_BYTES,
        ),
        "resumed_latest_checkpoint": _regular_file(
            training / RESUMED_CHECKPOINT_NAME,
            maximum_bytes=MAX_CHECKPOINT_BYTES,
        ),
        "training_summary": _regular_file(
            training / SUMMARY_NAME,
            maximum_bytes=MAX_SUMMARY_BYTES,
        ),
        "training_events": _regular_file(
            training / EVENT_LOG_NAME,
            maximum_bytes=MAX_EVENT_LOG_BYTES,
        ),
        "candidate": _regular_file(
            training / CANDIDATE_NAME,
            maximum_bytes=16 * 1024 * 1024,
        ),
        "best_checkpoint": _regular_file(
            training / BEST_CHECKPOINT_NAME,
            maximum_bytes=MAX_CHECKPOINT_BYTES,
        ),
        "training_manifest": _regular_file(
            training / MANIFEST_NAME,
            maximum_bytes=MAX_SUMMARY_BYTES,
        ),
        "rng_restore_attestation": _regular_file(
            training / RNG_ATTESTATION_NAME,
            maximum_bytes=MAX_SUMMARY_BYTES,
        ),
    }
    for label, path in paths.items():
        _portable_relative(root, path, label)
    if paths["nonterminal_checkpoint"] == paths["resumed_latest_checkpoint"]:
        raise ResumeProgressionVerificationError(
            "nonterminal and resumed checkpoints must be distinct artifacts"
        )

    output = raw_output.resolve()
    _portable_relative(root, output, "output_path")
    if output in paths.values():
        raise ResumeProgressionVerificationError(
            "resume progression evidence may not replace an input artifact"
        )
    if output.exists() or output.is_symlink():
        raise ResumeProgressionVerificationError(
            "resume progression evidence destination already exists"
        )

    profile = get_training_profile(profile_name)
    if profile.version != "2":
        raise ResumeProgressionVerificationError(
            "resume progression verification requires a version-2 profile"
        )
    profile.validate()
    seeds = TrainingSeedBundle.from_run_seed(
        _exact_int(run_seed, "run_seed")
    )
    candidate_hash = sha256_file(paths["candidate"])
    expected_dependency_hash = _dependency_lock_hash()
    expected_component_hash = trusted_component_set_sha256()

    hashes_before = {
        label: sha256_file(path) for label, path in paths.items()
    }
    initial = _load_checkpoint(
        paths["nonterminal_checkpoint"],
        label="nonterminal checkpoint",
    )
    latest = _load_checkpoint(
        paths["resumed_latest_checkpoint"],
        label="resumed latest checkpoint",
    )
    validation_arguments = {
        "candidate_hash": candidate_hash,
        "profile": profile,
        "task": DEFAULT_TASK,
        "seeds": seeds,
        "trusted_component_set_hash": expected_component_hash,
        "dependency_lock_hash": expected_dependency_hash,
    }
    for label, checkpoint in (
        ("nonterminal checkpoint", initial),
        ("resumed latest checkpoint", latest),
    ):
        try:
            _validate_resume(checkpoint, **validation_arguments)
        except ResumeMismatchError as error:
            raise ResumeProgressionVerificationError(
                f"{label} failed its exact v2 identity contract"
            ) from error
    mismatched_identity = [
        field for field in IDENTITY_FIELDS if initial.get(field) != latest.get(field)
    ]
    if mismatched_identity:
        raise ResumeProgressionVerificationError(
            "checkpoint resume identity changed: " + ", ".join(mismatched_identity)
        )

    initial_step, initial_parameter_states = _checkpoint_optimizer_step(
        initial,
        label="nonterminal_checkpoint",
    )
    latest_step, latest_parameter_states = _checkpoint_optimizer_step(
        latest,
        label="resumed_latest_checkpoint",
    )
    initial_examples = _exact_int(
        initial.get("examples_processed"),
        "nonterminal_checkpoint.examples_processed",
    )
    latest_examples = _exact_int(
        latest.get("examples_processed"),
        "resumed_latest_checkpoint.examples_processed",
    )
    maximum_examples = profile.max_steps * profile.global_batch_size
    if not 0 < initial_step < profile.max_steps:
        raise ResumeProgressionVerificationError(
            "retained checkpoint is not a positive nonterminal optimizer step"
        )
    if latest_step != profile.max_steps or latest_step <= initial_step:
        raise ResumeProgressionVerificationError(
            "resumed optimizer did not advance to the profile maximum"
        )
    if initial_examples != initial_step * profile.global_batch_size:
        raise ResumeProgressionVerificationError(
            "nonterminal examples do not reconstruct from optimizer step"
        )
    if latest_examples != maximum_examples or latest_examples <= initial_examples:
        raise ResumeProgressionVerificationError(
            "resumed examples did not advance to the profile maximum"
        )
    if hashes_before["nonterminal_checkpoint"] == hashes_before[
        "resumed_latest_checkpoint"
    ]:
        raise ResumeProgressionVerificationError(
            "nonterminal and resumed checkpoint bytes are identical"
        )

    summary_payload = _load_json_object(
        paths["training_summary"],
        maximum_bytes=MAX_SUMMARY_BYTES,
    )
    manifest_payload = _load_json_object(
        paths["training_manifest"],
        maximum_bytes=MAX_SUMMARY_BYTES,
    )
    manifest_seeds, manifest_dependency_hash, execution_context = (
        _validate_resumed_manifest(
            artifact_root=root,
            training=training,
            manifest_payload=manifest_payload,
            summary_payload=summary_payload,
            profile=profile,
            candidate_path=paths["candidate"],
            best_checkpoint_path=paths["best_checkpoint"],
            event_path=paths["training_events"],
        )
    )
    if manifest_seeds != seeds:
        raise ResumeProgressionVerificationError(
            "resumed TrainingManifest seed bundle differs from run_seed"
        )
    if manifest_dependency_hash != expected_dependency_hash:
        raise ResumeProgressionVerificationError(
            "resumed TrainingManifest dependency lock differs from source"
        )
    best_checkpoint = _load_checkpoint(
        paths["best_checkpoint"],
        label="best checkpoint",
    )
    try:
        _validate_cuda_checkpoints(
            output=training,
            best=best_checkpoint,
            partial=initial,
            latest=latest,
            profile=profile,
            candidate_hash=candidate_hash,
            summary=summary_payload,
            seeds=seeds,
            dependency_lock_hash=manifest_dependency_hash,
        )
    except ValueError as error:
        raise ResumeProgressionVerificationError(
            f"resumed checkpoint schema/binding is invalid: {error}"
        ) from error
    _validate_summary(
        summary_payload,
        profile=profile,
        candidate_hash=candidate_hash,
        seeds=seeds,
        latest=latest,
        event_path=paths["training_events"],
        best_checkpoint_path=paths["best_checkpoint"],
    )
    event_progression = _validate_events(
        paths["training_events"],
        max_steps=profile.max_steps,
        global_batch_size=profile.global_batch_size,
        initial_step=initial_step,
    )
    rng_attestation_payload = _load_json_object(
        paths["rng_restore_attestation"],
        maximum_bytes=MAX_SUMMARY_BYTES,
    )
    rng_progression = _validate_rng_attestation(
        rng_attestation_payload,
        initial=initial,
        latest=latest,
        initial_checkpoint_sha256=hashes_before["nonterminal_checkpoint"],
        latest_checkpoint_sha256=hashes_before["resumed_latest_checkpoint"],
        execution_context=execution_context,
    )

    hashes_after = {label: sha256_file(path) for label, path in paths.items()}
    if hashes_after != hashes_before:
        changed = sorted(
            label
            for label in hashes_before
            if hashes_before[label] != hashes_after[label]
        )
        raise ResumeProgressionVerificationError(
            "input artifacts changed during verification: " + ", ".join(changed)
        )

    identity = {
        "candidate_source_hash": candidate_hash,
        "dependency_lock_hash": _sha256_text(
            initial.get("dependency_lock_hash"),
            "dependency_lock_hash",
        ),
        "profile_hash": _sha256_text(initial.get("profile_hash"), "profile_hash"),
        "seed_bundle_hash": _sha256_text(
            initial.get("seed_bundle_hash"),
            "seed_bundle_hash",
        ),
        "task_adapter_hash": _sha256_text(
            initial.get("task_adapter_hash"),
            "task_adapter_hash",
        ),
        "task_adapter_version": initial.get("task_adapter_version"),
        "trusted_component_set_sha256": _sha256_text(
            initial.get("trusted_component_set_sha256"),
            "trusted_component_set_sha256",
        ),
    }
    evidence = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "verification_mode": VERIFICATION_MODE,
        "evidence_relative_path": _portable_relative(root, output, "output_path"),
        "training_output_relative_path": _portable_relative(
            root, training, "training_output_dir"
        ),
        "profile": {
            "name": profile.name,
            "version": profile.version,
            "hash": profile.profile_hash,
            "max_optimizer_steps": profile.max_steps,
            "global_batch_size": profile.global_batch_size,
            "max_examples_processed": maximum_examples,
        },
        "identity": identity,
        "artifacts": {
            label: _artifact_record(root, path, hashes_before[label])
            for label, path in sorted(paths.items())
        },
        "progression": {
            "nonterminal_optimizer_step": initial_step,
            "nonterminal_examples_processed": initial_examples,
            "nonterminal_optimizer_parameter_states": initial_parameter_states,
            "resumed_optimizer_step": latest_step,
            "resumed_examples_processed": latest_examples,
            "resumed_optimizer_parameter_states": latest_parameter_states,
            **rng_progression,
            **event_progression,
        },
        "checks": {
            "all_inputs_unchanged": True,
            "checkpoint_identity_bound": True,
            "examples_advanced_to_profile_maximum": True,
            "event_chain_contiguous": True,
            "optimizer_advanced_to_profile_maximum": True,
            "portable_paths_only": True,
            "rng_progressed_after_restore": True,
            "rng_restore_hash_attested": True,
            "summary_bound_to_resumed_checkpoint": True,
            "weights_only_checkpoint_loading": True,
        },
    }
    if set(evidence) != EVIDENCE_FIELDS:
        raise AssertionError("resume progression evidence schema drifted")
    _assert_portable_evidence(evidence)
    try:
        create_json_exclusive(output, evidence)
    except FileExistsError as error:
        raise ResumeProgressionVerificationError(
            "resume progression evidence destination already exists"
        ) from error
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--training-output-dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        evidence = verify_resume_progression(
            artifact_root=args.artifact_root,
            training_output_dir=args.training_output_dir,
            profile_name=args.profile,
            run_seed=args.seed,
            output_path=args.output,
        )
    except (OSError, ValueError, ResumeProgressionVerificationError) as error:
        raise SystemExit(
            "resume-progression verification failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
