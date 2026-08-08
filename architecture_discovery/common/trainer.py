"""Trusted evaluator-owned candidate optimization and checkpointing."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import random
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml

from containment.audit import audit_runtime
from containment.policy import (
    CandidateFormat,
    ScientificExecutionRequest,
    assess_scientific_execution,
)

from common.candidate_artifact import (
    build_candidate_artifact,
    inspect_candidate_artifact,
)
from common.device import (
    DeviceUnavailableError,
    mps_memory,
    resolve_training_device,
    synchronized_time,
    synchronize,
)
from common.task_adapter import DEFAULT_TASK, FixedAdditionTask
from common.training_config import (
    TrainingProfile,
    TrainingResult,
    TrainingSeedBundle,
)
from common.training_data import public_development_cases, training_batch


ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "outputs" / ".candidate_training.lock"


class OutputDirectoryError(RuntimeError):
    pass


class ResumeMismatchError(RuntimeError):
    pass


class TrainingTimeoutError(RuntimeError):
    pass


class TrainingNonfiniteError(RuntimeError):
    pass


class ContainmentGateError(RuntimeError):
    pass


class ReproducibilityBindingError(RuntimeError):
    """Trusted code or an immutable run artifact no longer matches its identity."""


def _enforce_training_wall_time(
    elapsed_seconds: float,
    profile: TrainingProfile,
    *,
    stage: str,
) -> None:
    """Fail closed whenever any evaluator-owned training work crosses its cap."""

    if elapsed_seconds >= profile.maximum_wall_seconds:
        raise TrainingTimeoutError(
            f"candidate exceeded {profile.maximum_wall_seconds}s wall-time cap "
            f"during {stage}"
        )


# Logical names, rather than absolute paths, are part of the stable component-set
# identity.  Keep this list explicit: adding executable code to any trusted model,
# training, or Layer-A evaluation path must be an intentional provenance change.
TRUSTED_EXECUTABLE_COMPONENT_PATHS: tuple[tuple[str, Path], ...] = (
    ("architecture_ir.__init__", ROOT / "architecture_ir" / "__init__.py"),
    ("architecture_ir.codec", ROOT / "architecture_ir" / "codec.py"),
    ("architecture_ir.graph", ROOT / "architecture_ir" / "graph.py"),
    ("architecture_ir.interpreter", ROOT / "architecture_ir" / "interpreter.py"),
    (
        "architecture_ir.runtime_evidence",
        ROOT / "architecture_ir" / "runtime_evidence.py",
    ),
    ("common.candidate_artifact", ROOT / "common" / "candidate_artifact.py"),
    ("common.candidate_contract", ROOT / "common" / "candidate_contract.py"),
    ("common.candidate_loader", ROOT / "common" / "candidate_loader.py"),
    ("common.descriptor_extractor", ROOT / "common" / "descriptor_extractor.py"),
    ("common.descriptor_schema", ROOT / "common" / "descriptor_schema.py"),
    ("common.device", ROOT / "common" / "device.py"),
    ("common.evaluation_profiles", ROOT / "common" / "evaluation_profiles.py"),
    ("common.evaluator", ROOT / "common" / "evaluator.py"),
    ("common.public_evaluation", ROOT / "common" / "public_evaluation.py"),
    ("common.task_adapter", ROOT / "common" / "task_adapter.py"),
    ("common.trainer", ROOT / "common" / "trainer.py"),
    ("common.training_client", ROOT / "common" / "training_client.py"),
    ("common.training_config", ROOT / "common" / "training_config.py"),
    ("common.training_data", ROOT / "common" / "training_data.py"),
    ("common.training_worker", ROOT / "common" / "training_worker.py"),
    ("containment.audit", ROOT / "containment" / "audit.py"),
    ("containment.policy", ROOT / "containment" / "policy.py"),
    ("containment.source_scan", ROOT / "containment" / "source_scan.py"),
    ("evaluation.records", ROOT / "evaluation" / "records.py"),
    (
        "scripts.training_worker_bootstrap",
        ROOT / "scripts" / "training_worker_bootstrap.py",
    ),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_source_copy(source: Path, destination: Path) -> None:
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(source.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
        destination.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return value


def _prepare_output_directory(output_dir: Path, resume: Path | None) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise OutputDirectoryError(
            f"output directory is non-empty: {output_dir}; pass --resume explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def _dependency_lock_hash() -> str:
    lock = ROOT / "uv.lock"
    return sha256_file(lock) if lock.exists() else ""


def trusted_component_hashes() -> dict[str, str]:
    """Return deterministic, reviewable hashes for the trusted execution path."""

    components: dict[str, str] = {}
    for logical_name, path in TRUSTED_EXECUTABLE_COMPONENT_PATHS:
        if logical_name in components:  # pragma: no cover - source invariant
            raise RuntimeError(f"duplicate trusted component name: {logical_name}")
        if not path.is_file():
            raise FileNotFoundError(f"trusted executable component is missing: {path}")
        components[logical_name] = sha256_file(path)
    return dict(sorted(components.items()))


def trusted_component_set_sha256(
    component_hashes: dict[str, str] | None = None,
) -> str:
    """Hash a named component set without depending on mapping insertion order."""

    components = (
        trusted_component_hashes()
        if component_hashes is None
        else dict(component_hashes)
    )
    normalized: dict[str, str] = {}
    for name, digest in sorted(components.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("trusted component names must be non-empty strings")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid SHA-256 for trusted component {name!r}")
        normalized[name] = digest
    encoded = json.dumps(
        {
            "schema": "trusted_executable_component_set_v1",
            "components": normalized,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def training_manifest(
    *,
    candidate_path: Path,
    candidate_hash: str,
    profile: TrainingProfile,
    seeds: TrainingSeedBundle,
    requested_device: str,
    selected_device: str | None,
    task: FixedAdditionTask,
    allow_cpu_for_tests: bool,
    containment_audit: dict[str, Any],
    containment_decision: dict[str, Any],
    candidate_format: CandidateFormat,
    candidate_graph_hash: str | None,
    component_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    mps_built = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    )
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    experiment_manifest_path = ROOT / "experiment_manifest.yaml"
    declared_machine: dict[str, Any] = {}
    if experiment_manifest_path.exists():
        declared_machine = (
            yaml.safe_load(experiment_manifest_path.read_text()).get("machine", {})
        )
    trusted_hashes = (
        trusted_component_hashes()
        if component_hashes is None
        else dict(sorted(component_hashes.items()))
    )
    trusted_set_hash = trusted_component_set_sha256(trusted_hashes)
    candidate_copy_name = (
        "candidate_graph.json"
        if candidate_format is CandidateFormat.ARCHITECTURE_IR
        else "candidate_source.py"
    )
    return {
        "created_at": _utc_now(),
        "candidate_path": str(candidate_path),
        "candidate_source_hash": candidate_hash,
        "candidate_artifact_hash": candidate_hash,
        "candidate_format": candidate_format.value,
        "candidate_graph_hash": candidate_graph_hash,
        "immutable_candidate_relative_path": candidate_copy_name,
        "candidate_initialization": "from_scratch",
        "profile": asdict(profile),
        "profile_hash": profile.profile_hash,
        "seed_bundle": asdict(seeds),
        "seed_bundle_hash": seeds.bundle_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "requested_device": requested_device,
        "selected_device": selected_device,
        "allow_cpu_for_tests": allow_cpu_for_tests,
        "hardware_matched_scientific_run": bool(
            profile.scientific and selected_device == "mps"
        ),
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "deterministic_algorithms": profile.deterministic_algorithms,
            "pytorch_enable_mps_fallback": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK", ""
            ),
            "mps_memory_fraction": profile.mps_memory_fraction,
            "declared_machine": declared_machine,
        },
        "dependency_lock_hash": _dependency_lock_hash(),
        "trusted_executable_component_hashes": trusted_hashes,
        "trusted_component_set_sha256": trusted_set_hash,
        # Backward-compatible name retained for existing evidence readers.  It
        # now identifies the complete named set above, not an opaque file concat.
        "controller_source_hash": trusted_set_hash,
        "parameter_count_role": "descriptive_metadata_only",
        "development_only_checkpoint_selection": profile.checkpoint_selection_rule,
        "scientific_limitations": (
            []
            if profile.scientific
            else [
                "Engineering only. Not valid for architecture ranking or "
                "scientific conclusions."
            ]
        ),
        "containment_audit": containment_audit,
        "containment_decision": containment_decision,
        "isolation_level": (
            "scientific_gate_allowed"
            if containment_decision["allowed"]
            and containment_decision["scientific"]
            else "engineering_only_or_scientific_gate_blocked"
        ),
        "reproducibility_note": (
            "PyTorch does not guarantee bitwise-identical results across releases, "
            "platforms, or devices even when all recorded seeds are fixed."
        ),
    }


def seed_everything(seed: int, *, deterministic: bool) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "keys": numpy_state[1].astype(np.uint32).tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
        try:
            state["torch_mps"] = torch.mps.get_rng_state()
        except RuntimeError:
            state["torch_mps"] = None
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, dict):
        raise ResumeMismatchError("resume checkpoint uses an unsafe legacy RNG state")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    mps_state = state.get("torch_mps")
    if mps_state is not None and hasattr(torch.mps, "set_rng_state"):
        torch.mps.set_rng_state(mps_state)


def _learning_rate_factor(step: int, profile: TrainingProfile) -> float:
    if profile.warmup_steps and step < profile.warmup_steps:
        return step / profile.warmup_steps
    progress = (step - profile.warmup_steps) / max(
        1, profile.max_steps - profile.warmup_steps
    )
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_is_better(
    *,
    accuracy: float,
    loss: float,
    step: int,
    best_accuracy: float,
    best_loss: float,
    best_step: int,
) -> bool:
    """Public-development-only deterministic checkpoint comparator."""

    if not math.isfinite(accuracy) or not math.isfinite(loss):
        return False
    if best_step < 0:
        return True
    return (accuracy, -loss, -step) > (best_accuracy, -best_loss, -best_step)


@torch.no_grad()
def evaluate_development(
    *,
    model: torch.nn.Module,
    task: FixedAdditionTask,
    cases: list[tuple[int, int]],
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset : offset + batch_size]
        input_ids, labels = task.collate(batch)
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        loss = task.teacher_forced_loss(model, input_ids, labels)
        total_loss += float(loss.detach().cpu()) * len(batch)
        total_examples += len(batch)
    exact_match, _ = task.exact_match(
        model,
        cases,
        device=device,
        batch_size=batch_size,
        failure_limit=0,
    )
    model.train(was_training)
    return total_loss / max(1, total_examples), exact_match


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    profile: TrainingProfile,
    candidate_hash: str,
    task: FixedAdditionTask,
    seeds: TrainingSeedBundle,
    step: int,
    examples_processed: int,
    elapsed_seconds: float,
    best_step: int,
    best_accuracy: float,
    best_loss: float,
    final_training_loss: float,
    trusted_component_set_hash: str | None = None,
) -> dict[str, Any]:
    component_set_hash = (
        trusted_component_set_sha256()
        if trusted_component_set_hash is None
        else trusted_component_set_hash
    )
    return {
        "checkpoint_kind": "trusted_resume_state_v1",
        "model_state": _cpu_copy(model.state_dict()),
        "optimizer_state": _cpu_copy(optimizer.state_dict()),
        "scheduler_state": _cpu_copy(scheduler.state_dict()),
        "global_step": step,
        "next_data_step": step,
        "examples_processed": examples_processed,
        "elapsed_seconds": elapsed_seconds,
        "best_development_step": best_step,
        "best_development_exact_match_accuracy": best_accuracy,
        "best_development_loss": best_loss,
        "final_training_loss": final_training_loss,
        "candidate_source_hash": candidate_hash,
        "profile_hash": profile.profile_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "seed_bundle": asdict(seeds),
        "seed_bundle_hash": seeds.bundle_hash,
        "trusted_component_set_sha256": component_set_hash,
        "rng_state": _rng_state(),
    }


def _best_evaluation_checkpoint_payload(
    *,
    model: torch.nn.Module,
    profile: TrainingProfile,
    candidate_hash: str,
    task: FixedAdditionTask,
    seeds: TrainingSeedBundle,
    step: int,
    examples_processed: int,
    best_accuracy: float,
    best_loss: float,
    trusted_component_set_hash: str | None = None,
) -> dict[str, Any]:
    """Plain tensor/primitives-only checkpoint safe for ``weights_only=True``."""

    component_set_hash = (
        trusted_component_set_sha256()
        if trusted_component_set_hash is None
        else trusted_component_set_hash
    )
    return {
        "checkpoint_kind": "best_evaluation_weights_v1",
        "model_state": _cpu_copy(model.state_dict()),
        "global_step": step,
        "examples_processed": examples_processed,
        "best_development_exact_match_accuracy": best_accuracy,
        "best_development_loss": best_loss,
        "candidate_source_hash": candidate_hash,
        "profile_hash": profile.profile_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "seed_bundle": asdict(seeds),
        "seed_bundle_hash": seeds.bundle_hash,
        "trusted_component_set_sha256": component_set_hash,
    }


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    candidate_hash: str,
    profile: TrainingProfile,
    task: FixedAdditionTask,
    seeds: TrainingSeedBundle,
    trusted_component_set_hash: str | None = None,
) -> None:
    expected_component_set_hash = (
        trusted_component_set_sha256()
        if trusted_component_set_hash is None
        else trusted_component_set_hash
    )
    expected = {
        "checkpoint_kind": "trusted_resume_state_v1",
        "candidate_source_hash": candidate_hash,
        "profile_hash": profile.profile_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "seed_bundle_hash": seeds.bundle_hash,
        "trusted_component_set_sha256": expected_component_set_hash,
    }
    mismatches = {
        key: {"expected": value, "observed": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    if mismatches:
        raise ResumeMismatchError(
            "resume checkpoint does not match this candidate/profile/task/seeds: "
            + json.dumps(mismatches, sort_keys=True)
        )
    required_fields = {
        "checkpoint_kind",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "global_step",
        "next_data_step",
        "examples_processed",
        "elapsed_seconds",
        "best_development_step",
        "best_development_exact_match_accuracy",
        "best_development_loss",
        "final_training_loss",
        "candidate_source_hash",
        "profile_hash",
        "task_adapter_version",
        "task_adapter_hash",
        "seed_bundle",
        "seed_bundle_hash",
        "trusted_component_set_sha256",
        "rng_state",
    }
    if set(checkpoint) != required_fields:
        raise ResumeMismatchError("resume checkpoint fields differ from trusted v1")
    for field in (
        "global_step",
        "next_data_step",
        "examples_processed",
        "best_development_step",
    ):
        value = checkpoint[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ResumeMismatchError(f"resume {field} must be a non-negative integer")
    step = checkpoint["global_step"]
    if checkpoint["next_data_step"] != step:
        raise ResumeMismatchError("resume data position is inconsistent")
    if step > profile.max_steps:
        raise ResumeMismatchError("resume step exceeds the frozen training profile")
    if checkpoint["examples_processed"] != step * profile.global_batch_size:
        raise ResumeMismatchError("resume example count does not reconstruct")
    if checkpoint["best_development_step"] > step:
        raise ResumeMismatchError("resume best step occurs after the current step")
    for field in (
        "elapsed_seconds",
        "best_development_exact_match_accuracy",
        "best_development_loss",
        "final_training_loss",
    ):
        value = checkpoint[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ResumeMismatchError(f"resume {field} must be finite numeric data")
    if checkpoint["elapsed_seconds"] < 0:
        raise ResumeMismatchError("resume elapsed time cannot be negative")
    if checkpoint["seed_bundle"] != asdict(seeds):
        raise ResumeMismatchError("resume seed bundle differs from its frozen hash")


@contextmanager
def exclusive_training_lock() -> Iterator[None]:
    import fcntl

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def validate_training_request(
    *,
    candidate_path: str | Path,
    profile: TrainingProfile,
    seeds: TrainingSeedBundle,
    requested_device: str,
    allow_cpu_for_tests: bool,
    output_dir: str | Path | None = None,
    resume: str | Path | None = None,
    task: FixedAdditionTask = DEFAULT_TASK,
) -> dict[str, Any]:
    candidate = Path(candidate_path).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"candidate does not exist: {candidate}")
    inspection = inspect_candidate_artifact(candidate)
    if not inspection.valid:
        raise ValueError(
            "candidate contract failed: " + "; ".join(inspection.reasons)
        )
    profile.validate()
    selection = resolve_training_device(
        profile,
        requested_device,
        allow_cpu_for_tests=allow_cpu_for_tests,
    )
    containment_audit = audit_runtime()
    containment_decision = assess_scientific_execution(
        containment_audit,
        ScientificExecutionRequest(
            candidate_format=inspection.candidate_format,
            requested_device=requested_device,
            scientific=profile.scientific,
            ir_validated=(
                inspection.candidate_format is CandidateFormat.ARCHITECTURE_IR
            ),
            trusted_ir_interpreter=(
                inspection.candidate_format is CandidateFormat.ARCHITECTURE_IR
            ),
            candidate_artifact_hash=sha256_file(candidate),
        ),
    )
    if not containment_decision.allowed:
        raise ContainmentGateError("; ".join(containment_decision.blockers))
    resolved_output: Path | None = None
    if output_dir is not None:
        raw_output = Path(output_dir).expanduser()
        if raw_output.is_symlink():
            raise OutputDirectoryError(
                f"output directory may not be a symlink: {raw_output}"
            )
        resolved_output = raw_output.resolve()
        resume_path = Path(resume).resolve() if resume else None
        if (
            resolved_output.exists()
            and any(resolved_output.iterdir())
            and resume_path is None
        ):
            raise OutputDirectoryError(
                f"output directory is non-empty: {resolved_output}; "
                "pass --resume explicitly"
            )
        if resume_path is not None:
            if (
                resume_path.parent != resolved_output
                or resume_path.name != "latest_resume_checkpoint.pt"
            ):
                raise ResumeMismatchError(
                    "--resume must reference this output directory's "
                    "latest_resume_checkpoint.pt"
                )
            checkpoint = torch.load(
                resume_path, map_location="cpu", weights_only=True
            )
            _validate_resume(
                checkpoint,
                candidate_hash=sha256_file(candidate),
                profile=profile,
                task=task,
                seeds=seeds,
            )
    component_hashes = trusted_component_hashes()
    return {
        "candidate": str(candidate),
        "candidate_source_hash": sha256_file(candidate),
        "candidate_artifact_hash": sha256_file(candidate),
        "candidate_format": inspection.candidate_format.value,
        "candidate_graph_hash": inspection.graph_hash,
        "profile_name": profile.name,
        "profile_version": profile.version,
        "profile_hash": profile.profile_hash,
        "seed_bundle": asdict(seeds),
        "seed_bundle_hash": seeds.bundle_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "device": str(selection.device),
        "scientific": profile.scientific,
        "hardware_matched": selection.hardware_matched,
        "containment_audit_hash": containment_audit.audit_hash,
        "containment_decision": containment_decision.to_dict(),
        "trusted_executable_component_hashes": component_hashes,
        "trusted_component_set_sha256": trusted_component_set_sha256(
            component_hashes
        ),
        "output_dir": str(resolved_output) if resolved_output else None,
        "resume": str(Path(resume).resolve()) if resume else None,
    }


def _failure_stage(error: BaseException) -> str:
    if isinstance(error, ReproducibilityBindingError):
        return "reproducibility_binding"
    if isinstance(error, ContainmentGateError):
        return "containment_unproven"
    if isinstance(error, DeviceUnavailableError):
        return "device_unavailable"
    if isinstance(error, ResumeMismatchError):
        return "checkpoint_resume_mismatch"
    if isinstance(error, OutputDirectoryError):
        return "checkpoint_write"
    if isinstance(error, TrainingTimeoutError):
        return "training_timeout"
    if isinstance(error, TrainingNonfiniteError):
        return "training_nonfinite"
    if isinstance(error, MemoryError) or (
        isinstance(error, RuntimeError)
        and "out of memory" in str(error).lower()
    ):
        return "training_oom"
    if isinstance(error, OSError):
        return "checkpoint_write"
    if isinstance(error, RuntimeError) and (
        "deterministic" in str(error).lower()
        or "not implemented for" in str(error).lower()
    ):
        return "unsupported_operation"
    return "model_initialization"


def train_candidate_in_process(
    *,
    candidate_path: str | Path,
    output_dir: str | Path,
    profile: TrainingProfile,
    seeds: TrainingSeedBundle,
    requested_device: str,
    allow_cpu_for_tests: bool,
    resume: str | Path | None = None,
    task: FixedAdditionTask = DEFAULT_TASK,
) -> TrainingResult:
    """Train one candidate. Call only inside the sanitized worker process."""

    candidate = Path(candidate_path).resolve()
    raw_destination = Path(output_dir).expanduser()
    destination_is_symlink = raw_destination.is_symlink()
    destination = raw_destination.resolve()
    resume_path = Path(resume).resolve() if resume else None
    candidate_hash = sha256_file(candidate) if candidate.is_file() else ""
    event_path = destination / "training_events.jsonl"
    best_path = destination / "best_checkpoint.pt"
    latest_path = destination / "latest_resume_checkpoint.pt"
    started = time.perf_counter()
    prior_elapsed = 0.0
    steps_completed = 0
    examples_processed = 0
    best_step = -1
    best_accuracy = 0.0
    best_loss = float("inf")
    final_loss = float("nan")
    parameter_count = 0
    selected_device = requested_device
    peak_mps = 0
    current_mps: int | None = None
    driver_mps: int | None = None
    recommended_mps: int | None = None
    checkpoint_hash = ""
    cleanup_completed = False
    output_prepared = False
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    result: TrainingResult | None = None
    current_stage = "candidate_contract"
    candidate_format = CandidateFormat.ARBITRARY_PYTHON
    candidate_graph_hash: str | None = None
    component_hashes = trusted_component_hashes()
    component_set_hash = trusted_component_set_sha256(component_hashes)

    try:
        if destination_is_symlink:
            raise OutputDirectoryError(
                f"output directory may not be a symlink: {raw_destination}"
            )
        _prepare_output_directory(destination, resume_path)
        output_prepared = True
        inspection = inspect_candidate_artifact(candidate)
        if not inspection.valid:
            raise ValueError(
                "candidate contract failed: " + "; ".join(inspection.reasons)
            )
        candidate_format = inspection.candidate_format
        candidate_graph_hash = inspection.graph_hash
        profile.validate()

        containment_audit = audit_runtime()
        containment_decision = assess_scientific_execution(
            containment_audit,
            ScientificExecutionRequest(
                candidate_format=candidate_format,
                requested_device=requested_device,
                scientific=profile.scientific,
                ir_validated=(
                    candidate_format is CandidateFormat.ARCHITECTURE_IR
                ),
                trusted_ir_interpreter=(
                    candidate_format is CandidateFormat.ARCHITECTURE_IR
                ),
                candidate_artifact_hash=candidate_hash,
            ),
        )

        manifest = training_manifest(
            candidate_path=candidate,
            candidate_hash=candidate_hash,
            profile=profile,
            seeds=seeds,
            requested_device=requested_device,
            selected_device=None,
            task=task,
            allow_cpu_for_tests=allow_cpu_for_tests,
            containment_audit=containment_audit.to_dict(),
            containment_decision=containment_decision.to_dict(),
            candidate_format=candidate_format,
            candidate_graph_hash=candidate_graph_hash,
            component_hashes=component_hashes,
        )
        _atomic_json(destination / "training_manifest.json", manifest)
        event_path.touch(exist_ok=True)
        if not containment_decision.allowed:
            raise ContainmentGateError(
                "; ".join(containment_decision.blockers)
            )

        resume_state: dict[str, Any] | None = None
        if resume_path is not None:
            if (
                resume_path.parent != destination
                or resume_path.name != "latest_resume_checkpoint.pt"
            ):
                raise ResumeMismatchError(
                    "--resume must reference this output directory's "
                    "latest_resume_checkpoint.pt"
                )
            resume_state = torch.load(
                resume_path, map_location="cpu", weights_only=True
            )
            _validate_resume(
                resume_state,
                candidate_hash=candidate_hash,
                profile=profile,
                task=task,
                seeds=seeds,
                trusted_component_set_hash=component_set_hash,
            )

        candidate_copy = destination / (
            "candidate_graph.json"
            if candidate_format is CandidateFormat.ARCHITECTURE_IR
            else "candidate_source.py"
        )
        if candidate_copy.exists():
            if sha256_file(candidate_copy) != candidate_hash:
                raise ResumeMismatchError(
                    "stored candidate source differs from requested candidate"
                )
        else:
            _atomic_source_copy(candidate, candidate_copy)

        selection = resolve_training_device(
            profile,
            requested_device,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        device = selection.device
        selected_device = str(device)
        manifest = training_manifest(
            candidate_path=candidate,
            candidate_hash=candidate_hash,
            profile=profile,
            seeds=seeds,
            requested_device=requested_device,
            selected_device=selected_device,
            task=task,
            allow_cpu_for_tests=allow_cpu_for_tests,
            containment_audit=containment_audit.to_dict(),
            containment_decision=containment_decision.to_dict(),
            candidate_format=candidate_format,
            candidate_graph_hash=candidate_graph_hash,
            component_hashes=component_hashes,
        )
        _atomic_json(destination / "training_manifest.json", manifest)

        dataloader_generator = seed_everything(
            seeds.model_initialization_seed,
            deterministic=profile.deterministic_algorithms,
        )
        dataloader_generator.manual_seed(seeds.dataloader_seed)
        current_stage = "model_initialization"
        built = build_candidate_artifact(
            candidate_copy,
            seed=seeds.model_initialization_seed,
        )
        model = built.model
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model = model.to(device=device, dtype=torch.float32)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=profile.peak_learning_rate,
            betas=profile.adamw_betas,
            weight_decay=profile.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: _learning_rate_factor(step, profile)
        )

        if resume_state is not None:
            model.load_state_dict(resume_state["model_state"])
            optimizer.load_state_dict(resume_state["optimizer_state"])
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_state[key] = value.to(device)
            scheduler.load_state_dict(resume_state["scheduler_state"])
            steps_completed = resume_state["global_step"]
            examples_processed = resume_state["examples_processed"]
            prior_elapsed = float(resume_state.get("elapsed_seconds", 0.0))
            best_step = resume_state["best_development_step"]
            best_accuracy = float(
                resume_state["best_development_exact_match_accuracy"]
            )
            best_loss = float(resume_state["best_development_loss"])
            final_loss = float(
                resume_state.get("final_training_loss", best_loss)
            )
            _restore_rng_state(resume_state["rng_state"])

        development_cases = public_development_cases(
            seeds.development_set_seed, profile.validation_examples
        )
        development_set = set(development_cases)
        microbatch = profile.microbatch_size or profile.global_batch_size
        trajectory_started = synchronized_time(device)
        current_stage = "training_execution"

        while steps_completed < profile.max_steps:
            elapsed = prior_elapsed + (time.perf_counter() - trajectory_started)
            _enforce_training_wall_time(elapsed, profile, stage="training_execution")

            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = 0.0
            for accumulation_index in range(profile.gradient_accumulation_steps):
                input_ids, labels, _ = training_batch(
                    task=task,
                    data_seed=seeds.training_data_seed,
                    optimizer_step=steps_completed,
                    batch_size=microbatch,
                    example_offset=accumulation_index * microbatch,
                    min_digits=profile.min_operand_digits,
                    max_digits=profile.max_operand_digits,
                    excluded_cases=development_set,
                )
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                loss = task.teacher_forced_loss(model, input_ids, labels)
                if not torch.isfinite(loss):
                    raise TrainingNonfiniteError(
                        f"non-finite loss at optimizer step {steps_completed}"
                    )
                scaled_loss = loss / profile.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_loss += float(loss.detach().cpu())

            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                model.parameters(), profile.gradient_clip_norm
            )
            grad_norm = float(grad_norm_tensor.detach().cpu())
            if not math.isfinite(grad_norm):
                raise TrainingNonfiniteError(
                    f"non-finite gradient norm at optimizer step {steps_completed}"
                )
            optimizer.step()
            scheduler.step()
            steps_completed += 1
            examples_processed += profile.global_batch_size
            final_loss = accumulated_loss / profile.gradient_accumulation_steps
            elapsed = prior_elapsed + (time.perf_counter() - trajectory_started)
            _enforce_training_wall_time(elapsed, profile, stage="training_execution")

            validation_loss: float | None = None
            validation_accuracy: float | None = None
            checkpoint_decision = "none"
            should_validate = (
                steps_completed % profile.validation_interval == 0
                or steps_completed == profile.max_steps
            )
            if should_validate:
                synchronize(device)
                current_stage = "development_evaluation"
                validation_loss, validation_accuracy = evaluate_development(
                    model=model,
                    task=task,
                    cases=development_cases,
                    device=device,
                    batch_size=min(profile.global_batch_size, len(development_cases)),
                )
                if not math.isfinite(validation_loss):
                    raise TrainingNonfiniteError(
                        f"non-finite development loss at step {steps_completed}"
                    )
                if checkpoint_is_better(
                    accuracy=validation_accuracy,
                    loss=validation_loss,
                    step=steps_completed,
                    best_accuracy=best_accuracy,
                    best_loss=best_loss,
                    best_step=best_step,
                ):
                    best_step = steps_completed
                    best_accuracy = validation_accuracy
                    best_loss = validation_loss
                    elapsed = prior_elapsed + (
                        time.perf_counter() - trajectory_started
                    )
                    best_payload = _best_evaluation_checkpoint_payload(
                        model=model,
                        profile=profile,
                        candidate_hash=candidate_hash,
                        task=task,
                        seeds=seeds,
                        step=steps_completed,
                        examples_processed=examples_processed,
                        best_accuracy=best_accuracy,
                        best_loss=best_loss,
                        trusted_component_set_hash=component_set_hash,
                    )
                    _atomic_torch_save(best_path, best_payload)
                    checkpoint_decision = "best_development"
                current_stage = "training_execution"
                elapsed = prior_elapsed + (
                    synchronized_time(device) - trajectory_started
                )
                _enforce_training_wall_time(
                    elapsed,
                    profile,
                    stage="development_evaluation",
                )

            memory = mps_memory(device)
            current_mps = memory["current"]
            driver_mps = memory["driver"]
            recommended_mps = memory["recommended"]
            if current_mps is not None:
                peak_mps = max(peak_mps, current_mps)
            elapsed = prior_elapsed + (time.perf_counter() - trajectory_started)
            _append_event(
                event_path,
                {
                    "timestamp": _utc_now(),
                    "optimizer_step": steps_completed,
                    "examples_processed": examples_processed,
                    "loss": final_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": grad_norm,
                    "validation_loss": validation_loss,
                    "validation_exact_match_accuracy": validation_accuracy,
                    "elapsed_seconds": elapsed,
                    "current_mps_allocated_bytes": current_mps,
                    "driver_mps_allocated_bytes": driver_mps,
                    "checkpoint_decision": checkpoint_decision,
                },
            )

            should_resume_checkpoint = (
                steps_completed % profile.checkpoint_interval == 0
                or steps_completed == profile.max_steps
            )
            if should_resume_checkpoint:
                current_stage = "checkpoint_write"
                resume_payload = _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    profile=profile,
                    candidate_hash=candidate_hash,
                    task=task,
                    seeds=seeds,
                    step=steps_completed,
                    examples_processed=examples_processed,
                    elapsed_seconds=elapsed,
                    best_step=best_step,
                    best_accuracy=best_accuracy,
                    best_loss=best_loss,
                    final_training_loss=final_loss,
                    trusted_component_set_hash=component_set_hash,
                )
                _atomic_torch_save(latest_path, resume_payload)
                current_stage = "training_execution"
                elapsed = prior_elapsed + (
                    synchronized_time(device) - trajectory_started
                )
                _enforce_training_wall_time(
                    elapsed,
                    profile,
                    stage="checkpoint_write",
                )

        train_seconds = prior_elapsed + (
            synchronized_time(device) - trajectory_started
        )
        _enforce_training_wall_time(
            train_seconds,
            profile,
            stage="training_completion",
        )
        if not best_path.exists():
            raise OSError("training completed without a best development checkpoint")
        if sha256_file(candidate_copy) != candidate_hash:
            raise RuntimeError("candidate source copy changed during execution")
        if sha256_file(candidate) != candidate_hash:
            raise RuntimeError("original candidate source changed during execution")
        if trusted_component_hashes() != component_hashes:
            raise ReproducibilityBindingError(
                "trusted executable components changed during training"
            )
        checkpoint_hash = sha256_file(best_path)
        result = TrainingResult(
            success=True,
            failure_stage="",
            error="",
            profile_name=profile.name,
            profile_version=profile.version,
            profile_hash=profile.profile_hash,
            candidate_source_hash=candidate_hash,
            initialization_seed=seeds.model_initialization_seed,
            data_seed=seeds.training_data_seed,
            development_seed=seeds.development_set_seed,
            dataloader_seed=seeds.dataloader_seed,
            device=selected_device,
            dtype=profile.dtype,
            steps_completed=steps_completed,
            examples_processed=examples_processed,
            best_development_step=best_step,
            best_development_exact_match_accuracy=best_accuracy,
            best_development_loss=(
                best_loss if math.isfinite(best_loss) else 0.0
            ),
            final_training_loss=(
                final_loss if math.isfinite(final_loss) else 0.0
            ),
            train_seconds=train_seconds,
            peak_mps_allocated_bytes=peak_mps if selected_device == "mps" else None,
            current_mps_allocated_bytes=current_mps,
            driver_mps_allocated_bytes=driver_mps,
            recommended_mps_memory_bytes=recommended_mps,
            parameter_count_metadata=parameter_count,
            checkpoint_path=str(best_path),
            checkpoint_sha256=checkpoint_hash,
            event_log_path=str(event_path),
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=selection.hardware_matched,
            cleanup_completed=False,
        )
    except BaseException as error:
        train_seconds = max(0.0, time.perf_counter() - started)
        stage = (
            "candidate_contract"
            if isinstance(error, ValueError)
            and "contract failed" in str(error)
            else _failure_stage(error)
        )
        if stage == "model_initialization" and current_stage != "model_initialization":
            stage = current_stage
        error_text = f"{type(error).__name__}: {error}"
        if output_prepared:
            _atomic_json(
                destination / "failure.json",
                {
                    "failure_stage": stage,
                    "error": error_text,
                    "traceback": traceback.format_exc(),
                    "timestamp": _utc_now(),
                },
            )
        result = TrainingResult(
            success=False,
            failure_stage=stage,
            error=error_text,
            profile_name=profile.name,
            profile_version=profile.version,
            profile_hash=profile.profile_hash,
            candidate_source_hash=candidate_hash,
            initialization_seed=seeds.model_initialization_seed,
            data_seed=seeds.training_data_seed,
            development_seed=seeds.development_set_seed,
            dataloader_seed=seeds.dataloader_seed,
            device=selected_device,
            dtype=profile.dtype,
            steps_completed=steps_completed,
            examples_processed=examples_processed,
            best_development_step=best_step,
            best_development_exact_match_accuracy=best_accuracy,
            best_development_loss=(
                best_loss if math.isfinite(best_loss) else 0.0
            ),
            final_training_loss=(
                final_loss if math.isfinite(final_loss) else 0.0
            ),
            train_seconds=train_seconds,
            peak_mps_allocated_bytes=peak_mps or None,
            current_mps_allocated_bytes=current_mps,
            driver_mps_allocated_bytes=driver_mps,
            recommended_mps_memory_bytes=recommended_mps,
            parameter_count_metadata=parameter_count,
            checkpoint_path=str(best_path) if best_path.exists() else "",
            checkpoint_sha256=checkpoint_hash,
            event_log_path=str(event_path),
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=False,
            cleanup_completed=False,
        )
    finally:
        del optimizer
        del model
        gc.collect()
        if selected_device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except RuntimeError:
                pass
        cleanup_completed = True

    assert result is not None
    result = TrainingResult(
        **{
            **result.to_dict(),
            "cleanup_completed": cleanup_completed,
        }
    )
    if output_prepared:
        _atomic_json(destination / "training_summary.json", result.to_dict())
    return result
