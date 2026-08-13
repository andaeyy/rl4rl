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
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    AcceleratorKind,
    DeviceUnavailableError,
    accelerator_memory,
    cleanup_accelerator,
    reset_peak_memory,
    resolve_training_device,
    synchronize,
    synchronized_time,
)
from common.runtime_context import ExecutionContextV1
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


def _create_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish one immutable JSON receipt without replacing prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        # A hard link is an atomic create-if-absent operation.  Unlike replace,
        # it cannot silently rewrite a prior resume attestation.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    accelerator_fingerprint: dict[str, Any] | None = None,
    component_hashes: dict[str, str] | None = None,
    execution_context: ExecutionContextV1 | None = None,
    dependency_lock_hash: str | None = None,
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
    manifest = {
        "created_at": _utc_now(),
        # V1 evidence historically recorded an executor-absolute source path.
        # V2 records only the immutable colocated artifact name; its SHA-256
        # below is the portable identity.
        "candidate_path": (
            str(candidate_path) if profile.version == "1" else candidate_copy_name
        ),
        "candidate_source_hash": candidate_hash,
        "candidate_artifact_hash": candidate_hash,
        "candidate_format": candidate_format.value,
        "candidate_graph_hash": candidate_graph_hash,
        "immutable_candidate_relative_path": candidate_copy_name,
        "candidate_initialization": "from_scratch",
        "profile": profile.to_dict(),
        "profile_hash": profile.profile_hash,
        "seed_bundle": asdict(seeds),
        "seed_bundle_hash": seeds.bundle_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "requested_device": requested_device,
        "selected_device": selected_device,
        "allow_cpu_for_tests": allow_cpu_for_tests,
        "hardware_matched_scientific_run": bool(
            profile.scientific
            and selected_device is not None
            and torch.device(selected_device).type == profile.device_requirement
        ),
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
            "cuda_available": bool(
                hasattr(torch, "cuda") and torch.cuda.is_available()
            ),
            "cuda_device_count": int(torch.cuda.device_count())
            if hasattr(torch, "cuda")
            else 0,
            "deterministic_algorithms": profile.deterministic_algorithms,
            "pytorch_enable_mps_fallback": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK", ""
            ),
            "accelerator_memory_fraction": profile.accelerator_memory_fraction,
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG", ""
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(
                getattr(torch.backends.cuda.matmul, "allow_tf32", False)
            ),
            "accelerator_fingerprint": accelerator_fingerprint or {},
            "declared_machine": declared_machine,
        },
        "dependency_lock_hash": (
            _dependency_lock_hash()
            if dependency_lock_hash is None
            else dependency_lock_hash
        ),
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
    if execution_context is not None:
        manifest["execution_context"] = execution_context.to_dict()
    if profile.version == "2":
        manifest["schema_name"] = "TrainingManifest"
        manifest["schema_version"] = "2.0"
    return manifest


def seed_everything(
    seed: int,
    *,
    deterministic: bool,
    device: torch.device,
) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    if device.type == AcceleratorKind.CUDA.value:
        torch.cuda.manual_seed_all(seed)
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
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        try:
            state["torch_cuda"] = [
                item.detach().cpu().clone()
                for item in torch.cuda.get_rng_state_all()
            ]
        except RuntimeError:
            state["torch_cuda"] = None
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
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and hasattr(torch, "cuda"):
        torch.cuda.set_rng_state_all(cuda_state)


def _canonical_rng_value(value: Any) -> Any:
    """Normalize the trusted RNG-state vocabulary for a portable digest."""

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ResumeMismatchError("RNG-state mapping keys must be text")
        return {
            key: _canonical_rng_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_rng_value(item) for item in value]
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise ResumeMismatchError(
        f"unsupported RNG-state value type: {type(value).__name__}"
    )


def rng_state_sha256(state: dict[str, Any]) -> str:
    """Hash RNG state using the frozen tensor-aware canonical representation."""

    if not isinstance(state, dict):
        raise ResumeMismatchError("RNG state must be a mapping")
    encoded = json.dumps(
        _canonical_rng_value(state),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _advance_checkpoint_rng_lineage() -> None:
    """Advance the persisted RNG lineage once per completed optimizer step.

    The sampled value is deliberately unused by training.  Its only purpose is
    to make checkpoint RNG continuation observable while leaving examples,
    gradients, optimizer updates, and model outputs unchanged.
    """

    random.getrandbits(64)


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
    dependency_lock_hash: str | None = None,
) -> dict[str, Any]:
    component_set_hash = (
        trusted_component_set_sha256()
        if trusted_component_set_hash is None
        else trusted_component_set_hash
    )
    payload = {
        "checkpoint_kind": (
            "trusted_resume_state_v1"
            if profile.version == "1"
            else "trusted_resume_state_v2"
        ),
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
    if profile.version == "2":
        payload["dependency_lock_hash"] = (
            _dependency_lock_hash()
            if dependency_lock_hash is None
            else dependency_lock_hash
        )
    return payload


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
    dependency_lock_hash: str | None = None,
) -> dict[str, Any]:
    """Plain tensor/primitives-only checkpoint safe for ``weights_only=True``."""

    component_set_hash = (
        trusted_component_set_sha256()
        if trusted_component_set_hash is None
        else trusted_component_set_hash
    )
    payload = {
        "checkpoint_kind": (
            "best_evaluation_weights_v1"
            if profile.version == "1"
            else "best_evaluation_weights_v2"
        ),
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
    if profile.version == "2":
        payload["dependency_lock_hash"] = (
            _dependency_lock_hash()
            if dependency_lock_hash is None
            else dependency_lock_hash
        )
    return payload


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    candidate_hash: str,
    profile: TrainingProfile,
    task: FixedAdditionTask,
    seeds: TrainingSeedBundle,
    trusted_component_set_hash: str | None = None,
    dependency_lock_hash: str | None = None,
) -> None:
    expected_component_set_hash = (
        trusted_component_set_sha256()
        if trusted_component_set_hash is None
        else trusted_component_set_hash
    )
    expected = {
        "checkpoint_kind": (
            "trusted_resume_state_v1"
            if profile.version == "1"
            else "trusted_resume_state_v2"
        ),
        "candidate_source_hash": candidate_hash,
        "profile_hash": profile.profile_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "seed_bundle_hash": seeds.bundle_hash,
        "trusted_component_set_sha256": expected_component_set_hash,
    }
    if profile.version == "2":
        expected["dependency_lock_hash"] = (
            _dependency_lock_hash()
            if dependency_lock_hash is None
            else dependency_lock_hash
        )
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
    if profile.version == "2":
        required_fields.add("dependency_lock_hash")
    if set(checkpoint) != required_fields:
        raise ResumeMismatchError(
            f"resume checkpoint fields differ from trusted v{profile.version}"
        )
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
            required_accelerator=profile.device_requirement,
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

    def portable_path(path: Path | None) -> str | None:
        if path is None:
            return None
        if profile.version == "1":
            return str(path)
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            pass
        # Controller candidates and training outputs are colocated below one
        # of these logical artifact roots. Retain that useful suffix without
        # recording the machine-specific prefix.
        for marker in ("artifacts", "candidate_training"):
            if marker in path.parts:
                index = path.parts.index(marker)
                return Path(*path.parts[index:]).as_posix()
        return path.name

    resolved_resume = Path(resume).resolve() if resume else None
    return {
        "candidate": portable_path(candidate),
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
        "accelerator_kind": selection.requested_kind.value,
        "accelerator_fingerprint": selection.fingerprint.to_dict(),
        "scientific": profile.scientific,
        "hardware_matched": selection.hardware_matched,
        "containment_audit_hash": containment_audit.audit_hash,
        "containment_decision": containment_decision.to_dict(),
        "trusted_executable_component_hashes": component_hashes,
        "trusted_component_set_sha256": trusted_component_set_sha256(
            component_hashes
        ),
        "output_dir": portable_path(resolved_output),
        "resume": portable_path(resolved_resume),
    }


def _failure_stage(
    error: BaseException,
    *,
    requested_device: str = "",
) -> str:
    accelerator = requested_device.split(":", 1)[0].lower()
    if isinstance(error, ReproducibilityBindingError):
        return "reproducibility_binding"
    if isinstance(error, ContainmentGateError):
        return "containment_unproven"
    if isinstance(error, DeviceUnavailableError):
        if accelerator == "cuda" or "cuda" in str(error).lower():
            return "cuda_unavailable"
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
        if accelerator == "cuda":
            return "cuda_deterministic_kernel_unavailable"
        return "unsupported_operation"
    if isinstance(error, RuntimeError) and accelerator == "cuda" and any(
        marker in str(error).lower()
        for marker in ("cuda", "cudnn", "cublas", "driver")
    ):
        return "cuda_driver_failure"
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
    execution_context: ExecutionContextV1 | None = None,
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
    partial_resume_path = destination / "partial_resume_checkpoint.pt"
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
    peak_accelerator = 0
    current_accelerator: int | None = None
    reserved_accelerator: int | None = None
    total_accelerator: int | None = None
    fingerprint: dict[str, Any] = {}
    checkpoint_hash = ""
    cleanup_completed = False
    output_prepared = False
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    result: TrainingResult | None = None
    current_stage = "candidate_contract"
    candidate_format = CandidateFormat.ARBITRARY_PYTHON
    candidate_graph_hash: str | None = None
    resume_source_checkpoint_sha256: str | None = None
    resume_source_rng_sha256: str | None = None
    resume_observed_rng_sha256: str | None = None
    resume_source_step: int | None = None
    resume_source_examples: int | None = None
    component_hashes = trusted_component_hashes()
    component_set_hash = trusted_component_set_sha256(component_hashes)
    dependency_lock_hash = _dependency_lock_hash()

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
                required_accelerator=profile.device_requirement,
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
            execution_context=execution_context,
            dependency_lock_hash=dependency_lock_hash,
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
                dependency_lock_hash=dependency_lock_hash,
            )
            # Capture the source identities before the latest-checkpoint path is
            # overwritten by the resumed trajectory.
            resume_source_checkpoint_sha256 = sha256_file(resume_path)
            resume_source_rng_sha256 = rng_state_sha256(resume_state["rng_state"])
            resume_source_step = int(resume_state["global_step"])
            resume_source_examples = int(resume_state["examples_processed"])

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
        fingerprint = selection.fingerprint.to_dict()
        reset_peak_memory(device)
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
            accelerator_fingerprint=fingerprint,
            component_hashes=component_hashes,
            execution_context=execution_context,
            dependency_lock_hash=dependency_lock_hash,
        )
        _atomic_json(destination / "training_manifest.json", manifest)

        dataloader_generator = seed_everything(
            seeds.model_initialization_seed,
            deterministic=profile.deterministic_algorithms,
            device=device,
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
            # This observation is intentionally adjacent to restore: no data,
            # model, optimizer, scheduler, or evaluator work may intervene.
            resume_observed_rng_sha256 = rng_state_sha256(_rng_state())
            if resume_observed_rng_sha256 != resume_source_rng_sha256:
                raise ReproducibilityBindingError(
                    "post-restore RNG state differs from the source checkpoint"
                )

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
            _advance_checkpoint_rng_lineage()
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
                        dependency_lock_hash=dependency_lock_hash,
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

            memory = accelerator_memory(device)
            current_accelerator = memory["current"]
            reserved_accelerator = memory["reserved_or_driver"]
            total_accelerator = memory["recommended_or_total"]
            measured_peak = memory["peak"] or current_accelerator
            if measured_peak is not None:
                peak_accelerator = max(peak_accelerator, measured_peak)
            elapsed = prior_elapsed + (time.perf_counter() - trajectory_started)
            memory_event = {
                "current_accelerator_allocated_bytes": current_accelerator,
                "reserved_accelerator_allocated_bytes": reserved_accelerator,
                "peak_accelerator_allocated_bytes": peak_accelerator or None,
                "accelerator_total_memory_bytes": total_accelerator,
            }
            if profile.version == "1":
                memory_event = {
                    "current_mps_allocated_bytes": current_accelerator,
                    "driver_mps_allocated_bytes": reserved_accelerator,
                }
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
                    **memory_event,
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
                    dependency_lock_hash=dependency_lock_hash,
                )
                _atomic_torch_save(latest_path, resume_payload)
                if (
                    profile.version == "2"
                    and steps_completed < profile.max_steps
                    and not partial_resume_path.exists()
                ):
                    _atomic_torch_save(partial_resume_path, resume_payload)
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
        if _dependency_lock_hash() != dependency_lock_hash:
            raise ReproducibilityBindingError(
                "dependency lock changed during training"
            )
        if resume_state is not None:
            if execution_context is None:
                local_run_id = "local-resume-" + hashlib.sha256(
                    str(destination).encode("utf-8")
                ).hexdigest()[:16]
                attestation_context = ExecutionContextV1.local(run_id=local_run_id)
            else:
                attestation_context = execution_context
            if any(
                value is None
                for value in (
                    resume_source_checkpoint_sha256,
                    resume_source_rng_sha256,
                    resume_observed_rng_sha256,
                    resume_source_step,
                    resume_source_examples,
                )
            ):
                raise ReproducibilityBindingError(
                    "resume RNG attestation inputs were not captured"
                )
            final_resume_checkpoint_sha256 = sha256_file(latest_path)
            final_resume_state = torch.load(
                latest_path, map_location="cpu", weights_only=True
            )
            _validate_resume(
                final_resume_state,
                candidate_hash=candidate_hash,
                profile=profile,
                task=task,
                seeds=seeds,
                trusted_component_set_hash=component_set_hash,
                dependency_lock_hash=dependency_lock_hash,
            )
            final_rng_sha256 = rng_state_sha256(final_resume_state["rng_state"])
            final_step = int(final_resume_state["global_step"])
            final_examples = int(final_resume_state["examples_processed"])
            restored_exactly = (
                resume_observed_rng_sha256 == resume_source_rng_sha256
            )
            rng_progressed = final_rng_sha256 != resume_observed_rng_sha256
            if (
                not restored_exactly
                or not rng_progressed
                or final_step <= resume_source_step
                or final_examples <= resume_source_examples
            ):
                raise ReproducibilityBindingError(
                    "resumed checkpoint did not prove RNG and optimizer progression"
                )
            _create_json(
                destination / "rng_restore_attestation.json",
                {
                    "schema_name": "RNGRestoreAttestation",
                    "schema_version": "1.0",
                    "source_checkpoint_sha256": resume_source_checkpoint_sha256,
                    "source_rng_state_sha256": resume_source_rng_sha256,
                    "observed_post_restore_rng_state_sha256": (
                        resume_observed_rng_sha256
                    ),
                    "restored_exactly": restored_exactly,
                    "source_optimizer_step": resume_source_step,
                    "source_examples_processed": resume_source_examples,
                    "final_checkpoint_sha256": final_resume_checkpoint_sha256,
                    "final_rng_state_sha256": final_rng_sha256,
                    "final_optimizer_step": final_step,
                    "final_examples_processed": final_examples,
                    "rng_progressed": rng_progressed,
                    "execution_context": attestation_context.to_dict(),
                },
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
            accelerator_kind=torch.device(selected_device).type,
            peak_accelerator_allocated_bytes=(peak_accelerator or None),
            current_accelerator_allocated_bytes=current_accelerator,
            reserved_accelerator_allocated_bytes=reserved_accelerator,
            accelerator_total_memory_bytes=total_accelerator,
            accelerator_fingerprint=fingerprint,
            parameter_count_metadata=parameter_count,
            checkpoint_path=(
                best_path.name if profile.version == "2" else str(best_path)
            ),
            checkpoint_sha256=checkpoint_hash,
            event_log_path=(
                event_path.name if profile.version == "2" else str(event_path)
            ),
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=selection.hardware_matched,
            cleanup_completed=False,
            schema_version="1.0" if profile.version == "1" else "2.0",
        )
    except BaseException as error:
        train_seconds = max(0.0, time.perf_counter() - started)
        stage = (
            "candidate_contract"
            if isinstance(error, ValueError)
            and "contract failed" in str(error)
            else _failure_stage(error, requested_device=selected_device)
        )
        if stage == "model_initialization" and current_stage != "model_initialization":
            stage = current_stage
        error_text = (
            f"{type(error).__name__}: training failed; details suppressed"
            if profile.version == "2"
            else f"{type(error).__name__}: {error}"
        )
        if output_prepared:
            failure_payload = (
                {
                    "failure_stage": stage,
                    "error_type": type(error).__name__,
                    "message": "training failed; details suppressed",
                    "timestamp": _utc_now(),
                }
                if profile.version == "2"
                else {
                    "failure_stage": stage,
                    "error": error_text,
                    "traceback": traceback.format_exc(),
                    "timestamp": _utc_now(),
                }
            )
            _atomic_json(destination / "failure.json", failure_payload)
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
            accelerator_kind=AcceleratorKind.parse(selected_device).value,
            peak_accelerator_allocated_bytes=peak_accelerator or None,
            current_accelerator_allocated_bytes=current_accelerator,
            reserved_accelerator_allocated_bytes=reserved_accelerator,
            accelerator_total_memory_bytes=total_accelerator,
            accelerator_fingerprint=fingerprint,
            parameter_count_metadata=parameter_count,
            checkpoint_path=(
                best_path.name
                if best_path.exists() and profile.version == "2"
                else str(best_path) if best_path.exists() else ""
            ),
            checkpoint_sha256=checkpoint_hash,
            event_log_path=(
                event_path.name if profile.version == "2" else str(event_path)
            ),
            unsupported_operation_fallback=False,
            scientific=profile.scientific,
            hardware_matched=False,
            cleanup_completed=False,
            schema_version="1.0" if profile.version == "1" else "2.0",
        )
    finally:
        del optimizer
        del model
        gc.collect()
        try:
            cleanup_accelerator(torch.device(selected_device))
        except (RuntimeError, ValueError, DeviceUnavailableError) as cleanup_error:
            cleanup_completed = False
            if output_prepared:
                _atomic_json(
                    destination / "cleanup_failure.json",
                    {
                        "failure_stage": "accelerator_cleanup_failure",
                        "error_type": type(cleanup_error).__name__,
                        "timestamp": _utc_now(),
                    },
                )
            if result is not None and result.success:
                result = TrainingResult.from_dict(
                    {
                        **result.to_dict(),
                        "success": False,
                        "failure_stage": "accelerator_cleanup_failure",
                        "error": (
                            f"{type(cleanup_error).__name__}: "
                            "accelerator cleanup failed"
                        ),
                        "cleanup_completed": False,
                    }
                )
        else:
            cleanup_completed = True

    assert result is not None
    result = TrainingResult.from_dict(
        {**result.to_dict(), "cleanup_completed": cleanup_completed}
    )
    if output_prepared:
        _atomic_json(destination / "training_summary.json", result.to_dict())
    return result
