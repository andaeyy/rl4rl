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

from common.candidate_contract import inspect_candidate_path, validate_candidate
from common.candidate_loader import load_candidate
from common.device import (
    ACCELERATOR_TELEMETRY_SCHEMA_VERSION,
    DeviceUnavailableError,
    accelerator_memory,
    accelerator_runtime_metadata,
    accelerator_telemetry,
    cleanup_accelerator,
    reset_accelerator_peak_memory,
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
from common.trusted_candidate import (
    sha256_bytes,
    validate_trusted_initial_candidate,
)


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


def _atomic_source_copy(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    private: bool,
) -> str:
    payload = source.read_bytes()
    observed_sha256 = sha256_bytes(payload)
    if observed_sha256 != expected_sha256:
        raise ResumeMismatchError(
            "candidate source changed while creating the immutable worker copy"
        )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    try:
        os.replace(temporary, destination)
        destination.chmod(0o400 if private else 0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return observed_sha256


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


def _parameter_state_sha256(model: torch.nn.Module) -> str:
    """Hash parameter values without retaining a second model-sized copy."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        value = parameter.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _prepare_output_directory(
    output_dir: Path,
    resume: Path | None,
    *,
    private: bool,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and resume is None:
        raise OutputDirectoryError(
            f"output directory is non-empty: {output_dir}; pass --resume explicitly"
        )
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700 if private else 0o777)
    if private:
        output_dir.chmod(0o700)


def _dependency_lock_hash() -> str:
    lock = ROOT / "uv.lock"
    return sha256_file(lock) if lock.exists() else ""


def _controller_source_hash() -> str:
    paths = [
        Path(__file__),
        ROOT / "common" / "evaluator.py",
        ROOT / "common" / "device.py",
        ROOT / "common" / "cuda_contract.py",
        ROOT / "common" / "task_adapter.py",
        ROOT / "common" / "training_data.py",
        ROOT / "common" / "training_config.py",
        ROOT / "common" / "training_worker.py",
        ROOT / "common" / "trusted_candidate.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def cuda_determinism_state() -> dict[str, Any]:
    """Capture every CUDA determinism control used by the frozen profiles."""

    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def assert_cuda_deterministic_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    expected = {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "float32_matmul_precision": "highest",
    }
    observed = cuda_determinism_state()
    if observed != expected:
        raise RuntimeError(
            "deterministic CUDA runtime controls changed during candidate execution: "
            + json.dumps({"expected": expected, "observed": observed}, sort_keys=True)
        )


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
    selected_backend = (
        torch.device(selected_device).type if selected_device is not None else None
    )
    accelerator = (
        accelerator_runtime_metadata(torch.device(selected_device))
        if selected_device is not None
        else {
            "schema_version": ACCELERATOR_TELEMETRY_SCHEMA_VERSION,
            "backend": None,
            "device": None,
            "identity": {},
            "runtime": {},
            "allocation": {},
        }
    )
    cuda_determinism = (
        cuda_determinism_state() if selected_backend == "cuda" else None
    )
    return {
        "created_at": _utc_now(),
        "candidate_path": str(candidate_path),
        "candidate_source_hash": candidate_hash,
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
            profile.scientific and selected_backend == profile.device_requirement
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
            "accelerator": accelerator,
            "cuda_determinism": cuda_determinism,
        },
        "dependency_lock_hash": _dependency_lock_hash(),
        "controller_source_hash": _controller_source_hash(),
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
        "hardware_condition_note": (
            "CUDA and MPS are separate hardware conditions. Identical seeds do not "
            "imply identical cross-backend trajectories, and results must not be "
            "pooled without an explicit analysis plan."
        ),
    }


def configure_deterministic_runtime(
    device: torch.device,
    *,
    deterministic: bool,
) -> None:
    torch.use_deterministic_algorithms(deterministic)
    if device.type != "cuda":
        return
    if not deterministic:
        raise ValueError("CUDA candidate training may not disable determinism")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    assert_cuda_deterministic_runtime(device)


def seed_everything(
    seed: int,
    *,
    deterministic: bool,
    device: torch.device | None = None,
) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    configure_deterministic_runtime(
        device or torch.device("cpu"),
        deterministic=deterministic,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _rng_state(device: torch.device | None = None) -> dict[str, Any]:
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
    if device is not None and device.type == "cuda":
        state["torch_cuda"] = [
            value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()
        ]
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
    if cuda_state is not None:
        if (
            not isinstance(cuda_state, list)
            or len(cuda_state) != 1
            or not all(
                isinstance(value, torch.Tensor)
                and value.dtype == torch.uint8
                and value.numel() > 0
                for value in cuda_state
            )
        ):
            raise ResumeMismatchError("resume checkpoint has invalid CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_state)


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
        assert_cuda_deterministic_runtime(device)
        total_loss += float(loss.detach().cpu()) * len(batch)
        total_examples += len(batch)
    exact_match, _ = task.exact_match(
        model,
        cases,
        device=device,
        batch_size=batch_size,
        failure_limit=0,
    )
    assert_cuda_deterministic_runtime(device)
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
    device: torch.device | None = None,
) -> dict[str, Any]:
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
        "rng_state": _rng_state(device),
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
) -> dict[str, Any]:
    """Plain tensor/primitives-only checkpoint safe for ``weights_only=True``."""

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
    }


def _validate_resume(
    checkpoint: dict[str, Any],
    *,
    candidate_hash: str,
    profile: TrainingProfile,
    task: FixedAdditionTask,
    seeds: TrainingSeedBundle,
) -> None:
    expected = {
        "checkpoint_kind": "trusted_resume_state_v1",
        "candidate_source_hash": candidate_hash,
        "profile_hash": profile.profile_hash,
        "task_adapter_version": task.version,
        "task_adapter_hash": task.config_hash,
        "seed_bundle_hash": seeds.bundle_hash,
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
    if profile.device_requirement == "cuda":
        rng_state = checkpoint.get("rng_state")
        cuda_state = rng_state.get("torch_cuda") if isinstance(rng_state, dict) else None
        if (
            not isinstance(cuda_state, list)
            or len(cuda_state) != 1
            or not isinstance(cuda_state[0], torch.Tensor)
            or cuda_state[0].dtype != torch.uint8
            or cuda_state[0].numel() == 0
        ):
            raise ResumeMismatchError(
                "CUDA resume checkpoint requires exactly one non-empty CUDA RNG state"
            )


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
    if profile.name == "smoke_train_cuda_v1":
        validate_trusted_initial_candidate(candidate)
    source_contract = inspect_candidate_path(candidate)
    if not source_contract.valid:
        raise ValueError(
            "candidate contract failed: " + "; ".join(source_contract.reasons)
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
            candidate_format=CandidateFormat.ARBITRARY_PYTHON,
            requested_device=requested_device,
            scientific=profile.scientific,
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
    return {
        "candidate": str(candidate),
        "candidate_source_hash": sha256_file(candidate),
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
        "output_dir": str(resolved_output) if resolved_output else None,
        "resume": str(Path(resume).resolve()) if resume else None,
    }


def _failure_stage(error: BaseException, *, requested_device: str) -> str:
    cuda_requested = requested_device.strip().lower() in {"cuda", "cuda:0"}
    error_text = str(error).lower()
    if isinstance(error, ContainmentGateError):
        return "containment_unproven"
    if isinstance(error, DeviceUnavailableError):
        return "cuda_unavailable" if cuda_requested else "device_unavailable"
    if isinstance(error, ResumeMismatchError):
        return "checkpoint_resume_mismatch"
    if isinstance(error, OutputDirectoryError):
        return "checkpoint_write"
    if isinstance(error, TrainingTimeoutError):
        return "training_timeout"
    if isinstance(error, TrainingNonfiniteError):
        return "training_nonfinite"
    if isinstance(error, MemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in error_text
    ):
        return "cuda_oom" if cuda_requested else "training_oom"
    if isinstance(error, OSError):
        return "checkpoint_write"
    if isinstance(error, RuntimeError) and "deterministic" in error_text:
        return (
            "unsupported_deterministic_cuda_operation"
            if cuda_requested
            else "unsupported_operation"
        )
    if cuda_requested and isinstance(error, (AssertionError, RuntimeError)) and any(
        marker in error_text
        for marker in ("cuda", "cudnn", "cublas", "nvidia", "driver")
    ):
        return "cuda_driver_runtime_failure"
    if isinstance(error, RuntimeError) and "not implemented for" in error_text:
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
    device: torch.device | None = None
    model: torch.nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    accelerator_snapshot: dict[str, Any] = {}
    timing_synchronized = False
    initial_parameter_hash = ""
    final_parameter_hash = ""
    parameters_changed = False
    result: TrainingResult | None = None
    failure_error: BaseException | None = None
    current_stage = "candidate_contract"

    try:
        if destination_is_symlink:
            raise OutputDirectoryError(
                f"output directory may not be a symlink: {raw_destination}"
            )
        if profile.name == "smoke_train_cuda_v1":
            pinned_hash = validate_trusted_initial_candidate(candidate)
            if candidate_hash != pinned_hash:
                raise ResumeMismatchError(
                    "trusted CUDA smoke candidate changed before training"
                )
        _prepare_output_directory(
            destination,
            resume_path,
            private=profile.device_requirement == "cuda",
        )
        output_prepared = True
        source_contract = inspect_candidate_path(candidate)
        if not source_contract.valid:
            raise ValueError(
                "candidate contract failed: " + "; ".join(source_contract.reasons)
            )
        profile.validate()

        containment_audit = audit_runtime()
        containment_decision = assess_scientific_execution(
            containment_audit,
            ScientificExecutionRequest(
                candidate_format=CandidateFormat.ARBITRARY_PYTHON,
                requested_device=requested_device,
                scientific=profile.scientific,
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
            )

        candidate_copy = destination / "candidate_source.py"
        if candidate_copy.exists():
            if sha256_file(candidate_copy) != candidate_hash:
                raise ResumeMismatchError(
                    "stored candidate source differs from requested candidate"
                )
        else:
            copied_hash = _atomic_source_copy(
                candidate,
                candidate_copy,
                expected_sha256=candidate_hash,
                private=profile.device_requirement == "cuda",
            )
            if copied_hash != candidate_hash:
                raise ResumeMismatchError("candidate copy hash does not reconstruct")

        copied_source_contract = inspect_candidate_path(candidate_copy)
        if not copied_source_contract.valid:
            raise ValueError(
                "candidate copy contract failed: "
                + "; ".join(copied_source_contract.reasons)
            )

        selection = resolve_training_device(
            profile,
            requested_device,
            allow_cpu_for_tests=allow_cpu_for_tests,
        )
        device = selection.device
        selected_device = str(device)
        dataloader_generator = seed_everything(
            seeds.model_initialization_seed,
            deterministic=profile.deterministic_algorithms,
            device=device,
        )
        dataloader_generator.manual_seed(seeds.dataloader_seed)
        reset_accelerator_peak_memory(device)
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
        )
        _atomic_json(destination / "training_manifest.json", manifest)
        current_stage = "model_initialization"
        if sha256_file(candidate_copy) != candidate_hash:
            raise ResumeMismatchError(
                "immutable candidate copy changed immediately before import"
            )
        module = load_candidate(candidate_copy)
        built = module.build_untrained_model(seeds.model_initialization_seed)
        if not isinstance(built, tuple) or len(built) != 2:
            raise TypeError(
                "build_untrained_model(seed) must return (torch.nn.Module, metadata)"
            )
        model, _metadata = built
        contract = validate_candidate(module, model)
        if not contract.valid:
            raise ValueError(
                "candidate runtime contract failed: " + "; ".join(contract.reasons)
            )
        # Candidate import, construction, and contract probes are arbitrary
        # Python. Reapply the frozen process-global controls before accepting
        # any training evidence.
        configure_deterministic_runtime(
            device,
            deterministic=profile.deterministic_algorithms,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model = model.to(device=device, dtype=torch.float32)
        initial_parameter_hash = _parameter_state_sha256(model)

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
        timing_synchronized = device.type in {"mps", "cuda"}
        current_stage = "training_execution"

        while steps_completed < profile.max_steps:
            elapsed = prior_elapsed + (
                synchronized_time(device) - trajectory_started
            )
            if elapsed >= profile.maximum_wall_seconds:
                raise TrainingTimeoutError(
                    f"candidate exceeded {profile.maximum_wall_seconds}s wall-time cap"
                )

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
                configure_deterministic_runtime(
                    device,
                    deterministic=profile.deterministic_algorithms,
                )
                loss = task.teacher_forced_loss(model, input_ids, labels)
                assert_cuda_deterministic_runtime(device)
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
            elapsed = prior_elapsed + (
                synchronized_time(device) - trajectory_started
            )
            if elapsed >= profile.maximum_wall_seconds:
                raise TrainingTimeoutError(
                    f"candidate exceeded {profile.maximum_wall_seconds}s wall-time cap"
                )

            validation_loss: float | None = None
            validation_accuracy: float | None = None
            checkpoint_decision = "none"
            should_validate = (
                steps_completed % profile.validation_interval == 0
                or steps_completed == profile.max_steps
            )
            if should_validate:
                synchronize(device)
                configure_deterministic_runtime(
                    device,
                    deterministic=profile.deterministic_algorithms,
                )
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
                        synchronized_time(device) - trajectory_started
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
                    )
                    _atomic_torch_save(best_path, best_payload)
                    checkpoint_decision = "best_development"
                current_stage = "training_execution"

            memory = accelerator_memory(device)
            backend_specific = memory["backend_specific"]
            current_mps = (
                memory["allocated_bytes"] if device.type == "mps" else None
            )
            driver_mps = (
                backend_specific.get("driver_allocated_bytes")
                if device.type == "mps"
                else None
            )
            recommended_mps = (
                backend_specific.get("recommended_max_memory_bytes")
                if device.type == "mps"
                else None
            )
            if current_mps is not None:
                peak_mps = max(peak_mps, current_mps)
                memory["peak_allocated_bytes"] = peak_mps
            accelerator_snapshot = {
                "schema_version": ACCELERATOR_TELEMETRY_SCHEMA_VERSION,
                "backend": device.type,
                "device": str(device),
                "memory": memory,
            }
            elapsed = prior_elapsed + (
                synchronized_time(device) - trajectory_started
            )
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
                    "accelerator_telemetry_schema_version": (
                        ACCELERATOR_TELEMETRY_SCHEMA_VERSION
                    ),
                    "accelerator_telemetry": accelerator_snapshot,
                    "timing_synchronized": timing_synchronized,
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
                    device=device,
                )
                _atomic_torch_save(latest_path, resume_payload)
                current_stage = "training_execution"

        train_seconds = prior_elapsed + (
            synchronized_time(device) - trajectory_started
        )
        assert_cuda_deterministic_runtime(device)
        final_parameter_hash = _parameter_state_sha256(model)
        parameters_changed = bool(initial_parameter_hash) and (
            final_parameter_hash != initial_parameter_hash
        )
        accelerator_snapshot = accelerator_telemetry(
            device,
            peak_mps_allocated_bytes=peak_mps or None,
        )
        if not best_path.exists():
            raise OSError("training completed without a best development checkpoint")
        if sha256_file(candidate_copy) != candidate_hash:
            raise RuntimeError("candidate source copy changed during execution")
        if sha256_file(candidate) != candidate_hash:
            raise RuntimeError("original candidate source changed during execution")
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
            accelerator_telemetry_schema_version=(
                ACCELERATOR_TELEMETRY_SCHEMA_VERSION
            ),
            accelerator_telemetry=accelerator_snapshot,
            timing_synchronized=timing_synchronized,
            initial_parameter_sha256=initial_parameter_hash,
            final_parameter_sha256=final_parameter_hash,
            parameters_changed=parameters_changed,
        )
    except BaseException as error:
        failure_error = error
        train_seconds = max(0.0, time.perf_counter() - started)
        if device is not None:
            try:
                accelerator_snapshot = accelerator_telemetry(
                    device,
                    peak_mps_allocated_bytes=peak_mps or None,
                )
            except (AssertionError, RuntimeError):
                pass
        if model is not None and initial_parameter_hash:
            try:
                final_parameter_hash = _parameter_state_sha256(model)
                parameters_changed = final_parameter_hash != initial_parameter_hash
            except (AssertionError, RuntimeError):
                pass
        stage = (
            "candidate_contract"
            if isinstance(error, ValueError)
            and "contract failed" in str(error)
            else _failure_stage(error, requested_device=requested_device)
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
            accelerator_telemetry_schema_version=(
                ACCELERATOR_TELEMETRY_SCHEMA_VERSION
            ),
            accelerator_telemetry=accelerator_snapshot,
            timing_synchronized=timing_synchronized,
            initial_parameter_sha256=initial_parameter_hash,
            final_parameter_sha256=final_parameter_hash,
            parameters_changed=parameters_changed,
        )
    finally:
        del optimizer
        del model
        gc.collect()
        cleanup_completed = True
        if device is not None:
            try:
                cleanup_accelerator(device)
            except (AssertionError, RuntimeError):
                cleanup_completed = False

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
