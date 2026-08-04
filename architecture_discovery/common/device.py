"""Strict accelerator selection, synchronization, and telemetry."""

from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

from common.cuda_contract import (
    CUDA_A40_REQUIRED_PROFILE_NAMES,
    CUDA_ALLOCATION_ENVIRONMENT_KEYS,
    CUDA_CUBLAS_WORKSPACE_CONFIG,
    cuda_allocation_gpu_counts,
    cuda_allocation_proves_exactly_one,
    is_nvidia_a40_name,
)
from common.training_config import TrainingProfile


ACCELERATOR_TELEMETRY_SCHEMA_VERSION = "accelerator_telemetry_v1"


class DeviceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceSelection:
    device: torch.device
    hardware_matched: bool
    fallback_requested: bool


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cuda_allocation_context() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return {
        "scheduler": "slurm",
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "visible_devices": visible,
        "allocation_markers": {
            key: os.environ.get(key, "") for key in CUDA_ALLOCATION_ENVIRONMENT_KEYS
        },
    }


def _validate_cuda_scheduler_allocation() -> dict[str, Any]:
    """Require evidence for one scheduler-isolated GPU before touching CUDA."""

    context = _cuda_allocation_context()
    if not context["job_id"]:
        raise DeviceUnavailableError(
            "CUDA training requires a Slurm allocation (SLURM_JOB_ID is absent); "
            "no CPU fallback occurred"
        )
    markers = context["allocation_markers"]
    if not cuda_allocation_proves_exactly_one(markers):
        raise DeviceUnavailableError(
            "CUDA training requires Slurm evidence for exactly one allocated GPU "
            f"from {CUDA_ALLOCATION_ENVIRONMENT_KEYS}; observed counts "
            f"{cuda_allocation_gpu_counts(markers)}; no CPU fallback occurred"
        )
    visible = str(context["visible_devices"]).strip()
    hidden_values = {"", "-1", "none", "nodevfiles", "void"}
    visible_devices = [item.strip() for item in visible.split(",") if item.strip()]
    if visible.lower() in hidden_values or len(visible_devices) != 1:
        raise DeviceUnavailableError(
            "CUDA training requires CUDA_VISIBLE_DEVICES to expose exactly one "
            f"scheduler-allocated GPU (observed {visible!r}); no CPU fallback occurred"
        )
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUDA_CUBLAS_WORKSPACE_CONFIG:
        raise DeviceUnavailableError(
            "deterministic CUDA training requires "
            f"CUBLAS_WORKSPACE_CONFIG={CUDA_CUBLAS_WORKSPACE_CONFIG}; "
            "no CPU fallback occurred"
        )
    return context


def resolve_training_device(
    profile: TrainingProfile,
    requested: str,
    *,
    allow_cpu_for_tests: bool,
) -> DeviceSelection:
    requested = requested.lower()
    fallback_requested = _truthy(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
    if requested == "mps":
        if profile.device_requirement != "mps":
            raise DeviceUnavailableError(
                f"profile {profile.name} requires {profile.device_requirement}; "
                "MPS cannot substitute for that accelerator"
            )
        if fallback_requested:
            raise DeviceUnavailableError(
                "PYTORCH_ENABLE_MPS_FALLBACK requests silent CPU fallback; "
                "set it to 0 for candidate training"
            )
        built = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        )
        available = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if not built or not available:
            raise DeviceUnavailableError(
                "MPS was required but is unavailable "
                f"(is_built={built}, is_available={available}); "
                "no CPU fallback occurred"
            )
        if profile.mps_memory_fraction is not None:
            torch.mps.set_per_process_memory_fraction(profile.mps_memory_fraction)
        return DeviceSelection(torch.device("mps"), True, False)
    if requested in {"cuda", "cuda:0"}:
        if profile.device_requirement != "cuda":
            raise DeviceUnavailableError(
                f"profile {profile.name} requires {profile.device_requirement}; "
                "CUDA results are a separate hardware condition"
            )
        _validate_cuda_scheduler_allocation()
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        if not available or count != 1:
            raise DeviceUnavailableError(
                "exactly one allocated CUDA GPU was required but is unavailable "
                f"(is_available={available}, visible_device_count={count}); "
                "no CPU fallback occurred"
            )
        device = torch.device("cuda:0")
        try:
            torch.cuda.set_device(device)
            current = int(torch.cuda.current_device())
        except (AssertionError, RuntimeError) as error:
            raise DeviceUnavailableError(
                "the scheduler-visible CUDA GPU could not be selected; "
                "no CPU fallback occurred"
            ) from error
        if current != 0:
            raise DeviceUnavailableError(
                f"CUDA resolved to unexpected logical device {current}; expected cuda:0"
            )
        try:
            device_name = str(torch.cuda.get_device_name(device))
        except (AssertionError, RuntimeError) as error:
            raise DeviceUnavailableError(
                "the scheduler-visible CUDA GPU identity could not be read; "
                "no CPU fallback occurred"
            ) from error
        if (
            profile.name in CUDA_A40_REQUIRED_PROFILE_NAMES
            and not is_nvidia_a40_name(device_name)
        ):
            raise DeviceUnavailableError(
                f"profile {profile.name} requires an NVIDIA A40; observed "
                f"{device_name!r}; no CPU fallback occurred"
            )
        return DeviceSelection(device, True, False)
    if requested == "cpu":
        if profile.device_requirement == "cuda":
            raise DeviceUnavailableError(
                f"profile {profile.name} requires CUDA; CPU fallback is forbidden"
            )
        if profile.scientific:
            raise DeviceUnavailableError(
                "scientific profile full_train_v1 requires MPS; CPU is not permitted"
            )
        if not allow_cpu_for_tests:
            raise DeviceUnavailableError(
                "CPU training is engineering-only and requires --allow-cpu-for-tests"
            )
        return DeviceSelection(torch.device("cpu"), False, fallback_requested)
    raise DeviceUnavailableError(
        f"unsupported training device {requested!r}; expected 'mps', 'cuda', "
        "or explicit test CPU"
    )


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def synchronized_time(device: torch.device) -> float:
    """Synchronize queued accelerator work immediately before reading the clock."""

    synchronize(device)
    return time.perf_counter()


def mps_memory(device: torch.device) -> dict[str, int | None]:
    """Return the legacy MPS memory fields without changing their semantics."""

    if device.type != "mps":
        return {
            "current": None,
            "driver": None,
            "recommended": None,
        }
    synchronize(device)
    recommended = (
        int(torch.mps.recommended_max_memory())
        if hasattr(torch.mps, "recommended_max_memory")
        else None
    )
    return {
        "current": int(torch.mps.current_allocated_memory()),
        "driver": int(torch.mps.driver_allocated_memory()),
        "recommended": recommended,
    }


def _cuda_driver_version() -> str | None:
    getter = getattr(torch.cuda, "driver_version", None)
    if getter is not None:
        try:
            value = getter() if callable(getter) else getter
            return str(value) if value is not None else None
        except (AssertionError, RuntimeError):
            pass
    internal_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if internal_getter is not None:
        try:
            return str(internal_getter())
        except (AssertionError, RuntimeError):
            pass
    try:
        nvml = ctypes.CDLL("libnvidia-ml.so.1")
        initialize = getattr(nvml, "nvmlInit_v2", None)
        if initialize is None:
            initialize = nvml.nvmlInit
        initialize.restype = ctypes.c_int
        get_version = nvml.nvmlSystemGetDriverVersion
        get_version.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        get_version.restype = ctypes.c_int
        shutdown = nvml.nvmlShutdown
        shutdown.restype = ctypes.c_int
        if initialize() != 0:
            return None
        try:
            buffer = ctypes.create_string_buffer(96)
            if get_version(buffer, len(buffer)) == 0:
                return buffer.value.decode("ascii", errors="replace")
        finally:
            shutdown()
    except (AttributeError, OSError):
        pass
    return None


def accelerator_runtime_metadata(device: torch.device) -> dict[str, Any]:
    """Describe the selected backend without conflating CUDA and MPS fields."""

    metadata: dict[str, Any] = {
        "schema_version": ACCELERATOR_TELEMETRY_SCHEMA_VERSION,
        "backend": device.type,
        "device": str(device),
        "identity": {
            "name": None,
            "uuid": None,
            "total_memory_bytes": None,
            "compute_capability": None,
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_runtime_version": None,
            "cuda_driver_version": None,
            "cudnn_version": None,
        },
        "allocation": {},
    }
    if device.type != "cuda":
        return metadata

    properties = torch.cuda.get_device_properties(device)
    if hasattr(properties, "major") and hasattr(properties, "minor"):
        capability = (int(properties.major), int(properties.minor))
    else:
        capability = tuple(
            int(value) for value in torch.cuda.get_device_capability(device)
        )
    uuid = getattr(properties, "uuid", None)
    cudnn_version = (
        torch.backends.cudnn.version()
        if hasattr(torch.backends, "cudnn")
        and hasattr(torch.backends.cudnn, "version")
        else None
    )
    metadata["identity"] = {
        "name": str(properties.name),
        "uuid": str(uuid) if uuid else None,
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [capability[0], capability[1]],
    }
    metadata["runtime"] = {
        "torch_version": torch.__version__,
        "cuda_runtime_version": str(torch.version.cuda)
        if torch.version.cuda is not None
        else None,
        "cuda_driver_version": _cuda_driver_version(),
        "cudnn_version": int(cudnn_version) if cudnn_version is not None else None,
    }
    metadata["allocation"] = _cuda_allocation_context()
    return metadata


def accelerator_memory(device: torch.device) -> dict[str, Any]:
    """Return versioned memory telemetry with backend-honest semantics."""

    memory: dict[str, Any] = {
        "allocated_bytes": None,
        "reserved_bytes": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "backend_specific": {},
    }
    if device.type == "cuda":
        synchronize(device)
        memory.update(
            {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    elif device.type == "mps":
        legacy = mps_memory(device)
        memory["allocated_bytes"] = legacy["current"]
        memory["backend_specific"] = {
            "driver_allocated_bytes": legacy["driver"],
            "recommended_max_memory_bytes": legacy["recommended"],
        }
    return memory


def accelerator_telemetry(
    device: torch.device,
    *,
    peak_mps_allocated_bytes: int | None = None,
) -> dict[str, Any]:
    telemetry = accelerator_runtime_metadata(device)
    memory = accelerator_memory(device)
    if device.type == "mps":
        memory["peak_allocated_bytes"] = peak_mps_allocated_bytes
    telemetry["memory"] = memory
    return telemetry


def reset_accelerator_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cleanup_accelerator(device: torch.device) -> None:
    """Synchronize and release backend caches after candidate execution."""

    if device.type == "cuda":
        synchronize(device)
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        synchronize(device)
        torch.mps.empty_cache()
