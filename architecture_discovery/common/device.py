"""Strict CPU, Apple MPS, and NVIDIA CUDA accelerator handling."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, ClassVar

import torch

from common.training_config import TrainingProfile


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested accelerator cannot be used without fallback."""


class DeterministicOperationUnavailableError(RuntimeError):
    """Raised when the frozen deterministic profile lacks a required kernel."""


class AcceleratorKind(StrEnum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"

    @classmethod
    def parse(cls, requested: str) -> AcceleratorKind:
        normalized = requested.strip().lower()
        if normalized.startswith("cuda:"):
            normalized = "cuda"
        try:
            return cls(normalized)
        except ValueError as error:
            raise DeviceUnavailableError(
                f"unsupported training device {requested!r}; expected one of "
                "'cpu', 'mps', or 'cuda'"
            ) from error


@dataclass(frozen=True)
class AcceleratorFingerprint:
    FIELD_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "requested_device",
            "selected_device",
            "accelerator_kind",
            "gpu_name",
            "gpu_count",
            "compute_capability",
            "cuda_runtime",
            "cuda_driver",
            "torch_version",
            "host_platform",
        }
    )

    requested_device: str
    selected_device: str
    accelerator_kind: str
    gpu_name: str | None
    gpu_count: int
    compute_capability: str | None
    cuda_runtime: str | None
    cuda_driver: str | None
    torch_version: str
    host_platform: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: object) -> AcceleratorFingerprint:
        """Parse the exact serialized fingerprint schema without coercion."""

        if not isinstance(payload, Mapping):
            raise ValueError("accelerator fingerprint must be an object")
        if set(payload) != cls.FIELD_NAMES:
            raise ValueError("accelerator fingerprint has an invalid exact schema")
        for field in (
            "requested_device",
            "selected_device",
            "accelerator_kind",
            "torch_version",
            "host_platform",
        ):
            value = payload[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"accelerator fingerprint {field} must be nonempty text"
                )
        if payload["accelerator_kind"] not in {
            kind.value for kind in AcceleratorKind
        }:
            raise ValueError("accelerator fingerprint kind is unsupported")
        gpu_count = payload["gpu_count"]
        if (
            not isinstance(gpu_count, int)
            or isinstance(gpu_count, bool)
            or gpu_count < 0
        ):
            raise ValueError(
                "accelerator fingerprint gpu_count must be a nonnegative integer"
            )
        for field in (
            "gpu_name",
            "compute_capability",
            "cuda_runtime",
            "cuda_driver",
        ):
            value = payload[field]
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise ValueError(
                    f"accelerator fingerprint {field} must be nonempty text or null"
                )
        return cls(
            requested_device=payload["requested_device"],
            selected_device=payload["selected_device"],
            accelerator_kind=payload["accelerator_kind"],
            gpu_name=payload["gpu_name"],
            gpu_count=gpu_count,
            compute_capability=payload["compute_capability"],
            cuda_runtime=payload["cuda_runtime"],
            cuda_driver=payload["cuda_driver"],
            torch_version=payload["torch_version"],
            host_platform=payload["host_platform"],
        )

    def validate_cuda(
        self,
        *,
        exact_gpu_count: int | None = None,
        require_driver: bool = False,
    ) -> AcceleratorFingerprint:
        """Validate the CUDA-specific semantics of a parsed fingerprint."""

        try:
            requested_kind = AcceleratorKind.parse(self.requested_device)
            selected_kind = AcceleratorKind.parse(self.selected_device)
        except DeviceUnavailableError as error:
            raise ValueError(str(error)) from error
        if requested_kind is not AcceleratorKind.CUDA:
            raise ValueError("accelerator fingerprint did not request CUDA")
        if selected_kind is not AcceleratorKind.CUDA:
            raise ValueError("accelerator fingerprint did not select CUDA")
        if self.accelerator_kind != AcceleratorKind.CUDA.value:
            raise ValueError("accelerator fingerprint kind is not CUDA")
        if self.gpu_count < 1:
            raise ValueError("CUDA accelerator fingerprint exposes no GPU")
        if exact_gpu_count is not None and self.gpu_count != exact_gpu_count:
            raise ValueError(
                "CUDA accelerator fingerprint expected exactly "
                f"{exact_gpu_count} GPU, observed {self.gpu_count}"
            )
        if self.gpu_name is None or not self.gpu_name.strip():
            raise ValueError("CUDA accelerator fingerprint GPU name is missing")
        capability = self.compute_capability
        if capability is None:
            raise ValueError(
                "CUDA accelerator fingerprint compute capability is missing"
            )
        capability_parts = capability.split(".")
        if len(capability_parts) != 2 or any(
            not part.isdigit() for part in capability_parts
        ):
            raise ValueError(
                "CUDA accelerator fingerprint compute capability is invalid"
            )
        if self.cuda_runtime is None or not self.cuda_runtime.strip():
            raise ValueError("CUDA accelerator fingerprint runtime is missing")
        if require_driver and (
            self.cuda_driver is None or not self.cuda_driver.strip()
        ):
            raise ValueError("CUDA accelerator fingerprint driver is missing")
        return self


@dataclass(frozen=True)
class DeviceSelection:
    device: torch.device
    hardware_matched: bool
    fallback_requested: bool
    requested_kind: AcceleratorKind
    fingerprint: AcceleratorFingerprint


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _cuda_driver_version() -> str | None:
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is None:
        return None
    driver = getattr(cuda_module, "driver_version", None)
    if callable(driver):
        try:
            value = driver()
        except (RuntimeError, TypeError, ValueError):
            pass
        else:
            if (
                isinstance(value, (str, int))
                and not isinstance(value, bool)
                and str(value).strip()
            ):
                return str(value).strip()
    torch_c = getattr(torch, "_C", None)
    for name in ("_cuda_getDriverVersion", "_cuda_get_driver_version"):
        probe = getattr(torch_c, name, None)
        if not callable(probe):
            continue
        try:
            value = probe()
        except (RuntimeError, TypeError, ValueError):
            continue
        if (
            isinstance(value, (str, int))
            and not isinstance(value, bool)
            and str(value).strip()
        ):
            return str(value).strip()
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    versions = tuple(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )
    if not versions or len(set(versions)) != 1:
        return None
    return versions[0]


def accelerator_fingerprint(
    device: torch.device,
    *,
    requested_device: str,
) -> AcceleratorFingerprint:
    kind = AcceleratorKind.parse(device.type)
    gpu_name: str | None = None
    gpu_count = 0
    compute_capability: str | None = None
    cuda_runtime: str | None = None
    cuda_driver: str | None = None
    if kind is AcceleratorKind.CUDA:
        index = device.index
        if index is None:
            index = int(torch.cuda.current_device())
        gpu_count = int(torch.cuda.device_count())
        gpu_name = str(torch.cuda.get_device_name(index))
        major, minor = torch.cuda.get_device_capability(index)
        compute_capability = f"{int(major)}.{int(minor)}"
        cuda_runtime = str(torch.version.cuda) if torch.version.cuda else None
        cuda_driver = _cuda_driver_version()
    elif kind is AcceleratorKind.MPS:
        gpu_count = 1
        gpu_name = platform.processor() or platform.machine() or "Apple MPS"
    return AcceleratorFingerprint(
        requested_device=requested_device,
        selected_device=str(device),
        accelerator_kind=kind.value,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        compute_capability=compute_capability,
        cuda_runtime=cuda_runtime,
        cuda_driver=cuda_driver,
        torch_version=str(torch.__version__),
        host_platform=platform.platform(),
    )


def _resolve_cuda_device(requested: str) -> torch.device:
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        built_runtime = getattr(torch.version, "cuda", None)
        device_count = (
            torch.cuda.device_count() if hasattr(torch, "cuda") else 0
        )
        raise DeviceUnavailableError(
            "CUDA was required but is unavailable "
            f"(torch.version.cuda={built_runtime!r}, "
            f"device_count={device_count}); "
            "no CPU fallback occurred"
        )
    count = int(torch.cuda.device_count())
    if count < 1:
        raise DeviceUnavailableError(
            "CUDA reported available without a visible GPU; no CPU fallback occurred"
        )
    if requested.strip().lower().startswith("cuda:"):
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError as error:
            raise DeviceUnavailableError(
                f"invalid CUDA device selector {requested!r}"
            ) from error
    else:
        index = int(torch.cuda.current_device())
    if index < 0 or index >= count:
        raise DeviceUnavailableError(
            f"CUDA device index {index} is outside the visible range 0..{count - 1}"
        )
    return torch.device("cuda", index)


def configure_determinism(profile: TrainingProfile, device: torch.device) -> None:
    """Apply the deterministic settings bound into the selected profile."""

    torch.use_deterministic_algorithms(profile.deterministic_algorithms)
    if device.type != AcceleratorKind.CUDA.value:
        return
    expected_workspace = profile.cublas_workspace_config
    observed_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if expected_workspace and observed_workspace != expected_workspace:
        raise DeviceUnavailableError(
            "deterministic CUDA requires "
            f"CUBLAS_WORKSPACE_CONFIG={expected_workspace!r}; "
            f"observed {observed_workspace!r}"
        )
    torch.backends.cudnn.deterministic = profile.cudnn_deterministic
    torch.backends.cudnn.benchmark = profile.cudnn_benchmark
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = profile.allow_tf32
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = profile.allow_tf32


def resolve_training_device(
    profile: TrainingProfile,
    requested: str,
    *,
    allow_cpu_for_tests: bool,
) -> DeviceSelection:
    kind = AcceleratorKind.parse(requested)
    mps_fallback_requested = _truthy(
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    )
    if kind is AcceleratorKind.MPS:
        if mps_fallback_requested:
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
        device = torch.device("mps")
        if profile.accelerator_memory_fraction is not None:
            torch.mps.set_per_process_memory_fraction(
                profile.accelerator_memory_fraction
            )
    elif kind is AcceleratorKind.CUDA:
        device = _resolve_cuda_device(requested)
        if profile.accelerator_memory_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(
                profile.accelerator_memory_fraction,
                device=device,
            )
    else:
        if profile.scientific:
            raise DeviceUnavailableError(
                f"scientific profile {profile.name} requires "
                f"{profile.device_requirement.upper()}; CPU is not permitted"
            )
        if not allow_cpu_for_tests:
            raise DeviceUnavailableError(
                "CPU training is engineering-only and requires --allow-cpu-for-tests"
            )
        device = torch.device("cpu")

    if (
        kind is not AcceleratorKind.CPU
        and profile.device_requirement != kind.value
    ):
        raise DeviceUnavailableError(
            f"profile {profile.name} requires {profile.device_requirement!r}, "
            f"not {kind.value!r}"
        )
    configure_determinism(profile, device)
    fingerprint = accelerator_fingerprint(device, requested_device=requested)
    return DeviceSelection(
        device=device,
        hardware_matched=(kind.value == profile.device_requirement),
        fallback_requested=mps_fallback_requested,
        requested_kind=kind,
        fingerprint=fingerprint,
    )


def synchronize(device: torch.device) -> None:
    if device.type == AcceleratorKind.MPS.value:
        torch.mps.synchronize()
    elif device.type == AcceleratorKind.CUDA.value:
        torch.cuda.synchronize(device)


def synchronized_time(device: torch.device) -> float:
    """Synchronize queued accelerator work before reading the host clock."""

    synchronize(device)
    return time.perf_counter()


def accelerator_memory(device: torch.device) -> dict[str, int | None]:
    if device.type == AcceleratorKind.MPS.value:
        synchronize(device)
        recommended = (
            int(torch.mps.recommended_max_memory())
            if hasattr(torch.mps, "recommended_max_memory")
            else None
        )
        return {
            "current": int(torch.mps.current_allocated_memory()),
            "reserved_or_driver": int(torch.mps.driver_allocated_memory()),
            "peak": None,
            "recommended_or_total": recommended,
        }
    if device.type == AcceleratorKind.CUDA.value:
        synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        return {
            "current": int(torch.cuda.memory_allocated(device)),
            "reserved_or_driver": int(torch.cuda.memory_reserved(device)),
            "peak": int(torch.cuda.max_memory_allocated(device)),
            "recommended_or_total": int(properties.total_memory),
        }
    return {
        "current": None,
        "reserved_or_driver": None,
        "peak": None,
        "recommended_or_total": None,
    }


def reset_peak_memory(device: torch.device) -> None:
    if device.type == AcceleratorKind.CUDA.value:
        torch.cuda.reset_peak_memory_stats(device)


def cleanup_accelerator(device: torch.device) -> None:
    if device.type == AcceleratorKind.MPS.value:
        torch.mps.empty_cache()
    elif device.type == AcceleratorKind.CUDA.value:
        synchronize(device)
        torch.cuda.empty_cache()


# Historical import compatibility. New writers use accelerator_memory.
def mps_memory(device: torch.device) -> dict[str, int | None]:
    values = accelerator_memory(device)
    return {
        "current": values["current"],
        "driver": values["reserved_or_driver"],
        "recommended": values["recommended_or_total"],
    }
