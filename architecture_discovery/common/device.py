"""Strict device selection and Apple-MPS telemetry."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from common.training_config import TrainingProfile


class DeviceUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceSelection:
    device: torch.device
    hardware_matched: bool
    fallback_requested: bool


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def resolve_training_device(
    profile: TrainingProfile,
    requested: str,
    *,
    allow_cpu_for_tests: bool,
) -> DeviceSelection:
    requested = requested.lower()
    fallback_requested = _truthy(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
    if requested == "mps":
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
                f"(is_built={built}, is_available={available}); no CPU fallback occurred"
            )
        if profile.mps_memory_fraction is not None:
            torch.mps.set_per_process_memory_fraction(profile.mps_memory_fraction)
        return DeviceSelection(torch.device("mps"), True, False)
    if requested == "cpu":
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
        f"unsupported training device {requested!r}; expected 'mps' or explicit test CPU"
    )


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def synchronized_time(device: torch.device) -> float:
    """Synchronize queued device work immediately before reading the clock."""

    synchronize(device)
    return time.perf_counter()


def mps_memory(device: torch.device) -> dict[str, int | None]:
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
