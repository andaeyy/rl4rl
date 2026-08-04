"""Shared, side-effect-free CUDA/A40 allocation identity checks."""

from __future__ import annotations

import re
from collections.abc import Mapping


CUDA_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
CUDA_ALLOCATION_ENVIRONMENT_KEYS = (
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_ON_NODE",
)
CUDA_A40_REQUIRED_PROFILE_NAMES = frozenset(
    {"full_train_cuda_a40_v1", "smoke_train_cuda_v1"}
)
_A40_NAME_PATTERN = re.compile(r"(?<![A-Z0-9])A40(?![A-Z0-9])", re.IGNORECASE)
_INDEX_RANGE_PATTERN = re.compile(r"^(\d+)-(\d+)$")


def is_nvidia_a40_name(name: object) -> bool:
    """Match the A40 model token without accepting A4000/A40-like names."""

    return isinstance(name, str) and _A40_NAME_PATTERN.search(name) is not None


def _assignment_count(name: str, raw_value: str) -> int | None:
    value = raw_value.strip()
    if not value:
        return None
    if name == "SLURM_GPUS_ON_NODE":
        if value.isdigit():
            return int(value)
        typed = re.fullmatch(r"(?:gpu:)?[A-Za-z0-9_.-]+:(\d+)", value)
        return int(typed.group(1)) if typed else None

    count = 0
    for token in (item.strip() for item in value.split(",")):
        if not token:
            return None
        match = _INDEX_RANGE_PATTERN.fullmatch(token)
        if match:
            start, end = (int(match.group(1)), int(match.group(2)))
            if end < start:
                return None
            count += end - start + 1
        else:
            count += 1
    return count


def cuda_allocation_gpu_counts(
    allocation_markers: Mapping[str, object],
) -> dict[str, int]:
    """Parse every present Slurm GPU-count signal conservatively."""

    counts: dict[str, int] = {}
    for name in CUDA_ALLOCATION_ENVIRONMENT_KEYS:
        raw_value = allocation_markers.get(name)
        if raw_value is None or not str(raw_value).strip():
            continue
        count = _assignment_count(name, str(raw_value))
        if count is None:
            return {}
        counts[name] = count
    return counts


def cuda_allocation_proves_exactly_one(
    allocation_markers: Mapping[str, object],
) -> bool:
    counts = cuda_allocation_gpu_counts(allocation_markers)
    return bool(counts) and set(counts.values()) == {1}


def slurm_tres_has_exact_count(
    raw_tres: object,
    resource_name: str,
    expected_count: int,
) -> bool:
    """Match one complete Slurm TRES entry, never a numeric substring."""

    if not isinstance(raw_tres, str) or expected_count < 0:
        return False
    expected = f"{resource_name}={expected_count}"
    return expected in {
        entry.strip() for entry in raw_tres.split(",") if entry.strip()
    }
