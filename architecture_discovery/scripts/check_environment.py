from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.cuda_contract import (
    CUDA_ALLOCATION_ENVIRONMENT_KEYS,
    cuda_allocation_gpu_counts,
    cuda_allocation_proves_exactly_one,
    is_nvidia_a40_name,
)
from common.gpt56_sol import TARGET_MODEL
from common.training_config import PROFILES, get_training_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report accelerator readiness without treating login-node GPU "
            "visibility as allocated-GPU validation."
        )
    )
    parser.add_argument(
        "--device",
        choices=("mps", "cuda", "cpu"),
        default=os.environ.get("DISCOVERY_TRAIN_DEVICE", "mps"),
    )
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(PROFILES)),
        default=os.environ.get("DISCOVERY_TRAINING_PROFILE"),
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="exit nonzero unless the requested accelerator is strictly ready",
    )
    parser.add_argument(
        "--require-a40",
        action="store_true",
        help="also require the allocated CUDA device name to contain A40",
    )
    return parser


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _cuda_allocation() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_devices = [value.strip() for value in visible.split(",") if value.strip()]
    markers = {
        key: os.environ.get(key, "") for key in CUDA_ALLOCATION_ENVIRONMENT_KEYS
    }
    job_id = os.environ.get("SLURM_JOB_ID", "")
    valid = bool(
        job_id
        and cuda_allocation_proves_exactly_one(markers)
        and len(visible_devices) == 1
        and visible.lower() not in {"-1", "none", "nodevfiles", "void"}
    )
    return {
        "scheduler": "slurm",
        "job_id": job_id or None,
        "gpu_allocation_markers": {
            key: value or None for key, value in markers.items()
        },
        "gpu_allocation_counts": cuda_allocation_gpu_counts(markers),
        "exactly_one_allocated_gpu": cuda_allocation_proves_exactly_one(markers),
        "cuda_visible_devices": visible or None,
        "exactly_one_visible_device": len(visible_devices) == 1,
        "allocation_evidence_valid": valid,
    }


def _nvidia_smi() -> tuple[list[dict[str, str]], str | None]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return [], f"{type(error).__name__}: {error}"[:500]
    if completed.returncode != 0:
        return [], (
            f"nvidia-smi exit {completed.returncode}: {completed.stderr.strip()}"
        )[:500]
    field_names = ("index", "name", "uuid", "memory_total_mib", "driver")
    rows = []
    for row in csv.reader(io.StringIO(completed.stdout)):
        if len(row) != len(field_names):
            continue
        rows.append(
            {
                key: value.strip()
                for key, value in zip(field_names, row, strict=True)
            }
        )
    return rows, None


def _cuda_report(allocation: dict[str, Any]) -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if available else 0
    device: dict[str, Any] | None = None
    nvidia_rows: list[dict[str, str]] = []
    nvidia_error: str | None = None
    validation_performed = bool(
        allocation["allocation_evidence_valid"] and available and device_count == 1
    )
    if validation_performed:
        properties = torch.cuda.get_device_properties(0)
        capability = torch.cuda.get_device_capability(0)
        uuid = getattr(properties, "uuid", None)
        device = {
            "logical_index": 0,
            "name": str(properties.name),
            "uuid": str(uuid) if uuid else None,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(capability[0]), int(capability[1])],
        }
        nvidia_rows, nvidia_error = _nvidia_smi()
    return {
        "built": torch.version.cuda is not None,
        "available": available,
        "visible_device_count": device_count,
        "runtime_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if hasattr(torch.backends, "cudnn")
            else None
        ),
        "selected_device": device,
        "driver_inventory": nvidia_rows,
        "driver_inventory_error": nvidia_error,
        "allocated_gpu_validation_performed": validation_performed,
        "validation_scope": (
            "slurm_compute_allocation"
            if validation_performed
            else "not_performed_without_valid_allocation"
        ),
    }


def environment_report(
    *,
    requested_device: str,
    profile_name: str | None,
) -> dict[str, Any]:
    allocation = _cuda_allocation()
    cuda = _cuda_report(allocation)
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested_device == "mps" and mps_available:
        device_status = "mps_ready"
    elif requested_device == "mps":
        device_status = "mps_unavailable_no_fallback"
    elif requested_device == "cuda" and not allocation["allocation_evidence_valid"]:
        device_status = "cuda_unallocated_no_fallback"
    elif requested_device == "cuda" and not cuda["available"]:
        device_status = "cuda_unavailable_no_fallback"
    elif requested_device == "cuda" and cuda["visible_device_count"] != 1:
        device_status = "cuda_visible_device_count_mismatch_no_fallback"
    elif requested_device == "cuda":
        device_status = "cuda_ready"
    else:
        device_status = "cpu_test_requested_no_fallback"

    profile = get_training_profile(profile_name) if profile_name else None
    operating_system = platform.platform()
    mps_built = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
    )
    cpu_test_flag = os.environ.get("DISCOVERY_ALLOW_CPU_TRAINING", "0")
    mps_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")
    report = {
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "operating_system": operating_system,
        # Backward-compatible aliases retained for existing environment readers.
        "platform": operating_system,
        "torch": torch.__version__,
        "git_commit": _git_commit(),
        "training_device_requested": requested_device,
        "training_device_status": device_status,
        "scientific_cpu_fallback": False,
        "cpu_training_test_flag": cpu_test_flag,
        "pytorch_mps_fallback": mps_fallback,
        "mps_built": mps_built,
        "mps_available": mps_available,
        "training": {
            "requested_device": requested_device,
            "device_status": device_status,
            "profile_name": profile.name if profile else None,
            "profile_version": profile.version if profile else None,
            "profile_hash": profile.profile_hash if profile else None,
            "cpu_fallback": False,
            "cpu_training_test_flag": cpu_test_flag,
            "pytorch_mps_fallback": mps_fallback,
        },
        "scheduler_allocation": allocation,
        "cuda": cuda,
        "mps": {
            "built": mps_built,
            "available": mps_available,
        },
        "generation": {
            "target_model": TARGET_MODEL,
            "configured_model": os.environ.get("DISCOVERY_MODEL"),
            "model_matches_target": (
                os.environ.get("DISCOVERY_MODEL") == TARGET_MODEL
            ),
            "reasoning_effort": os.environ.get(
                "DISCOVERY_REASONING_EFFORT", "high (config default)"
            ),
            "max_completion_tokens": os.environ.get(
                "DISCOVERY_MAX_COMPLETION_TOKENS", "16384 (config default)"
            ),
        },
        # Report booleans only: never serialize provider values or environment names.
        "provider_configuration": {
            "key_present": bool(os.environ.get("DISCOVERY_API_KEY")),
            "endpoint_present": bool(os.environ.get("DISCOVERY_API_BASE")),
            "model_present": bool(os.environ.get("DISCOVERY_MODEL")),
        },
    }
    return report


def _readiness_errors(report: dict[str, Any], *, require_a40: bool) -> list[str]:
    requested = report["training"]["requested_device"]
    status = report["training"]["device_status"]
    errors: list[str] = []
    if status != f"{requested}_ready":
        errors.append(f"requested accelerator is not ready: {status}")
    if requested == "cuda":
        cuda = report["cuda"]
        if not cuda["allocated_gpu_validation_performed"]:
            errors.append("CUDA was not validated inside a Slurm GPU allocation")
        if cuda["driver_inventory_error"]:
            errors.append(str(cuda["driver_inventory_error"]))
        if not cuda["driver_inventory"]:
            errors.append("NVIDIA driver/GPU inventory is unavailable")
        if require_a40:
            selected = cuda["selected_device"] or {}
            if not is_nvidia_a40_name(selected.get("name")):
                errors.append("the allocated CUDA device is not an NVIDIA A40")
    elif require_a40:
        errors.append("--require-a40 requires --device cuda")
    return errors


def main() -> None:
    args = build_parser().parse_args()
    report = environment_report(
        requested_device=args.device,
        profile_name=args.profile,
    )
    errors = _readiness_errors(report, require_a40=args.require_a40)
    report["strict_readiness"] = {
        "required": bool(args.require_ready or args.require_a40),
        "passed": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if (args.require_ready or args.require_a40) and errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
