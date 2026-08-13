from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.gpt56_sol import TARGET_MODEL


def main() -> None:
    requested_device = os.environ.get("DISCOVERY_TRAIN_DEVICE", "cuda")
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    cuda_device_count = int(torch.cuda.device_count())
    if requested_device.startswith("cuda") and cuda_available:
        device_status = "cuda_ready"
    elif requested_device.startswith("cuda"):
        device_status = "cuda_unavailable_no_fallback"
    elif requested_device == "mps" and mps_available:
        device_status = "mps_ready"
    elif requested_device == "mps":
        device_status = "mps_unavailable_no_fallback"
    else:
        device_status = f"{requested_device}_requested"
    report = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "training_device_requested": requested_device,
        "training_device_status": device_status,
        "scientific_cpu_fallback": False,
        "cpu_training_test_flag": os.environ.get(
            "DISCOVERY_ALLOW_CPU_TRAINING", "0"
        ),
        "pytorch_mps_fallback": os.environ.get(
            "PYTORCH_ENABLE_MPS_FALLBACK", "unset"
        ),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps_available,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "compute_capability": ".".join(
                    str(part) for part in torch.cuda.get_device_capability(index)
                ),
                "total_memory_bytes": int(
                    torch.cuda.get_device_properties(index).total_memory
                ),
            }
            for index in range(cuda_device_count)
        ]
        if cuda_available
        else [],
        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG", "unset"
        ),
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
        "credentials": {
            name: bool(os.environ.get(name))
            for name in (
                "DISCOVERY_API_KEY",
                "DISCOVERY_API_BASE",
                "DISCOVERY_MODEL",
            )
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
