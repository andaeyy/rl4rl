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
    requested_device = os.environ.get("DISCOVERY_TRAIN_DEVICE", "mps")
    mps_available = torch.backends.mps.is_available()
    if requested_device == "mps" and mps_available:
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
