#!/usr/bin/env python3
"""Create a hash-linked full-profile MPS evidence record after training.

This command never trains a model and never calls a provider. It validates an
already completed ``full_train_v1`` output directory and creates the readiness
receipt exactly once.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training_config import FULL_TRAIN_V1
from study.serialization import create_json_exclusive


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_bool(payload: dict[str, Any], field: str, expected: bool) -> None:
    value = payload.get(field)
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")


def _validate_training_output(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = output_dir / "training_summary.json"
    manifest_path = output_dir / "training_manifest.json"
    for path in (summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(manifest, dict):
        raise ValueError("training summary and manifest must be JSON objects")
    candidate_format = manifest.get("candidate_format", "arbitrary_python")
    if candidate_format not in {"architecture_ir", "arbitrary_python"}:
        raise ValueError("training manifest has an unsupported candidate_format")
    candidate_path = output_dir / (
        "candidate_graph.json"
        if candidate_format == "architecture_ir"
        else "candidate_source.py"
    )
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)

    for field, expected in {
        "success": True,
        "scientific": True,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
    }.items():
        _exact_bool(summary, field, expected)
    if (
        not isinstance(summary.get("steps_completed"), int)
        or isinstance(summary["steps_completed"], bool)
        or summary["steps_completed"] != FULL_TRAIN_V1.max_steps
    ):
        raise ValueError("training did not complete the exact full_train_v1 step count")
    expected_summary = {
        "profile_name": FULL_TRAIN_V1.name,
        "profile_version": FULL_TRAIN_V1.version,
        "profile_hash": FULL_TRAIN_V1.profile_hash,
        "device": "mps",
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"training summary {field} does not match full_train_v1")

    for field, expected in {
        "allow_cpu_for_tests": False,
        "hardware_matched_scientific_run": True,
    }.items():
        _exact_bool(manifest, field, expected)
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("training manifest lacks runtime evidence")
    _exact_bool(runtime, "mps_built", True)
    _exact_bool(runtime, "mps_available", True)
    if runtime.get("pytorch_enable_mps_fallback") not in {"", "0"}:
        raise ValueError("training manifest requested PyTorch MPS fallback")
    expected_manifest = {
        "candidate_source_hash": summary.get("candidate_source_hash"),
        "profile_hash": FULL_TRAIN_V1.profile_hash,
        "requested_device": "mps",
        "selected_device": "mps",
        "parameter_count_role": "descriptive_metadata_only",
        "isolation_level": "scientific_gate_allowed",
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"training manifest {field} does not match the run")
    candidate_hash = _sha256_file(candidate_path)
    if candidate_hash != summary.get("candidate_source_hash"):
        raise ValueError("candidate source hash does not match the training summary")
    return summary, manifest


def record_mps_validation(
    *, training_output_dir: str | Path, output_path: str | Path
) -> dict[str, Any]:
    if not (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS must be built and available in this recording process")
    output_dir = Path(training_output_dir).resolve()
    summary, _manifest = _validate_training_output(output_dir)
    evidence = {
        "schema_name": "FullProfileMPSValidationEvidence",
        "schema_version": "1.0",
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "profile_name": FULL_TRAIN_V1.name,
        "profile_version": FULL_TRAIN_V1.version,
        "profile_hash": FULL_TRAIN_V1.profile_hash,
        "requested_device": "mps",
        "selected_device": "mps",
        "mps_available": True,
        "cpu_fallback": False,
        "success": True,
        "steps_completed": FULL_TRAIN_V1.max_steps,
        "candidate_source_hash": summary["candidate_source_hash"],
        "training_manifest_hash": _sha256_file(
            output_dir / "training_manifest.json"
        ),
        "training_summary_hash": _sha256_file(output_dir / "training_summary.json"),
        "training_output_dir": str(output_dir),
    }
    create_json_exclusive(Path(output_path).resolve(), evidence)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a completed full_train_v1 MPS run without training or API calls."
    )
    parser.add_argument("--training-output-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "readiness" / "full_train_v1_mps_evidence.json",
    )
    arguments = parser.parse_args()
    evidence = record_mps_validation(
        training_output_dir=arguments.training_output_dir,
        output_path=arguments.output,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
