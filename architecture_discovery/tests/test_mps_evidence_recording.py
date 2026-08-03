from __future__ import annotations

import hashlib
import json

import torch

from common.training_config import FULL_TRAIN_V1
from scripts.audit_scientific_readiness import audit_readiness
from scripts.record_mps_validation import record_mps_validation


def test_full_mps_receipt_hashes_and_revalidates_training_artifacts(
    tmp_path, monkeypatch
) -> None:
    training = tmp_path / "full-training"
    training.mkdir()
    candidate = b"def build_untrained_model(seed):\n    raise NotImplementedError\n"
    candidate_hash = hashlib.sha256(candidate).hexdigest()
    (training / "candidate_source.py").write_bytes(candidate)
    summary = {
        "success": True,
        "scientific": True,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
        "steps_completed": FULL_TRAIN_V1.max_steps,
        "profile_name": FULL_TRAIN_V1.name,
        "profile_version": FULL_TRAIN_V1.version,
        "profile_hash": FULL_TRAIN_V1.profile_hash,
        "device": "mps",
        "candidate_source_hash": candidate_hash,
    }
    manifest = {
        "allow_cpu_for_tests": False,
        "hardware_matched_scientific_run": True,
        "candidate_source_hash": candidate_hash,
        "profile_hash": FULL_TRAIN_V1.profile_hash,
        "requested_device": "mps",
        "selected_device": "mps",
        "parameter_count_role": "descriptive_metadata_only",
        "isolation_level": "scientific_gate_allowed",
        "runtime": {
            "mps_built": True,
            "mps_available": True,
            "pytorch_enable_mps_fallback": "0",
        },
    }
    (training / "training_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (training / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    receipt = tmp_path / "mps-evidence.json"

    evidence = record_mps_validation(
        training_output_dir=training,
        output_path=receipt,
    )

    assert evidence["mps_available"] is True
    assert evidence["cpu_fallback"] is False
    assert evidence["steps_completed"] == FULL_TRAIN_V1.max_steps
    report = audit_readiness(mps_evidence=receipt)
    gate = next(
        item for item in report["gates"] if item["gate"] == "full_profile_mps_validation"
    )
    assert gate["passed"], gate["blockers"]
