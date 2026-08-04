from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from common.training_config import SMOKE_TRAIN_CUDA_V1, TrainingSeedBundle
from scripts.audit_scientific_readiness import audit_readiness
from scripts.record_cuda_a40_validation import (
    record_cuda_a40_validation,
    validate_cuda_a40_receipt,
)
from study.budget import (
    AcceleratorBackend,
    AcceleratorResourceLedger,
    AcceleratorResourceSpec,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _completed_cuda_smoke(tmp_path: Path) -> Path:
    output = tmp_path / "cuda-smoke"
    output.mkdir(parents=True)
    candidate = (ROOT / "common" / "initial_candidate.py").read_bytes()
    candidate_hash = _sha256(candidate)
    (output / "candidate_source.py").write_bytes(candidate)
    checkpoint = b"mock trusted checkpoint bytes"
    checkpoint_hash = _sha256(checkpoint)
    checkpoint_path = output / "best_checkpoint.pt"
    checkpoint_path.write_bytes(checkpoint)
    event_path = output / "events.jsonl"
    event_path.write_text(
        json.dumps(
            {
                "optimizer_step": 10,
                "validation_loss": 0.5,
                "validation_exact_match_accuracy": 0.25,
                "timing_synchronized": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    identity = {
        "name": "NVIDIA A40",
        "uuid": "GPU-a40-test",
        "total_memory_bytes": 48 * 1024**3,
        "compute_capability": [8, 6],
    }
    runtime = {
        "torch_version": "2.7.1+cu126",
        "cuda_runtime_version": "12.6",
        "cuda_driver_version": "570.00",
        "cudnn_version": 90501,
    }
    memory = {
        "allocated_bytes": 1024,
        "reserved_bytes": 2048,
        "peak_allocated_bytes": 4096,
        "peak_reserved_bytes": 8192,
        "backend_specific": {},
    }
    telemetry = {
        "schema_version": "accelerator_telemetry_v1",
        "backend": "cuda",
        "device": "cuda:0",
        "identity": identity,
        "runtime": runtime,
        "allocation": {
            "scheduler": "slurm",
            "job_id": "6499999",
            "visible_devices": "0",
            "allocation_markers": {
                "SLURM_JOB_GPUS": "2",
                "SLURM_STEP_GPUS": "",
                "SLURM_GPUS_ON_NODE": "1",
            },
        },
        "memory": memory,
    }
    seeds = TrainingSeedBundle.from_run_seed(1)
    summary = {
        "success": True,
        "failure_stage": "",
        "error": "",
        "profile_name": SMOKE_TRAIN_CUDA_V1.name,
        "profile_version": SMOKE_TRAIN_CUDA_V1.version,
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "candidate_source_hash": candidate_hash,
        "initialization_seed": seeds.model_initialization_seed,
        "data_seed": seeds.training_data_seed,
        "development_seed": seeds.development_set_seed,
        "dataloader_seed": seeds.dataloader_seed,
        "device": "cuda:0",
        "dtype": "float32",
        "steps_completed": 10,
        "examples_processed": 160,
        "best_development_step": 10,
        "best_development_exact_match_accuracy": 0.25,
        "best_development_loss": 0.5,
        "final_training_loss": 0.75,
        "train_seconds": 1.5,
        "parameter_count_metadata": 1234,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "event_log_path": str(event_path),
        "unsupported_operation_fallback": False,
        "scientific": False,
        "hardware_matched": True,
        "cleanup_completed": True,
        "accelerator_telemetry_schema_version": "accelerator_telemetry_v1",
        "accelerator_telemetry": telemetry,
        "timing_synchronized": True,
        "initial_parameter_sha256": "1" * 64,
        "final_parameter_sha256": "2" * 64,
        "parameters_changed": True,
    }
    manifest = {
        "candidate_source_hash": candidate_hash,
        "profile": asdict(SMOKE_TRAIN_CUDA_V1),
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "seed_bundle": asdict(seeds),
        "requested_device": "cuda",
        "selected_device": "cuda:0",
        "allow_cpu_for_tests": False,
        "hardware_matched_scientific_run": False,
        "runtime": {
            "accelerator": {
                key: value for key, value in telemetry.items() if key != "memory"
            },
            "cuda_determinism": {
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "float32_matmul_precision": "highest",
            },
        },
        "controller_source_hash": "3" * 64,
        "dependency_lock_hash": "4" * 64,
        "parameter_count_role": "descriptive_metadata_only",
        "scientific_limitations": ["Engineering only."],
        "isolation_level": "engineering_only_or_scientific_gate_blocked",
    }
    summary_path = output / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (output / "training_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    condition = (
        f"nvidia_a40:{SMOKE_TRAIN_CUDA_V1.name}:"
        f"{SMOKE_TRAIN_CUDA_V1.profile_hash}"
    )
    ledger = AcceleratorResourceLedger(
        AcceleratorResourceSpec(
            backend=AcceleratorBackend.CUDA,
            hardware_condition=condition,
            accelerator_seconds=float(SMOKE_TRAIN_CUDA_V1.maximum_wall_seconds),
        )
    )
    ledger.record(
        summary["train_seconds"],
        backend=AcceleratorBackend.CUDA,
        hardware_condition=condition,
    )
    (output / "accelerator_resource_usage.json").write_text(
        json.dumps(
            {
                "schema_name": "AcceleratorTrainingResourceRecord",
                "schema_version": "2.0",
                "accelerator_key": "cuda:uuid:GPU-a40-test",
                "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
                "candidate_source_hash": candidate_hash,
                "selected_device": "cuda:0",
                "training_summary_sha256": _sha256(summary_path.read_bytes()),
                "resource_ledger": ledger.to_dict(),
            }
        ),
        encoding="utf-8",
    )
    return output


def test_cuda_a40_receipt_hashes_artifacts_without_authorizing_science(
    tmp_path,
) -> None:
    output = _completed_cuda_smoke(tmp_path)
    receipt_path = tmp_path / "cuda-a40-receipt.json"

    receipt = record_cuda_a40_validation(
        training_output_dir=output,
        output_path=receipt_path,
    )

    assert receipt["profile"]["optimizer_steps"] == 10
    assert receipt["execution"]["selected_device"] == "cuda:0"
    assert receipt["execution"]["parameters_changed"] is True
    assert receipt["accelerator"]["memory"]["peak_allocated_bytes"] > 0
    assert receipt["credential_scan"]["passed"] is True
    assert receipt["scientific_readiness"] is False
    assert receipt["formal_scientific_execution_authorized"] is False
    assert receipt["cross_backend_pooling_authorized"] is False
    assert validate_cuda_a40_receipt(receipt_path) == receipt

    readiness = audit_readiness(cuda_a40_evidence=receipt_path)
    gate = next(
        item
        for item in readiness["gates"]
        if item["gate"] == "cuda_a40_engineering_validation_receipt"
    )
    assert gate["passed"], gate["blockers"]
    assert readiness["readiness_levels"]["cuda_a40_smoke_validated"] is True
    assert readiness["ready"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("steps_completed", 9, "exactly 10"),
        ("device", "cpu", "does not match CUDA smoke"),
        ("parameters_changed", False, "must be exactly True"),
        ("timing_synchronized", False, "must be exactly True"),
    ],
)
def test_cuda_a40_receipt_rejects_incomplete_or_fallback_like_results(
    tmp_path,
    field: str,
    value: object,
    message: str,
) -> None:
    output = _completed_cuda_smoke(tmp_path)
    summary_path = output / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        record_cuda_a40_validation(
            training_output_dir=output,
            output_path=tmp_path / "rejected.json",
        )


def test_cuda_a40_receipt_rejects_credential_names_in_artifacts(tmp_path) -> None:
    output = _completed_cuda_smoke(tmp_path)
    (output / "worker.err").write_text(
        "forbidden variable name: OPENAI_API_KEY\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="credential marker names"):
        record_cuda_a40_validation(
            training_output_dir=output,
            output_path=tmp_path / "rejected.json",
        )


def test_cuda_a40_receipt_rejects_a4000_name(tmp_path) -> None:
    output = _completed_cuda_smoke(tmp_path)
    for filename in ("training_summary.json", "training_manifest.json"):
        path = output / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        if filename == "training_summary.json":
            payload["accelerator_telemetry"]["identity"]["name"] = (
                "NVIDIA RTX A4000"
            )
        else:
            payload["runtime"]["accelerator"]["identity"]["name"] = (
                "NVIDIA RTX A4000"
            )
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="NVIDIA A40"):
        record_cuda_a40_validation(
            training_output_dir=output,
            output_path=tmp_path / "rejected.json",
        )


def test_cuda_a40_receipt_rejects_wrong_seed_candidate_and_checkpoint(tmp_path) -> None:
    wrong_seed = _completed_cuda_smoke(tmp_path / "seed")
    summary_path = wrong_seed / "training_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["initialization_seed"] += 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match seed"):
        record_cuda_a40_validation(
            training_output_dir=wrong_seed,
            output_path=tmp_path / "wrong-seed.json",
        )

    wrong_candidate = _completed_cuda_smoke(tmp_path / "candidate")
    (wrong_candidate / "candidate_source.py").write_text(
        "# modified candidate\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="trusted common/initial_candidate.py"):
        record_cuda_a40_validation(
            training_output_dir=wrong_candidate,
            output_path=tmp_path / "wrong-candidate.json",
        )

    wrong_checkpoint = _completed_cuda_smoke(tmp_path / "checkpoint")
    with (wrong_checkpoint / "best_checkpoint.pt").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash"):
        record_cuda_a40_validation(
            training_output_dir=wrong_checkpoint,
            output_path=tmp_path / "wrong-checkpoint.json",
        )


def test_cuda_a40_receipt_detects_linked_artifact_tampering(tmp_path) -> None:
    output = _completed_cuda_smoke(tmp_path)
    receipt_path = tmp_path / "cuda-a40-receipt.json"
    record_cuda_a40_validation(
        training_output_dir=output,
        output_path=receipt_path,
    )
    (output / "events.jsonl").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match revalidated artifacts"):
        validate_cuda_a40_receipt(receipt_path)
