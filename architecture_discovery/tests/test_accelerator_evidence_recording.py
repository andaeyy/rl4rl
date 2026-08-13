from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import scripts.record_accelerator_validation as accelerator_receipts
from common.candidate_artifact import inspect_candidate_artifact
from common.runtime_context import ExecutionContextV1
from common.training_config import FULL_TRAIN_CUDA_V2, TrainingSeedBundle
from modal_boundary import ImageSourceManifestV1, SourceFileV1, canonical_sha256
from scripts.audit_scientific_readiness import audit_readiness
from scripts.record_accelerator_validation import (
    EVIDENCE_FIELDS,
    record_accelerator_validation,
    validate_accelerator_validation_evidence,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "run"
    training = root / "training"
    training.mkdir(parents=True)
    dependency_hash = "d" * 64
    source_manifest = ImageSourceManifestV1(
        dependency_lock_sha256=dependency_hash,
        files=(
            SourceFileV1(
                relative_path="uv.lock",
                sha256=dependency_hash,
                size_bytes=1,
            ),
        ),
    ).to_dict()
    image_source_hash = canonical_sha256(source_manifest)
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="full-cuda-1",
        app_name="rl4rl-architecture-discovery",
        function_name="full_profile_validation",
        modal_app_id="ap-abc123",
        modal_function_id="fu-def456",
        modal_call_id="fc-ghi789",
        modal_image_id="im-jkl012",
        image_source_sha256=image_source_hash,
        artifact_uri=("volume://rl4rl-architecture-artifacts/runs/full-cuda-1"),
    )
    fingerprint = {
        "requested_device": "cuda",
        "selected_device": "cuda:0",
        "accelerator_kind": "cuda",
        "gpu_name": "NVIDIA T4",
        "gpu_count": 1,
        "compute_capability": "7.5",
        "cuda_runtime": "12.8",
        "cuda_driver": "570.00",
        "torch_version": "2.7.1",
        "host_platform": "Linux-test",
    }
    _write_json(root / "execution_context.json", context.to_dict())
    _write_json(root / "image_source_manifest.json", source_manifest)
    _write_json(
        root / "cuda_environment.json",
        {
            "python": "3.12.13",
            "platform": "Linux-test",
            "torch": "2.7.1",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": "NVIDIA T4",
            "cuda_compute_capability": [7, 5],
            "cuda_runtime": "12.8",
            "cuda_driver": "570.00",
            "cuda_total_memory_bytes": 16_000_000_000,
            "accelerator_fingerprint": fingerprint,
            "execution_context": context.to_dict(),
        },
    )

    candidate = training / "candidate_graph.json"
    candidate.write_bytes(
        (PROJECT_ROOT / "common/initial_candidate.ir.json").read_bytes()
    )
    candidate_hash = _sha256(candidate)
    graph_hash = inspect_candidate_artifact(candidate).graph_hash
    assert graph_hash is not None
    checkpoint = training / "best_checkpoint.pt"
    checkpoint.write_bytes(b"bounded synthetic checkpoint")
    event_log = training / "training_events.jsonl"
    event_log.write_text('{"step":30000}\n', encoding="utf-8")
    seeds = TrainingSeedBundle(
        model_initialization_seed=11,
        training_data_seed=12,
        development_set_seed=13,
        dataloader_seed=14,
    )
    _write_json(
        training / "training_manifest.json",
        {
            "schema_name": "TrainingManifest",
            "schema_version": "2.0",
            "candidate_source_hash": candidate_hash,
            "candidate_artifact_hash": candidate_hash,
            "candidate_format": "architecture_ir",
            "candidate_graph_hash": graph_hash,
            "candidate_path": "candidate_graph.json",
            "immutable_candidate_relative_path": "candidate_graph.json",
            "profile": FULL_TRAIN_CUDA_V2.to_dict(),
            "profile_hash": FULL_TRAIN_CUDA_V2.profile_hash,
            "seed_bundle": {
                "model_initialization_seed": seeds.model_initialization_seed,
                "training_data_seed": seeds.training_data_seed,
                "development_set_seed": seeds.development_set_seed,
                "dataloader_seed": seeds.dataloader_seed,
            },
            "seed_bundle_hash": seeds.bundle_hash,
            "requested_device": "cuda",
            "selected_device": "cuda:0",
            "allow_cpu_for_tests": False,
            "hardware_matched_scientific_run": True,
            "dependency_lock_hash": dependency_hash,
            "execution_context": context.to_dict(),
            "runtime": {
                "cuda_available": True,
                "cuda_device_count": 1,
                "deterministic_algorithms": True,
                "pytorch_enable_mps_fallback": "0",
                "cublas_workspace_config": ":4096:8",
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "cuda_matmul_allow_tf32": False,
                "accelerator_fingerprint": fingerprint,
            },
        },
    )
    _write_json(
        training / "training_summary.json",
        {
            "schema_name": "TrainingResult",
            "schema_version": "2.0",
            "success": True,
            "scientific": True,
            "hardware_matched": True,
            "unsupported_operation_fallback": False,
            "cleanup_completed": True,
            "steps_completed": FULL_TRAIN_CUDA_V2.max_steps,
            "profile_name": FULL_TRAIN_CUDA_V2.name,
            "profile_version": FULL_TRAIN_CUDA_V2.version,
            "profile_hash": FULL_TRAIN_CUDA_V2.profile_hash,
            "device": "cuda:0",
            "accelerator_kind": "cuda",
            "accelerator_fingerprint": fingerprint,
            "candidate_source_hash": candidate_hash,
            "initialization_seed": seeds.model_initialization_seed,
            "data_seed": seeds.training_data_seed,
            "development_seed": seeds.development_set_seed,
            "dataloader_seed": seeds.dataloader_seed,
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": _sha256(checkpoint),
            "event_log_path": event_log.name,
        },
    )
    return root, training, tmp_path / "accelerator-evidence.json"


def _mutate(path: Path, field: str, value) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    _write_json(path, payload)


def test_accelerator_receipt_binds_modal_cuda_and_all_training_artifacts(
    tmp_path,
) -> None:
    root, training, receipt = _fixture(tmp_path)
    evidence = record_accelerator_validation(
        training_output_dir=training,
        artifact_root=root,
        output_path=receipt,
        code_revision="a" * 40,
    )

    assert set(evidence) == EVIDENCE_FIELDS
    assert evidence["schema_name"] == "AcceleratorValidationEvidence"
    assert evidence["schema_version"] == "2.0"
    assert evidence["artifact_root"] == "run"
    assert evidence["training_output_relative_path"] == "training"
    assert evidence["execution_backend"] == "modal"
    assert evidence["requested_gpu_kind"] == "cuda"
    assert evidence["observed_gpu_kind"] == "cuda"
    assert evidence["observed_gpu_name"] == "NVIDIA T4"
    assert evidence["observed_gpu_count"] == 1
    assert evidence["observed_gpu_compute_capability"] == "7.5"
    assert evidence["training_profile_hash"] == FULL_TRAIN_CUDA_V2.profile_hash
    assert evidence["steps_completed"] == FULL_TRAIN_CUDA_V2.max_steps
    assert validate_accelerator_validation_evidence(receipt) == evidence


@pytest.mark.parametrize(
    ("relative", "field", "value", "message"),
    (
        ("training/training_summary.json", "hardware_matched", 1, "exactly True"),
        ("training/training_summary.json", "steps_completed", True, "integer"),
        ("training/training_manifest.json", "requested_device", "mps", "request CUDA"),
        (
            "training/training_manifest.json",
            "runtime.cuda_available",
            1,
            "exactly True",
        ),
        ("cuda_environment.json", "cuda_device_count", True, "integer"),
        (
            "cuda_environment.json",
            "accelerator_fingerprint.cuda_driver",
            None,
            "driver is missing",
        ),
        (
            "cuda_environment.json",
            "cuda_driver",
            None,
            "cuda_driver differs",
        ),
        ("execution_context.json", "image_source_sha256", None, "image source"),
        (
            "training/training_manifest.json",
            "runtime.accelerator_fingerprint.gpu_name",
            "",
            "GPU name",
        ),
        (
            "training/training_manifest.json",
            "profile.max_steps",
            29_999,
            "exact CUDA profile",
        ),
        (
            "training/training_manifest.json",
            "candidate_graph_hash",
            "e" * 64,
            "does not reconstruct",
        ),
        (
            "training/training_manifest.json",
            "seed_bundle.model_initialization_seed",
            True,
            "integer",
        ),
        (
            "training/training_manifest.json",
            "execution_context.modal_call_id",
            "fc-other",
            "execution context differs",
        ),
    ),
)
def test_accelerator_receipt_rejects_type_spoofing_and_identity_mismatch(
    tmp_path, relative, field, value, message
) -> None:
    root, training, receipt = _fixture(tmp_path)
    _mutate(root / relative, field, value)
    with pytest.raises(ValueError, match=message):
        record_accelerator_validation(
            training_output_dir=training,
            artifact_root=root,
            output_path=receipt,
        )


def test_v2_receipt_rejects_absolute_artifact_paths(tmp_path) -> None:
    root, training, receipt = _fixture(tmp_path)
    _mutate(
        training / "training_summary.json",
        "checkpoint_path",
        str((training / "best_checkpoint.pt").resolve()),
    )
    with pytest.raises(ValueError, match="relative"):
        record_accelerator_validation(
            training_output_dir=training,
            artifact_root=root,
            output_path=receipt,
        )


def test_receipt_detects_modified_artifacts_and_refuses_overwrite(tmp_path) -> None:
    root, training, receipt = _fixture(tmp_path)
    record_accelerator_validation(
        training_output_dir=training,
        artifact_root=root,
        output_path=receipt,
    )
    with pytest.raises(FileExistsError):
        record_accelerator_validation(
            training_output_dir=training,
            artifact_root=root,
            output_path=receipt,
        )

    (training / "training_events.jsonl").write_text(
        '{"step":30000,"modified":true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact mismatch"):
        validate_accelerator_validation_evidence(receipt)


def test_stored_receipt_rejects_absolute_artifact_root(tmp_path) -> None:
    root, training, receipt = _fixture(tmp_path)
    record_accelerator_validation(
        training_output_dir=training,
        artifact_root=root,
        output_path=receipt,
    )
    _mutate(receipt, "artifact_root", str(root.resolve()))
    with pytest.raises(ValueError, match="relative"):
        validate_accelerator_validation_evidence(receipt)


def test_accelerator_receipt_satisfies_only_active_hardware_gate(tmp_path) -> None:
    root, training, receipt = _fixture(tmp_path)
    record_accelerator_validation(
        training_output_dir=training,
        artifact_root=root,
        output_path=receipt,
        code_revision="a" * 40,
    )

    report = audit_readiness(accelerator_evidence=receipt)
    gates = {gate["gate"]: gate for gate in report["gates"]}
    assert gates["full_profile_accelerator_validation"]["passed"]
    assert gates["historical_mps_evidence_compatibility"]["passed"]
    assert "mps_available_no_fallback" not in gates
    assert not gates["principal_investigator_decisions"]["passed"]
    assert report["readiness_levels"]["accelerator_validated"] is True
    assert "mps_validated" not in report["readiness_levels"]
    assert report["readiness_levels"]["modal_infrastructure_validated"] is False
    assert report["ready"] is False
    assert report["provider_calls"] == 0
    assert report["training_runs"] == 0


def test_project_receipt_uses_project_relative_artifact_root(
    tmp_path, monkeypatch
) -> None:
    project = tmp_path / "project"
    fixture_root, training, _ = _fixture(tmp_path / "fixture")
    run = project / "outputs" / "development" / "modal_downloads" / "run"
    run.parent.mkdir(parents=True)
    fixture_root.rename(run)
    training = run / training.name
    receipt = (
        project
        / "outputs"
        / "readiness"
        / "full_train_cuda_v2_accelerator_evidence.json"
    )
    monkeypatch.setattr(accelerator_receipts, "ROOT", project.resolve())

    evidence = record_accelerator_validation(
        training_output_dir=training,
        artifact_root=run,
        output_path=receipt,
        code_revision="a" * 40,
    )

    assert evidence["artifact_root"] == ("outputs/development/modal_downloads/run")
    assert validate_accelerator_validation_evidence(receipt) == evidence
