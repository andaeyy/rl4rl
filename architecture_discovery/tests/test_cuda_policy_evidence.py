from __future__ import annotations

from types import SimpleNamespace

import torch

from common.training_config import FULL_TRAIN_V1, SMOKE_TRAIN_V1
from containment.audit import audit_runtime
from containment.policy import (
    CandidateFormat,
    ScientificExecutionRequest,
    assess_scientific_execution,
)


FROZEN_FULL_MPS_HASH = (
    "046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e"
)
FROZEN_SMOKE_MPS_HASH = (
    "1a2b04bcb966f4189f90d6b8f6ef3aa8f83fb537f0f031004d0e58d69192cb61"
)


def _cuda_audit(monkeypatch):
    properties = SimpleNamespace(
        name="NVIDIA A40",
        uuid="GPU-test-a40",
        total_memory=48 * 1024**3,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: properties)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (8, 6))
    return audit_runtime(
        environment={
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_GPUS": "2",
            "CUDA_VISIBLE_DEVICES": "0",
        }
    )


def test_cuda_capability_requires_scheduler_allocation_evidence(monkeypatch) -> None:
    audit = _cuda_audit(monkeypatch)

    assert audit.cuda_available
    assert audit.cuda_device_count == 1
    assert audit.cuda_allocation_validated
    assert audit.cuda_devices[0]["name"] == "NVIDIA A40"
    assert audit.to_dict()["cuda"]["scheduler_job_id"] == "12345"


def test_scientific_cuda_ir_requires_a40_and_forbids_cpu_fallback(monkeypatch) -> None:
    audit = _cuda_audit(monkeypatch)
    request = ScientificExecutionRequest(
        candidate_format=CandidateFormat.ARCHITECTURE_IR,
        requested_device="cuda",
        ir_validated=True,
        trusted_ir_interpreter=True,
        required_cuda_device_name="A40",
    )
    allowed = assess_scientific_execution(audit, request)
    assert allowed.allowed, allowed.blockers
    assert any("distinct hardware conditions" in item for item in allowed.warnings)

    fallback = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="cuda",
            ir_validated=True,
            trusted_ir_interpreter=True,
            required_cuda_device_name="A40",
            cpu_fallback_allowed=True,
        ),
    )
    assert not fallback.allowed
    assert any("forbids CPU fallback" in item for item in fallback.blockers)


def test_cuda_does_not_weaken_arbitrary_python_scientific_gate(monkeypatch) -> None:
    audit = _cuda_audit(monkeypatch)
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARBITRARY_PYTHON,
            requested_device="cuda",
            candidate_artifact_hash="a" * 64,
            required_cuda_device_name="A40",
        ),
    )

    assert not decision.allowed
    assert any("arbitrary Python" in item for item in decision.blockers)
    assert not audit.strong_containment_proven


def test_frozen_mps_profiles_remain_hash_and_device_identical() -> None:
    assert FULL_TRAIN_V1.profile_hash == FROZEN_FULL_MPS_HASH
    assert SMOKE_TRAIN_V1.profile_hash == FROZEN_SMOKE_MPS_HASH
    assert FULL_TRAIN_V1.device_requirement == "mps"
    assert SMOKE_TRAIN_V1.device_requirement == "mps"
    assert FULL_TRAIN_V1.max_steps == 30_000
    assert SMOKE_TRAIN_V1.max_steps == 10
