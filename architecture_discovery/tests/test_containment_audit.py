from dataclasses import replace
from types import SimpleNamespace

import containment.audit as audit_module

from containment.audit import (
    BoundaryAttestation,
    CapabilityAudit,
    CapabilityState,
    CUDADeviceMetadata,
    LEGACY_SCHEMA_VERSION,
    REQUIRED_ADVERSARIAL_TESTS,
    REQUIRED_STRONG_CONTROLS,
    SCHEMA_VERSION,
    audit_runtime,
)
from containment.policy import (
    CandidateFormat,
    GatePhase,
    ScientificExecutionRequest,
    assess_scientific_execution,
)


def _attestation(*, authenticated: bool) -> BoundaryAttestation:
    return BoundaryAttestation(
        runner="trusted-study-runner-v1",
        candidate_artifact_hash="a" * 64,
        report_artifact_hash="b" * 64,
        created_at_utc="2026-07-31T00:00:00+00:00",
        test_results={name: True for name in REQUIRED_ADVERSARIAL_TESTS.values()},
        authenticated_by_trusted_runner=authenticated,
    )


def test_runtime_detection_is_not_mislabeled_as_enforcement():
    audit = audit_runtime(environment={"DISCOVERY_API_KEY": "never-serialize-me"})
    assert not audit.strong_containment_proven
    assert all(
        audit.controls[name].state is not CapabilityState.PROVEN
        for name in REQUIRED_STRONG_CONTROLS
    )
    assert audit.visible_credential_names == ("DISCOVERY_API_KEY",)
    assert "never-serialize-me" not in audit.to_json()
    assert audit.to_dict()["audit_hash"] == audit.audit_hash


def test_v2_runtime_audit_exposes_cpu_mps_and_cuda_device_metadata(monkeypatch):
    properties = SimpleNamespace(
        name="NVIDIA T4",
        major=7,
        minor=5,
        total_memory=16_000_000_000,
        multi_processor_count=40,
        uuid="GPU-test-t4",
    )
    monkeypatch.setattr(audit_module.torch.version, "cuda", "12.8")
    monkeypatch.setattr(audit_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(audit_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        audit_module.torch.cuda,
        "driver_version",
        lambda: "550.54",
        raising=False,
    )
    monkeypatch.setattr(
        audit_module.torch.cuda,
        "get_device_properties",
        lambda index: properties,
    )

    audit = audit_runtime(environment={})
    payload = audit.to_dict()

    assert audit.schema_version == SCHEMA_VERSION == "2.0"
    assert audit.accelerator_available("cpu")
    assert audit.accelerator_available("cuda:0")
    assert payload["accelerators"]["cpu"] == {"available": True}
    assert payload["accelerators"]["mps"]["available"] == audit.mps_available
    assert payload["accelerators"]["cuda"]["runtime_version"] == "12.8"
    assert payload["accelerators"]["cuda"]["driver_version"] == "550.54"
    assert payload["accelerators"]["cuda"]["devices"] == [
        {
            "index": 0,
            "name": "NVIDIA T4",
            "compute_capability": [7, 5],
            "total_memory_bytes": 16_000_000_000,
            "multi_processor_count": 40,
            "device_uuid": "GPU-test-t4",
        }
    ]
    assert CapabilityAudit.from_json(audit.to_json()) == audit


def test_historical_v1_mps_audit_round_trips_without_hash_or_shape_changes():
    legacy = replace(
        audit_runtime(environment={}),
        schema_version=LEGACY_SCHEMA_VERSION,
        mps_built=True,
        mps_available=True,
    )
    payload = legacy.to_dict()

    assert set(payload) == {
        "schema_name",
        "schema_version",
        "created_at_utc",
        "platform",
        "mps",
        "visible_credential_names",
        "attested_candidate_artifact_hash",
        "attestation_report_artifact_hash",
        "controls",
        "strong_containment_proven",
        "notes",
        "audit_hash",
    }
    loaded = CapabilityAudit.from_dict(payload)
    assert loaded.to_dict() == payload
    assert loaded.audit_hash == payload["audit_hash"]
    assert loaded.mps_built and loaded.mps_available
    assert not loaded.cuda_available


def test_untrusted_attestation_cannot_promote_boundary_controls():
    audit = audit_runtime(
        environment={},
        trusted_attestation=_attestation(authenticated=False),
    )
    assert not audit.strong_containment_proven
    assert any("not authenticated" in note for note in audit.notes)


def test_arbitrary_python_scientific_lane_fails_closed_without_os_proof():
    audit = replace(audit_runtime(environment={}), mps_built=True, mps_available=True)
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARBITRARY_PYTHON,
            requested_device="mps",
            candidate_artifact_hash="a" * 64,
        ),
    )
    assert not decision.allowed
    assert any("arbitrary Python" in blocker for blocker in decision.blockers)


def test_authenticated_complete_attestation_can_satisfy_python_boundary_gate():
    audit = audit_runtime(
        environment={},
        trusted_attestation=_attestation(authenticated=True),
    )
    audit = replace(audit, mps_built=True, mps_available=True)
    assert audit.strong_containment_proven
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARBITRARY_PYTHON,
            requested_device="mps",
            candidate_artifact_hash="a" * 64,
        ),
    )
    assert decision.allowed, decision.blockers

    wrong_artifact = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARBITRARY_PYTHON,
            requested_device="mps",
            candidate_artifact_hash="c" * 64,
        ),
    )
    assert not wrong_artifact.allowed
    assert any("exact candidate" in blocker for blocker in wrong_artifact.blockers)


def test_ir_gate_requires_typed_validation_trusted_interpreter_and_runtime_evidence():
    audit = replace(audit_runtime(environment={}), mps_built=True, mps_available=True)
    preflight = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="mps",
            ir_validated=True,
            trusted_ir_interpreter=True,
        ),
    )
    assert preflight.allowed, preflight.blockers

    postflight = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="mps",
            phase=GatePhase.POST_EXECUTION,
            ir_validated=True,
            trusted_ir_interpreter=True,
            runtime_validity_passed=False,
        ),
    )
    assert not postflight.allowed
    assert any("runtime" in blocker for blocker in postflight.blockers)


def test_scientific_gate_rejects_cpu_and_mps_fallback():
    audit = replace(
        audit_runtime(environment={}),
        mps_built=True,
        mps_available=True,
        mps_fallback_requested=True,
    )
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="cpu",
            ir_validated=True,
            trusted_ir_interpreter=True,
        ),
    )
    assert not decision.allowed
    assert any("requested_device='mps'" in blocker for blocker in decision.blockers)
    assert any("CPU fallback" in blocker for blocker in decision.blockers)


def test_cuda_scientific_profile_requires_and_audits_its_requested_accelerator():
    audit = replace(
        audit_runtime(environment={}),
        mps_built=False,
        mps_available=False,
        cuda_built=True,
        cuda_available=True,
        cuda_device_count=1,
        cuda_runtime_version="12.8",
        cuda_devices=(
            CUDADeviceMetadata(
                index=0,
                name="NVIDIA T4",
                compute_capability=(7, 5),
            ),
        ),
    )
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="cuda:0",
            ir_validated=True,
            trusted_ir_interpreter=True,
        ),
    )
    assert decision.allowed, decision.blockers

    wrong_accelerator = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="mps",
            required_accelerator="cuda",
            ir_validated=True,
            trusted_ir_interpreter=True,
        ),
    )
    assert not wrong_accelerator.allowed
    assert any("requested_device='cuda'" in item for item in wrong_accelerator.blockers)


def test_cuda_scientific_profile_fails_closed_when_cuda_is_unavailable():
    audit = replace(
        audit_runtime(environment={}),
        cuda_built=True,
        cuda_available=False,
        cuda_device_count=0,
    )
    decision = assess_scientific_execution(
        audit,
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARCHITECTURE_IR,
            requested_device="cuda",
            required_accelerator="cuda",
            ir_validated=True,
            trusted_ir_interpreter=True,
        ),
    )
    assert not decision.allowed
    assert any("CUDA" in blocker for blocker in decision.blockers)


def test_engineering_permission_is_explicitly_non_scientific():
    decision = assess_scientific_execution(
        audit_runtime(environment={}),
        ScientificExecutionRequest(
            candidate_format=CandidateFormat.ARBITRARY_PYTHON,
            requested_device="cpu",
            scientific=False,
        ),
    )
    assert decision.allowed
    assert not decision.scientific
    assert any("not scientific" in warning for warning in decision.warnings)
