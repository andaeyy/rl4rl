from dataclasses import replace

from containment.audit import (
    BoundaryAttestation,
    CapabilityState,
    REQUIRED_ADVERSARIAL_TESTS,
    REQUIRED_STRONG_CONTROLS,
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
