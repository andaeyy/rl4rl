import json

from scripts.audit_scientific_readiness import audit_readiness


def test_readiness_audit_is_provider_free_and_fails_closed():
    report = audit_readiness()
    gates = {gate["gate"]: gate for gate in report["gates"]}
    assert not report["ready"]
    assert report["provider_calls"] == 0
    assert report["training_runs"] == 0
    assert gates["common_c0_c3_engine"]["passed"]
    assert gates["evaluation_and_novelty_firewall"]["passed"]
    assert gates["scientific_evaluation_profiles"]["passed"]
    assert not gates["principal_investigator_decisions"]["passed"]
    assert not gates["strong_candidate_containment"]["passed"]
    assert not gates["frozen_populated_reference_corpus"]["passed"]
    assert not gates["recorded_unit_test_evidence"]["passed"]
    assert not gates["recorded_offline_smoke_evidence"]["passed"]
    assert "not bound to a source revision" in gates[
        "recorded_unit_test_evidence"
    ]["blockers"][0]
    assert report["readiness_levels"]["unit_tested"] is False
    assert report["readiness_levels"]["offline_smoke_tested"] is False


def test_mps_receipt_rejects_integer_boolean_spoofing(tmp_path):
    evidence = {
        "schema_name": "FullProfileMPSValidationEvidence",
        "schema_version": "1.0",
        "recorded_at_utc": "2026-07-31T12:00:00Z",
        "profile_name": "full_train_v1",
        "profile_version": "1",
        "profile_hash": "0" * 64,
        "requested_device": "mps",
        "selected_device": "mps",
        "mps_available": 1,
        "cpu_fallback": 0,
        "success": 1,
        "steps_completed": 30000.0,
        "candidate_source_hash": "1" * 64,
        "training_manifest_hash": "2" * 64,
        "training_summary_hash": "3" * 64,
        "training_output_dir": str(tmp_path / "missing"),
    }
    path = tmp_path / "mps.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    report = audit_readiness(mps_evidence=path)
    gate = next(
        item for item in report["gates"] if item["gate"] == "full_profile_mps_validation"
    )
    assert not gate["passed"]
    assert "must be boolean" in gate["blockers"][0]
