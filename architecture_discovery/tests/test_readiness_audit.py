import copy
import json
from dataclasses import replace

import scripts.audit_scientific_readiness as readiness_audit
from scripts.audit_scientific_readiness import audit_readiness
from study.contracts import StudySpec


def test_modal_action_journal_manifest_inventory_is_exact() -> None:
    manifest = readiness_audit.yaml.safe_load(
        (readiness_audit.ROOT / "experiment_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    remote = manifest["remote_execution"]

    assert {
        "provider_canary_aggregate_outcome_schema": remote[
            "provider_canary_aggregate_outcome_schema"
        ],
        "modal_local_host_anchor_schema": remote[
            "modal_local_host_anchor_schema"
        ],
        "modal_remote_run_reservation_schema": remote[
            "modal_remote_run_reservation_schema"
        ],
        "modal_action_intent_schema": remote["modal_action_intent_schema"],
        "modal_local_process_start_schema": remote[
            "modal_local_process_start_schema"
        ],
        "modal_action_attempt_receipt_schema": remote[
            "modal_action_attempt_receipt_schema"
        ],
        "modal_action_recovery_request_schema": remote[
            "modal_action_recovery_request_schema"
        ],
        "modal_action_recovery_intent_schema": remote[
            "modal_action_recovery_intent_schema"
        ],
        "modal_action_recovery_host_containment_schema": remote[
            "modal_action_recovery_host_containment_schema"
        ],
        "modal_action_recovery_resolution_schema": remote[
            "modal_action_recovery_resolution_schema"
        ],
        "modal_prior_cohort_quarantine_accounting_schema": remote[
            "modal_prior_cohort_quarantine_accounting_schema"
        ],
        "modal_action_lock_path": remote["modal_action_lock_path"],
        "modal_action_lock_scope": remote["modal_action_lock_scope"],
        "modal_global_action_journal_scanner_implemented": remote[
            "modal_global_action_journal_scanner_implemented"
        ],
        "modal_global_action_journal_prelaunch_gate_wired": remote[
            "modal_global_action_journal_prelaunch_gate_wired"
        ],
        "modal_action_orphan_recovery_status": remote[
            "modal_action_orphan_recovery_status"
        ],
    } == {
        "provider_canary_aggregate_outcome_schema": (
            "ProviderCanaryAggregateOutcomeReceipt/1.1"
        ),
        "modal_local_host_anchor_schema": "ModalLocalHostAnchor/1.0",
        "modal_remote_run_reservation_schema": "ModalRemoteRunReservation/1.2",
        "modal_action_intent_schema": "ModalActionIntent/1.6",
        "modal_local_process_start_schema": "ModalLocalProcessStart/1.1",
        "modal_action_attempt_receipt_schema": "ModalActionAttemptReceipt/3.6",
        "modal_action_recovery_request_schema": "ModalActionRecoveryRequest/1.0",
        "modal_action_recovery_intent_schema": "ModalActionRecoveryIntent/1.0",
        "modal_action_recovery_host_containment_schema": (
            "ModalActionRecoveryHostContainment/1.0"
        ),
        "modal_action_recovery_resolution_schema": (
            "ModalActionRecoveryResolution/1.0"
        ),
        "modal_prior_cohort_quarantine_accounting_schema": (
            "ModalPriorCohortQuarantineAccounting/1.1"
        ),
        "modal_action_lock_path": "outputs/readiness/.modal_action.lock",
        "modal_action_lock_scope": (
            "launcher_recovery_snapshot_capture_prior_accounting_lineage_roster_"
            "resource_cleanup_migration_bundle_and_global_journal_scan"
        ),
        "modal_global_action_journal_scanner_implemented": True,
        "modal_global_action_journal_prelaunch_gate_wired": True,
        "modal_action_orphan_recovery_status": (
            "operational_exact_v1_cli_scanner_validated"
        ),
    }


def test_readiness_audit_rejects_disabled_action_journal_prelaunch_wiring(
    monkeypatch,
) -> None:
    original_safe_load = readiness_audit.yaml.safe_load

    def manifest_with_false_wiring_claim(stream):
        payload = original_safe_load(stream)
        if (
            isinstance(payload, dict)
            and payload.get("schema_name")
            == "architecture_discovery_experiment_manifest"
        ):
            payload = copy.deepcopy(payload)
            payload["remote_execution"][
                "modal_global_action_journal_prelaunch_gate_wired"
            ] = False
        return payload

    monkeypatch.setattr(
        readiness_audit.yaml,
        "safe_load",
        manifest_with_false_wiring_claim,
    )

    report = audit_readiness()
    gate = next(
        item
        for item in report["gates"]
        if item["gate"] == "manifest_security_invariants"
    )

    assert not gate["passed"]
    assert (
        "modal_global_action_journal_prelaunch_gate_wired"
        in gate["blockers"][0]
    )


def test_readiness_audit_is_provider_free_and_fails_closed(monkeypatch):
    def reject_local_receipt(*_args, **_kwargs):
        raise ValueError("forced local receipt rejection")

    monkeypatch.setattr(
        readiness_audit,
        "validate_local_engineering_receipt",
        reject_local_receipt,
    )
    report = audit_readiness()
    gates = {gate["gate"]: gate for gate in report["gates"]}
    assert not report["ready"]
    assert report["provider_calls"] == 0
    assert report["training_runs"] == 0
    assert gates["common_c0_c3_engine"]["passed"]
    assert gates["evaluation_and_novelty_firewall"]["passed"]
    assert gates["scientific_evaluation_profiles"]["passed"]
    assert gates["manifest_security_invariants"]["passed"]
    assert gates["modal_cuda_execution_configured"]["passed"]
    for gate_name in (
        "modal_cuda_environment_validated",
        "modal_artifact_round_trip_validated",
        "modal_resource_cleanup_validated",
        "modal_migration_validation_bundle_validated",
    ):
        assert not gates[gate_name]["passed"]
        assert "receipt path is pending" in gates[gate_name]["blockers"][0]
    assert gates["historical_mps_evidence_compatibility"]["passed"]
    assert "mps_available_no_fallback" not in gates
    assert not gates["principal_investigator_decisions"]["passed"]
    assert not gates["full_profile_accelerator_validation"]["passed"]
    assert (
        "AcceleratorValidationEvidence"
        not in gates["full_profile_accelerator_validation"]["evidence"]
    )
    assert not gates["strong_candidate_containment"]["passed"]
    assert not gates["frozen_populated_reference_corpus"]["passed"]
    assert not gates["recorded_unit_test_evidence"]["passed"]
    assert not gates["recorded_offline_smoke_evidence"]["passed"]
    assert gates["recorded_unit_test_evidence"]["blockers"]
    assert report["readiness_levels"]["unit_tested"] is False
    assert report["readiness_levels"]["offline_smoke_tested"] is False
    assert "mps_validated" not in report["readiness_levels"]
    assert report["readiness_levels"]["accelerator_validated"] is False
    assert report["readiness_levels"]["modal_infrastructure_validated"] is False
    assert not gates["external_scientific_evidence_attestation"]["passed"]


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
        item
        for item in report["gates"]
        if item["gate"] == "full_profile_mps_validation"
    )
    assert not gate["passed"]
    assert "must be boolean" in gate["blockers"][0]


def test_local_engineering_receipts_do_not_require_external_attestation(
    monkeypatch,
) -> None:
    original_safe_load = readiness_audit.yaml.safe_load

    def readiness_with_local_receipts(stream):
        payload = original_safe_load(stream)
        if (
            isinstance(payload, dict)
            and payload.get("schema_name") == "scientific_readiness_evidence"
        ):
            payload = copy.deepcopy(payload)
            payload["engineering_evidence_provenance"][
                "source_revision_bound"
            ] = True
            assert (
                payload["engineering_evidence_provenance"]["externally_attested"]
                is False
            )
            for level_name in ("unit_tested", "offline_smoke_tested"):
                contract = readiness_audit.LOCAL_ENGINEERING_RECEIPT_CONTRACTS[
                    level_name
                ]
                payload["levels"][level_name]["passed"] = True
                payload["levels"][level_name]["receipt_path"] = (
                    readiness_audit.current_local_engineering_receipt_path(
                        level_name,
                        root=readiness_audit.ROOT,
                    ).as_posix()
                )
                payload["levels"][level_name]["receipt_contract"] = contract[
                    "receipt_contract"
                ]
        return payload

    monkeypatch.setattr(
        readiness_audit.yaml,
        "safe_load",
        readiness_with_local_receipts,
    )
    monkeypatch.setattr(
        readiness_audit,
        "validate_local_engineering_receipt",
        lambda level_name, receipt_path, *, root: f"local {level_name}",
    )
    monkeypatch.setattr(
        readiness_audit,
        "validate_local_engineering_freeze_receipt",
        lambda **_kwargs: {"passed": True},
    )

    report = readiness_audit.audit_readiness()
    gates = {gate["gate"]: gate for gate in report["gates"]}

    assert gates["recorded_unit_test_evidence"]["passed"]
    assert gates["recorded_offline_smoke_evidence"]["passed"]
    assert gates["local_engineering_freeze_validated"]["passed"]
    assert not gates["external_scientific_evidence_attestation"]["passed"]


def test_frozen_executable_contract_rejects_non_cuda_budget(tmp_path) -> None:
    spec = replace(StudySpec.toy(), scientific=True)
    path = tmp_path / "scientific-study.json"
    path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")

    report = audit_readiness(study_spec=path)
    gate = next(
        item
        for item in report["gates"]
        if item["gate"] == "frozen_executable_study_contract"
    )

    assert not gate["passed"]
    assert "accelerator_kind" in gate["blockers"][0]
