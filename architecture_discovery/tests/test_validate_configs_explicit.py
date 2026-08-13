from __future__ import annotations

import ast
import copy
import subprocess
import sys
from pathlib import Path

import pytest
from common.task_adapter import DEFAULT_TASK
from common.training_config import (
    FULL_TRAIN_CUDA_V2,
    FULL_TRAIN_V1,
    SEED_DERIVATION_METHOD,
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
)
from modal_boundary import (
    CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS,
    FUNCTION_CPU_REQUEST_CORES,
    FUNCTION_CPU_SOFT_LIMIT_CORES,
    FUNCTION_MEMORY_LIMIT_MIB,
    FUNCTION_MEMORY_REQUEST_MIB,
    FUNCTION_TIMEOUT_SECONDS,
    IMAGE_BUILD_CPU_REQUEST_CORES,
    IMAGE_BUILD_MEMORY_REQUEST_MIB,
    IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
    PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS,
    PROVIDER_REQUEST_TIMEOUT_SECONDS,
)
from scripts.record_modal_readiness import MODAL_READINESS_RECEIPT_CONTRACTS
from scripts.validate_configs import (
    MODAL_IMAGE_SOURCE_SHA256_NULL_REASON,
    MODAL_LIVE_EVIDENCE_AUTHORITY,
    MODAL_LIVE_VALIDATION_PENDING_STATUS,
    _validate_controller_training_reference,
    _validate_experiment_manifest_profiles,
    _validate_modal_engineering_evidence,
    _validate_openevolve_source_provenance,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts" / "validate_configs.py"


def test_configuration_validator_contains_no_optimization_sensitive_asserts():
    tree = ast.parse(VALIDATOR.read_text(encoding="utf-8"))
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree))


def test_explicit_invariant_failure_survives_optimized_python():
    program = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT)!r}); "
        "from scripts.validate_configs import _require; "
        "_require(False, 'optimized-sentinel')"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert "optimized-sentinel" in result.stderr


def _manifest_fixture(version: str) -> dict:
    active = version == "3"
    full = FULL_TRAIN_CUDA_V2 if active else FULL_TRAIN_V1
    smoke = SMOKE_TRAIN_CUDA_V2 if active else SMOKE_TRAIN_V1
    device = "cuda" if active else "mps"
    manifest = {
        "schema_name": "architecture_discovery_experiment_manifest",
        "schema_version": version,
        "training": {
            "profile": full.name,
            "profile_version": full.version,
            "profile_hash": full.profile_hash,
            "device": device,
            "task_adapter": DEFAULT_TASK.version,
            "seed_derivation": SEED_DERIVATION_METHOD,
        },
        "engineering_smoke": {
            "profile": smoke.name,
            "profile_version": smoke.version,
            "profile_hash": smoke.profile_hash,
        },
    }
    if active:
        manifest["remote_execution"] = {
            "image_source_sha256": None,
            "live_validation_status": MODAL_LIVE_VALIDATION_PENDING_STATUS,
            "function_timeout_seconds": FUNCTION_TIMEOUT_SECONDS,
            "controller_subprocess_timeout_seconds": (
                CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS
            ),
            "provider_request_timeout_seconds": PROVIDER_REQUEST_TIMEOUT_SECONDS,
            "provider_attempt_finalization_reserve_seconds": (
                PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS
            ),
            "function_cpu_request_cores": FUNCTION_CPU_REQUEST_CORES,
            "function_cpu_soft_limit_cores": FUNCTION_CPU_SOFT_LIMIT_CORES,
            "function_cpu_limit_kind": "soft_throttle_threshold",
            "function_memory_request_mib": FUNCTION_MEMORY_REQUEST_MIB,
            "function_memory_limit_mib": FUNCTION_MEMORY_LIMIT_MIB,
            "function_memory_limit_kind": "hard",
            "function_platform_compute_cost_ceiling_enforced": False,
            "runtime_functions_preemptible": True,
            "platform_preemption_restart_possible": True,
            "logical_call_count_is_not_container_attempt_ceiling": True,
            "function_region": None,
            "image_build_cpu_request_cores": IMAGE_BUILD_CPU_REQUEST_CORES,
            "image_build_cpu_soft_limit_cores": None,
            "image_build_memory_request_mib": IMAGE_BUILD_MEMORY_REQUEST_MIB,
            "image_build_memory_limit_mib": None,
            "image_build_region": None,
            "image_build_subprocess_thread_limit": (
                IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT
            ),
            "image_build_resource_limits_exposed": False,
            "image_build_platform_compute_cost_ceiling_enforced": False,
            "modal_price_basis_schema": "ModalPriceBasis/1.0",
            "modal_price_basis_max_age_hours": 48,
            "modal_cost_gate_scope": (
                "local_pre_popen_request_rate_and_one_gib_month_storage_"
                "estimate_not_platform_billing_cap"
            ),
            "provider_canary_aggregate_outcome_schema": (
                "ProviderCanaryAggregateOutcomeReceipt/1.1"
            ),
            "modal_local_host_anchor_schema": "ModalLocalHostAnchor/1.0",
            "modal_remote_run_reservation_schema": (
                "ModalRemoteRunReservation/1.2"
            ),
            "modal_action_intent_schema": "ModalActionIntent/1.6",
            "modal_local_process_start_schema": "ModalLocalProcessStart/1.1",
            "modal_action_attempt_receipt_schema": (
                "ModalActionAttemptReceipt/3.6"
            ),
            "modal_action_recovery_request_schema": (
                "ModalActionRecoveryRequest/1.0"
            ),
            "modal_action_recovery_intent_schema": (
                "ModalActionRecoveryIntent/1.0"
            ),
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
        manifest["modal_engineering_evidence"] = {
            "live_evidence_authority": MODAL_LIVE_EVIDENCE_AUTHORITY,
            "receipt_contracts": copy.deepcopy(MODAL_READINESS_RECEIPT_CONTRACTS),
            "remote_execution_image_source_sha256_null_reason": (
                MODAL_IMAGE_SOURCE_SHA256_NULL_REASON
            ),
        }
        manifest["engineering_smoke"]["device"] = "cuda"
        manifest["historical_mps_compatibility"] = {
            "full_profile": FULL_TRAIN_V1.name,
            "smoke_profile": SMOKE_TRAIN_V1.name,
        }
        manifest["scientific_evidence_status"] = {
            "full_accelerator_validation": "required",
            "historical_mps_validation": "retained",
        }
    return manifest


def test_modal_engineering_evidence_is_static_and_exact() -> None:
    manifest = _manifest_fixture("3")

    _validate_modal_engineering_evidence(manifest)

    evidence = manifest["modal_engineering_evidence"]
    assert evidence["live_evidence_authority"] == "external_create_only_receipts"
    assert evidence["receipt_contracts"] == MODAL_READINESS_RECEIPT_CONTRACTS
    assert len(evidence["receipt_contracts"]) == 4
    assert manifest["remote_execution"]["image_source_sha256"] is None
    assert (
        manifest["remote_execution"]["live_validation_status"]
        == "pending_explicit_cost_approval"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("image_source_sha256", "a" * 64, "must remain null"),
        ("live_validation_status", "validated", "must remain pending"),
    ),
)
def test_modal_engineering_evidence_rejects_dynamic_remote_state(
    field: str,
    value: str,
    message: str,
) -> None:
    manifest = _manifest_fixture("3")
    manifest["remote_execution"][field] = value

    with pytest.raises(RuntimeError, match=message):
        _validate_modal_engineering_evidence(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("function_cpu_soft_limit_cores", 3.0),
        ("function_memory_limit_mib", 16384),
        ("function_region", "us-east"),
        ("image_build_cpu_soft_limit_cores", 2.0),
        ("image_build_region", "us-east"),
        ("image_build_platform_compute_cost_ceiling_enforced", True),
    ),
)
def test_modal_engineering_evidence_rejects_resource_contract_drift(
    field: str,
    value: object,
) -> None:
    manifest = _manifest_fixture("3")
    manifest["remote_execution"][field] = value

    with pytest.raises(RuntimeError, match=f"resource contract {field} changed"):
        _validate_modal_engineering_evidence(manifest)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provider_canary_aggregate_outcome_schema", "stale"),
        ("modal_local_host_anchor_schema", "stale"),
        ("modal_remote_run_reservation_schema", "stale"),
        ("modal_action_intent_schema", "stale"),
        ("modal_local_process_start_schema", "stale"),
        ("modal_action_attempt_receipt_schema", "stale"),
        ("modal_action_recovery_request_schema", "stale"),
        ("modal_action_recovery_intent_schema", "stale"),
        ("modal_action_recovery_host_containment_schema", "stale"),
        ("modal_action_recovery_resolution_schema", "stale"),
        (
            "modal_prior_cohort_quarantine_accounting_schema",
            "stale",
        ),
        ("modal_action_lock_path", "outputs/readiness/wrong.lock"),
        ("modal_action_lock_scope", "launcher_only"),
        ("modal_global_action_journal_scanner_implemented", False),
        ("modal_global_action_journal_prelaunch_gate_wired", False),
        ("modal_action_orphan_recovery_status", "operational"),
    ),
)
def test_modal_engineering_evidence_rejects_action_journal_contract_drift(
    field: str,
    value: object,
) -> None:
    manifest = _manifest_fixture("3")
    manifest["remote_execution"][field] = value

    with pytest.raises(
        RuntimeError,
        match=f"Modal action journal contract {field} changed",
    ):
        _validate_modal_engineering_evidence(manifest)


def test_modal_engineering_evidence_rejects_contract_drift() -> None:
    manifest = _manifest_fixture("3")
    manifest["modal_engineering_evidence"]["receipt_contracts"][
        "modal_cuda_environment_validated"
    ]["receipt_contract"]["schema_version"] = "1.0"

    with pytest.raises(RuntimeError, match="contract changed"):
        _validate_modal_engineering_evidence(manifest)


@pytest.mark.parametrize(
    ("version", "expected_contract"),
    [("2", "historical_v1_mps"), ("3", "active_v2_cuda")],
)
def test_manifest_profile_reader_accepts_v3_cuda_and_historical_v2_mps_read_only(
    version: str,
    expected_contract: str,
) -> None:
    manifest = _manifest_fixture(version)
    before = copy.deepcopy(manifest)

    profile, contract = _validate_experiment_manifest_profiles(manifest)

    assert contract == expected_contract
    assert profile.name == manifest["training"]["profile"]
    assert manifest == before


def test_active_controller_reference_requires_cuda_v2_but_historical_is_readable() -> None:
    historical = {
        "profile": "full_train_v1",
        "profile_version": "1",
        "device": "mps",
        "allow_cpu_for_tests": False,
    }
    profile, contract = _validate_controller_training_reference(
        historical,
        require_active=False,
    )
    assert profile is FULL_TRAIN_V1
    assert contract == "historical_v1_mps"
    with pytest.raises(RuntimeError, match="active controllers"):
        _validate_controller_training_reference(historical, require_active=True)

    active = {
        "profile": "full_train_cuda_v2",
        "profile_version": "2",
        "device": "cuda",
        "allow_cpu_for_tests": False,
    }
    profile, contract = _validate_controller_training_reference(
        active,
        require_active=True,
    )
    assert profile is FULL_TRAIN_CUDA_V2
    assert contract == "active_v2_cuda"


def test_historical_mps_profile_hashes_are_unchanged() -> None:
    assert (
        FULL_TRAIN_V1.profile_hash
        == "046034a7949f3563fc13dcb38df4b34e997cb5a1ffe6b90e755e2f44bfd9f06e"
    )
    assert (
        SMOKE_TRAIN_V1.profile_hash
        == "1a2b04bcb966f4189f90d6b8f6ef3aa8f83fb537f0f031004d0e58d69192cb61"
    )


def test_openevolve_local_patches_are_bound_to_exact_files() -> None:
    manifest = __import__("yaml").safe_load(
        (ROOT / "experiment_manifest.yaml").read_text(encoding="utf-8")
    )
    _validate_openevolve_source_provenance(manifest)
    isolation_files = manifest["sources"]["openevolve"]["patches"][
        "isolate_provider_key_from_process_pool_payload"
    ]["files"]
    assert isolation_files[
        "vendor_patches/openevolve_process_isolation.patch"
    ] == "f39fa2a2ed50b7d22a28a5c5ce5547838f66b8445b2f17ad24f899a4e92560a8"
    provider_files = manifest["sources"]["openevolve"]["patches"][
        "provider_transport_single_shot_and_attempt_ledger"
    ]["files"]
    assert provider_files[
        "vendor_patches/openevolve_provider_attempt_ledger.patch"
    ] == "b0f731fa87fda394188dadc8fdab9c687b22c0a635eebef06d04cc4740017dc6"
    assert provider_files[
        "vendor/openevolve/openevolve/llm/openai.py"
    ] == "c01228a2a47b7d22096206e9f9da999dfb031fed728568583da6fbc667fec1d1"

    drifted = copy.deepcopy(manifest)
    drifted["sources"]["openevolve"]["patches"][
        "isolate_provider_key_from_process_pool_payload"
    ]["files"]["vendor/openevolve/openevolve/process_parallel.py"] = "0" * 64
    with pytest.raises(RuntimeError, match="source hash changed"):
        _validate_openevolve_source_provenance(drifted)

    wrong_base = copy.deepcopy(manifest)
    wrong_base["sources"]["openevolve"]["commit"] = "0" * 40
    with pytest.raises(RuntimeError, match="base commit"):
        _validate_openevolve_source_provenance(wrong_base)
