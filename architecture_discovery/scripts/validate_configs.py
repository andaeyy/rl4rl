from __future__ import annotations

import copy
import hashlib
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml
from architecture_ir.interpreter import validate_ir_candidate_path
from common.evaluation_profiles import EVALUATION_PROFILES
from common.gpt56_sol import API_MODE, TARGET_MODEL
from common.openevolve_policy import _quality
from common.task_adapter import DEFAULT_TASK
from common.trainer import checkpoint_is_better
from common.training_config import (
    FULL_TRAIN_CUDA_V2,
    FULL_TRAIN_V1,
    SEED_DERIVATION_METHOD,
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
    TrainingProfile,
    get_training_profile,
)
from evaluation.dependency_audit import assert_controller_dependencies_clean
from evaluation.records import CONTROLLER_SEARCH_FIELDS
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
from scripts.audit_scientific_readiness import audit_readiness
from scripts.openevolve_patch_bundle import (
    OPENEVOLVE_BASE_COMMIT,
    OPENEVOLVE_ISOLATION_PATCHED_FILE_SHA256,
    OPENEVOLVE_PATCH_RELATIVE_PATH,
    OPENEVOLVE_PATCH_SHA256,
    OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH,
    OPENEVOLVE_PROVIDER_PATCH_SHA256,
    OPENEVOLVE_PROVIDER_PATCHED_FILE_SHA256,
    OpenEvolvePatchBundleError,
    validate_applied_patch_bundle,
)
from scripts.record_local_engineering_evidence import (
    LEGACY_LOCAL_ENGINEERING_RECEIPT_CONTRACTS,
    LOCAL_ENGINEERING_FREEZE_RECEIPT_CONTRACT,
    LOCAL_ENGINEERING_RECEIPT_CONTRACTS,
    current_local_engineering_receipt_path,
)
from scripts.record_modal_readiness import MODAL_READINESS_RECEIPT_CONTRACTS
from scripts.validate_engineering_canaries import validate_controller_surfaces
from study.contracts import ConditionSpec

MODAL_LIVE_EVIDENCE_AUTHORITY = "external_create_only_receipts"
MODAL_IMAGE_SOURCE_SHA256_NULL_REASON = (
    "experiment_manifest_is_source_bound_and_embedding_its_derived_image_digest_"
    "would_create_a_self_reference_and_invalidate_the_live_cohort"
)
MODAL_LIVE_VALIDATION_PENDING_STATUS = "pending_explicit_cost_approval"


def _config(name: str) -> dict:
    path = ROOT / "agents" / name / "config.yaml"
    return yaml.safe_load(path.read_text())


def _require(condition: bool, message: str) -> None:
    """Fail explicitly even when Python optimization disables assertions."""

    if not condition:
        raise RuntimeError(f"configuration invariant failed: {message}")


def _validate_openevolve_source_provenance(manifest: dict) -> None:
    """Bind every local OpenEvolve patch to its exact implementation bytes."""

    expected = {
        "provider_transport_single_shot_and_attempt_ledger": {
            *(
                f"vendor/openevolve/{path}"
                for path in OPENEVOLVE_PROVIDER_PATCHED_FILE_SHA256
            ),
            OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH,
        },
        "isolate_provider_key_from_process_pool_payload": {
            *(
                f"vendor/openevolve/{path}"
                for path in OPENEVOLVE_ISOLATION_PATCHED_FILE_SHA256
            ),
            OPENEVOLVE_PATCH_RELATIVE_PATH,
        },
    }
    source = manifest.get("sources", {}).get("openevolve", {})
    _require(
        isinstance(source, dict) and source.get("commit") == OPENEVOLVE_BASE_COMMIT,
        "OpenEvolve base commit differs from the patch bundle",
    )
    try:
        validate_applied_patch_bundle(ROOT)
    except OpenEvolvePatchBundleError as error:
        raise RuntimeError(f"configuration invariant failed: {error}") from None
    patches = source.get("patches") if isinstance(source, dict) else None
    _require(
        isinstance(patches, dict) and set(patches) == set(expected),
        "OpenEvolve patch roster is incomplete",
    )
    if not isinstance(patches, dict):  # pragma: no cover - _require raises
        raise RuntimeError("unreachable OpenEvolve patch state")
    for patch_name, expected_files in expected.items():
        patch = patches.get(patch_name)
        files = patch.get("files") if isinstance(patch, dict) else None
        _require(
            isinstance(files, dict) and set(files) == expected_files,
            f"OpenEvolve patch {patch_name} file roster changed",
        )
        if not isinstance(files, dict):  # pragma: no cover - _require raises
            raise RuntimeError("unreachable OpenEvolve patch file state")
        for logical_path in sorted(expected_files):
            recorded = files.get(logical_path)
            _require(
                isinstance(recorded, str)
                and len(recorded) == 64
                and all(character in "0123456789abcdef" for character in recorded),
                f"OpenEvolve patch {patch_name} has an invalid SHA-256",
            )
            digest = hashlib.sha256((ROOT / logical_path).read_bytes()).hexdigest()
            _require(
                recorded == digest,
                f"OpenEvolve patch {patch_name} source hash changed",
            )
    isolation_files = patches["isolate_provider_key_from_process_pool_payload"][
        "files"
    ]
    _require(
        isolation_files.get(OPENEVOLVE_PATCH_RELATIVE_PATH)
        == OPENEVOLVE_PATCH_SHA256,
        "OpenEvolve reviewed patch identity changed",
    )
    for vendor_relative, expected_hash in (
        OPENEVOLVE_ISOLATION_PATCHED_FILE_SHA256.items()
    ):
        logical_path = f"vendor/openevolve/{vendor_relative}"
        _require(
            isolation_files.get(logical_path) == expected_hash,
            f"OpenEvolve patched file identity changed: {logical_path}",
        )
    provider_files = patches[
        "provider_transport_single_shot_and_attempt_ledger"
    ]["files"]
    _require(
        provider_files.get(OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH)
        == OPENEVOLVE_PROVIDER_PATCH_SHA256,
        "OpenEvolve reviewed provider patch identity changed",
    )
    for vendor_relative, expected_hash in (
        OPENEVOLVE_PROVIDER_PATCHED_FILE_SHA256.items()
    ):
        logical_path = f"vendor/openevolve/{vendor_relative}"
        _require(
            provider_files.get(logical_path) == expected_hash,
            f"OpenEvolve provider file identity changed: {logical_path}",
        )


_CONTROLLER_TRAINING_BINDINGS = {
    ("full_train_v1", "1", "mps"): "historical_v1_mps",
    ("full_train_cuda_v2", "2", "cuda"): "active_v2_cuda",
}

_MANIFEST_BINDINGS = {
    "2": {
        "contract": "historical_v1_mps",
        "full_profile": FULL_TRAIN_V1,
        "smoke_profile": SMOKE_TRAIN_V1,
        "device": "mps",
    },
    "3": {
        "contract": "active_v2_cuda",
        "full_profile": FULL_TRAIN_CUDA_V2,
        "smoke_profile": SMOKE_TRAIN_CUDA_V2,
        "device": "cuda",
    },
}


def _validate_modal_engineering_evidence(manifest: dict) -> None:
    expected = {
        "live_evidence_authority": MODAL_LIVE_EVIDENCE_AUTHORITY,
        "receipt_contracts": MODAL_READINESS_RECEIPT_CONTRACTS,
        "remote_execution_image_source_sha256_null_reason": (
            MODAL_IMAGE_SOURCE_SHA256_NULL_REASON
        ),
    }
    _require(
        manifest.get("modal_engineering_evidence") == expected,
        "manifest Modal engineering evidence contract changed",
    )
    for gate, contract in MODAL_READINESS_RECEIPT_CONTRACTS.items():
        template = contract["receipt_path_template"]
        _require(
            template.count("{source_tree_sha256}") == 1
            and template.count("{image_source_sha256}") == 1
            and template.count("{cohort_id}") == 1,
            f"manifest receipt template {gate} is not identity-parameterized",
        )
    remote_execution = manifest.get("remote_execution")
    _require(
        isinstance(remote_execution, dict),
        "manifest remote execution mapping is missing",
    )
    if not isinstance(remote_execution, dict):  # pragma: no cover - _require raises
        raise RuntimeError("unreachable remote execution state")
    _require(
        remote_execution.get("image_source_sha256") is None,
        "manifest image source digest must remain null to avoid self-reference",
    )
    _require(
        remote_execution.get("live_validation_status")
        == MODAL_LIVE_VALIDATION_PENDING_STATUS,
        "manifest live validation status must remain pending",
    )
    expected_deadlines = {
        "function_timeout_seconds": FUNCTION_TIMEOUT_SECONDS,
        "controller_subprocess_timeout_seconds": (
            CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS
        ),
        "provider_request_timeout_seconds": PROVIDER_REQUEST_TIMEOUT_SECONDS,
        "provider_attempt_finalization_reserve_seconds": (
            PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS
        ),
    }
    for field, expected_value in expected_deadlines.items():
        _require(
            remote_execution.get(field) == expected_value,
            f"manifest remote execution deadline {field} changed",
        )
    expected_resources = {
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
            "local_pre_popen_request_rate_and_one_gib_month_storage_estimate_"
            "not_platform_billing_cap"
        ),
    }
    for field, expected_value in expected_resources.items():
        _require(
            remote_execution.get(field) == expected_value,
            f"manifest remote execution resource contract {field} changed",
        )
    expected_action_journal = {
        "provider_canary_aggregate_outcome_schema": (
            "ProviderCanaryAggregateOutcomeReceipt/1.1"
        ),
        "modal_local_host_anchor_schema": "ModalLocalHostAnchor/1.0",
        "modal_remote_run_reservation_schema": "ModalRemoteRunReservation/1.2",
        "modal_action_intent_schema": "ModalActionIntent/1.6",
        "modal_local_process_start_schema": "ModalLocalProcessStart/1.1",
        "modal_action_attempt_receipt_schema": "ModalActionAttemptReceipt/3.6",
        "modal_action_recovery_request_schema": (
            "ModalActionRecoveryRequest/1.0"
        ),
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
    for field, expected_value in expected_action_journal.items():
        _require(
            remote_execution.get(field) == expected_value,
            f"manifest Modal action journal contract {field} changed",
        )
    _require(
        FUNCTION_CPU_REQUEST_CORES == FUNCTION_CPU_SOFT_LIMIT_CORES
        and FUNCTION_MEMORY_REQUEST_MIB == FUNCTION_MEMORY_LIMIT_MIB,
        "runtime Modal Function CPU soft and memory hard limits must equal requests",
    )
    _require(
        PROVIDER_REQUEST_TIMEOUT_SECONDS
        + PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS
        == CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS
        < FUNCTION_TIMEOUT_SECONDS,
        "provider request deadline lacks bounded finalization reserve",
    )


def _validate_controller_training_reference(
    reference: dict,
    *,
    require_active: bool,
) -> tuple[TrainingProfile, str]:
    profile_name = reference.get("profile")
    profile_version = reference.get("profile_version")
    device = reference.get("device")
    _require(isinstance(profile_name, str), "controller profile name is invalid")
    _require(
        isinstance(profile_version, (str, int))
        and not isinstance(profile_version, bool),
        "controller profile version is invalid",
    )
    _require(isinstance(device, str), "controller device is invalid")
    binding = (profile_name, str(profile_version), device)
    contract = _CONTROLLER_TRAINING_BINDINGS.get(binding)
    _require(contract is not None, f"unsupported controller training binding {binding!r}")
    if require_active:
        _require(contract == "active_v2_cuda", "active controllers do not use CUDA v2")
    profile = get_training_profile(profile_name)
    _require(profile.version == str(profile_version), "controller profile version changed")
    _require(profile.device_requirement == device, "controller profile device changed")
    _require(
        reference.get("allow_cpu_for_tests") is False,
        "controller permits CPU fallback",
    )
    return profile, str(contract)


def _validate_experiment_manifest_profiles(
    manifest: dict,
) -> tuple[TrainingProfile, str]:
    _require(
        manifest.get("schema_name") == "architecture_discovery_experiment_manifest",
        "experiment manifest schema name changed",
    )
    schema_version = str(manifest.get("schema_version", ""))
    binding = _MANIFEST_BINDINGS.get(schema_version)
    _require(binding is not None, "experiment manifest schema version is unsupported")
    if binding is None:  # pragma: no cover - _require above always raises
        raise RuntimeError("unreachable manifest binding state")
    full_profile = binding["full_profile"]
    smoke_profile = binding["smoke_profile"]
    device = binding["device"]
    _require(
        isinstance(full_profile, TrainingProfile)
        and isinstance(smoke_profile, TrainingProfile)
        and isinstance(device, str),
        "internal manifest binding is invalid",
    )

    training = manifest.get("training")
    smoke = manifest.get("engineering_smoke")
    _require(isinstance(training, dict), "manifest training mapping is missing")
    _require(isinstance(smoke, dict), "manifest engineering smoke mapping is missing")
    if not isinstance(training, dict) or not isinstance(smoke, dict):
        raise RuntimeError("unreachable manifest profile state")
    for field, expected in {
        "profile": full_profile.name,
        "profile_version": full_profile.version,
        "profile_hash": full_profile.profile_hash,
        "device": device,
        "task_adapter": DEFAULT_TASK.version,
        "seed_derivation": SEED_DERIVATION_METHOD,
    }.items():
        observed = training.get(field)
        if field == "profile_version":
            observed = str(observed)
        _require(observed == expected, f"manifest training field {field} changed")
    for field, expected in {
        "profile": smoke_profile.name,
        "profile_version": smoke_profile.version,
        "profile_hash": smoke_profile.profile_hash,
    }.items():
        observed = smoke.get(field)
        if field == "profile_version":
            observed = str(observed)
        _require(observed == expected, f"manifest smoke field {field} changed")
    if schema_version == "3":
        _validate_modal_engineering_evidence(manifest)
        _require(smoke.get("device") == "cuda", "manifest smoke device is not CUDA")
        _require(
            isinstance(manifest.get("historical_mps_compatibility"), dict),
            "manifest lacks historical MPS compatibility metadata",
        )
        scientific_status = manifest.get("scientific_evidence_status")
        _require(
            isinstance(scientific_status, dict),
            "manifest scientific evidence status is missing",
        )
        if not isinstance(scientific_status, dict):
            raise RuntimeError("unreachable scientific evidence status")
        _require(
            "full_accelerator_validation" in scientific_status,
            "manifest lacks full accelerator validation status",
        )
        _require(
            "historical_mps_validation" in scientific_status,
            "manifest lacks historical MPS validation status",
        )
    return full_profile, str(binding["contract"])


def _validate_readiness_contract(readiness: dict, *, manifest_version: str) -> None:
    _require(
        readiness.get("schema_name") == "scientific_readiness_evidence"
        and str(readiness.get("schema_version")) == "4",
        "active readiness evidence is not schema v4",
    )
    _require(
        LOCAL_ENGINEERING_FREEZE_RECEIPT_CONTRACT
        == {
            "schema_name": "LocalEngineeringFreezeReceipt",
            "schema_version": "1.0",
        },
        "local engineering freeze aggregate contract changed",
    )
    levels = readiness.get("levels")
    _require(isinstance(levels, dict), "readiness levels mapping is missing")
    if not isinstance(levels, dict):
        raise RuntimeError("unreachable readiness level state")
    accelerator_level = "accelerator_validated" if manifest_version == "3" else "mps_validated"
    _require(
        set(levels)
        == {
            "infrastructure_implemented",
            "unit_tested",
            "offline_smoke_tested",
            accelerator_level,
            "pilot_ready",
            "pilot_validated",
            "main_study_ready",
        },
        "readiness level roster changed",
    )
    for name, record in levels.items():
        _require(isinstance(record, dict), f"readiness level {name} is not a mapping")
        expected_fields = {"passed", "evidence"}
        if manifest_version == "3" and name in LOCAL_ENGINEERING_RECEIPT_CONTRACTS:
            expected_fields |= {"receipt_path", "receipt_contract"}
        _require(
            set(record) == expected_fields,
            f"readiness level {name} schema changed",
        )
        _require(
            type(record.get("passed")) is bool
            and isinstance(record.get("evidence"), str)
            and record["evidence"].strip(),
            f"readiness level {name} requires an exact boolean and evidence",
        )
        if name in LOCAL_ENGINEERING_RECEIPT_CONTRACTS:
            contract = LOCAL_ENGINEERING_RECEIPT_CONTRACTS[name]
            legacy = LEGACY_LOCAL_ENGINEERING_RECEIPT_CONTRACTS[name]
            current_path = current_local_engineering_receipt_path(
                name,
                root=ROOT,
            ).as_posix()
            if record["passed"]:
                _require(
                    record["receipt_contract"] == contract["receipt_contract"],
                    f"readiness level {name} passed with a stale receipt contract",
                )
                _require(
                    record["receipt_path"] == current_path,
                    f"readiness level {name} passed without the current freeze",
                )
            else:
                _require(
                    record["receipt_contract"]
                    in {
                        "current": contract["receipt_contract"],
                        "legacy": legacy["receipt_contract"],
                    }.values(),
                    f"readiness level {name} pending receipt contract changed",
                )
                _require(
                    record["receipt_path"]
                    in {None, current_path, legacy["receipt_path"]},
                    f"readiness level {name} pending receipt path changed",
                )
    if manifest_version == "3":
        _require(
            isinstance(readiness.get("historical_evidence"), dict),
            "readiness lacks historical MPS evidence metadata",
        )
        gates = readiness.get("gates")
        _require(isinstance(gates, dict), "readiness gates mapping is missing")
        if not isinstance(gates, dict):
            raise RuntimeError("unreachable readiness gates state")
        for gate_name, contract in MODAL_READINESS_RECEIPT_CONTRACTS.items():
            record = gates.get(gate_name)
            _require(
                isinstance(record, dict)
                and set(record)
                == {
                    "passed",
                    "evidence",
                    "receipt_path",
                    "receipt_sha256",
                    "selected_cohort_identity",
                    "receipt_contract",
                },
                f"readiness gate {gate_name} schema changed",
            )
            if not isinstance(record, dict):
                raise RuntimeError("unreachable Modal readiness record")
            _require(
                type(record["passed"]) is bool
                and isinstance(record["evidence"], str)
                and record["evidence"].strip(),
                f"readiness gate {gate_name} lacks typed status/evidence",
            )
            _require(
                record["receipt_contract"] == contract["receipt_contract"],
                f"readiness gate {gate_name} receipt contract changed",
            )
            if record["passed"]:
                _require(
                    record["receipt_path"] is not None
                    and record["receipt_sha256"] is not None
                    and record["selected_cohort_identity"] is not None,
                    f"readiness gate {gate_name} passed without a full binding",
                )
            else:
                _require(
                    record["receipt_path"] is None
                    and record["receipt_sha256"] is None
                    and record["selected_cohort_identity"] is None,
                    f"readiness gate {gate_name} pending binding must be null",
                )
    if levels["main_study_ready"]["passed"]:
        _require(
            all(levels[name]["passed"] for name in levels),
            "main study is marked ready while a lower level is false",
        )
    if levels["pilot_validated"]["passed"]:
        _require(
            levels["pilot_ready"]["passed"],
            "pilot is marked validated without pilot readiness",
        )


def _check_fitness_source() -> None:
    for function in (_quality, checkpoint_is_better):
        source = inspect.getsource(function)
        _require(
            "parameter_count_metadata" not in source,
            f"{function.__name__} uses parameter-count metadata",
        )
        _require(
            "parameter_count" not in source,
            f"{function.__name__} uses parameter count",
        )
    _require(
        "parameter_count_metadata" not in CONTROLLER_SEARCH_FIELDS,
        "controller view exposes parameter-count metadata",
    )
    _require(
        "shadow_accuracy" not in CONTROLLER_SEARCH_FIELDS,
        "controller view exposes shadow accuracy",
    )
    _require(
        "sealed_metrics" not in CONTROLLER_SEARCH_FIELDS,
        "controller view exposes sealed metrics",
    )


def main() -> None:
    greedy = _config("greedy_autoresearch")
    semantic_autoresearch = _config("semantic_autoresearch")
    generic = _config("openevolve_generic")
    semantic = _config("openevolve_semantic")
    _require(len(ConditionSpec.primary()) == 4, "primary condition count is not four")
    _require(
        set(EVALUATION_PROFILES)
        == {
            "unit_eval_v1",
            "smoke_eval_v1",
            "development_eval_v1",
            "scientific_layer_a_v1",
            "scientific_layer_b_v1",
            "scientific_layer_c_v1",
        },
        "evaluation profile roster changed",
    )
    _require(
        greedy["acceptance"]["use_parameter_count"] is False,
        "greedy acceptance uses parameter count",
    )
    _require(
        semantic_autoresearch["condition"] == "semantic_autoresearch",
        "semantic Autoresearch condition ID changed",
    )
    _require(
        semantic_autoresearch["acceptance"]["use_parameter_count"] is False,
        "semantic Autoresearch acceptance uses parameter count",
    )
    _require(
        all(
            name.startswith("semantic_")
            for name in semantic_autoresearch["archive"]["axes"]
        ),
        "semantic Autoresearch archive contains a non-semantic axis",
    )
    _require(generic["early_stopping_patience"] is None, "generic early stopping enabled")
    _require(semantic["early_stopping_patience"] is None, "semantic early stopping enabled")
    _require(
        generic["evaluator"]["parallel_evaluations"] == 1,
        "generic evaluator is not sequential",
    )
    _require(
        semantic["evaluator"]["parallel_evaluations"] == 1,
        "semantic evaluator is not sequential",
    )
    _require(generic["evaluator"]["timeout"] > 1800, "generic timeout is too short")
    _require(semantic["evaluator"]["timeout"] > 1800, "semantic timeout is too short")
    _require(generic["evaluator"]["max_retries"] == 0, "generic evaluator retries enabled")
    _require(semantic["evaluator"]["max_retries"] == 0, "semantic evaluator retries enabled")
    _require(
        generic["database"]["feature_dimensions"] == ["complexity", "diversity"],
        "generic archive dimensions changed",
    )
    _require(
        all(
            name.startswith("semantic_")
            for name in semantic["database"]["feature_dimensions"]
        ),
        "semantic OpenEvolve archive contains a non-semantic axis",
    )
    for config in (generic, semantic):
        trace = config["evolution_trace"]
        _require(trace["enabled"] is True, "evolution trace is disabled")
        _require(trace["include_code"] is True, "evolution trace omits code")
        _require(trace["include_prompts"] is True, "evolution trace omits prompts")

    generic_control = copy.deepcopy(generic)
    semantic_control = copy.deepcopy(semantic)
    generic_control["database"].pop("feature_dimensions")
    generic_control["database"].pop("feature_bins")
    semantic_control["database"].pop("feature_dimensions")
    semantic_control["database"].pop("feature_bins")
    _require(
        generic_control == semantic_control,
        "OpenEvolve conditions differ outside archive descriptors and prompts",
    )

    shared_generation = (
        greedy["iterations"],
        greedy["reasoning_effort"],
        greedy["temperature"],
        greedy["top_p"],
        greedy["max_tokens"],
        greedy["timeout_seconds"],
        greedy["retries"],
        greedy["retry_delay_seconds"],
    )
    _require(
        (
            semantic_autoresearch["iterations"],
            semantic_autoresearch["reasoning_effort"],
            semantic_autoresearch["temperature"],
            semantic_autoresearch["top_p"],
            semantic_autoresearch["max_tokens"],
            semantic_autoresearch["timeout_seconds"],
            semantic_autoresearch["retries"],
            semantic_autoresearch["retry_delay_seconds"],
        )
        == shared_generation,
        "semantic Autoresearch generation settings are not shared",
    )
    _require(
        greedy["timeout_seconds"] == PROVIDER_REQUEST_TIMEOUT_SECONDS,
        "provider request timeout differs from the frozen safety deadline",
    )
    for config in (generic, semantic):
        _require(
            (
                config["max_iterations"],
                config["llm"]["reasoning_effort"],
                config["llm"]["temperature"],
                config["llm"]["top_p"],
                config["llm"]["max_tokens"],
                config["llm"]["timeout"],
                config["llm"]["retries"],
                config["llm"]["retry_delay"],
            )
            == shared_generation,
            "OpenEvolve generation settings are not shared",
        )
        _require(
            config["llm"]["models"]
            == [{"name": TARGET_MODEL, "weight": 1.0}],
            "OpenEvolve model roster changed",
        )
        _require(
            config["llm"]["api_base"] == "https://api.openai.com/v1",
            "OpenEvolve API base changed",
        )
        _require(
            config["diff_based_evolution"] is False,
            "OpenEvolve must request complete IR documents, not source diffs",
        )
        _require(config["language"] == "json", "OpenEvolve language is not JSON")
        _require(
            config["file_suffix"] == ".json",
            "OpenEvolve candidate suffix is not .json",
        )
        _require(
            config["evaluator"]["use_llm_feedback"] is False,
            "evaluator LLM feedback is enabled",
        )
    _require(greedy["temperature"] is None, "greedy temperature must be unset")
    _require(greedy["top_p"] is None, "greedy top_p must be unset")

    training_references = [
        greedy["training"],
        semantic_autoresearch["training"],
        generic["training"],
        semantic["training"],
    ]
    _require(
        all(
            reference == training_references[0]
            for reference in training_references[1:]
        ),
        "controller training references differ",
    )
    training_reference = training_references[0]
    profile, training_contract = _validate_controller_training_reference(
        training_reference,
        require_active=True,
    )
    _require(
        training_contract == "active_v2_cuda",
        "active controller training contract is not CUDA v2",
    )
    expected_profile_fields = {
        "version": str(training_reference["profile_version"]),
        "optimizer": "AdamW",
        "peak_learning_rate": 0.001,
        "adamw_betas": (0.9, 0.98),
        "weight_decay": 0.1,
        "scheduler": "cosine_decay_to_zero",
        "warmup_steps": 300,
        "global_batch_size": 512,
        "microbatch_size": None,
        "gradient_accumulation_steps": 1,
        "max_steps": 30_000,
        "validation_interval": 1_000,
        "validation_examples": 2_000,
        "checkpoint_interval": 1_000,
        "maximum_wall_seconds": 1_800,
        "device_requirement": "cuda",
        "dtype": "float32",
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "cublas_workspace_config": ":4096:8",
    }
    for field, expected in expected_profile_fields.items():
        _require(
            getattr(profile, field) == expected,
            f"training profile field {field} changed",
        )
    _require(
        profile.max_steps * profile.global_batch_size == 15_360_000,
        "full training example budget changed",
    )
    _require(
        training_reference["task_adapter"] == DEFAULT_TASK.version,
        "task adapter version changed",
    )
    _require(
        training_reference["seed_derivation"] == SEED_DERIVATION_METHOD,
        "seed derivation changed",
    )
    _require(
        profile.checkpoint_selection_rule.startswith(
            "higher_development_exact_match"
        ),
        "checkpoint selection rule changed",
    )

    manifest = yaml.safe_load((ROOT / "experiment_manifest.yaml").read_text())
    _require(isinstance(manifest, dict), "experiment manifest is not a mapping")
    if not isinstance(manifest, dict):
        raise RuntimeError("unreachable experiment manifest state")
    _validate_openevolve_source_provenance(manifest)
    manifest_profile, manifest_contract = _validate_experiment_manifest_profiles(
        manifest
    )
    _require(
        str(manifest["schema_version"]) == "3",
        "active experiment manifest is not schema v3",
    )
    _require(
        manifest_contract == "active_v2_cuda",
        "active experiment manifest does not use CUDA v2",
    )
    _require(manifest_profile == profile, "controller and manifest profiles differ")
    manifest_generation = manifest["shared_generation"]
    generation_expectations = {
        "target_model": TARGET_MODEL,
        "api_mode": API_MODE,
        "reasoning_effort": greedy["reasoning_effort"],
        "max_completion_tokens": greedy["max_tokens"],
        "request_timeout_seconds": greedy["timeout_seconds"],
        "retries": greedy["retries"],
        "retry_delay_seconds": greedy["retry_delay_seconds"],
    }
    for field, expected in generation_expectations.items():
        _require(
            manifest_generation.get(field) == expected,
            f"manifest generation field {field} changed",
        )
    readiness = yaml.safe_load((ROOT / "readiness_evidence.yaml").read_text())
    _require(isinstance(readiness, dict), "readiness evidence is not a mapping")
    if not isinstance(readiness, dict):
        raise RuntimeError("unreachable readiness evidence state")
    _validate_readiness_contract(readiness, manifest_version="3")
    provenance = readiness.get("engineering_evidence_provenance")
    _require(
        isinstance(provenance, dict)
        and set(provenance)
        == {
            "authority",
            "source_revision_bound",
            "externally_attested",
            "scientific_launch_authority",
        },
        "engineering evidence provenance schema is incomplete",
    )
    _require(
        provenance["authority"] == "local_self_report",
        "engineering evidence authority is misstated",
    )
    for field in (
        "source_revision_bound",
        "externally_attested",
        "scientific_launch_authority",
    ):
        _require(
            type(provenance[field]) is bool,
            f"engineering evidence provenance field {field} is not boolean",
        )
    if readiness["status"] != "blocked":
        _require(
            provenance["source_revision_bound"]
            and provenance["externally_attested"]
            and provenance["scientific_launch_authority"],
            "an unblocked launch requires revision-bound external evidence",
        )
    _require(
        manifest["study"]["launch_status"] == readiness["status"],
        "manifest and readiness launch statuses differ",
    )

    decisions = yaml.safe_load((ROOT / "scientific_decisions.yaml").read_text())
    if decisions["status"] == "unresolved":
        _require(
            readiness["status"] == "blocked",
            "unresolved decisions do not block readiness",
        )
    readiness_report = audit_readiness()
    _require(readiness_report["provider_calls"] == 0, "readiness audit called provider")
    _require(readiness_report["training_runs"] == 0, "readiness audit started training")
    if readiness["status"] == "blocked":
        _require(
            not readiness_report["main_study_ready"],
            "blocked readiness reports main-study readiness",
        )

    forbidden_incentives = (
        "smallest model",
        "minimize parameter",
        "fewer parameter",
        "low-parameter",
        "compress the model",
    )
    prompt_paths = list((ROOT / "common" / "prompts").glob("*.md"))
    prompt_paths += list((ROOT / "agents").glob("**/*.md"))
    for path in prompt_paths:
        text = path.read_text().lower()
        for phrase in forbidden_incentives:
            _require(
                phrase not in text,
                f"{path} contains forbidden incentive {phrase}",
            )

    generic_prompt = (
        ROOT / "agents" / "openevolve_generic" / "system_prompt.md"
    ).read_text().lower()
    semantic_axis_terms = (
        "semantic_token_representation",
        "semantic_positional_integration",
        "semantic_attention_organization",
        "semantic_feedforward_mechanism",
        "semantic_normalization",
        "semantic_depth_topology",
        "semantic_output_readout",
        "semantic_tokenization",
        "token representation",
        "positional integration",
        "attention organization",
        "feedforward mechanism",
        "depth topology",
        "output readout",
        "tokenization",
    )
    _require(
        not any(term in generic_prompt for term in semantic_axis_terms),
        "generic prompt exposes semantic archive axes",
    )

    ir_seed = ROOT / "common" / "initial_candidate.ir.json"
    seed_validation = validate_ir_candidate_path(ir_seed)
    _require(
        seed_validation.valid,
        "initial architecture IR is invalid: "
        + "; ".join(issue.message for issue in seed_validation.issues),
    )
    _require(
        (ROOT / "architecture_ir" / "interpreter.py").is_file(),
        "trusted architecture-IR interpreter is missing",
    )

    controller_prompts = [
        ROOT / "agents" / "greedy_autoresearch" / "program.md",
        ROOT / "agents" / "semantic_autoresearch" / "program.md",
        ROOT / "agents" / "openevolve_generic" / "system_prompt.md",
        ROOT / "agents" / "openevolve_semantic" / "system_prompt.md",
    ]
    for path in controller_prompts:
        prompt = " ".join(path.read_text(encoding="utf-8").lower().split())
        _require(
            "complete replacement" in prompt and "json" in prompt,
            f"{path} does not require a complete replacement IR JSON document",
        )
        _require(
            "never return executable" in prompt or "do not return python" in prompt,
            f"{path} does not explicitly prohibit executable Python candidates",
        )

    runner_sources = [
        (ROOT / "agents" / "greedy_autoresearch" / "run.py").read_text(),
        (ROOT / "agents" / "semantic_autoresearch" / "run.py").read_text(),
        (ROOT / "common" / "openevolve_runner.py").read_text(),
    ]
    for source in runner_sources:
        _require(
            "initial_candidate.ir.json" in source,
            "controller runner does not reference the shared initial IR candidate",
        )
        _require(
            "common\" / \"evaluator.py" in source,
            "controller runner does not reference the shared evaluator",
        )
        _require(
            "--engineering-pilot" in source,
            "controller runner lacks the explicit engineering-pilot mode",
        )
    training_sources = [
        (ROOT / "common" / "trainer.py").read_text(),
        (ROOT / "common" / "training_data.py").read_text(),
    ]
    for source in training_sources:
        _require("private_eval" not in source, "training source imports private evaluation")
        _require(
            "DISCOVERY_SHADOW_SEED" not in source,
            "training source reads the shadow seed",
        )
        _require("2025" not in source, "training source embeds the legacy shadow seed")
    assert_controller_dependencies_clean(
        (
            ROOT / "agents" / "greedy_autoresearch" / "run.py",
            ROOT / "agents" / "semantic_autoresearch" / "run.py",
            ROOT / "common" / "openevolve_runner.py",
        ),
        project_root=ROOT,
    )
    _check_fitness_source()
    canary = validate_controller_surfaces(ROOT)
    _require(canary["passed"], f"static controller surfaces failed: {canary['errors']}")
    _require(canary["real_provider_calls"] == 0, "surface validator called provider")
    _require(canary["local_fixture_calls"] == 4, "surface fixture count is not four")
    _require(
        canary["entrypoint_execution_runs"] == 0,
        "surface validator executed an entrypoint",
    )
    _require(
        canary["candidate_execution_runs"] == 0,
        "surface validator executed a candidate",
    )
    _require(canary["training_runs"] == 0, "surface validator started training")
    print("configuration invariants: PASS")


if __name__ == "__main__":
    main()
