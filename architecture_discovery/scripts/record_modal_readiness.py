#!/usr/bin/env python3
# ruff: noqa: E402
"""Create and revalidate provider-free receipts for completed Modal checks.

This module never imports Modal, contacts a provider, or starts training.  It
only consumes already-downloaded artifacts and operator-saved JSON snapshots.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.device import AcceleratorFingerprint
from common.evaluation_profiles import EvaluationLayer, EvaluationPlan
from common.gpt56_sol import (
    API_MODE,
    OFFICIAL_OPENAI_API_BASE,
    TARGET_MODEL,
)
from common.modal_action_lock import (
    acquire_modal_action_lock,
    assert_modal_action_lock_identity,
    release_modal_action_lock,
)
from common.network_denial import (
    PROVIDER_FREE_MODAL_FUNCTIONS,
    validate_provider_free_action_outer_roster,
    validate_provider_free_network_denial_probe,
)
from common.provider_attempts import ProviderAttemptRecord
from common.public_evaluation import (
    PUBLIC_LAYER_A_SOURCE_ID,
    PUBLIC_LAYER_A_SOURCE_SHA256,
)
from common.runtime_context import ExecutionContextV1
from common.training_config import SMOKE_TRAIN_CUDA_V2
from common.evolution_run import (
    EVOLUTION_ACTION,
    EVOLUTION_FUNCTION_NAME,
    EvolutionRunSpec,
)
from evaluation.records import search_evaluation_from_dict
from modal_action_journal import (
    ModalActionJournalIntegrityError,
    ModalGlobalJournalScan,
    build_modal_global_launch_rejection_seal_payload,
    require_modal_global_action_journal_resolved,
    scan_modal_global_action_journal,
)
from modal_boundary import (
    APP_NAME,
    ARTIFACT_MANIFEST_FILENAMES,
    CANARY_ORDER,
    GPU_TYPE,
    IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
    IMAGE_BUILD_CPU_REQUEST_CORES,
    IMAGE_BUILD_MEMORY_REQUEST_MIB,
    IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
    IMAGE_RECIPE_VERSION,
    MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MAX_MODAL_BILLING_WINDOW,
    MODAL_LIVE_COHORT_ROOT,
    MODAL_VERSION,
    OPENEVOLVE_60_ACTION,
    PYTHON_VERSION,
    UV_VERSION,
    VOLUME_NAME,
    ArtifactVerificationV1,
    ImageSourceManifestV1,
    ModalLiveCohortIdentity,
    SourceFileV1,
    build_image_source_manifest,
    canonical_sha256,
    load_raw_artifact_manifest,
    modal_action_attempt_directory,
    modal_action_intent_receipt_path,
    modal_action_terminal_receipt_path,
    modal_artifact_verifier_capture_directory_path,
    modal_cli_command_sha256,
    modal_global_launch_rejection_seal_path,
    modal_live_cohort_root,
    modal_local_host_anchor_path,
    modal_local_process_start_receipt_path,
    modal_migration_lineage_path,
    modal_remote_run_reservation_path,
    modal_remote_verification_receipt_path,
    provider_canary_aggregate_outcome_receipt_path,
    safe_relative_path,
    validate_modal_action_identity,
    validate_provider_canary_aggregate_outcome_receipt,
    validate_run_id,
    verify_artifact_manifest,
    volume_artifact_uri,
    function_spec,
)
from modal_boundary import (
    MODAL_ENVIRONMENT_NAME as MODAL_ENVIRONMENT,
)
from reconstruction.downloaded_offline import validate_downloaded_offline_bundle
from scripts.capture_modal_cleanup_snapshots import (
    CAPTURE_MANIFEST_FILENAME,
    CAPTURE_MANIFEST_SCHEMA_NAME,
    CAPTURE_MANIFEST_SCHEMA_VERSION,
    build_modal_cleanup_snapshot_commands,
    modal_cleanup_snapshot_capture_manifest_path,
)
from scripts.capture_modal_cleanup_snapshots import (
    COMMAND_TIMEOUT_SECONDS as SNAPSHOT_COMMAND_TIMEOUT_SECONDS,
)
from scripts.capture_modal_cleanup_snapshots import (
    MODAL_ENVIRONMENT as SNAPSHOT_MODAL_ENVIRONMENT,
)
from scripts.capture_modal_cleanup_snapshots import (
    MODAL_PROFILE as SNAPSHOT_MODAL_PROFILE,
)
from scripts.capture_modal_cleanup_snapshots import (
    OUTER_TIMEOUT_SECONDS as SNAPSHOT_OUTER_TIMEOUT_SECONDS,
)
from scripts.capture_modal_cleanup_snapshots import (
    SNAPSHOT_NAMES as CAPTURE_SNAPSHOT_NAMES,
)
from scripts.provider_canary_plan import (
    build_provider_canary_approval_plan,
    verify_provider_canary_approval_plan,
)
from scripts.openevolve_60_plan import (
    build_openevolve_60_approval_plan,
    verify_openevolve_60_approval_plan,
)
from scripts.evolution_plan import (
    build_evolution_approval_plan,
    verify_evolution_approval_plan,
)
from scripts.record_local_engineering_evidence import (
    LOCAL_ENGINEERING_FREEZE_GATES,
    historical_local_engineering_freeze_predecessor_bindings,
    source_tree_sha256,
)
from scripts.validate_engineering_canaries import (
    load_modal_canary_selector,
    validate_downloaded_modal_canaries,
    validate_existing_cuda_smoke,
)
from scripts.verify_resume_contract import verify_resume_contract
from scripts.verify_resume_progression import verify_resume_progression
from study.serialization import create_json_exclusive, json_value

_GIT_VERSION = re.compile(
    r"\Agit version "
    r"[0-9]+(?:\.[0-9]+)+"
    r"(?:[.-][0-9A-Za-z]+)*"
    r"(?: \([0-9A-Za-z ._+-]+\))?\Z"
)

MODAL_READINESS_RECEIPT_CONTRACTS = {
    "modal_cuda_environment_validated": {
        "receipt_path_template": (
            "outputs/readiness/modal_only_final/modal_live_cohorts/"
            "{source_tree_sha256}/{image_source_sha256}/{cohort_id}/components/"
            "modal_cuda_environment_validation_receipt.v2.0.json"
        ),
        "receipt_contract": {
            "schema_name": "ModalCUDAEnvironmentValidationReceipt",
            "schema_version": "2.0",
        },
    },
    "modal_artifact_round_trip_validated": {
        "receipt_path_template": (
            "outputs/readiness/modal_only_final/modal_live_cohorts/"
            "{source_tree_sha256}/{image_source_sha256}/{cohort_id}/components/"
            "modal_artifact_round_trip_validation_receipt.v3.0.json"
        ),
        "receipt_contract": {
            "schema_name": "ModalArtifactRoundTripValidationReceipt",
            "schema_version": "3.0",
        },
    },
    "modal_resource_cleanup_validated": {
        "receipt_path_template": (
            "outputs/readiness/modal_only_final/modal_live_cohorts/"
            "{source_tree_sha256}/{image_source_sha256}/{cohort_id}/components/"
            "modal_resource_cleanup_validation_receipt.v4.0.json"
        ),
        "receipt_contract": {
            "schema_name": "ModalResourceCleanupValidationReceipt",
            "schema_version": "4.0",
        },
    },
    "modal_migration_validation_bundle_validated": {
        "receipt_path_template": (
            "outputs/readiness/modal_only_final/modal_live_cohorts/"
            "{source_tree_sha256}/{image_source_sha256}/{cohort_id}/components/"
            "modal_migration_validation_bundle_receipt.v4.0.json"
        ),
        "receipt_contract": {
            "schema_name": "ModalMigrationValidationBundleReceipt",
            "schema_version": "4.0",
        },
    },
}

MODAL_OFFLINE_SMOKE_VALIDATION_RECEIPT_PATH_TEMPLATE = (
    "outputs/readiness/modal_only_final/modal_live_cohorts/"
    "{source_tree_sha256}/{image_source_sha256}/{cohort_id}/components/"
    "modal_offline_smoke_validation_receipt.v2.0.json"
)
MODAL_COHORT_ROSTER_PATH_TEMPLATE = (
    "outputs/readiness/modal_only_final/modal_live_cohorts/"
    "{source_tree_sha256}/{image_source_sha256}/{cohort_id}/"
    "cohort_roster.v4.0.json"
)
MODAL_CANDIDATE_RESUME_PREFLIGHT_PATH_TEMPLATE = (
    "outputs/readiness/modal_only_final/modal_live_cohorts/"
    "{source_tree_sha256}/{image_source_sha256}/{cohort_id}/components/"
    "candidate_resume_preflight_receipts/v2.0/{binding_sha256}.json"
)

_MODAL_COMPONENT_FILENAMES = {
    "modal_cuda_environment_validated": (
        "modal_cuda_environment_validation_receipt.v2.0.json"
    ),
    "modal_offline_smoke_validated": (
        "modal_offline_smoke_validation_receipt.v2.0.json"
    ),
    "modal_artifact_round_trip_validated": (
        "modal_artifact_round_trip_validation_receipt.v3.0.json"
    ),
    "modal_resource_cleanup_validated": (
        "modal_resource_cleanup_validation_receipt.v4.0.json"
    ),
    "modal_migration_validation_bundle_validated": (
        "modal_migration_validation_bundle_receipt.v4.0.json"
    ),
}
_ROSTER_COMPONENT_KEYS = (
    "modal_cuda_environment_validated",
    "modal_offline_smoke_validated",
    "modal_artifact_round_trip_validated",
    "candidate_resume_preflight_validated",
)


def modal_cohort_identity_dict(
    identity: ModalLiveCohortIdentity,
) -> dict[str, str]:
    """Return the exact JSON identity shared by all live-cohort evidence."""

    if not isinstance(identity, ModalLiveCohortIdentity):
        raise TypeError("Modal live cohort identity has the wrong type")
    return {
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
    }


def modal_component_receipt_path(
    identity: ModalLiveCohortIdentity,
    component: str,
) -> PurePosixPath:
    """Return one canonical versioned component path within a cohort leaf."""

    try:
        filename = _MODAL_COMPONENT_FILENAMES[component]
    except KeyError as error:
        raise ValueError(f"unknown Modal cohort component: {component}") from error
    return modal_live_cohort_root(identity) / "components" / filename


def modal_cohort_roster_path(identity: ModalLiveCohortIdentity) -> PurePosixPath:
    return modal_live_cohort_root(identity) / "cohort_roster.v4.0.json"


def modal_cleanup_snapshot_directory(
    identity: ModalLiveCohortIdentity,
) -> PurePosixPath:
    return modal_live_cohort_root(identity) / "resource_cleanup"


def modal_provider_price_basis_path(
    identity: ModalLiveCohortIdentity,
) -> PurePosixPath:
    return modal_cleanup_snapshot_directory(identity) / "provider_price_basis.json"


def modal_prior_quarantine_accounting_path(
    identity: ModalLiveCohortIdentity,
) -> PurePosixPath:
    return modal_live_cohort_root(identity) / "quarantine_accounting.v1.1.json"


def modal_candidate_resume_preflight_binding_sha256(
    *,
    identity: ModalLiveCohortIdentity,
    execution_run_ids: Mapping[str, str],
    verifier_bindings: Mapping[str, Mapping[str, str]],
    predecessor_receipts: Mapping[str, Mapping[str, str]],
) -> str:
    """Hash the complete canonical preflight selection before it is recorded."""

    labels = (
        "cuda_environment",
        "offline_smoke",
        "candidate_smoke",
        "resume_attempt",
    )
    if set(execution_run_ids) != set(labels):
        raise ValueError("preflight execution roster must use the exact labels")
    if set(verifier_bindings) != set(labels):
        raise ValueError("preflight verifier roster must use the exact labels")
    canonical_verifiers: dict[str, dict[str, str]] = {}
    for label in labels:
        source_run_id = validate_run_id(execution_run_ids[label])
        record = verifier_bindings[label]
        if not isinstance(record, Mapping) or set(record) != {
            "verifier_run_id",
            "verifier_attempt_id",
        }:
            raise ValueError(f"preflight verifier binding {label} is invalid")
        verifier_run_id = validate_run_id(record["verifier_run_id"])
        attempt_id = _attempt_id(
            record["verifier_attempt_id"],
            f"preflight.{label}.verifier_attempt_id",
        )
        if source_run_id == verifier_run_id:
            raise ValueError("preflight verifier cannot reuse its source run ID")
        canonical_verifiers[label] = {
            "verifier_run_id": verifier_run_id,
            "verifier_attempt_id": attempt_id,
        }
    if set(predecessor_receipts) != {
        "modal_cuda_environment_validated",
        "modal_offline_smoke_validated",
        "modal_artifact_round_trip_validated",
    }:
        raise ValueError("preflight predecessor roster is invalid")
    canonical_predecessors: dict[str, dict[str, str]] = {}
    for gate in sorted(predecessor_receipts):
        record = predecessor_receipts[gate]
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ValueError(f"preflight predecessor {gate} is invalid")
        logical = _text(record["path"], f"preflight.{gate}.path")
        safe_relative_path(logical)
        canonical_predecessors[gate] = {
            "path": logical,
            "sha256": _sha256(
                record["sha256"], f"preflight.{gate}.sha256"
            ),
        }
    return canonical_sha256(
        {
            "schema_name": "CandidateResumePreflightBinding",
            "schema_version": "1.0",
            "cohort_identity": modal_cohort_identity_dict(identity),
            "execution_run_ids": {
                label: validate_run_id(execution_run_ids[label]) for label in labels
            },
            "verifier_bindings": canonical_verifiers,
            "predecessor_receipts": canonical_predecessors,
        }
    )


def modal_candidate_resume_preflight_receipt_path(
    identity: ModalLiveCohortIdentity,
    binding_sha256: str,
) -> PurePosixPath:
    return (
        modal_live_cohort_root(identity)
        / "components"
        / "candidate_resume_preflight_receipts"
        / "v2.0"
        / f"{_sha256(binding_sha256, 'binding_sha256')}.json"
    )

MODAL_PRICE_BASIS_ROOT = PurePosixPath(
    "outputs/readiness/modal_only_final/modal_price_bases"
)
MODAL_PRICE_BASIS_MAX_AGE = timedelta(hours=48)
MODAL_PRICE_BASIS_FUTURE_SKEW = timedelta(minutes=5)
MODAL_APP_LIFECYCLE_CLOCK_TOLERANCE = timedelta(seconds=30)
MODAL_PRICE_BASIS_OFFICIAL_SOURCE_URL = "https://modal.com/pricing"
MODAL_DOWNLOAD_TRANSFER_PRICING = (
    "not_separately_listed_on_official_pricing_page"
)
MODAL_COST_ESTIMATE_SCOPE = (
    "local_pre_popen_request_rate_and_one_gib_month_storage_estimate_"
    "not_platform_billing_cap"
)
_GIB_BYTES = Decimal(1024**3)
_MODAL_PRICE_TIMESTAMP = re.compile(
    r"\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z"
)
_CANONICAL_DECIMAL = re.compile(r"\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_MODAL_PRICE_BASIS_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "image_source_sha256",
        "official_source_url",
        "retrieved_at_utc",
        "region",
        "cpu_usd_per_core_second",
        "memory_usd_per_gib_second",
        "t4_usd_per_gpu_second",
        "volume_storage_usd_per_gib_month",
        "included_volume_storage_gib_per_month",
        "download_transfer_pricing",
    }
)
_MODAL_COST_ESTIMATE_FIELDS = frozenset(
    {
        "runtime_request_rate_estimate_usd",
        "cache_miss_image_build_request_rate_estimate_usd",
        "new_remote_run_count",
        "per_remote_run_storage_bound_gib",
        "one_month_storage_estimate_usd",
        "download_transfer_bound_gib",
        "download_transfer_pricing",
        "download_transfer_estimate_usd",
        "action_estimate_usd",
        "scope",
    }
)

_CUDA_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "cohort_id",
        "recorded_at_utc",
        "run_id",
        "downloaded_run_path",
        "execution_backend",
        "app_name",
        "function_name",
        "requested_gpu",
        "observed_gpu_name",
        "cuda_available",
        "cuda_device_count",
        "artifact_uri",
        "image_source_sha256",
        "execution_context_sha256",
        "cuda_environment_sha256",
        "remote_action_result_sha256",
        "artifact_manifest_sha256",
        "files_verified",
        "validated",
    }
)
_OFFLINE_SMOKE_VALIDATION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "cohort_id",
        "recorded_at_utc",
        "run_id",
        "downloaded_run_path",
        "execution_backend",
        "app_name",
        "function_name",
        "artifact_uri",
        "modal_app_id",
        "modal_function_id",
        "modal_call_id",
        "modal_image_id",
        "image_source_sha256",
        "dependency_lock_sha256",
        "execution_context_sha256",
        "image_source_manifest_sha256",
        "provider_free_network_denial_probe_sha256",
        "remote_action_result_sha256",
        "manifest_filename",
        "raw_manifest_sha256",
        "raw_manifest_size_bytes",
        "artifact_manifest_sha256",
        "files_verified",
        "artifact_bytes",
        "offline_validation_sha256",
        "offline_study_id",
        "offline_study_run_count",
        "offline_study_sha256",
        "remote_verifier_run_id",
        "remote_verifier_attempt_id",
        "remote_verification_path",
        "remote_verification_sha256",
        "validation_mode",
        "validation_network_calls",
        "validation_provider_calls",
        "validation_remote_calls_started",
        "validation_training_runs_started",
        "validated",
    }
)
_ROUND_TRIP_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "recorded_at_utc",
        "source_run_id",
        "verifier_run_id",
        "verifier_attempt_id",
        "downloaded_run_path",
        "artifact_uri",
        "manifest_filename",
        "remote_verification_path",
        "remote_verification_sha256",
        "verifier_execution_context_sha256",
        "remote_raw_manifest_sha256",
        "remote_raw_manifest_size_bytes",
        "local_raw_manifest_sha256",
        "local_raw_manifest_size_bytes",
        "remote_canonical_manifest_sha256",
        "local_canonical_manifest_sha256",
        "files_verified",
        "remote_verification_completed",
        "local_verification_completed",
        "validated",
    }
)
_CLEANUP_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "recorded_at_utc",
        "cohort_roster_path",
        "cohort_roster_sha256",
        "app_name",
        "volume_name",
        "modal_cli_version",
        "accepted_primary_executions",
        "artifact_verifier_executions",
        "additional_artifact_verifier_executions",
        "evidence_backed_failed_ordinary_executions",
        "action_attempts",
        "final_accepted_roster",
        "failed_run_ids",
        "quarantined_run_ids",
        "recovery_run_ids",
        "validation_only_attempt_ids",
        "recovery_links",
        "active_app_count",
        "active_container_count",
        "active_endpoint_count",
        "volume_present",
        "task_function_call_inventory",
        "direct_detached_call_inventory",
        "detached_calls_prohibited",
        "detached_call_policy_source_path",
        "detached_call_policy_source_sha256",
        "bound_image_source_sha256",
        "artifact_verifier_network_policy",
        "billing_window_start_utc",
        "billing_window_end_utc",
        "cohort_billing_total_usd",
        "superseded_usage_usd",
        "migration_total_usd",
        "billing_scope",
        "billing_attributions",
        "modal_compute_exposure",
        "provider_spend_estimate",
        "migration_provider_spend_estimate",
        "volume_run_directory_inventory",
        "retained_storage_estimate",
        "snapshot_capture_manifest_path",
        "snapshot_capture_manifest_sha256",
        "migration_lineage_path",
        "migration_lineage_sha256",
        "snapshots",
        "validated",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "cohort_id",
        "recorded_at_utc",
        "cohort_roster",
        "executions",
        "image_source_sha256",
        "migration_lineage",
        "evidence",
        "required_artifacts",
        "validated",
    }
)

_PRIMARY_FUNCTIONS = {
    "cuda_environment": "cuda_environment",
    "offline_smoke": "offline_smoke",
    "candidate_smoke": "candidate_smoke",
    "resume_attempt": "checkpoint_resume",
    **{f"canary_{harness}": f"canary_{harness}" for harness in CANARY_ORDER},
}
_PRIMARY_LABELS = tuple(_PRIMARY_FUNCTIONS)
_ORDINARY_ACTION_FUNCTIONS = {
    "cuda-environment": "cuda_environment",
    "offline-smoke": "offline_smoke",
    "candidate-smoke": "candidate_smoke",
    "checkpoint-resume": "checkpoint_resume",
    OPENEVOLVE_60_ACTION: "openevolve_generic_60",
    EVOLUTION_ACTION: EVOLUTION_FUNCTION_NAME,
}
_PROVIDER_LAUNCH_ACTIONS = frozenset(
    {
        "canary",
        "canaries",
        "exploratory_c0c3_pilot",
        OPENEVOLVE_60_ACTION,
        EVOLUTION_ACTION,
    }
)
_COHORT_ROSTER_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "cleanup_run_id",
        "component_receipts",
        "accepted_primary_runs",
        "accepted_attempt_ids",
        "artifact_verifiers",
        "additional_artifact_verifiers",
        "provider_canary_outcomes",
        "provider_canary_selector_path",
        "provider_canary_selector_sha256",
        "action_intent_receipts",
        "action_attempt_receipts",
        "provider_canary_aggregate_outcome_receipts",
        "attempt_classifications",
        "billing_attributions",
        "terminal_run_dispositions",
        "recovery_links",
        "declared_failed_run_ids",
        "declared_quarantined_run_ids",
        "declared_recovery_run_ids",
        "billing_window_start_utc",
        "billing_window_end_utc",
        "snapshot_captured_at_utc",
        "snapshot_capture_manifest_path",
        "snapshot_capture_manifest_sha256",
        "migration_lineage_path",
        "migration_lineage_sha256",
        "superseded_usage",
        "provider_price_basis_path",
    }
)
_VERIFIER_REMOTE_RECEIPT_ROSTER = (
    "execution_context.json",
    "image_source_manifest.json",
    "artifact_verification_result.json",
    "artifact_manifest.json",
)
_FAILED_VERIFIER_REMOTE_RECEIPT_ROSTER = (
    "execution_context.json",
    "image_source_manifest.json",
    "artifact_verification_failure.json",
    "artifact_manifest.json",
)
_ARTIFACT_VERIFIER_NETWORK_POLICY_FIELDS = frozenset(
    {
        "function_name",
        "provider_secret",
        "block_network",
        "proof_kind",
        "sources",
    }
)
_VERIFIER_ROSTER_FIELDS = frozenset(
    {
        "source_label",
        "source_run_id",
        "verifier_run_id",
        "attempt_id",
        "remote_verification_path",
        "remote_verification_sha256",
        "verifier_execution_context",
        "expected_remote_receipt_roster",
    }
)
_FAILED_VERIFIER_RECEIPT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_run_id",
        "verifier_run_id",
        "error_type",
        "message",
        "verifier_execution_context",
    }
)
_ADDITIONAL_VERIFIER_FIELDS = frozenset(
    {
        "source_run_id",
        "verifier_run_id",
        "attempt_id",
        "status",
        "remote_verifier_outcome",
        "remote_evidence_kind",
        "billing_object_ids",
        "remote_verification_path",
        "remote_verification_sha256",
        "verifier_execution_context",
        "failure_receipt_path",
        "failure_receipt_sha256",
        "failure_execution_context",
        "recovery_verifier_attempt_id",
        "expected_remote_receipt_roster",
    }
)
_OUTCOME_VERIFIER_FIELDS = frozenset(
    {
        "attempt_id",
        "verifier_run_id",
        "remote_verification_path",
        "remote_verification_sha256",
    }
)
_PROVIDER_CANARY_OUTCOME_FIELDS = frozenset(
    {
        "launcher_attempt_id",
        "launcher_attempt_receipt_path",
        "launcher_attempt_receipt_sha256",
        "harness",
        "concrete_run_id",
        "outcome",
        "provider_attempt_ledger_path",
        "provider_attempt_ledger_sha256",
        "provider_start_uncertain_evidence_path",
        "provider_start_uncertain_evidence_sha256",
        "artifact_verifier",
    }
)
_PROVIDER_START_UNCERTAIN_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "harness",
        "action",
        "execution_backend",
        "action_run_id",
        "modal_call_id",
        "api_endpoint",
        "model",
        "provider_attempt_count_lower_bound",
        "provider_attempt_count_upper_bound",
        "provider_request_started",
        "provider_attempt_ledger_state",
        "billing_treatment",
        "reason",
    }
)
_BILLING_ATTRIBUTION_FIELDS = frozenset({"attempt_id", "disposition", "object_ids"})
_RECOVERY_LINK_FIELDS = frozenset(
    {"failed_attempt_id", "recovery_attempt_id", "recovered_run_ids"}
)
_ATTEMPT_CLASSIFICATION_FIELDS = frozenset({"attempt_id", "roles"})
_ATTEMPT_ROLES = frozenset(
    {
        "accepted_primary",
        "artifact_verifier",
        "failed",
        "quarantined",
        "recovery",
        "validation_only",
    }
)
_ACTION_ATTEMPT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempt_id",
        "started_at_utc",
        "finished_at_utc",
        "status",
        "failure_kind",
        "action",
        "run_id",
        "concrete_remote_run_ids",
        "remote_run_reservations",
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
        "source_run_id",
        "verifier_run_id",
        "harness",
        "source_tree_sha256",
        "cohort_id",
        "approved_image_source_sha256",
        "modal_command_sha256",
        "launch_capability_sha256",
        "modal_profile",
        "modal_environment",
        "outer_cli_timeout_seconds",
        "modal_cost_cap_usd",
        "modal_resource_profile",
        "modal_price_basis_path",
        "modal_price_basis_sha256",
        "modal_cost_estimate",
        "modal_cost_approved",
        "provider_cost_approved",
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
        "predecessor_receipts",
        "source_evidence_recovery",
        "local_process_start_receipt_path",
        "local_process_start_receipt_sha256",
        "local_process_id",
        "local_process_group_id",
        "local_session_id",
        "modal_cli_process_started",
        "remote_execution_state",
        "returncode",
        "process_group_closed",
    }
)
_ACTION_INTENT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempt_id",
        "created_at_utc",
        "action",
        "run_id",
        "concrete_remote_run_ids",
        "remote_run_reservations",
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
        "source_run_id",
        "verifier_run_id",
        "harness",
        "source_tree_sha256",
        "cohort_id",
        "approved_image_source_sha256",
        "modal_command_sha256",
        "launch_capability_sha256",
        "modal_profile",
        "modal_environment",
        "outer_cli_timeout_seconds",
        "modal_cost_cap_usd",
        "modal_resource_profile",
        "modal_price_basis_path",
        "modal_price_basis_sha256",
        "modal_cost_estimate",
        "modal_cost_approved",
        "provider_cost_approved",
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
        "predecessor_receipts",
        "source_evidence_recovery",
    }
)
_INTENT_TERMINAL_SHARED_FIELDS = frozenset(
    _ACTION_INTENT_FIELDS
    - {
        "schema_name",
        "schema_version",
        "created_at_utc",
        "attempt_id",
    }
)
_ACTION_STATUSES = frozenset(
    {
        "preflight_failed",
        "preflight_rejected",
        "lock_contended",
        "interrupted",
        "timed_out",
        "cli_failed",
        "succeeded",
        "failed",
        "cleanup_failed",
    }
)
_ACTIONS = frozenset(
    {
        "canaries",
        "canary",
        "candidate-smoke",
        "checkpoint-resume",
        "cuda-environment",
        "exploratory_c0c3_pilot",
        OPENEVOLVE_60_ACTION,
        EVOLUTION_ACTION,
        "download",
        "offline-smoke",
        "verify",
    }
)
_SUPERSEDED_RUN_ID = "modal-cuda-env-20260809-02"
_SUPERSEDED_USAGE_USD = Decimal("0.00643852")
_TASK_FUNCTION_CALL_INVENTORY = "unavailable_in_modal_cli_1_5_3"
_DIRECT_DETACHED_CALL_INVENTORY = "unavailable_in_modal_cli_1_5_3"
_PRICE_BASIS_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "model",
        "official_source_url",
        "retrieved_at_utc",
        "uncached_input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "per_request_fee_usd",
    }
)
_EXECUTION_FIELDS = frozenset(
    {
        "run_id",
        "function_name",
        "modal_app_id",
        "modal_function_id",
        "modal_call_id",
        "modal_image_id",
        "image_source_sha256",
        "artifact_manifest_sha256",
    }
)
_BUNDLE_EXECUTION_KEYS = frozenset(
    {
        "cuda_environment",
        "candidate_smoke",
        "resume_attempt",
        "offline_smoke",
        "canaries",
    }
)
_CANARY_SUFFIXES = {
    harness: harness.replace("_autoresearch", "-ar").replace("_", "-")
    for harness in CANARY_ORDER
}
_SNAPSHOT_NAMES = (
    "app_list",
    "container_list",
    "endpoint_list",
    "volume_list",
    "run_directory_list",
    "billing_report",
)
if _SNAPSHOT_NAMES != CAPTURE_SNAPSHOT_NAMES:
    raise AssertionError("cleanup recorder and capture helper snapshot rosters drifted")
_SNAPSHOT_CAPTURE_MANIFEST_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "capture_id",
        "modal_profile",
        "modal_environment",
        "modal_cli_version",
        "billing_window_start_utc",
        "billing_window_end_utc",
        "started_at_utc",
        "finished_at_utc",
        "command_timeout_seconds",
        "outer_timeout_seconds",
        "command_retry_count",
        "snapshots",
    }
)
_SNAPSHOT_CAPTURE_RECORD_FIELDS = frozenset(
    {"path", "sha256", "size_bytes", "argv", "captured_at_utc"}
)
_FILE_BINDING_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_JOURNAL_BINDING_FIELDS = frozenset(
    {"intent_receipts", "terminal_receipts", "aggregate_receipts"}
)
_LINEAGE_SELECTED_FINAL_FIELDS = frozenset(
    {
        "identity",
        "accepted_primary_runs",
        "accepted_attempt_ids",
        "action_journal",
        "remote_run_reservations",
        "run_dispositions",
        "aggregate_receipts",
        "remote_executions",
        "remote_object_ids",
        "provider_attempt_evidence",
        "provider_spend_estimate",
        "artifact_manifests",
    }
)
_LINEAGE_PRIOR_FIELDS = frozenset(
    {
        "identity",
        "disposition",
        "accounting_receipt",
        "action_journal",
        "remote_run_reservations",
        "provider_spend_estimate",
        "modal_compute_exposure",
    }
)
_MIGRATION_LINEAGE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "recorded_at_utc",
        "selected_final",
        "prior_quarantined_cohorts",
        "global_remote_run_reservations",
        "legacy_superseded_usage",
        "prior_app_compute_total_usd",
        "final_provider_spend_bound_usd",
        "prior_provider_spend_bound_usd",
        "migration_provider_spend_bound_usd",
        "prior_modal_measured_app_billing_usd",
        "prior_modal_unresolved_compute_reserve_usd",
        "prior_modal_conservative_exposure_usd",
        "retained_storage_estimate",
        "global_uniqueness_validated",
        "validated",
    }
)
_PRIOR_QUARANTINE_ACCOUNTING_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "recorded_at_utc",
        "action_journal",
        "remote_run_reservations",
        "attempt_dispositions",
        "remote_run_dispositions",
        "remote_executions",
        "provider_attempt_evidence",
        "unbound_provider_evidence",
        "provider_spend_estimate",
        "modal_compute_exposure",
        "snapshot_capture_manifest_path",
        "snapshot_capture_manifest_sha256",
        "snapshot_capture_manifest_size_bytes",
        "app_lifecycles",
        "selected_billing_rows",
        "app_compute_subtotal_usd",
        "volume_dispositions",
        "modal_price_basis",
        "active_app_count",
        "active_container_count",
        "active_endpoint_count",
        "accepted_contexts",
        "retained_storage_estimate",
        "validated",
    }
)
_PRIOR_QUARANTINE_ACCOUNTING_REQUEST_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "recorded_at_utc",
        "snapshot_capture_manifest",
    }
)
_PRIOR_QUARANTINE_ACCOUNTING_INSPECTION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "request",
        "canonical_receipt_path",
        "candidate",
        "blockers",
    }
)
_MIGRATION_LINEAGE_INPUT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "accepted_primary_runs",
        "accepted_attempt_ids",
        "prior_quarantine_accounting_paths",
    }
)
_PRIOR_REMOTE_EXECUTION_FIELDS = frozenset(
    {
        "attempt_id",
        "run_id",
        "execution_context_path",
        "execution_context_sha256",
        "execution_context_size_bytes",
    }
)
_PRIOR_REMOTE_RUN_DISPOSITION_FIELDS = frozenset(
    {
        "attempt_id",
        "run_id",
        "execution_disposition",
        "provider_disposition",
        "snapshot_disposition",
        "snapshot_app_ids",
        "volume_disposition",
    }
)
_PRIOR_PROVIDER_EVIDENCE_FIELDS = frozenset(
    {
        "attempt_id",
        "run_id",
        "state",
        "evidence_path",
        "evidence_sha256",
        "evidence_size_bytes",
        "provider_attempt_count",
        "request_ids",
        "response_ids",
    }
)
_PRIOR_UNBOUND_PROVIDER_EVIDENCE_FIELDS = frozenset(
    {
        "attempt_id",
        "run_id",
        "evidence_kind",
        "evidence_path",
        "evidence_sha256",
        "evidence_size_bytes",
        "parse_disposition",
        "provider_attempt_count_lower_bound",
        "request_ids",
        "response_ids",
    }
)
_PRIOR_APP_LIFECYCLE_FIELDS = frozenset(
    {"attempt_id", "app_id", "created_at_utc", "stopped_at_utc"}
)
_PRIOR_BILLING_ROW_FIELDS = frozenset(
    {"attempt_id", "app_id", "row_sha256", "row"}
)
_PRIOR_VOLUME_DISPOSITION_FIELDS = frozenset(
    {
        "run_id",
        "entry_sha256",
        "entry",
        "artifact_manifest_disposition",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "artifact_manifest_size_bytes",
    }
)
_RETAINED_STORAGE_ESTIMATE_FIELDS = frozenset(
    {
        "retained_run_count",
        "conservative_bytes_per_run",
        "conservative_total_bytes",
        "estimated_gib",
        "volume_rate_usd_per_gib_month",
        "estimated_monthly_usd",
        "basis",
    }
)
_RAW_SNAPSHOT_FIELDS = {
    "app_list": {
        "app_id",
        "description",
        "state",
        "tasks",
        "created_at",
        "stopped_at",
    },
    "container_list": {"container_id", "app_id", "app_name", "start_time"},
    "endpoint_list": {"name", "endpoint_id", "status", "created_at", "created_by"},
    "volume_list": {"name", "created_at", "created_by"},
    "run_directory_list": {"filename", "type", "created_modified", "size"},
    "billing_report": {
        "object_id",
        "description",
        "environment",
        "interval_start",
        "resource",
        "cost",
    },
}
_REMOTE_VERIFICATION_FIELDS = ArtifactVerificationV1.FIELDS

_MAX_JSON_OBJECT_BYTES = 16 * 1024 * 1024
_MAX_MODAL_CLI_SNAPSHOT_BYTES = 16 * 1024 * 1024
_MAX_PROVIDER_LEDGER_BYTES = 4 * 1024 * 1024
_MAX_HASHABLE_FILE_BYTES = (
    MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES + MAX_ARTIFACT_MANIFEST_BYTES
)


class _PriorQuarantineAccountingIncomplete(ValueError):
    """Structurally valid evidence is not yet complete enough to scaffold."""

    def __init__(self, *messages: str) -> None:
        if not messages or any(not message for message in messages):
            raise ValueError("prior accounting blocker messages must be non-empty")
        self.messages = tuple(messages)
        super().__init__("; ".join(messages))


def _open_regular_file_descriptor(path: Path) -> tuple[int, os.stat_result]:
    """Open one file through stable directory descriptors without symlinks."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or not absolute.name:
        raise ValueError("readiness input must name an anchored regular file")
    components = absolute.parts
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, directory_flags)
    try:
        for component in components[1:-1]:
            if component in {"", ".", ".."}:
                raise ValueError("readiness input contains an unsafe path component")
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"required readiness input parent is missing: {absolute.parent}"
                ) from None
            if stat.S_ISLNK(before.st_mode):
                raise ValueError("readiness input path may not traverse symlinks")
            if not stat.S_ISDIR(before.st_mode):
                raise ValueError("readiness input ancestor must be a directory")
            try:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise ValueError(
                    "readiness input parent changed while it was opened"
                ) from error
            opened = os.fstat(next_descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (before.st_dev, before.st_ino)
                != (opened.st_dev, opened.st_ino)
            ):
                os.close(next_descriptor)
                raise ValueError(
                    "readiness input parent changed while it was opened"
                )
            os.close(descriptor)
            descriptor = next_descriptor

        leaf = absolute.name
        try:
            before = os.stat(leaf, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"required regular file is missing: {absolute}"
            ) from None
        if stat.S_ISLNK(before.st_mode):
            raise ValueError("readiness input may not be a symbolic link")
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
        ):
            raise ValueError("readiness input must be one regular file")
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_flags |= getattr(os, "O_NONBLOCK", 0)
        try:
            file_descriptor = os.open(leaf, file_flags, dir_fd=descriptor)
        except OSError as error:
            raise ValueError(
                "readiness input changed while it was opened"
            ) from error
        opened = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            os.close(file_descriptor)
            raise ValueError("readiness input changed while it was opened")
        return file_descriptor, opened
    finally:
        os.close(descriptor)


def _read_regular_file_bytes(
    path: Path,
    *,
    maximum_bytes: int,
    required_mode: int | None = None,
) -> bytes:
    """Read one stable regular-file snapshot within an explicit byte limit."""

    if maximum_bytes <= 0:
        raise ValueError("readiness input byte limit must be positive")
    descriptor, before = _open_regular_file_descriptor(path)
    try:
        if required_mode is not None and stat.S_IMODE(before.st_mode) != required_mode:
            raise ValueError(
                f"readiness input mode must be exactly {required_mode:04o}"
            )
        if before.st_size > maximum_bytes:
            raise ValueError("readiness input exceeds its size limit")
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            try:
                chunk = os.read(
                    descriptor,
                    min(64 * 1024, maximum_bytes + 1 - len(payload)),
                )
            except InterruptedError:
                continue
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum_bytes:
            raise ValueError("readiness input exceeds its size limit")
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
            before.st_uid,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
            after.st_uid,
        )
        if before_identity != after_identity or len(payload) != after.st_size:
            raise ValueError("readiness input changed while it was read")
        reopened_descriptor, reopened = _open_regular_file_descriptor(path)
        os.close(reopened_descriptor)
        reopened_identity = (
            reopened.st_dev,
            reopened.st_ino,
            reopened.st_size,
            reopened.st_mtime_ns,
            reopened.st_ctime_ns,
            reopened.st_nlink,
            reopened.st_uid,
        )
        if reopened_identity != after_identity:
            raise ValueError("readiness input path changed while it was read")
        if (
            required_mode is not None
            and stat.S_IMODE(reopened.st_mode) != required_mode
        ):
            raise ValueError(
                f"readiness input mode must be exactly {required_mode:04o}"
            )
        return bytes(payload)
    finally:
        os.close(descriptor)


def _decode_utf8(payload: bytes, path: Path) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path.name} must be UTF-8") from error


def _load_object_with_sha256(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_regular_file_bytes(path, maximum_bytes=_MAX_JSON_OBJECT_BYTES)
    payload = json.loads(
        _decode_utf8(raw, path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _hash_regular_file_snapshot(path: Path) -> tuple[str, int]:
    descriptor, before = _open_regular_file_descriptor(path)
    try:
        if before.st_size > _MAX_HASHABLE_FILE_BYTES:
            raise ValueError("readiness input exceeds its hash size limit")
        digest = hashlib.sha256()
        observed_size = 0
        while observed_size <= _MAX_HASHABLE_FILE_BYTES:
            try:
                chunk = os.read(
                    descriptor,
                    min(
                        1024 * 1024,
                        _MAX_HASHABLE_FILE_BYTES + 1 - observed_size,
                    ),
                )
            except InterruptedError:
                continue
            if not chunk:
                break
            observed_size += len(chunk)
            digest.update(chunk)
        if observed_size > _MAX_HASHABLE_FILE_BYTES:
            raise ValueError("readiness input exceeds its hash size limit")
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
            before.st_uid,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
            after.st_uid,
        )
        if before_identity != after_identity or observed_size != after.st_size:
            raise ValueError("readiness input changed while it was hashed")
        reopened_descriptor, reopened = _open_regular_file_descriptor(path)
        os.close(reopened_descriptor)
        reopened_identity = (
            reopened.st_dev,
            reopened.st_ino,
            reopened.st_size,
            reopened.st_mtime_ns,
            reopened.st_ctime_ns,
            reopened.st_nlink,
            reopened.st_uid,
        )
        if reopened_identity != after_identity:
            raise ValueError("readiness input path changed while it was hashed")
        return digest.hexdigest(), observed_size
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return _hash_regular_file_snapshot(path)[0]


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _attempt_id(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError(f"{field} must be 32 lowercase hexadecimal digits")
    return value


def _cohort_identity_from_payload(
    payload: Mapping[str, Any],
    *,
    field: str = "cohort_identity",
) -> ModalLiveCohortIdentity:
    try:
        return ModalLiveCohortIdentity(
            source_tree_sha256=_sha256(
                payload["source_tree_sha256"],
                f"{field}.source_tree_sha256",
            ),
            image_source_sha256=_sha256(
                payload["image_source_sha256"],
                f"{field}.image_source_sha256",
            ),
            cohort_id=validate_run_id(payload["cohort_id"]),
        )
    except KeyError as error:
        raise ValueError(f"{field} is incomplete") from error


def _assert_identity_matches(
    payload: Mapping[str, Any],
    identity: ModalLiveCohortIdentity,
    *,
    field: str,
) -> None:
    observed = _cohort_identity_from_payload(payload, field=field)
    if observed != identity:
        raise ValueError(f"{field} differs from the selected Modal live cohort")


def _identity_for_recording(
    *,
    project_root: Path,
    image_source_sha256: str,
    cohort_id: str,
) -> ModalLiveCohortIdentity:
    return ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256(project_root),
        image_source_sha256=_sha256(
            image_source_sha256, "image_source_sha256"
        ),
        cohort_id=validate_run_id(cohort_id),
    )


def _exact_bool(
    value: object,
    field: str,
    expected: bool | None = True,
) -> bool:
    if type(value) is not bool:
        if expected is None:
            raise ValueError(f"{field} must be boolean")
        raise ValueError(f"{field} must be exactly {expected}")
    if expected is not None and value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")
    return value


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def exact_json_equal(observed: object, expected: object) -> bool:
    """Compare JSON-shaped values without Python numeric type coercion.

    Python considers ``True == 1`` and ``1 == 1.0``.  Approval receipts must
    preserve the exact JSON scalar type as well as the value, including inside
    nested resource profiles and estimates.  Tuples are supported for the
    equivalent in-memory dataclass structures used before JSON serialization.
    """

    if type(observed) is not type(expected):
        return False
    if type(expected) is dict:
        if any(type(key) is not str for key in observed) or any(
            type(key) is not str for key in expected
        ):
            return False
        if set(observed) != set(expected):
            return False
        return all(
            exact_json_equal(observed[key], expected[key]) for key in expected
        )
    if type(expected) in {list, tuple}:
        return len(observed) == len(expected) and all(
            exact_json_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected, strict=True)
        )
    if expected is None:
        return True
    if type(expected) in {str, bool, int, float}:
        return observed == expected
    return False


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an explicit UTC timestamp") from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must use UTC")
    return parsed


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        payload[key] = value
    return payload


def _load_object(path: Path) -> dict[str, Any]:
    payload, _raw_sha256 = _load_object_with_sha256(path)
    return payload


def _text(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "text" if allow_empty else "non-empty text"
        raise ValueError(f"{field} must be {suffix}")
    return value


def _git_version(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 160
        or _GIT_VERSION.fullmatch(value) is None
    ):
        raise ValueError(
            f"{field} must be canonical `git version <numeric-version>` output"
        )
    return value


def _raw_timestamp_utc(value: object, field: str, *, naive_utc: bool) -> datetime:
    text = _text(value, field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        if not naive_utc:
            raise ValueError(f"{field} must include a timezone")
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _modal_billing_accounting_key(
    row: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str, str, datetime, str]:
    """Return the Modal charge identity, excluding descriptive metadata and cost."""

    return (
        _text(row["object_id"], f"{field}.object_id"),
        _text(row["environment"], f"{field}.environment", allow_empty=True),
        _raw_timestamp_utc(
            row["interval_start"],
            f"{field}.interval_start",
            naive_utc=False,
        ),
        _text(row["resource"], f"{field}.resource"),
    )


def _utc_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _cli_rows_from_bytes(
    raw: bytes,
    *,
    path: Path,
    snapshot_name: str,
) -> list[dict[str, Any]]:
    payload = json.loads(
        _decode_utf8(raw, path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, list):
        raise ValueError(f"{snapshot_name} must be a Modal CLI JSON row list")
    expected = _RAW_SNAPSHOT_FIELDS[snapshot_name]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{snapshot_name}[{index}] must be an object")
        if set(row) != expected:
            raise ValueError(
                f"{snapshot_name}[{index}] differs from the Modal 1.5.3 JSON schema"
            )
        rows.append(row)
    return rows


def _load_cli_rows(path: Path, snapshot_name: str) -> list[dict[str, Any]]:
    raw = _read_regular_file_bytes(
        path,
        maximum_bytes=_MAX_MODAL_CLI_SNAPSHOT_BYTES,
    )
    return _cli_rows_from_bytes(raw, path=path, snapshot_name=snapshot_name)


def _load_cleanup_snapshot_capture(
    root: Path,
    roster: Mapping[str, Any],
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, Any], Path, str, dict[str, list[dict[str, Any]]]]:
    """Load one exact, immutable six-command cleanup capture."""

    logical = _text(
        roster["snapshot_capture_manifest_path"],
        "snapshot_capture_manifest_path",
    )
    expected_sha256 = _sha256(
        roster["snapshot_capture_manifest_sha256"],
        "snapshot_capture_manifest_sha256",
    )
    manifest_path = _contained_path(
        root,
        logical,
        "snapshot_capture_manifest_path",
        kind="file",
    )
    manifest_raw = _read_regular_file_bytes(
        manifest_path,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        required_mode=0o600,
    )
    observed_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("cleanup snapshot capture manifest digest changed")
    manifest = json.loads(
        _decode_utf8(manifest_raw, manifest_path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(manifest, dict) or set(manifest) != (
        _SNAPSHOT_CAPTURE_MANIFEST_FIELDS
    ):
        raise ValueError("cleanup snapshot capture manifest has an invalid schema")
    if (
        manifest["schema_name"] != CAPTURE_MANIFEST_SCHEMA_NAME
        or manifest["schema_version"] != CAPTURE_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("cleanup snapshot capture manifest contract drifted")
    _assert_identity_matches(
        manifest,
        identity,
        field="snapshot_capture_manifest",
    )
    capture_id = validate_run_id(manifest["capture_id"])
    expected_logical = modal_cleanup_snapshot_capture_manifest_path(
        identity,
        capture_id,
    ).as_posix()
    if logical != expected_logical or manifest_path.name != CAPTURE_MANIFEST_FILENAME:
        raise ValueError("cleanup snapshot capture manifest path is not canonical")
    if (
        manifest["modal_profile"] != SNAPSHOT_MODAL_PROFILE
        or manifest["modal_environment"] != SNAPSHOT_MODAL_ENVIRONMENT
        or manifest["modal_cli_version"] != MODAL_VERSION
        or manifest["billing_window_start_utc"]
        != roster["billing_window_start_utc"]
        or manifest["billing_window_end_utc"] != roster["billing_window_end_utc"]
        or type(manifest["command_timeout_seconds"]) is not float
        or manifest["command_timeout_seconds"] != SNAPSHOT_COMMAND_TIMEOUT_SECONDS
        or type(manifest["outer_timeout_seconds"]) is not float
        or manifest["outer_timeout_seconds"] != SNAPSHOT_OUTER_TIMEOUT_SECONDS
        or _exact_int(
            manifest["command_retry_count"],
            "snapshot_capture_manifest.command_retry_count",
        )
        != 0
    ):
        raise ValueError("cleanup snapshot capture policy drifted")
    started = _utc(
        manifest["started_at_utc"],
        "snapshot_capture_manifest.started_at_utc",
    )
    finished = _utc(
        manifest["finished_at_utc"],
        "snapshot_capture_manifest.finished_at_utc",
    )
    billing_start = _utc(
        roster["billing_window_start_utc"],
        "billing_window_start_utc",
    )
    billing_end = _utc(
        roster["billing_window_end_utc"],
        "billing_window_end_utc",
    )
    if (
        billing_start.minute
        or billing_start.second
        or billing_start.microsecond
        or billing_end.minute
        or billing_end.second
        or billing_end.microsecond
        or billing_end <= billing_start
        or billing_end - billing_start > MAX_MODAL_BILLING_WINDOW
    ):
        raise ValueError(
            "cleanup snapshot billing window is invalid or exceeds 31 days"
        )
    if (
        started < billing_end
        or finished < started
        or roster["snapshot_captured_at_utc"] != _utc_z(finished)
    ):
        raise ValueError("cleanup snapshot capture timestamps are invalid")

    snapshots = manifest["snapshots"]
    if not isinstance(snapshots, dict) or set(snapshots) != set(_SNAPSHOT_NAMES):
        raise ValueError("cleanup snapshot capture roster is not exact")
    expected_commands = build_modal_cleanup_snapshot_commands(
        modal_executable=Path("/dev/fd/0"),
        billing_window_start_utc=roster["billing_window_start_utc"],
        billing_window_end_utc=roster["billing_window_end_utc"],
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    previous_capture_time = started
    modal_descriptor_argv: str | None = None
    capture_root = modal_cleanup_snapshot_capture_manifest_path(
        identity,
        capture_id,
    ).parent
    for name, expected_command in zip(
        _SNAPSHOT_NAMES,
        expected_commands,
        strict=True,
    ):
        record = snapshots[name]
        if not isinstance(record, dict) or set(record) != (
            _SNAPSHOT_CAPTURE_RECORD_FIELDS
        ):
            raise ValueError(f"snapshot capture {name} record has an invalid schema")
        expected_path = (capture_root / f"{name}.json").as_posix()
        argv = record["argv"]
        if (
            not isinstance(argv, list)
            or len(argv) != len(expected_command)
            or not all(isinstance(item, str) for item in argv)
            or re.fullmatch(r"/dev/fd/[0-9]+", argv[0]) is None
            or argv[1:] != list(expected_command[1:])
        ):
            raise ValueError(f"snapshot capture {name} command or path drifted")
        if modal_descriptor_argv is None:
            modal_descriptor_argv = argv[0]
        elif argv[0] != modal_descriptor_argv:
            raise ValueError("snapshot capture executable descriptor drifted")
        if record["path"] != expected_path:
            raise ValueError(f"snapshot capture {name} command or path drifted")
        size_bytes = _exact_int(
            record["size_bytes"],
            f"snapshot_capture.{name}.size_bytes",
        )
        leaf_sha256 = _sha256(
            record["sha256"],
            f"snapshot_capture.{name}.sha256",
        )
        captured_at = _utc(
            record["captured_at_utc"],
            f"snapshot_capture.{name}.captured_at_utc",
        )
        if captured_at < previous_capture_time or captured_at > finished:
            raise ValueError("cleanup snapshot capture timestamps are not ordered")
        previous_capture_time = captured_at
        leaf_path = _contained_path(
            root,
            expected_path,
            f"snapshot_capture.{name}.path",
            kind="file",
        )
        raw = _read_regular_file_bytes(
            leaf_path,
            maximum_bytes=_MAX_MODAL_CLI_SNAPSHOT_BYTES,
            required_mode=0o600,
        )
        if len(raw) != size_bytes or hashlib.sha256(raw).hexdigest() != leaf_sha256:
            raise ValueError(f"snapshot capture {name} bytes changed")
        rows[name] = _cli_rows_from_bytes(
            raw,
            path=leaf_path,
            snapshot_name=name,
        )
    return manifest, manifest_path, observed_sha256, rows


def _parse_volume_run_directory_row(
    raw: Mapping[str, Any],
    index: int,
) -> tuple[str, str, str, str, str, datetime]:
    """Parse one raw ``modal volume ls ... /runs --json`` row."""

    filename = _text(raw["filename"], f"run_directory_list[{index}].filename")
    kind = _text(raw["type"], f"run_directory_list[{index}].type")
    timestamp = _text(
        raw["created_modified"],
        f"run_directory_list[{index}].created_modified",
    )
    size = _text(raw["size"], f"run_directory_list[{index}].size")
    if kind not in {"dir", "file", "fifo", "link", "socket"}:
        raise ValueError("run directory inventory contains an unsupported type")
    if re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} UTC",
        timestamp,
    ) is None:
        raise ValueError("run directory inventory timestamp is not Modal CLI output")
    try:
        observed_at = datetime.strptime(timestamp[:16], "%Y-%m-%d %H:%M").replace(
            tzinfo=UTC
        )
    except ValueError as error:
        raise ValueError("run directory inventory timestamp is invalid") from error
    if re.fullmatch(
        r"(?:0|[1-9][0-9]*) B|(?:0|[1-9][0-9]*)\.[0-9] "
        r"(?:KiB|MiB|GiB|TiB|PiB|EiB|ZiB)",
        size,
    ) is None:
        raise ValueError("run directory inventory size is not Modal CLI output")
    normalized = filename.removeprefix("/").removesuffix("/")
    parts = PurePosixPath(normalized).parts
    if len(parts) != 2 or parts[0] != "runs":
        raise ValueError("run directory inventory path is outside /runs")
    run_id = validate_run_id(parts[1])
    return filename, kind, timestamp, size, run_id, observed_at


def _validate_owned_volume_run_start_times(
    rows: Sequence[Mapping[str, Any]],
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    observed_times: dict[str, datetime] = {}
    for index, row in enumerate(rows):
        parsed = _parse_volume_run_directory_row(row, index)
        observed_times[parsed[4]] = parsed[5]
    for attempt in attempts:
        started = _utc(attempt["started_at_utc"], "volume_owner.started_at_utc")
        for run_id in attempt["concrete_remote_run_ids"]:
            observed_at = observed_times.get(validate_run_id(run_id))
            if observed_at is not None and observed_at + timedelta(minutes=1) < started:
                raise ValueError(
                    "owned Volume /runs entry predates its launcher attempt"
                )


def _validate_owned_volume_run_time_bounds(
    rows: Sequence[Mapping[str, Any]],
    *,
    owned_run_ids: set[str],
    captured_at: datetime,
    recorded_at: datetime,
    observed_now: datetime,
) -> None:
    """Bound owned run-directory timestamps by their observation horizons."""

    for index, row in enumerate(rows):
        parsed = _parse_volume_run_directory_row(row, index)
        if parsed[4] in owned_run_ids:
            _validate_observed_timestamp_horizon(
                parsed[5],
                captured_at=captured_at,
                recorded_at=recorded_at,
                observed_now=observed_now,
                field="owned Volume /runs entry",
            )


def _volume_run_directory_inventory(
    rows: list[dict[str, Any]],
    roster: Mapping[str, Any],
    *,
    prior_quarantined_run_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Bind all cohort run IDs to the raw non-recursive ``volume ls`` rows."""

    by_run_id: dict[str, dict[str, str]] = {}
    unrelated_directory_count = 0
    required_current = {
        record["run_id"]
        for record in roster["terminal_run_dispositions"]
        if record["volume_disposition"] == "present_bound"
    }
    absent_current = {
        record["run_id"]
        for record in roster["terminal_run_dispositions"]
        if record["volume_disposition"] in {"absent", "absent_after_failure"}
    }
    required_prior = set(prior_quarantined_run_ids or set())
    if required_current.intersection(required_prior):
        raise ValueError("current and prior Volume run IDs overlap")
    required = required_current | required_prior | {_SUPERSEDED_RUN_ID}
    for index, raw in enumerate(rows):
        filename, kind, timestamp, size, run_id, _observed_at = (
            _parse_volume_run_directory_row(raw, index)
        )
        if run_id in by_run_id:
            raise ValueError("run directory inventory contains a duplicate run ID")
        by_run_id[run_id] = {
            "filename": filename,
            "type": kind,
            "created_modified": timestamp,
            "size": size,
        }
        if run_id not in required and kind == "dir":
            unrelated_directory_count += 1
    missing = sorted(required - set(by_run_id))
    if missing:
        raise ValueError(f"Volume /runs inventory lacks cohort directories: {missing}")
    mistyped = sorted(
        run_id for run_id in required if by_run_id[run_id]["type"] != "dir"
    )
    if mistyped:
        raise ValueError(f"Volume /runs cohort entries are not directories: {mistyped}")
    unexpected_absent = sorted(absent_current.intersection(by_run_id))
    if unexpected_absent:
        raise ValueError(
            "Volume /runs inventory contradicts absent terminal dispositions: "
            f"{unexpected_absent}"
        )
    return {
        "required_current_run_ids": sorted(required_current),
        "required_prior_quarantined_run_ids": sorted(required_prior),
        "superseded_run_id": _SUPERSEDED_RUN_ID,
        "observed_required_run_ids": sorted(required),
        "snapshot_entry_count": len(rows),
        "unrelated_directory_count": unrelated_directory_count,
    }


def _sorted_unique_text(
    value: object,
    field: str,
    *,
    run_ids: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a sorted list")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _text(item, f"{field}[{index}]")
        result.append(validate_run_id(text) if run_ids else text)
    if result != sorted(set(result)):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _decimal_text(value: object, field: str) -> Decimal:
    text = _text(value, field)
    if len(text) > 128:
        raise ValueError(f"{field} is not a decimal amount")
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(f"{field} is not a decimal amount") from error
    exponent_zero = (
        parsed.is_zero()
        and not parsed.is_signed()
        and re.fullmatch(r"0E[+-][1-9][0-9]*", text) is not None
        and str(parsed) == text
    )
    if (
        not parsed.is_finite()
        or parsed.is_signed()
        or parsed < 0
        or (format(parsed, "f") != text and not exponent_zero)
    ):
        raise ValueError(f"{field} must be canonical, finite, and non-negative")
    return parsed


def _canonical_modal_decimal(
    value: object,
    field: str,
    *,
    require_positive: bool,
) -> Decimal:
    if (
        not isinstance(value, str)
        or len(value) > 64
        or _CANONICAL_DECIMAL.fullmatch(value) is None
    ):
        raise ValueError(f"{field} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field} must be a canonical decimal string") from error
    if (
        not parsed.is_finite()
        or parsed.is_signed()
        or parsed < 0
        or (require_positive and parsed <= 0)
        or format(parsed, "f") != value
    ):
        qualifier = "positive" if require_positive else "non-negative"
        raise ValueError(f"{field} must be canonical, finite, and {qualifier}")
    return parsed


def _canonical_decimal_output(value: Decimal) -> str:
    if not value.is_finite() or value.is_signed():
        raise ValueError("derived Modal estimate must be finite and non-negative")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def _canonical_modal_price_timestamp(value: object) -> tuple[str, datetime]:
    if not isinstance(value, str) or _MODAL_PRICE_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(
            "Modal price basis timestamp must be second-precision UTC Z-form"
        )
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Modal price basis timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.astimezone(UTC).isoformat().replace(
        "+00:00", "Z"
    ) != value:
        raise ValueError("Modal price basis timestamp must be canonical UTC")
    return value, parsed.astimezone(UTC)


def modal_price_basis_logical_path(
    image_source_sha256: str,
    retrieved_at_utc: str,
) -> PurePosixPath:
    """Return the sole create-only path for one source-bound rate snapshot."""

    image_sha256 = _sha256(image_source_sha256, "image_source_sha256")
    timestamp, _parsed = _canonical_modal_price_timestamp(retrieved_at_utc)
    timestamp_component = re.sub(r"[-:]", "", timestamp)
    return MODAL_PRICE_BASIS_ROOT / image_sha256 / f"{timestamp_component}.json"


def validate_modal_price_basis_payload(
    payload: Mapping[str, Any],
    *,
    expected_image_source_sha256: str,
    now_utc: datetime | None = None,
    require_freshness: bool,
) -> dict[str, Decimal]:
    """Validate one exact Modal rate record without contacting Modal."""

    if set(payload) != _MODAL_PRICE_BASIS_FIELDS:
        raise ValueError("Modal price basis has an invalid exact schema")
    expected_image = _sha256(
        expected_image_source_sha256,
        "expected_image_source_sha256",
    )
    if (
        payload["schema_name"] != "ModalPriceBasis"
        or payload["schema_version"] != "1.0"
        or payload["image_source_sha256"] != expected_image
    ):
        raise ValueError("Modal price basis has the wrong contract or source")
    if payload["official_source_url"] != MODAL_PRICE_BASIS_OFFICIAL_SOURCE_URL:
        raise ValueError("Modal price basis must cite the exact official pricing page")
    if payload["region"] is not None:
        raise ValueError(
            "Modal price basis must use base rates with no region selected"
        )
    if payload["download_transfer_pricing"] != MODAL_DOWNLOAD_TRANSFER_PRICING:
        raise ValueError("Modal download-transfer pricing disclosure changed")
    _timestamp, retrieved = _canonical_modal_price_timestamp(
        payload["retrieved_at_utc"]
    )
    if require_freshness:
        observed_now = datetime.now(UTC) if now_utc is None else now_utc
        if observed_now.tzinfo is None:
            raise ValueError("Modal price-basis validation time needs a timezone")
        observed_now = observed_now.astimezone(UTC)
        if retrieved > observed_now + MODAL_PRICE_BASIS_FUTURE_SKEW:
            raise ValueError("Modal price basis timestamp is too far in the future")
        if observed_now - retrieved > MODAL_PRICE_BASIS_MAX_AGE:
            raise ValueError("Modal price basis is older than 48 hours")
    rates = {
        "cpu": _canonical_modal_decimal(
            payload["cpu_usd_per_core_second"],
            "cpu_usd_per_core_second",
            require_positive=True,
        ),
        "memory": _canonical_modal_decimal(
            payload["memory_usd_per_gib_second"],
            "memory_usd_per_gib_second",
            require_positive=True,
        ),
        "t4": _canonical_modal_decimal(
            payload["t4_usd_per_gpu_second"],
            "t4_usd_per_gpu_second",
            require_positive=True,
        ),
        "volume": _canonical_modal_decimal(
            payload["volume_storage_usd_per_gib_month"],
            "volume_storage_usd_per_gib_month",
            require_positive=True,
        ),
        "included_volume": _canonical_modal_decimal(
            payload["included_volume_storage_gib_per_month"],
            "included_volume_storage_gib_per_month",
            require_positive=False,
        ),
    }
    return rates


def load_modal_price_basis(
    root: Path,
    logical: object,
    *,
    expected_raw_sha256: object,
    expected_image_source_sha256: str,
    now_utc: datetime | None = None,
    require_freshness: bool,
) -> tuple[dict[str, Any], dict[str, Decimal], Path]:
    """Load, path-bind, hash-bind, and validate one immutable price basis."""

    path = _contained_path(root, logical, "modal_price_basis_path", kind="file")
    payload, observed_digest = _load_object_with_sha256(path)
    rates = validate_modal_price_basis_payload(
        payload,
        expected_image_source_sha256=expected_image_source_sha256,
        now_utc=now_utc,
        require_freshness=require_freshness,
    )
    expected_logical = modal_price_basis_logical_path(
        expected_image_source_sha256,
        payload["retrieved_at_utc"],
    ).as_posix()
    if logical != expected_logical:
        raise ValueError("Modal price basis is outside its canonical source path")
    expected_digest = _sha256(
        expected_raw_sha256,
        "modal_price_basis_sha256",
    )
    if observed_digest != expected_digest:
        raise ValueError("Modal price-basis raw SHA-256 changed")
    return payload, rates, path


def _expected_attempt_timeout(action: str, harness: str | None = None) -> int:
    if action == EVOLUTION_ACTION:
        return EvolutionRunSpec.parse(harness).outer_cli_timeout_seconds
    if action == "canaries":
        runtime_seconds = sum(
            function_spec(f"canary_{harness}").timeout_seconds
            for harness in CANARY_ORDER
        )
    elif action == "canary":
        runtime_seconds = function_spec(
            f"canary_{CANARY_ORDER[0]}"
        ).timeout_seconds
    elif action in {"download", "verify"}:
        runtime_seconds = function_spec("artifact_verify").timeout_seconds
    else:
        runtime_seconds = function_spec(action.replace("-", "_")).timeout_seconds
    return IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS + runtime_seconds + 300


def _reconstructed_modal_command_sha256(
    payload: Mapping[str, Any],
    *,
    root: Path,
) -> str:
    action = payload["action"]
    run_id = payload["run_id"]
    image_source_sha256 = payload["approved_image_source_sha256"]
    if not all(
        isinstance(value, str)
        for value in (action, run_id, image_source_sha256)
    ):
        raise ValueError("Modal command receipt lacks its canonical identity")
    return modal_cli_command_sha256(
        python_executable=sys.executable,
        project_root=root,
        action=action,
        run_id=run_id,
        source_run_id=payload["source_run_id"],
        verifier_run_id=payload["verifier_run_id"],
        harness=payload["harness"],
        source_tree_sha256=payload["source_tree_sha256"],
        cohort_id=payload["cohort_id"],
        image_source_sha256=image_source_sha256,
        provider_approved=payload["provider_cost_approved"],
    )


def _expected_modal_resource_profile(
    action: str,
    harness: str | None,
) -> dict[str, Any]:
    """Recompute the exact request/limit disclosure emitted by the launcher."""

    dynamic_timeout: int | None = None
    if action == EVOLUTION_ACTION:
        evolution = EvolutionRunSpec.parse(harness)
        function_names = [EVOLUTION_FUNCTION_NAME]
        dynamic_timeout = evolution.function_timeout_seconds
    elif action == "canaries":
        function_names = [f"canary_{item}" for item in CANARY_ORDER]
    elif action == "canary":
        if harness not in CANARY_ORDER:
            raise ValueError("single-canary resource profile requires a harness")
        function_names = [f"canary_{harness}"]
    elif action in {"download", "verify"}:
        function_names = ["artifact_verify"]
    else:
        function_names = [action.replace("-", "_")]
    runtime_calls = []
    for function_name in function_names:
        spec = function_spec(function_name)
        runtime_calls.append(
            {
                "function_name": function_name,
                "call_count": 1,
                "cpu_request_cores": spec.cpu_request_cores,
                "cpu_soft_limit_cores": spec.cpu_soft_limit_cores,
                "memory_request_mib": spec.memory_request_mib,
                "memory_limit_mib": spec.memory_limit_mib,
                "gpu": spec.gpu,
                "region": spec.region,
                "timeout_seconds": dynamic_timeout or spec.timeout_seconds,
                "max_containers": spec.max_containers,
                "min_containers": spec.min_containers,
                "retries": spec.retries,
                "provider_secret_attached": spec.provider_secret,
                "network_mode": (
                    "provider_egress_enabled"
                    if spec.provider_secret
                    else "blocked"
                ),
            }
        )
    return {
        "modal_environment": MODAL_ENVIRONMENT,
        "runtime_function_calls": runtime_calls,
        "runtime_cpu_request_equals_soft_limit": True,
        "runtime_memory_request_equals_hard_limit": True,
        "runtime_platform_compute_cost_ceiling_enforced": False,
        "runtime_functions_preemptible": True,
        "platform_preemption_restart_possible": True,
        "logical_call_count_is_not_container_attempt_ceiling": True,
        "image_build": {
            "invocation_condition": "backend_cache_miss",
            "cpu_request_cores": IMAGE_BUILD_CPU_REQUEST_CORES,
            "cpu_soft_limit_cores": None,
            "memory_request_mib": IMAGE_BUILD_MEMORY_REQUEST_MIB,
            "memory_limit_mib": None,
            "gpu": None,
            "region": None,
            "timeout_seconds": IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
            "subprocess_thread_limit": IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
            "resource_limits_exposed": False,
            "platform_compute_cost_ceiling_enforced": False,
            "provider_secret_attached": False,
            "network_mode": "dependency_install_egress_required",
        },
    }


def derive_modal_action_cost_estimate(
    *,
    action: str,
    harness: str | None,
    resource_profile: Mapping[str, Any],
    price_basis: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the deterministic estimate a local approval cap must cover.

    This is neither a platform billing floor nor a platform billing ceiling.
    """

    expected_profile = _expected_modal_resource_profile(action, harness)
    if not exact_json_equal(resource_profile, expected_profile):
        raise ValueError("Modal resource profile changed before cost estimation")
    rates = validate_modal_price_basis_payload(
        price_basis,
        expected_image_source_sha256=price_basis.get("image_source_sha256"),
        require_freshness=False,
    )
    runtime_cost = Decimal("0")
    calls = resource_profile["runtime_function_calls"]
    for index, call in enumerate(calls):
        if (
            call["cpu_request_cores"] != call["cpu_soft_limit_cores"]
            or call["memory_request_mib"] != call["memory_limit_mib"]
            or call["region"] is not None
            or call["max_containers"] != 1
            or call["min_containers"] != 0
            or call["retries"] != 0
        ):
            raise ValueError(
                f"runtime call {index} differs from the frozen request, soft-CPU, "
                "hard-memory, time, or concurrency contract"
            )
        seconds = Decimal(call["call_count"] * call["timeout_seconds"])
        cpu = Decimal(str(call["cpu_request_cores"]))
        memory_gib = Decimal(call["memory_request_mib"]) / Decimal(1024)
        runtime_cost += seconds * (
            cpu * rates["cpu"] + memory_gib * rates["memory"]
        )
        if call["gpu"] == GPU_TYPE:
            runtime_cost += seconds * rates["t4"]
        elif call["gpu"] is not None:
            raise ValueError("Modal price basis supports only the frozen T4 GPU")

    build = resource_profile["image_build"]
    if (
        build["cpu_soft_limit_cores"] is not None
        or build["memory_limit_mib"] is not None
        or build["gpu"] is not None
        or build["region"] is not None
        or build["resource_limits_exposed"] is not False
        or build["platform_compute_cost_ceiling_enforced"] is not False
    ):
        raise ValueError("Modal image-build estimate overclaims platform limits")
    build_seconds = Decimal(build["timeout_seconds"])
    build_cost = build_seconds * (
        Decimal(str(build["cpu_request_cores"])) * rates["cpu"]
        + Decimal(build["memory_request_mib"])
        / Decimal(1024)
        * rates["memory"]
    )

    per_run_storage_gib = Decimal(
        MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES + MAX_ARTIFACT_MANIFEST_BYTES
    ) / _GIB_BYTES
    new_remote_run_count = len(CANARY_ORDER) if action == "canaries" else 1
    storage_cost = (
        Decimal(new_remote_run_count) * per_run_storage_gib * rates["volume"]
    )
    transfer_bound_gib = (
        Decimal(2)
        * Decimal(MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES + MAX_ARTIFACT_MANIFEST_BYTES)
        / _GIB_BYTES
        if action == "download"
        else Decimal("0")
    )
    transfer_cost = Decimal("0")
    total = runtime_cost + build_cost + storage_cost + transfer_cost
    estimate = {
        "runtime_request_rate_estimate_usd": _canonical_decimal_output(
            runtime_cost
        ),
        "cache_miss_image_build_request_rate_estimate_usd": (
            _canonical_decimal_output(build_cost)
        ),
        "new_remote_run_count": new_remote_run_count,
        "per_remote_run_storage_bound_gib": _canonical_decimal_output(
            per_run_storage_gib
        ),
        "one_month_storage_estimate_usd": _canonical_decimal_output(
            storage_cost
        ),
        "download_transfer_bound_gib": _canonical_decimal_output(
            transfer_bound_gib
        ),
        "download_transfer_pricing": MODAL_DOWNLOAD_TRANSFER_PRICING,
        "download_transfer_estimate_usd": _canonical_decimal_output(
            transfer_cost
        ),
        "action_estimate_usd": _canonical_decimal_output(total),
        "scope": MODAL_COST_ESTIMATE_SCOPE,
    }
    if set(estimate) != _MODAL_COST_ESTIMATE_FIELDS:
        raise AssertionError("Modal cost estimate schema drifted")
    return estimate


def _validate_predecessor_bindings(
    value: object,
    *,
    root: Path,
    field: str,
) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an ordered list")
    records: list[dict[str, str]] = []
    observed: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != {"gate", "path", "sha256"}:
            raise ValueError(f"{field}[{index}] has an invalid exact schema")
        gate = _text(raw["gate"], f"{field}[{index}].gate")
        if not gate.replace("_", "").isalnum():
            raise ValueError(f"{field}[{index}].gate is invalid")
        logical = _text(raw["path"], f"{field}[{index}].path")
        path = _contained_path(
            root,
            logical,
            f"{field}[{index}].path",
            kind="file",
        )
        digest = _sha256(raw["sha256"], f"{field}[{index}].sha256")
        if digest != _sha256_file(path):
            raise ValueError(f"{field}[{index}] predecessor digest changed")
        key = (gate, logical)
        if key in observed:
            raise ValueError(f"{field} contains a duplicate predecessor")
        observed.add(key)
        records.append({"gate": gate, "path": logical, "sha256": digest})
    return records


def _candidate_preflight_binding(
    bindings: list[dict[str, str]],
) -> dict[str, str]:
    matches = [
        record
        for record in bindings
        if record["gate"] == "candidate_resume_preflight_validated"
    ]
    if len(matches) != 1:
        raise ValueError("provider approval requires one exact preflight binding")
    return matches[0]


_REMOTE_RUN_RESERVATION_BINDING_FIELDS = frozenset({"run_id", "path", "sha256"})
_REMOTE_RUN_RESERVATION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "remote_run_id",
        "owner_attempt_id",
        "action",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "modal_environment",
        "created_at_utc",
        "launch_capability_sha256",
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
    }
)

_LOCAL_PROCESS_START_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempt_id",
        "created_at_utc",
        "action",
        "run_id",
        "intent_path",
        "intent_sha256",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "modal_command_sha256",
        "launch_capability_sha256",
        "modal_cost_cap_usd",
        "provider_cost_cap_usd",
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
        "process_id",
        "expected_process_group_id",
        "expected_session_id",
        "process_birth_identity_sha256",
    }
)

_MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS = 946_684_800_000_000


def _validate_local_containment_fields(
    payload: Mapping[str, Any],
    *,
    field: str,
    not_after_utc: datetime | None = None,
) -> dict[str, Any]:
    """Validate historical containment bindings without inventing a boot UUID."""

    host_path = _text(payload["local_host_anchor_path"], f"{field}.path")
    if host_path != modal_local_host_anchor_path().as_posix():
        raise ValueError(f"{field} host-anchor path is not canonical")
    host_sha256 = _sha256(
        payload["local_host_anchor_sha256"],
        f"{field}.host_anchor_sha256",
    )
    boot_started = _exact_int(
        payload["local_boot_started_at_unix_microseconds"],
        f"{field}.boot_started_at_unix_microseconds",
        minimum=_MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS,
    )
    if not_after_utc is not None and boot_started > int(
        not_after_utc.timestamp() * 1_000_000
    ):
        raise ValueError(f"{field} boot session begins after its receipt")
    session_sha256 = _sha256(
        payload["local_boot_session_sha256"],
        f"{field}.boot_session_sha256",
    )
    return {
        "local_host_anchor_path": host_path,
        "local_host_anchor_sha256": host_sha256,
        "local_boot_started_at_unix_microseconds": boot_started,
        "local_boot_session_sha256": session_sha256,
    }


def _validate_remote_run_reservation_bindings(
    root: Path,
    value: object,
    *,
    concrete_remote_run_ids: list[str],
    attempt_id: str,
    action: str,
    identity: ModalLiveCohortIdentity,
    created_at_utc: str,
    launch_capability_sha256: str,
    local_containment: Mapping[str, Any],
) -> list[dict[str, str]]:
    if not isinstance(value, list) or len(value) != len(concrete_remote_run_ids):
        raise ValueError("remote run reservation roster is not exact")
    expected_runs = [validate_run_id(item) for item in concrete_remote_run_ids]
    created_at = _utc(created_at_utc, "reservation.created_at_utc")
    containment = _validate_local_containment_fields(
        local_containment,
        field="reservation.local_containment",
        not_after_utc=created_at,
    )
    records: list[dict[str, str]] = []
    for index, (raw, run_id) in enumerate(zip(value, expected_runs, strict=True)):
        if not isinstance(raw, dict) or set(raw) != (
            _REMOTE_RUN_RESERVATION_BINDING_FIELDS
        ):
            raise ValueError(
                f"remote_run_reservations[{index}] has an invalid exact schema"
            )
        expected_path = modal_remote_run_reservation_path(run_id).as_posix()
        digest = _sha256(
            raw["sha256"],
            f"remote_run_reservations[{index}].sha256",
        )
        if raw["run_id"] != run_id or raw["path"] != expected_path:
            raise ValueError("remote run reservation binding is not canonical")
        reservation_path = _contained_path(
            root,
            expected_path,
            f"remote_run_reservations[{index}].path",
            kind="file",
        )
        reservation, observed_sha256 = _load_object_with_sha256(reservation_path)
        if observed_sha256 != digest:
            raise ValueError("remote run reservation raw digest changed")
        if set(reservation) != _REMOTE_RUN_RESERVATION_FIELDS or reservation != {
            "schema_name": "ModalRemoteRunReservation",
            "schema_version": "1.2",
            "remote_run_id": run_id,
            "owner_attempt_id": attempt_id,
            "action": action,
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "modal_environment": MODAL_ENVIRONMENT,
            "created_at_utc": created_at_utc,
            "launch_capability_sha256": launch_capability_sha256,
            **containment,
        }:
            raise ValueError("remote run reservation owner or identity changed")
        records.append({"run_id": run_id, "path": expected_path, "sha256": digest})
    return records


def _expected_predecessor_gates(
    action: str,
    bindings: list[dict[str, str]],
    *,
    root: Path,
    image_source_sha256: str | None,
    identity: ModalLiveCohortIdentity,
) -> list[str]:
    if len(bindings) < 3:
        raise ValueError("action lacks its local engineering freeze bindings")
    expected_local_bindings = (
        historical_local_engineering_freeze_predecessor_bindings(
            bindings[:3],
            root=root,
            expected_image_source_sha256=image_source_sha256,
        )
    )
    if not exact_json_equal(tuple(bindings[:3]), expected_local_bindings):
        raise ValueError("action local engineering freeze bindings changed")
    local_expected = [
        (record["gate"], record["path"]) for record in expected_local_bindings
    ]
    if [(record["gate"], record["path"]) for record in bindings[:3]] != (
        local_expected
    ):
        raise ValueError("action local engineering freeze paths are invalid")
    action_bindings = bindings[3:]
    fixed = {
        "cuda-environment": [],
        "offline-smoke": [
            (
                "modal_cuda_environment_validated",
                modal_component_receipt_path(
                    identity, "modal_cuda_environment_validated"
                ).as_posix(),
            )
        ],
        "candidate-smoke": [
            (
                "modal_cuda_environment_validated",
                modal_component_receipt_path(
                    identity, "modal_cuda_environment_validated"
                ).as_posix(),
            ),
            (
                "modal_offline_smoke_validated",
                modal_component_receipt_path(
                    identity, "modal_offline_smoke_validated"
                ).as_posix(),
            ),
        ],
        "checkpoint-resume": [
            (
                "modal_artifact_round_trip_validated",
                modal_component_receipt_path(
                    identity, "modal_artifact_round_trip_validated"
                ).as_posix(),
            )
        ],
    }
    if action in {"download", "verify"}:
        pairs = [(record["gate"], record["path"]) for record in action_bindings]
        if len(pairs) not in {3, 4}:
            raise ValueError("verifier predecessor receipt order is invalid")
        attempt_directory = modal_action_attempt_directory(identity).as_posix()
        intent_match = re.fullmatch(
            rf"{re.escape(attempt_directory)}/([0-9a-f]{{32}})\.intent\.json",
            pairs[0][1],
        )
        if intent_match is None:
            raise ValueError("verifier source intent path is not canonical")
        source_attempt_id = intent_match.group(1)
        expected_pairs = [
            (
                "source_action_intent",
                modal_action_intent_receipt_path(
                    identity, source_attempt_id
                ).as_posix(),
            ),
            (
                "source_action_attempt_terminal",
                modal_action_terminal_receipt_path(
                    identity, source_attempt_id
                ).as_posix(),
            ),
            (
                "source_local_process_start",
                modal_local_process_start_receipt_path(
                    source_attempt_id
                ).as_posix(),
            ),
        ]
        if len(pairs) == 4:
            expected_pairs.append(
                (
                    "provider_canary_aggregate_outcomes",
                    provider_canary_aggregate_outcome_receipt_path(
                        identity, source_attempt_id
                    ).as_posix(),
                )
            )
        if not exact_json_equal(pairs, expected_pairs):
            raise ValueError("verifier predecessor paths are not source-attempt bound")
        return [record["gate"] for record in bindings]
    if action in _PROVIDER_LAUNCH_ACTIONS:
        if len(action_bindings) != 1:
            raise ValueError("provider action requires one preflight predecessor")
        gate, logical = action_bindings[0]["gate"], action_bindings[0]["path"]
        expected_parent = (
            modal_live_cohort_root(identity)
            / "components"
            / "candidate_resume_preflight_receipts"
            / "v2.0"
        )
        relative = safe_relative_path(logical)
        if (
            gate != "candidate_resume_preflight_validated"
            or relative.parent != expected_parent
            or re.fullmatch(r"[0-9a-f]{64}\.json", relative.name) is None
        ):
            raise ValueError("provider preflight predecessor path is not canonical")
        return [*LOCAL_ENGINEERING_FREEZE_GATES, gate]
    expected = fixed[action]
    if not exact_json_equal(
        [(record["gate"], record["path"]) for record in action_bindings],
        expected,
    ):
        raise ValueError("action predecessor receipt order is invalid")
    return [*LOCAL_ENGINEERING_FREEZE_GATES, *(gate for gate, _path in expected)]


def _validate_action_intent_receipt(
    payload: Mapping[str, Any],
    *,
    expected_attempt_id: str,
    root: Path,
) -> dict[str, Any]:
    if set(payload) != _ACTION_INTENT_FIELDS:
        raise ValueError("Modal action intent has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalActionIntent"
        or payload["schema_version"] != "1.6"
        or payload["attempt_id"] != expected_attempt_id
    ):
        raise ValueError("Modal action intent has the wrong contract")
    created_at = _utc(payload["created_at_utc"], "intent.created_at_utc")
    _validate_local_containment_fields(
        payload,
        field="intent.local_containment",
        not_after_utc=created_at,
    )
    action, run_id, _source_run_id, verifier_run_id, harness = (
        validate_modal_action_identity(
            action=payload["action"],
            run_id=payload["run_id"],
            source_run_id=payload["source_run_id"],
            verifier_run_id=payload["verifier_run_id"],
            harness=payload["harness"],
        )
    )
    concrete = payload["concrete_remote_run_ids"]
    if not isinstance(concrete, list):
        raise ValueError("intent concrete remote run IDs must be a list")
    for value in concrete:
        validate_run_id(value)
    if action in {"download", "verify"}:
        expected_concrete = [verifier_run_id]
    elif action == "canaries":
        expected_concrete = [
            f"{run_id}-{_CANARY_SUFFIXES[item]}" for item in CANARY_ORDER
        ]
    else:
        expected_concrete = [run_id]
    if concrete != expected_concrete:
        raise ValueError("Modal action intent concrete run roster is invalid")
    _sha256(payload["approved_image_source_sha256"], "intent.image_source_sha256")
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=_sha256(
            payload["source_tree_sha256"], "intent.source_tree_sha256"
        ),
        image_source_sha256=payload["approved_image_source_sha256"],
        cohort_id=validate_run_id(payload["cohort_id"]),
    )
    launch_capability_sha256 = _sha256(
        payload["launch_capability_sha256"],
        "intent.launch_capability_sha256",
    )
    _validate_remote_run_reservation_bindings(
        root,
        payload["remote_run_reservations"],
        concrete_remote_run_ids=concrete,
        attempt_id=expected_attempt_id,
        action=action,
        identity=identity,
        created_at_utc=payload["created_at_utc"],
        launch_capability_sha256=launch_capability_sha256,
        local_containment=payload,
    )
    bindings = _validate_predecessor_bindings(
        payload["predecessor_receipts"],
        root=root,
        field="intent.predecessor_receipts",
    )
    _expected_predecessor_gates(
        action,
        bindings,
        root=root,
        image_source_sha256=payload["approved_image_source_sha256"],
        identity=identity,
    )
    _sha256(payload["modal_command_sha256"], "intent.modal_command_sha256")
    if payload["modal_command_sha256"] != _reconstructed_modal_command_sha256(
        payload,
        root=root,
    ):
        raise ValueError("Modal action intent command digest changed")
    if payload["modal_profile"] != "scalingintelligence":
        raise ValueError("Modal action intent used the wrong profile")
    if payload["modal_environment"] != MODAL_ENVIRONMENT:
        raise ValueError("Modal action intent used the wrong environment")
    if payload["outer_cli_timeout_seconds"] != _expected_attempt_timeout(
        action, harness
    ):
        raise ValueError("Modal action intent timeout differs from its action")
    if not exact_json_equal(
        payload["modal_resource_profile"],
        _expected_modal_resource_profile(action, harness),
    ):
        raise ValueError("Modal action intent resource profile changed")
    modal_cap = _decimal_text(
        payload["modal_cost_cap_usd"],
        "intent.modal_cost_cap_usd",
    )
    if modal_cap <= 0 or payload["modal_cost_approved"] is not True:
        raise ValueError("Modal action intent lacks positive Modal cost approval")
    modal_price, _rates, _price_path = load_modal_price_basis(
        root,
        payload["modal_price_basis_path"],
        expected_raw_sha256=payload["modal_price_basis_sha256"],
        expected_image_source_sha256=payload["approved_image_source_sha256"],
        require_freshness=False,
    )
    expected_modal_estimate = derive_modal_action_cost_estimate(
        action=action,
        harness=harness,
        resource_profile=payload["modal_resource_profile"],
        price_basis=modal_price,
    )
    if not exact_json_equal(
        payload["modal_cost_estimate"], expected_modal_estimate
    ):
        raise ValueError("Modal action intent cost estimate changed")
    if modal_cap < _canonical_modal_decimal(
        expected_modal_estimate["action_estimate_usd"],
        "intent.modal_cost_estimate.action_estimate_usd",
        require_positive=True,
    ):
        raise ValueError("Modal action intent cap is below its exact estimate")
    if type(payload["provider_cost_approved"]) is not bool:
        raise ValueError("intent.provider_cost_approved must be boolean")
    provider_action = action in _PROVIDER_LAUNCH_ACTIONS
    provider_fields = (
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
    )
    if provider_action:
        if payload["provider_cost_approved"] is not True:
            raise ValueError("provider action intent lacks provider approval")
        provider_cap = _decimal_text(
            payload["provider_cost_cap_usd"],
            "intent.provider_cost_cap_usd",
        )
        if provider_cap <= 0:
            raise ValueError("provider action intent cap must be positive")
        _load_provider_approval_plan(
            root,
            payload["provider_approval_plan_path"],
            expected_approval_sha256=payload["approval_plan_sha256"],
            expected_image_source_sha256=payload[
                "approved_image_source_sha256"
            ],
            expected_identity=identity,
            expected_preflight_binding=_candidate_preflight_binding(bindings),
            expected_evolution_spec=(
                harness if action == EVOLUTION_ACTION else None
            ),
        )
        _price, _price_path, price_digest = _load_price_basis(
            root, payload["provider_price_basis_path"]
        )
        if price_digest != payload["provider_price_basis_sha256"]:
            raise ValueError("intent provider price-basis digest changed")
    elif payload["provider_cost_approved"] is not False or any(
        payload[field] is not None for field in provider_fields
    ):
        raise ValueError("non-provider intent contains provider approval fields")
    if type(payload["source_evidence_recovery"]) is not bool or (
        payload["source_evidence_recovery"] and action not in {"download", "verify"}
    ):
        raise ValueError("Modal action intent evidence-recovery flag is invalid")
    return dict(payload)


def _positive_process_identity(value: object, field: str) -> int:
    if type(value) is not int or value <= 1:
        raise ValueError(f"{field} must be an exact positive process identifier")
    return value


def _validate_local_process_start_for_terminal(
    root: Path,
    terminal: Mapping[str, Any],
    *,
    identity: ModalLiveCohortIdentity,
    started_at: datetime,
    finished_at: datetime,
) -> dict[str, Any]:
    attempt_id = _attempt_id(
        terminal["attempt_id"],
        "local_process_start.attempt_id",
    )
    logical = _text(
        terminal["local_process_start_receipt_path"],
        "attempt.local_process_start_receipt_path",
    )
    if logical != modal_local_process_start_receipt_path(attempt_id).as_posix():
        raise ValueError("local process-start receipt path is not canonical")
    expected_sha256 = _sha256(
        terminal["local_process_start_receipt_sha256"],
        "attempt.local_process_start_receipt_sha256",
    )
    path = _contained_path(
        root,
        logical,
        "attempt.local_process_start_receipt_path",
        kind="file",
    )
    raw = _read_regular_file_bytes(
        path,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        required_mode=0o600,
    )
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("local process-start receipt bytes changed")
    marker = json.loads(
        _decode_utf8(raw, path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(marker, dict) or set(marker) != _LOCAL_PROCESS_START_FIELDS:
        raise ValueError("local process-start receipt has an invalid exact schema")
    if (
        marker["schema_name"] != "ModalLocalProcessStart"
        or marker["schema_version"] != "1.1"
        or marker["attempt_id"] != attempt_id
    ):
        raise ValueError("local process-start receipt has the wrong contract")
    created_at = _utc(
        marker["created_at_utc"],
        "local_process_start.created_at_utc",
    )
    if created_at < started_at or created_at > finished_at:
        raise ValueError("local process-start timestamp is outside its attempt")
    intent_logical = modal_action_intent_receipt_path(
        identity,
        attempt_id,
    ).as_posix()
    if marker["intent_path"] != intent_logical:
        raise ValueError("local process-start intent path is not canonical")
    intent_path = _contained_path(
        root,
        intent_logical,
        "local_process_start.intent_path",
        kind="file",
    )
    if _sha256(
        marker["intent_sha256"],
        "local_process_start.intent_sha256",
    ) != _sha256_file(intent_path):
        raise ValueError("local process-start intent bytes changed")
    process_id = _positive_process_identity(
        marker["process_id"],
        "local_process_start.process_id",
    )
    if (
        _positive_process_identity(
            marker["expected_process_group_id"],
            "local_process_start.expected_process_group_id",
        )
        != process_id
        or _positive_process_identity(
            marker["expected_session_id"],
            "local_process_start.expected_session_id",
        )
        != process_id
    ):
        raise ValueError("local process-start process identity changed")
    _sha256(
        marker["process_birth_identity_sha256"],
        "local_process_start.process_birth_identity_sha256",
    )
    _validate_local_containment_fields(
        marker,
        field="local_process_start.local_containment",
        not_after_utc=created_at,
    )
    expected = {
        "action": terminal["action"],
        "run_id": terminal["run_id"],
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
        "modal_command_sha256": terminal["modal_command_sha256"],
        "launch_capability_sha256": terminal["launch_capability_sha256"],
        "modal_cost_cap_usd": terminal["modal_cost_cap_usd"],
        "provider_cost_cap_usd": terminal["provider_cost_cap_usd"],
        "local_host_anchor_path": terminal["local_host_anchor_path"],
        "local_host_anchor_sha256": terminal["local_host_anchor_sha256"],
        "local_boot_started_at_unix_microseconds": terminal[
            "local_boot_started_at_unix_microseconds"
        ],
        "local_boot_session_sha256": terminal["local_boot_session_sha256"],
        "process_id": terminal["local_process_id"],
        "expected_process_group_id": terminal["local_process_group_id"],
        "expected_session_id": terminal["local_session_id"],
    }
    if any(
        not exact_json_equal(marker[field], value)
        for field, value in expected.items()
    ):
        raise ValueError("local process-start receipt differs from its terminal")
    return marker


def _validate_action_attempt_receipt(
    payload: Mapping[str, Any],
    *,
    expected_attempt_id: str,
    root: Path,
) -> dict[str, Any]:
    if set(payload) != _ACTION_ATTEMPT_FIELDS:
        raise ValueError("Modal action attempt receipt has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalActionAttemptReceipt"
        or payload["schema_version"] != "3.6"
    ):
        raise ValueError("Modal action attempt receipt has the wrong contract")
    attempt_id = _text(payload["attempt_id"], "attempt_id")
    if re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
        raise ValueError("Modal action attempt ID is invalid")
    if attempt_id != expected_attempt_id:
        raise ValueError("Modal action attempt ID differs from its filename")
    started = _utc(payload["started_at_utc"], "attempt.started_at_utc")
    finished = _utc(payload["finished_at_utc"], "attempt.finished_at_utc")
    if finished < started:
        raise ValueError("Modal action attempt finished before it started")
    status = _text(payload["status"], "attempt.status")
    if status not in _ACTION_STATUSES:
        raise ValueError("Modal action attempt status is unsupported")

    action = payload["action"]
    if action is not None and action not in _ACTIONS:
        raise ValueError("Modal action attempt action is unsupported")
    for field in ("run_id", "source_run_id", "verifier_run_id"):
        value = payload[field]
        if value is not None:
            validate_run_id(value)
    harness = payload["harness"]
    if harness is not None:
        if action == EVOLUTION_ACTION:
            EvolutionRunSpec.parse(harness)
        elif harness not in CANARY_ORDER:
            raise ValueError("Modal action attempt harness is unsupported")
    if action is not None and payload["run_id"] is not None:
        validate_modal_action_identity(
            action=action,
            run_id=payload["run_id"],
            source_run_id=payload["source_run_id"],
            verifier_run_id=payload["verifier_run_id"],
            harness=harness,
        )
    concrete_run_ids = payload["concrete_remote_run_ids"]
    if not isinstance(concrete_run_ids, list):
        raise ValueError("attempt.concrete_remote_run_ids must be a list")
    for index, value in enumerate(concrete_run_ids):
        if not isinstance(value, str):
            raise ValueError(
                f"attempt.concrete_remote_run_ids[{index}] must be text"
            )
        validate_run_id(value)
    if len(concrete_run_ids) != len(set(concrete_run_ids)):
        raise ValueError("attempt concrete remote run IDs are duplicated")
    for field in ("failure_kind",):
        value = payload[field]
        if value is not None:
            _text(value, f"attempt.{field}")
    for field in (
        "approved_image_source_sha256",
        "modal_command_sha256",
        "launch_capability_sha256",
    ):
        value = payload[field]
        if value is not None:
            _sha256(value, f"attempt.{field}")
    selected_identity: ModalLiveCohortIdentity | None = None
    if payload["source_tree_sha256"] is not None:
        if payload["approved_image_source_sha256"] is None:
            raise ValueError("cohort-scoped attempt lacks its image source digest")
        selected_identity = ModalLiveCohortIdentity(
            source_tree_sha256=_sha256(
                payload["source_tree_sha256"], "attempt.source_tree_sha256"
            ),
            image_source_sha256=payload["approved_image_source_sha256"],
            cohort_id=validate_run_id(payload["cohort_id"]),
        )
    elif payload["cohort_id"] is not None:
        raise ValueError("attempt cohort identity is only partially populated")
    containment_fields = (
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
    )
    if selected_identity is not None:
        _validate_local_containment_fields(
            payload,
            field="attempt.local_containment",
            not_after_utc=started,
        )
    elif any(payload[field] is not None for field in containment_fields):
        raise ValueError("unbound attempt contains local containment fields")
    if (
        action is not None
        and selected_identity is not None
        and payload["launch_capability_sha256"] is not None
    ):
        _validate_remote_run_reservation_bindings(
            root,
            payload["remote_run_reservations"],
            concrete_remote_run_ids=concrete_run_ids,
            attempt_id=attempt_id,
            action=action,
            identity=selected_identity,
            created_at_utc=payload["started_at_utc"],
            launch_capability_sha256=payload["launch_capability_sha256"],
            local_containment=payload,
        )
    elif payload["remote_run_reservations"] != []:
        raise ValueError("unbound attempt contains remote run reservations")
    if payload["modal_profile"] != "scalingintelligence":
        raise ValueError("Modal action attempt used the wrong profile")
    if payload["modal_environment"] != MODAL_ENVIRONMENT:
        raise ValueError("Modal action attempt used the wrong environment")
    timeout = payload["outer_cli_timeout_seconds"]
    if timeout is not None:
        timeout = _exact_int(timeout, "attempt.outer_cli_timeout_seconds", minimum=1)
        if action is None or timeout != _expected_attempt_timeout(action, harness):
            raise ValueError("Modal action attempt timeout differs from its action")
    resource_profile = payload["modal_resource_profile"]
    if resource_profile is not None and (
        action is None
        or not exact_json_equal(
            resource_profile,
            _expected_modal_resource_profile(action, harness),
        )
    ):
        raise ValueError("Modal action attempt resource profile changed")
    for field in ("modal_cost_approved", "provider_cost_approved"):
        if type(payload[field]) is not bool:
            raise ValueError(f"attempt.{field} must be boolean")
    if payload["modal_command_sha256"] is not None and (
        payload["modal_command_sha256"]
        != _reconstructed_modal_command_sha256(payload, root=root)
    ):
        raise ValueError("Modal action attempt command digest changed")
    process_started = payload["modal_cli_process_started"]
    if type(process_started) is not bool:
        raise ValueError("attempt.modal_cli_process_started must be boolean")
    process_start_fields = (
        "local_process_start_receipt_path",
        "local_process_start_receipt_sha256",
        "local_process_id",
        "local_process_group_id",
        "local_session_id",
    )
    if not process_started:
        if any(payload[field] is not None for field in process_start_fields):
            raise ValueError("unstarted Modal attempt has process-start evidence")
    else:
        expected_marker_path = modal_local_process_start_receipt_path(
            attempt_id
        ).as_posix()
        if payload["local_process_start_receipt_path"] != expected_marker_path:
            raise ValueError("started Modal attempt lacks its canonical start marker")
        process_id = payload["local_process_id"]
        process_group_id = payload["local_process_group_id"]
        session_id = payload["local_session_id"]
        if process_id is None:
            if process_group_id is not None or session_id is not None:
                raise ValueError(
                    "started Modal process identity is partially populated"
                )
        else:
            process_id = _positive_process_identity(
                process_id,
                "attempt.local_process_id",
            )
            if (
                _positive_process_identity(
                    process_group_id,
                    "attempt.local_process_group_id",
                )
                != process_id
                or _positive_process_identity(
                    session_id,
                    "attempt.local_session_id",
                )
                != process_id
            ):
                raise ValueError("started Modal process identity changed")
        marker_sha256 = payload["local_process_start_receipt_sha256"]
        if marker_sha256 is not None:
            if selected_identity is None or process_id is None:
                raise ValueError("process-start marker lacks a bound process identity")
            _validate_local_process_start_for_terminal(
                root,
                payload,
                identity=selected_identity,
                started_at=started,
                finished_at=finished,
            )
    modal_cap = payload["modal_cost_cap_usd"]
    if modal_cap is not None:
        _decimal_text(modal_cap, "attempt.modal_cost_cap_usd")
    modal_basis_complete = all(
        payload[field] is not None
        for field in (
            "modal_price_basis_path",
            "modal_price_basis_sha256",
            "modal_cost_estimate",
        )
    )
    if modal_basis_complete:
        if (
            action is None
            or resource_profile is None
            or payload["approved_image_source_sha256"] is None
            or modal_cap is None
        ):
            raise ValueError("Modal attempt cost binding lacks its action contract")
        modal_price, _rates, _price_path = load_modal_price_basis(
            root,
            payload["modal_price_basis_path"],
            expected_raw_sha256=payload["modal_price_basis_sha256"],
            expected_image_source_sha256=payload[
                "approved_image_source_sha256"
            ],
            require_freshness=False,
        )
        expected_modal_estimate = derive_modal_action_cost_estimate(
            action=action,
            harness=harness,
            resource_profile=resource_profile,
            price_basis=modal_price,
        )
        if not exact_json_equal(
            payload["modal_cost_estimate"], expected_modal_estimate
        ):
            raise ValueError("Modal attempt cost estimate changed")
        if _decimal_text(
            modal_cap, "attempt.modal_cost_cap_usd"
        ) < _canonical_modal_decimal(
            expected_modal_estimate["action_estimate_usd"],
            "attempt.modal_cost_estimate.action_estimate_usd",
            require_positive=True,
        ):
            raise ValueError("Modal attempt cap is below its exact estimate")
    elif payload["modal_cost_estimate"] is not None:
        raise ValueError("partial Modal action cost estimate is not allowed")
    bindings = _validate_predecessor_bindings(
        payload["predecessor_receipts"],
        root=root,
        field="attempt.predecessor_receipts",
    )
    provider_fields = (
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
    )
    if action in _PROVIDER_LAUNCH_ACTIONS:
        if payload["provider_cost_cap_usd"] is not None:
            _decimal_text(
                payload["provider_cost_cap_usd"],
                "attempt.provider_cost_cap_usd",
            )
        for field in ("provider_approval_plan_path", "provider_price_basis_path"):
            if payload[field] is not None:
                _contained_path(root, payload[field], f"attempt.{field}", kind="file")
        for field in ("approval_plan_sha256", "provider_price_basis_sha256"):
            if payload[field] is not None:
                _sha256(payload[field], f"attempt.{field}")
        if process_started and (
            payload["provider_cost_approved"] is not True
            or any(payload[field] is None for field in provider_fields)
        ):
            raise ValueError(
                "started provider attempt lacks complete approval artifacts"
            )
        if all(payload[field] is not None for field in provider_fields):
            if selected_identity is None:
                raise ValueError("provider attempt lacks its cohort identity")
            _load_provider_approval_plan(
                root,
                payload["provider_approval_plan_path"],
                expected_approval_sha256=payload["approval_plan_sha256"],
                expected_image_source_sha256=payload[
                    "approved_image_source_sha256"
                ],
                expected_identity=selected_identity,
                expected_preflight_binding=_candidate_preflight_binding(bindings),
                expected_evolution_spec=(
                    harness if action == EVOLUTION_ACTION else None
                ),
            )
            _price, _price_path, price_digest = _load_price_basis(
                root, payload["provider_price_basis_path"]
            )
            if price_digest != payload["provider_price_basis_sha256"]:
                raise ValueError("attempt provider price-basis digest changed")
    elif any(payload[field] is not None for field in provider_fields):
        raise ValueError("non-provider attempt contains provider approval artifacts")
    if type(payload["source_evidence_recovery"]) is not bool or (
        payload["source_evidence_recovery"] and action not in {"download", "verify"}
    ):
        raise ValueError("attempt source-evidence-recovery flag is invalid")
    returncode = payload["returncode"]
    if returncode is not None and (
        not isinstance(returncode, int) or isinstance(returncode, bool)
    ):
        raise ValueError("attempt.returncode must be an exact integer or null")
    closed = payload["process_group_closed"]
    if closed is not None and type(closed) is not bool:
        raise ValueError("attempt.process_group_closed must be boolean or null")
    remote_state = payload["remote_execution_state"]
    if not isinstance(remote_state, str) or remote_state not in {
        "definitely_not_started",
        "may_have_started",
    }:
        raise ValueError("attempt.remote_execution_state is unsupported")
    if (remote_state == "may_have_started") is not process_started:
        raise ValueError("attempt process start and remote execution state disagree")
    if process_started:
        if action is None:
            raise ValueError("started Modal attempt lacks its action")
        if selected_identity is None:
            raise ValueError("started Modal attempt lacks its cohort identity")
        _expected_predecessor_gates(
            action,
            bindings,
            root=root,
            image_source_sha256=payload["approved_image_source_sha256"],
            identity=selected_identity,
        )
        if (
            payload["modal_command_sha256"] is None
            or payload["launch_capability_sha256"] is None
            or payload["approved_image_source_sha256"] is None
            or timeout is None
            or modal_cap is None
            or resource_profile is None
            or not modal_basis_complete
            or _decimal_text(modal_cap, "attempt.modal_cost_cap_usd") <= 0
            or payload["modal_cost_approved"] is not True
            or payload["provider_cost_approved"]
            is not (action in _PROVIDER_LAUNCH_ACTIONS)
        ):
            raise ValueError("started Modal attempt lacks its approved launch contract")
    elif returncode is not None or closed is not None:
        raise ValueError("unstarted Modal attempt has process terminal fields")

    if status == "succeeded":
        if (
            action is None
            or payload["failure_kind"] is not None
            or returncode != 0
            or closed is not True
            or payload["modal_cost_approved"] is not True
            or payload["approved_image_source_sha256"] is None
            or payload["modal_command_sha256"] is None
            or payload["launch_capability_sha256"] is None
            or process_started is not True
        ):
            raise ValueError("successful Modal attempt fields do not reconcile")
        if payload["provider_cost_approved"] is not (
            action in _PROVIDER_LAUNCH_ACTIONS
        ):
            raise ValueError("successful Modal attempt provider approval is invalid")
    elif status == "failed":
        if (
            payload["failure_kind"] != "modal_cli_exit"
            or returncode is None
            or returncode == 0
            or closed is not True
            or process_started is not True
        ):
            raise ValueError("failed Modal attempt fields do not reconcile")
    elif status == "timed_out":
        if (
            payload["failure_kind"] != "outer_cli_timeout"
            or returncode is not None
            or closed is not True
            or process_started is not True
        ):
            raise ValueError("timed-out Modal attempt fields do not reconcile")
    elif status in {"preflight_failed", "preflight_rejected", "lock_contended"}:
        allowed_failure_kinds = {
            "preflight_failed": {
                "preflight",
                "action_intent_persistence",
                "action_intent_post_persistence",
                "action_intent_persistence_uncertain",
            },
            "preflight_rejected": {"preflight"},
            "lock_contended": {"local_launcher_lock"},
        }[status]
        if (
            payload["failure_kind"] not in allowed_failure_kinds
            or returncode is not None
            or closed is not None
            or process_started is not False
        ):
            raise ValueError("pre-start Modal attempt fields do not reconcile")
    elif status == "interrupted":
        if payload["failure_kind"] != "interrupt" or returncode is not None:
            raise ValueError("interrupted Modal attempt fields do not reconcile")
        if process_started and closed is not True:
            raise ValueError("started interrupted Modal attempt was not closed")
        if not process_started and closed is not None:
            raise ValueError("pre-start interruption has process closure state")
    elif status == "cli_failed":
        allowed_failures = (
            {
                "process_launch",
                "modal_cli",
                "process_start_receipt_persistence",
            }
            if process_started
            else {"process_launch"}
        )
        if (
            payload["failure_kind"] not in allowed_failures
            or returncode is not None
            or (process_started and closed is not True)
            or (not process_started and closed is not None)
        ):
            raise ValueError("CLI-failed Modal attempt fields do not reconcile")
    elif status == "cleanup_failed":
        cleanup_kind = payload["failure_kind"]
        if cleanup_kind == "process_group_cleanup":
            reconciled = process_started is True and closed is False
        elif cleanup_kind == "python_execution_cleanup":
            reconciled = closed is None or closed is True
        elif cleanup_kind in {
            "process_group_and_python_execution_cleanup",
            "process_start_receipt_and_process_group_cleanup",
        }:
            reconciled = process_started is True and closed is False
        elif cleanup_kind == "process_start_receipt_and_python_execution_cleanup":
            reconciled = process_started is True and closed is True
        elif cleanup_kind == (
            "process_start_receipt_process_group_and_python_execution_cleanup"
        ):
            reconciled = process_started is True and closed is False
        else:
            reconciled = False
        if returncode is not None or not reconciled:
            raise ValueError(
                "cleanup-failed Modal attempt fields do not reconcile"
            )

    if (
        process_started
        and payload["local_process_start_receipt_sha256"] is None
        and payload["failure_kind"]
        not in {
            "process_start_receipt_persistence",
            "process_start_receipt_and_process_group_cleanup",
            "process_start_receipt_and_python_execution_cleanup",
            "process_start_receipt_process_group_and_python_execution_cleanup",
        }
    ):
        raise ValueError(
            "started Modal attempt lacks its durable process-start receipt"
        )
    if action in {"download", "verify"}:
        if payload["run_id"] is None or payload["verifier_run_id"] is None:
            raise ValueError("verifier attempt lacks source or verifier run ID")
        if payload["source_run_id"] is not None or harness is not None:
            raise ValueError("verifier attempt contains unrelated identity fields")
    elif action == "checkpoint-resume":
        if payload["run_id"] is None or payload["source_run_id"] is None:
            raise ValueError("resume attempt lacks source or attempt run ID")
        if payload["verifier_run_id"] is not None or harness is not None:
            raise ValueError("resume attempt contains unrelated identity fields")
    elif action == EVOLUTION_ACTION:
        if payload["run_id"] is None or harness is None:
            raise ValueError("evolution attempt lacks its run identity")
        EvolutionRunSpec.parse(harness)
        if payload["source_run_id"] is not None or payload["verifier_run_id"] is not None:
            raise ValueError("evolution attempt contains unrelated run IDs")
    elif action == "canary":
        if payload["run_id"] is None or harness is None:
            raise ValueError("single-canary attempt lacks its harness identity")
        if (
            payload["source_run_id"] is not None
            or payload["verifier_run_id"] is not None
        ):
            raise ValueError("single-canary attempt contains unrelated run IDs")
    elif action is not None:
        if any(
            payload[field] is not None
            for field in ("source_run_id", "verifier_run_id", "harness")
        ):
            raise ValueError("Modal action attempt contains unrelated identity fields")
    if action in {"download", "verify"} and payload["verifier_run_id"] is not None:
        expected_concrete = [payload["verifier_run_id"]]
    elif action == "canaries" and payload["run_id"] is not None:
        expected_concrete = [
            f"{payload['run_id']}-{_CANARY_SUFFIXES[harness_id]}"
            for harness_id in CANARY_ORDER
        ]
    elif action is not None and payload["run_id"] is not None:
        expected_concrete = [payload["run_id"]]
    else:
        expected_concrete = []
    if concrete_run_ids != expected_concrete:
        raise ValueError("attempt concrete remote run IDs differ from the action")
    return dict(payload)


def _load_action_attempts(
    root: Path,
    identity: ModalLiveCohortIdentity,
    declared_terminal_paths: list[str],
    declared_intent_paths: list[str],
    declared_aggregate_paths: list[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    attempt_directory = modal_action_attempt_directory(identity).as_posix()
    directory = _contained_path(
        root, attempt_directory, "attempt_directory", kind="directory"
    )
    entries = list(directory.iterdir())
    if any(
        item.is_symlink()
        or not item.is_file()
        or re.fullmatch(
            r"[0-9a-f]{32}(?:\.intent|\.aggregate)?\.json",
            item.name,
        )
        is None
        for item in entries
    ):
        raise ValueError("Modal action attempt directory contains an unsupported entry")
    observed_paths = sorted(f"{attempt_directory}/{item.name}" for item in entries)
    declared_paths = sorted(
        [
            *declared_terminal_paths,
            *declared_intent_paths,
            *declared_aggregate_paths,
        ]
    )
    if observed_paths != declared_paths:
        raise ValueError(
            "cohort roster does not classify every Modal action journal file"
        )

    records: list[dict[str, Any]] = []
    terminal_by_id: dict[str, dict[str, Any]] = {}
    terminal_path_by_id: dict[str, str] = {}
    for logical in declared_terminal_paths:
        path = _contained_path(root, logical, "action_attempt_receipt", kind="file")
        attempt_id = path.stem
        record = _validate_action_attempt_receipt(
            _load_object(path),
            expected_attempt_id=attempt_id,
            root=root,
        )
        if (
            record["source_tree_sha256"] != identity.source_tree_sha256
            or record["approved_image_source_sha256"]
            != identity.image_source_sha256
            or record["cohort_id"] != identity.cohort_id
        ):
            raise ValueError("action attempt differs from the selected cohort")
        if attempt_id in terminal_by_id:
            raise ValueError("Modal action attempt IDs are duplicated")
        terminal_by_id[attempt_id] = record
        terminal_path_by_id[attempt_id] = logical
        records.append(record)

    intent_by_id: dict[str, dict[str, Any]] = {}
    intent_path_by_id: dict[str, str] = {}
    for logical in declared_intent_paths:
        path = _contained_path(root, logical, "action_intent_receipt", kind="file")
        match = re.fullmatch(r"([0-9a-f]{32})\.intent\.json", path.name)
        if match is None:
            raise ValueError("Modal action intent path has an invalid filename")
        attempt_id = match.group(1)
        intent = _validate_action_intent_receipt(
            _load_object(path),
            expected_attempt_id=attempt_id,
            root=root,
        )
        if (
            intent["source_tree_sha256"] != identity.source_tree_sha256
            or intent["approved_image_source_sha256"]
            != identity.image_source_sha256
            or intent["cohort_id"] != identity.cohort_id
        ):
            raise ValueError("action intent differs from the selected cohort")
        if attempt_id in intent_by_id:
            raise ValueError("Modal action intent IDs are duplicated")
        intent_by_id[attempt_id] = intent
        intent_path_by_id[attempt_id] = logical

    intent_attempt_ids = set(intent_by_id)
    terminal_attempt_ids = set(terminal_by_id)
    if intent_attempt_ids != terminal_attempt_ids:
        raise ValueError(
            "unresolved_remote_identity: cohort action intent and terminal "
            "attempt ID sets differ "
            f"(intent_only={sorted(intent_attempt_ids - terminal_attempt_ids)}, "
            f"terminal_only={sorted(terminal_attempt_ids - intent_attempt_ids)})"
        )
    for attempt_id, intent in intent_by_id.items():
        terminal = terminal_by_id[attempt_id]
        if any(
            not exact_json_equal(intent[field], terminal[field])
            for field in _INTENT_TERMINAL_SHARED_FIELDS
        ):
            raise ValueError("Modal action intent and terminal receipt differ")
        if intent["created_at_utc"] != terminal["started_at_utc"]:
            raise ValueError(
                "Modal action intent timestamp differs from its attempt start"
            )
    for terminal in terminal_by_id.values():
        failure_kind = terminal["failure_kind"]
        if failure_kind == "action_intent_persistence":
            raise ValueError(
                "action-intent-persistence is global-rejection-only"
            )
        if failure_kind == "action_intent_persistence_uncertain":
            raise ValueError(
                "action-intent-persistence-uncertain is global-rejection-only"
            )

    aggregates: dict[str, dict[str, Any]] = {}
    aggregate_path_by_id: dict[str, str] = {}
    for logical in declared_aggregate_paths:
        path = _contained_path(
            root,
            logical,
            "provider_canary_aggregate_outcome_receipt",
            kind="file",
        )
        match = re.fullmatch(r"([0-9a-f]{32})\.aggregate\.json", path.name)
        if match is None:
            raise ValueError("provider aggregate receipt filename is invalid")
        attempt_id = match.group(1)
        terminal = terminal_by_id.get(attempt_id)
        if (
            terminal is None
            or terminal["action"] != "canaries"
            or terminal["modal_cli_process_started"] is not True
            or terminal["returncode"] not in {0, 2}
        ):
            raise ValueError("provider aggregate receipt lacks one eligible terminal")
        aggregate = validate_provider_canary_aggregate_outcome_receipt(
            _load_object(path),
            expected_attempt_id=attempt_id,
            expected_run_id_prefix=terminal["run_id"],
            expected_source_tree_sha256=identity.source_tree_sha256,
            expected_image_source_sha256=terminal[
                "approved_image_source_sha256"
            ],
            expected_cohort_id=identity.cohort_id,
        )
        expected_all_succeeded = terminal["returncode"] == 0
        if (
            aggregate["all_succeeded"] is not expected_all_succeeded
            or (expected_all_succeeded and terminal["status"] != "succeeded")
            or (not expected_all_succeeded and terminal["status"] != "failed")
        ):
            raise ValueError("provider aggregate and terminal statuses differ")
        if attempt_id in aggregates:
            raise ValueError("provider aggregate attempt IDs are duplicated")
        aggregates[attempt_id] = aggregate
        aggregate_path_by_id[attempt_id] = logical
    expected_aggregate_ids = {
        attempt_id
        for attempt_id, terminal in terminal_by_id.items()
        if terminal["action"] == "canaries"
        and terminal["modal_cli_process_started"]
        and terminal["returncode"] in {0, 2}
    }
    if set(aggregates) != expected_aggregate_ids:
        raise ValueError(
            "provider aggregate outcome receipts do not cover every completed "
            "aggregate terminal"
        )

    evidence: list[dict[str, Any]] = []
    for record in records:
        attempt_id = record["attempt_id"]
        terminal_logical = terminal_path_by_id[attempt_id]
        terminal_path = _contained_path(
            root,
            terminal_logical,
            "action_attempt_receipt",
            kind="file",
        )
        intent_logical = intent_path_by_id.get(attempt_id)
        aggregate_logical = aggregate_path_by_id.get(attempt_id)
        evidence.append(
            {
                "attempt_id": attempt_id,
                "intent": (
                    {
                        "path": intent_logical,
                        "sha256": _sha256_file(
                            _contained_path(
                                root,
                                intent_logical,
                                "action_intent_receipt",
                                kind="file",
                            )
                        ),
                        "receipt": intent_by_id[attempt_id],
                    }
                    if intent_logical is not None
                    else None
                ),
                "terminal": {
                    "path": terminal_logical,
                    "sha256": _sha256_file(terminal_path),
                    "receipt": record,
                },
                "provider_canary_aggregate_outcomes": (
                    {
                        "path": aggregate_logical,
                        "sha256": _sha256_file(
                            _contained_path(
                                root,
                                aggregate_logical,
                                "provider_canary_aggregate_outcome_receipt",
                                kind="file",
                            )
                        ),
                        "receipt": aggregates[attempt_id],
                    }
                    if aggregate_logical is not None
                    else None
                ),
            }
        )
    return records, evidence, aggregates


def _bound_file_record(root: Path, logical: str, *, field: str) -> dict[str, Any]:
    path = _contained_path(root, logical, field, kind="file")
    raw = _read_regular_file_bytes(path, maximum_bytes=_MAX_JSON_OBJECT_BYTES)
    return {
        "path": logical,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _cohort_action_journal(
    root: Path,
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    logical_root = modal_action_attempt_directory(identity).as_posix()
    directory = _contained_path(
        root,
        logical_root,
        "migration_lineage.action_attempts",
        kind="directory",
    )
    intents: list[str] = []
    terminals: list[str] = []
    aggregates: list[str] = []
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file():
            raise ValueError("migration lineage journal contains a non-file entry")
        logical = f"{logical_root}/{entry.name}"
        if re.fullmatch(r"[0-9a-f]{32}\.intent\.json", entry.name):
            intents.append(logical)
        elif re.fullmatch(r"[0-9a-f]{32}\.aggregate\.json", entry.name):
            aggregates.append(logical)
        elif re.fullmatch(r"[0-9a-f]{32}\.json", entry.name):
            terminals.append(logical)
        else:
            raise ValueError("migration lineage journal filename is unsupported")
    attempts, _evidence, _outcomes = _load_action_attempts(
        root,
        identity,
        terminals,
        intents,
        aggregates,
    )
    journal = {
        "intent_receipts": [
            _bound_file_record(root, logical, field="lineage.intent")
            for logical in intents
        ],
        "terminal_receipts": [
            _bound_file_record(root, logical, field="lineage.terminal")
            for logical in terminals
        ],
        "aggregate_receipts": [
            _bound_file_record(root, logical, field="lineage.aggregate")
            for logical in aggregates
        ],
    }
    return journal, attempts


def _discover_cohort_journal_identities(root: Path) -> list[ModalLiveCohortIdentity]:
    live_root = _contained_path(
        root,
        MODAL_LIVE_COHORT_ROOT.as_posix(),
        "modal_live_cohort_root",
        kind="directory",
    )
    identities: list[ModalLiveCohortIdentity] = []
    for source_directory in sorted(live_root.iterdir(), key=lambda item: item.name):
        if (
            source_directory.is_symlink()
            or not source_directory.is_dir()
            or re.fullmatch(r"[0-9a-f]{64}", source_directory.name) is None
        ):
            raise ValueError("Modal live cohort source namespace is not canonical")
        for image_directory in sorted(
            source_directory.iterdir(), key=lambda item: item.name
        ):
            if (
                image_directory.is_symlink()
                or not image_directory.is_dir()
                or re.fullmatch(r"[0-9a-f]{64}", image_directory.name) is None
            ):
                raise ValueError("Modal live cohort image namespace is not canonical")
            for cohort_directory in sorted(
                image_directory.iterdir(), key=lambda item: item.name
            ):
                if cohort_directory.is_symlink() or not cohort_directory.is_dir():
                    raise ValueError("Modal live cohort leaf is not a directory")
                cohort_id = validate_run_id(cohort_directory.name)
                action_directory = cohort_directory / "action_attempts"
                if not action_directory.exists():
                    continue
                if action_directory.is_symlink() or not action_directory.is_dir():
                    raise ValueError("Modal cohort action journal is not a directory")
                identities.append(
                    ModalLiveCohortIdentity(
                        source_tree_sha256=source_directory.name,
                        image_source_sha256=image_directory.name,
                        cohort_id=cohort_id,
                    )
                )
    return identities


def _scan_global_remote_run_reservations(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    logical_root = modal_remote_run_reservation_path("reservation-scan").parent
    directory = _contained_path(
        root,
        logical_root.as_posix(),
        "remote_run_reservation_root",
        kind="directory",
    )
    bindings: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
            raise ValueError("remote run reservation namespace is not canonical")
        run_id = validate_run_id(entry.stem)
        logical = (logical_root / entry.name).as_posix()
        raw = _read_regular_file_bytes(entry, maximum_bytes=_MAX_JSON_OBJECT_BYTES)
        payload = json.loads(
            _decode_utf8(raw, entry),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(payload, dict) or set(payload) != (
            _REMOTE_RUN_RESERVATION_FIELDS
        ):
            raise ValueError("global remote run reservation schema drifted")
        identity = _cohort_identity_from_payload(
            payload,
            field="global_remote_run_reservation",
        )
        if (
            payload["schema_name"] != "ModalRemoteRunReservation"
            or payload["schema_version"] != "1.2"
            or payload["remote_run_id"] != run_id
            or payload["action"] not in _ACTIONS
            or payload["modal_environment"] != MODAL_ENVIRONMENT
        ):
            raise ValueError("global remote run reservation contract drifted")
        _attempt_id(payload["owner_attempt_id"], "reservation.owner_attempt_id")
        created_at = _utc(
            payload["created_at_utc"],
            "reservation.created_at_utc",
        )
        _validate_local_containment_fields(
            payload,
            field="reservation.local_containment",
            not_after_utc=created_at,
        )
        _sha256(
            payload["launch_capability_sha256"],
            "reservation.launch_capability_sha256",
        )
        if identity not in _discover_cohort_journal_identities(root):
            raise ValueError("remote run reservation names an unlisted cohort")
        binding = {
            "path": logical,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        bindings.append(binding)
        records[logical] = {"binding": binding, "payload": payload}
    return bindings, records


def _validate_file_binding(
    root: Path,
    value: object,
    *,
    expected_path: str | None,
    field: str,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(value, dict) or set(value) != _FILE_BINDING_FIELDS:
        raise ValueError(f"{field} has an invalid exact schema")
    logical = _text(value["path"], f"{field}.path")
    if expected_path is not None and logical != expected_path:
        raise ValueError(f"{field} path is not canonical")
    path = _contained_path(root, logical, f"{field}.path", kind="file")
    raw = _read_regular_file_bytes(path, maximum_bytes=_MAX_JSON_OBJECT_BYTES)
    if (
        _sha256(value["sha256"], f"{field}.sha256")
        != hashlib.sha256(raw).hexdigest()
        or _exact_int(value["size_bytes"], f"{field}.size_bytes") != len(raw)
    ):
        raise ValueError(f"{field} bytes changed")
    return dict(value), raw


def _assert_attempts_finished_before_snapshot(
    attempts: Sequence[Mapping[str, Any]],
    snapshot_manifest: Mapping[str, Any],
    *,
    field: str,
) -> None:
    """Prove that one quiescence capture began after its frozen journal."""

    capture_started = _utc(
        snapshot_manifest["started_at_utc"],
        f"{field}.snapshot_started_at_utc",
    )
    late_attempts = sorted(
        _attempt_id(attempt["attempt_id"], f"{field}.attempt_id")
        for attempt in attempts
        if _utc(
            attempt["finished_at_utc"],
            f"{field}.finished_at_utc",
        )
        > capture_started
    )
    if late_attempts:
        raise ValueError(
            f"{field} contains terminal receipts after snapshot capture began: "
            f"{late_attempts}"
        )


def _validate_snapshot_artifact_volume(
    rows: Sequence[Mapping[str, Any]],
    *,
    captured_at: datetime,
    recorded_at: datetime,
    observed_now: datetime,
    field_prefix: str,
    timestamp_field: str,
    missing_message: str,
    duplicate_message: str,
    missing_is_incomplete: bool,
) -> None:
    """Require and time-bound one target Volume while allowing unrelated ones."""

    matching_count = 0
    for index, row in enumerate(rows):
        name = _text(row["name"], f"{field_prefix}[{index}].name")
        created_at = _raw_timestamp_utc(
            row["created_at"],
            f"{field_prefix}[{index}].created_at",
            naive_utc=False,
        )
        _text(row["created_by"], f"{field_prefix}[{index}].created_by")
        if name == VOLUME_NAME:
            matching_count += 1
            _validate_observed_timestamp_horizon(
                created_at,
                captured_at=captured_at,
                recorded_at=recorded_at,
                observed_now=observed_now,
                field=timestamp_field,
            )
    if matching_count > 1:
        raise ValueError(duplicate_message)
    if matching_count == 0:
        if missing_is_incomplete:
            raise _PriorQuarantineAccountingIncomplete(missing_message)
        raise ValueError(missing_message)


def _validate_prior_snapshot_artifact_volume(
    rows: Sequence[Mapping[str, Any]],
    *,
    captured_at: datetime,
    recorded_at: datetime,
    observed_now: datetime,
    missing_is_incomplete: bool,
) -> None:
    _validate_snapshot_artifact_volume(
        rows,
        captured_at=captured_at,
        recorded_at=recorded_at,
        observed_now=observed_now,
        field_prefix="prior_volume_list",
        timestamp_field="prior artifact Volume creation",
        missing_message="selected prior snapshot lacks the artifact Volume",
        duplicate_message="selected prior snapshot repeats the artifact Volume",
        missing_is_incomplete=missing_is_incomplete,
    )


def _validate_observed_timestamp_horizon(
    timestamp: datetime,
    *,
    captured_at: datetime,
    recorded_at: datetime,
    observed_now: datetime,
    field: str,
) -> None:
    """Bound one observation by its command, accounting, and wall-clock horizons."""

    if any(
        timestamp > upper_bound
        for upper_bound in (
            captured_at,
            recorded_at,
            observed_now + MODAL_PRICE_BASIS_FUTURE_SKEW,
        )
    ):
        raise ValueError(
            f"{field} timestamp exceeds snapshot, accounting, or clock-skew bounds"
        )


def _validate_prior_app_lifecycle_time_bounds(
    *,
    created: datetime,
    stopped: datetime,
    captured_at: datetime,
    recorded_at: datetime,
    observed_now: datetime,
) -> None:
    """Reject lifecycle timestamps outside the evidence/accounting horizon."""

    for timestamp in (created, stopped):
        _validate_observed_timestamp_horizon(
            timestamp,
            captured_at=captured_at,
            recorded_at=recorded_at,
            observed_now=observed_now,
            field="prior App lifecycle",
        )


def _prior_snapshot_captured_at(
    snapshot_manifest: Mapping[str, Any],
    snapshot_name: str,
) -> datetime:
    return _utc(
        snapshot_manifest["snapshots"][snapshot_name]["captured_at_utc"],
        f"prior_quarantine.{snapshot_name}.captured_at_utc",
    )


def _prior_billing_window(
    snapshot_manifest: Mapping[str, Any],
) -> tuple[datetime, datetime]:
    start = _utc(
        snapshot_manifest["billing_window_start_utc"],
        "prior_quarantine.billing_window_start_utc",
    )
    end = _utc(
        snapshot_manifest["billing_window_end_utc"],
        "prior_quarantine.billing_window_end_utc",
    )
    if (
        start.minute
        or start.second
        or start.microsecond
        or end.minute
        or end.second
        or end.microsecond
        or end <= start
        or end - start > MAX_MODAL_BILLING_WINDOW
    ):
        raise ValueError(
            "prior billing window is not a completed hourly interval within 31 days"
        )
    return start, end


def _validate_prior_attempt_billing_window(
    attempts: Sequence[Mapping[str, Any]],
    *,
    billing_start: datetime,
    billing_end: datetime,
) -> None:
    for attempt in attempts:
        started = _utc(attempt["started_at_utc"], "prior_attempt.started_at_utc")
        finished = _utc(attempt["finished_at_utc"], "prior_attempt.finished_at_utc")
        if not billing_start <= started <= finished <= billing_end:
            raise ValueError("prior action attempt falls outside the billing window")


def _prior_run_directory_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    captured_at: datetime,
    recorded_at: datetime,
    observed_now: datetime,
) -> dict[str, dict[str, Any]]:
    by_run_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        _filename, kind, _timestamp, _size, run_id, created_modified = (
            _parse_volume_run_directory_row(row, index)
        )
        _validate_observed_timestamp_horizon(
            created_modified,
            captured_at=captured_at,
            recorded_at=recorded_at,
            observed_now=observed_now,
            field="prior Volume /runs entry",
        )
        if kind != "dir":
            raise ValueError("prior Volume /runs entry is not a directory")
        if run_id in by_run_id:
            raise ValueError("prior snapshot repeats a Volume /runs entry")
        by_run_id[run_id] = dict(row)
    return by_run_id


def _validate_prior_billing_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    billing_start: datetime,
    billing_end: datetime,
) -> None:
    accounting_keys: set[tuple[str, str, datetime, str]] = set()
    for index, row in enumerate(rows):
        _text(
            row["description"],
            f"prior_billing[{index}].description",
            allow_empty=True,
        )
        accounting_key = _modal_billing_accounting_key(
            row,
            field=f"prior_billing[{index}]",
        )
        interval = accounting_key[2]
        if interval.minute or interval.second or interval.microsecond:
            raise ValueError("prior billing report is not hourly aligned")
        _decimal_text(row["cost"], f"prior_billing[{index}].cost")
        if accounting_key in accounting_keys:
            raise ValueError(
                "prior billing report contains a duplicate accounting row"
            )
        accounting_keys.add(accounting_key)
        if interval < billing_start or interval >= billing_end:
            raise ValueError(
                "prior billing report contains a row outside the completed query window"
            )


def _assert_attempts_contained_for_seal(
    attempts: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> None:
    """Reject started terminals without closed, durable process-start evidence."""

    provisional = sorted(
        _attempt_id(attempt["attempt_id"], f"{field}.attempt_id")
        for attempt in attempts
        if attempt["modal_cli_process_started"]
        and (
            attempt["process_group_closed"] is not True
            or attempt.get("local_process_start_receipt_sha256") is None
        )
    )
    if provisional:
        raise ValueError(
            "provisional_unsealable: started attempts lack durable process-start "
            f"evidence or proven process-group containment ({provisional})"
        )


def _derive_modal_compute_exposure(
    attempts: Sequence[Mapping[str, Any]],
    *,
    measured_by_attempt: Mapping[str, Decimal],
    unresolved_attempt_ids: set[str],
    accounting_label: str,
) -> dict[str, Any]:
    """Keep measured Modal billing distinct from authorization-based reserves."""

    known_attempt_ids = {
        _attempt_id(attempt["attempt_id"], "modal_exposure.attempt_id")
        for attempt in attempts
    }
    if set(measured_by_attempt) != known_attempt_ids:
        raise ValueError("Modal measured-billing roster is not journal-complete")
    if not unresolved_attempt_ids <= known_attempt_ids:
        raise ValueError("Modal unresolved-reserve roster invents attempts")
    measured_total = Decimal("0")
    reserve_total = Decimal("0")
    overage_total = Decimal("0")
    per_attempt: list[dict[str, Any]] = []
    breach_attempt_ids: list[str] = []
    for attempt in sorted(attempts, key=lambda item: item["attempt_id"]):
        attempt_id = attempt["attempt_id"]
        measured = measured_by_attempt[attempt_id]
        if measured < 0:
            raise ValueError("Modal measured App billing cannot be negative")
        started = attempt["modal_cli_process_started"] is True
        cap_text = attempt["modal_cost_cap_usd"]
        cap = (
            _decimal_text(cap_text, "modal_exposure.local_authorization_cap")
            if cap_text is not None
            else Decimal("0")
        )
        if started and cap_text is None:
            raise ValueError("started Modal attempt lacks its authorization cap")
        unresolved = attempt_id in unresolved_attempt_ids
        if not started:
            if measured != 0 or unresolved:
                raise ValueError(
                    "definitely-not-started attempt has measured or reserved exposure"
                )
            reserve = Decimal("0")
            reserve_basis = "definitely_not_started_zero_exposure"
        elif unresolved:
            reserve = cap
            reserve_basis = (
                "full_local_authorization_cap_reserved_for_unresolved_start; "
                "not_a_platform_hard_bound"
            )
        elif attempt["status"] != "succeeded" or measured == 0:
            reserve = max(cap - measured, Decimal("0"))
            reserve_basis = (
                "remaining_local_authorization_cap_reserved_for_failure_or_"
                "billing_lag; not_a_platform_hard_bound"
            )
        else:
            reserve = Decimal("0")
            reserve_basis = "completed_measured_execution_no_additional_reserve"
        overage = max(measured - cap, Decimal("0")) if started else Decimal("0")
        breached = overage > 0
        if breached:
            breach_attempt_ids.append(attempt_id)
        conservative = measured + reserve
        measured_total += measured
        reserve_total += reserve
        overage_total += overage
        per_attempt.append(
            {
                "attempt_id": attempt_id,
                "modal_cli_process_started": started,
                "execution_disposition": (
                    "may_have_started_unresolved_quarantined"
                    if unresolved
                    else "definitely_not_started"
                    if not started
                    else "remote_execution_bound"
                ),
                "measured_app_billing_usd": format(measured, "f"),
                "unresolved_compute_reserve_usd": format(reserve, "f"),
                "conservative_compute_exposure_usd": format(conservative, "f"),
                "local_authorization_cap_usd": (
                    format(cap, "f") if cap_text is not None else None
                ),
                "local_authorization_cap_breached": breached,
                "measured_over_local_authorization_cap_usd": format(overage, "f"),
                "reserve_basis": reserve_basis,
            }
        )
    return {
        "accounting_label": accounting_label,
        "measured_app_billing_usd": format(measured_total, "f"),
        "unresolved_compute_reserve_usd": format(reserve_total, "f"),
        "conservative_compute_exposure_usd": format(
            measured_total + reserve_total,
            "f",
        ),
        "measured_over_local_authorization_cap_usd": format(overage_total, "f"),
        "local_authorization_cap_breach_attempt_ids": breach_attempt_ids,
        "local_authorization_is_platform_hard_bound": False,
        "attempts": per_attempt,
    }


def _path_has_any_entry(root: Path, logical: str) -> bool:
    """Return true for a file, directory, or link at one canonical path."""

    path = root.resolve().joinpath(*safe_relative_path(logical).parts)
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _prior_execution_evidence_candidates(
    identity: ModalLiveCohortIdentity,
    attempt: Mapping[str, Any],
    run_id: str,
) -> tuple[str, ...]:
    candidates = [f"{_expected_download_path(run_id)}/execution_context.json"]
    if attempt["action"] in {"download", "verify"}:
        source_run_id = validate_run_id(attempt["run_id"])
        attempt_id = _attempt_id(attempt["attempt_id"], "prior.attempt_id")
        candidates.append(
            _remote_verification_logical(
                identity,
                source_run_id,
                run_id,
                attempt_id,
            )
        )
        capture_root = modal_artifact_verifier_capture_directory_path(
            identity,
            source_run_id,
            run_id,
            attempt_id,
        )
        candidates.extend(
            (capture_root / name).as_posix()
            for name in (
                "execution_context.json",
                "artifact_verification_result.json",
                "artifact_verification_failure.json",
            )
        )
    return tuple(dict.fromkeys(candidates))


def _prior_provider_evidence_candidates(run_id: str) -> tuple[str, str]:
    controller = f"{_expected_download_path(run_id)}/controller"
    return (
        f"{controller}/provider_attempts.jsonl",
        f"{controller}/provider_request_start_uncertain.json",
    )


def _validate_prior_remote_run_dispositions(
    root: Path,
    identity: ModalLiveCohortIdentity,
    attempts: Sequence[Mapping[str, Any]],
    value: object,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Validate an exhaustive, explicit remote-start disposition per reservation."""

    if not isinstance(value, list):
        raise ValueError("prior remote-run dispositions must be a sorted list")
    attempt_by_id = {
        _attempt_id(item["attempt_id"], "prior.attempt_id"): item
        for item in attempts
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    observed_keys: list[tuple[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != (
            _PRIOR_REMOTE_RUN_DISPOSITION_FIELDS
        ):
            raise ValueError("prior remote-run disposition schema drifted")
        attempt_id = _attempt_id(
            raw["attempt_id"],
            f"prior.remote_run_dispositions[{index}].attempt_id",
        )
        run_id = validate_run_id(raw["run_id"])
        attempt = attempt_by_id.get(attempt_id)
        if attempt is None or run_id not in attempt["concrete_remote_run_ids"]:
            raise ValueError("prior remote-run disposition is not journal-bound")
        execution = _text(
            raw["execution_disposition"],
            f"prior.remote_run_dispositions[{index}].execution_disposition",
        )
        provider = _text(
            raw["provider_disposition"],
            f"prior.remote_run_dispositions[{index}].provider_disposition",
        )
        snapshot = _text(
            raw["snapshot_disposition"],
            f"prior.remote_run_dispositions[{index}].snapshot_disposition",
        )
        snapshot_app_ids = _sorted_unique_text(
            raw["snapshot_app_ids"],
            f"prior.remote_run_dispositions[{index}].snapshot_app_ids",
        )
        volume_disposition = _text(
            raw["volume_disposition"],
            f"prior.remote_run_dispositions[{index}].volume_disposition",
        )
        provider_action = attempt["action"] in _PROVIDER_LAUNCH_ACTIONS
        if not attempt["modal_cli_process_started"]:
            disposition_valid = (execution, provider, snapshot) == (
                "definitely_not_started",
                (
                    "definitely_not_started"
                    if provider_action
                    else "not_applicable"
                ),
                "no_remote_resources_observed",
            )
            expected_resource_shape = not snapshot_app_ids and (
                volume_disposition == "absent"
            )
        elif execution == "remote_execution_bound":
            provider_valid = provider == (
                "not_applicable"
                if not provider_action
                else provider
            ) and (
                not provider_action
                or provider
                in {"evidence_bound", "start_unresolved_conservative"}
            )
            if attempt["status"] == "succeeded":
                disposition_valid = (
                    snapshot == "app_volume_and_billing_bound"
                    and provider_valid
                )
                expected_resource_shape = bool(snapshot_app_ids) and (
                    volume_disposition == "present_bound"
                )
            else:
                disposition_valid = (
                    snapshot == "stopped_resources_bound" and provider_valid
                )
                expected_resource_shape = bool(snapshot_app_ids) and (
                    volume_disposition in {"present_bound", "absent_after_failure"}
                )
        else:
            disposition_valid = (
                attempt["status"] != "succeeded"
                and execution == "may_have_started_unresolved_quarantined"
                and provider
                == (
                    "start_unresolved_conservative"
                    if provider_action
                    else "not_applicable"
                )
            )
            if snapshot == "no_remote_resources_observed":
                expected_resource_shape = not snapshot_app_ids and (
                    volume_disposition == "absent"
                )
            elif snapshot == "stopped_resources_bound":
                expected_resource_shape = (
                    volume_disposition
                    in {"present_bound", "absent_after_failure"}
                    and (
                        bool(snapshot_app_ids)
                        or volume_disposition == "present_bound"
                    )
                )
            else:
                expected_resource_shape = False
        if not disposition_valid or not expected_resource_shape:
            raise ValueError(
                "prior remote-run disposition disagrees with terminal start state"
            )
        if execution != "remote_execution_bound" and any(
            _path_has_any_entry(root, logical)
            for logical in _prior_execution_evidence_candidates(
                identity, attempt, run_id
            )
        ):
            raise ValueError(
                "known prior remote execution evidence was omitted from accounting"
            )
        key = (attempt_id, run_id)
        observed_keys.append(key)
        records[key] = {
            "attempt_id": attempt_id,
            "run_id": run_id,
            "execution_disposition": execution,
            "provider_disposition": provider,
            "snapshot_disposition": snapshot,
            "snapshot_app_ids": snapshot_app_ids,
            "volume_disposition": volume_disposition,
        }
    expected_keys = sorted(
        (attempt["attempt_id"], run_id)
        for attempt in attempts
        for run_id in attempt["concrete_remote_run_ids"]
    )
    if observed_keys != sorted(set(observed_keys)):
        raise ValueError("prior remote-run dispositions must be sorted and unique")
    if observed_keys != expected_keys:
        raise ValueError(
            "prior remote-run dispositions do not cover every reserved remote run"
        )
    app_ids_by_attempt: dict[str, list[str]] = {}
    for (attempt_id, _run_id), record in records.items():
        previous = app_ids_by_attempt.setdefault(
            attempt_id,
            record["snapshot_app_ids"],
        )
        if previous != record["snapshot_app_ids"]:
            raise ValueError(
                "prior aggregate child dispositions disagree on snapshot App IDs"
            )
    return records


def _provider_harness_for_run(
    attempt: Mapping[str, Any],
    run_id: str,
) -> str:
    if attempt["action"] == "canary":
        harness = attempt["harness"]
        if harness not in CANARY_ORDER or run_id != attempt["run_id"]:
            raise ValueError("provider run differs from its single-canary harness")
        return harness
    if attempt["action"] != "canaries":
        raise ValueError("provider cost accounting received a non-provider action")
    matches = [
        harness
        for harness in CANARY_ORDER
        if run_id == f"{attempt['run_id']}-{_CANARY_SUFFIXES[harness]}"
    ]
    if len(matches) != 1:
        raise ValueError("provider aggregate child does not identify one harness")
    return matches[0]


def _validate_prior_execution_context_identity(
    context: ExecutionContextV1,
    *,
    identity: ModalLiveCohortIdentity,
    attempt: Mapping[str, Any],
    run_id: str,
) -> None:
    if context.artifact_uri != _prior_execution_artifact_uri(attempt, run_id):
        raise ValueError("prior remote execution artifact URI is not canonical")
    if (
        context.execution_backend != "modal"
        or context.run_id != run_id
        or context.app_name != APP_NAME
        or context.image_source_sha256 != identity.image_source_sha256
        or context.modal_app_id is None
        or context.modal_function_id is None
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise ValueError("prior remote execution identity is incomplete")
    action = attempt["action"]
    expected_function = (
        "artifact_verify"
        if action in {"download", "verify"}
        else _ORDINARY_ACTION_FUNCTIONS[action]
        if action in _ORDINARY_ACTION_FUNCTIONS
        else f"canary_{attempt['harness']}"
        if action == "canary"
        else f"canary_{_provider_harness_for_run(attempt, run_id)}"
        if action == "canaries"
        else None
    )
    if expected_function is not None and context.function_name != expected_function:
        raise ValueError("prior remote execution function differs from its action")


def _derive_journal_provider_spend_estimate(
    root: Path,
    *,
    identity: ModalLiveCohortIdentity,
    attempts: Sequence[Mapping[str, Any]],
    remote_run_dispositions: Mapping[tuple[str, str], Mapping[str, str]],
    provider_evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    accounting_label: str,
) -> dict[str, Any]:
    """Derive a fail-closed provider bound from a complete launcher journal."""

    provider_attempts = sorted(
        (
            attempt
            for attempt in attempts
            if attempt["action"] in {"canary", "canaries"}
        ),
        key=lambda item: item["attempt_id"],
    )
    input_total = 0
    output_total = 0
    successful_count = 0
    failed_count = 0
    terminal_record_count = 0
    request_lower_bound = 0
    request_upper_bound = 0
    known_success = Decimal("0")
    failed_reserve = Decimal("0")
    uncertain_reserve = Decimal("0")
    approved_cap_total = Decimal("0")
    run_bounds: list[dict[str, Any]] = []
    launcher_bounds: list[dict[str, Any]] = []
    consumed_evidence_keys: set[tuple[str, str]] = set()
    provider_request_ids: set[str] = set()
    provider_response_ids: set[str] = set()

    for attempt in provider_attempts:
        attempt_id = _attempt_id(attempt["attempt_id"], "provider.attempt_id")
        process_started = _exact_bool(
            attempt["modal_cli_process_started"],
            "provider.modal_cli_process_started",
            expected=None,
        )
        provider_fields = (
            "provider_cost_cap_usd",
            "provider_approval_plan_path",
            "approval_plan_sha256",
            "provider_price_basis_path",
            "provider_price_basis_sha256",
        )
        populated = [attempt[field] is not None for field in provider_fields]
        if any(populated) and not all(populated):
            raise ValueError("provider launcher has partial approval artifacts")
        if attempt["provider_cost_approved"] and not all(populated):
            raise ValueError("approved provider launcher lacks approval artifacts")
        if process_started and not all(populated):
            raise ValueError("started provider launcher lacks approval artifacts")

        harness_plans: dict[str, dict[str, Any]] = {}
        request_reserves: dict[str, Decimal] = {}
        approved_cap = Decimal("0")
        source_bound = Decimal("0")
        approval_binding: dict[str, Any] | None = None
        price_binding: dict[str, Any] | None = None
        if all(populated):
            plan, plan_path = _load_provider_approval_plan(
                root,
                attempt["provider_approval_plan_path"],
                expected_approval_sha256=attempt["approval_plan_sha256"],
                expected_image_source_sha256=attempt[
                    "approved_image_source_sha256"
                ],
                expected_identity=identity,
                expected_preflight_binding=_candidate_preflight_binding(
                    attempt["predecessor_receipts"]
                ),
                expected_evolution_spec=(
                    attempt["harness"]
                    if attempt["action"] == EVOLUTION_ACTION
                    else None
                ),
            )
            price, price_path, price_sha256 = _load_price_basis(
                root,
                attempt["provider_price_basis_path"],
            )
            if price_sha256 != attempt["provider_price_basis_sha256"]:
                raise ValueError("provider launcher price-basis digest changed")
            plan_raw = _read_regular_file_bytes(
                plan_path,
                maximum_bytes=_MAX_JSON_OBJECT_BYTES,
            )
            price_raw = _read_regular_file_bytes(
                price_path,
                maximum_bytes=_MAX_JSON_OBJECT_BYTES,
            )
            approval_binding = {
                "path": attempt["provider_approval_plan_path"],
                "approval_plan_sha256": attempt["approval_plan_sha256"],
                "file_sha256": hashlib.sha256(plan_raw).hexdigest(),
                "size_bytes": len(plan_raw),
            }
            price_binding = {
                "path": attempt["provider_price_basis_path"],
                "sha256": price_sha256,
                "size_bytes": len(price_raw),
            }
            harness_plans = {item["harness"]: item for item in plan["harnesses"]}
            input_rate = _decimal_text(
                price["uncached_input_usd_per_million_tokens"],
                "provider.input_rate",
            )
            output_rate = _decimal_text(
                price["output_usd_per_million_tokens"],
                "provider.output_rate",
            )
            request_fee = _decimal_text(
                price["per_request_fee_usd"],
                "provider.request_fee",
            )
            for harness, harness_plan in harness_plans.items():
                request_reserves[harness] = (
                    Decimal(
                        harness_plan["first_opportunity"][
                            "conservative_input_token_ceiling"
                        ]
                    )
                    * input_rate
                    / Decimal(1_000_000)
                    + Decimal(
                        harness_plan["request_settings"][
                            "max_completion_tokens"
                        ]
                    )
                    * output_rate
                    / Decimal(1_000_000)
                    + request_fee
                )
            selected_harnesses = (
                list(CANARY_ORDER)
                if attempt["action"] == "canaries"
                else [attempt["harness"]]
            )
            source_bound = sum(
                (request_reserves[harness] for harness in selected_harnesses),
                Decimal("0"),
            )
            approved_cap = _decimal_text(
                attempt["provider_cost_cap_usd"],
                "provider.provider_cost_cap_usd",
            )
            if approved_cap < source_bound:
                raise ValueError(
                    "provider launcher cap is below its source-bound approval ceiling"
                )
            if attempt["provider_cost_approved"]:
                approved_cap_total += approved_cap

        launcher_known = Decimal("0")
        launcher_failed = Decimal("0")
        launcher_uncertain = Decimal("0")
        for run_id in attempt["concrete_remote_run_ids"]:
            key = (attempt_id, validate_run_id(run_id))
            disposition = remote_run_dispositions.get(key)
            if disposition is None:
                raise ValueError("provider accounting omits one reserved remote run")
            provider_disposition = disposition["provider_disposition"]
            harness = _provider_harness_for_run(attempt, run_id)
            run_known = Decimal("0")
            run_failed = Decimal("0")
            run_uncertain = Decimal("0")
            run_lower = 0
            run_upper = 0
            evidence = provider_evidence.get(key)
            records: list[ProviderAttemptRecord] = []
            if provider_disposition == "definitely_not_started":
                if process_started:
                    raise ValueError(
                        "started provider run claims definitely-not-started"
                    )
                if evidence is not None:
                    raise ValueError(
                        "definitely-not-started provider run has provider evidence"
                    )
                cost_disposition = "definitely_not_started_zero_exposure"
            elif provider_disposition == "start_unresolved_conservative":
                if not process_started or not all(populated):
                    raise ValueError(
                        "unresolved provider run lacks an approved launcher"
                    )
                if evidence is None:
                    run_uncertain = request_reserves[harness]
                    run_upper = 1
                    cost_disposition = (
                        "may_have_started_full_approved_request_reserve"
                    )
                else:
                    consumed_evidence_keys.add(key)
                    state = evidence.get("state")
                    records = evidence.get("records")
                    evidence_harness = evidence.get("harness")
                    parse_dispositions = evidence.get("parse_dispositions")
                    if (
                        state != "unbound_observed"
                        or not isinstance(records, list)
                        or len(records) > 1
                        or evidence_harness != harness
                        or not isinstance(parse_dispositions, list)
                        or not parse_dispositions
                    ):
                        raise ValueError(
                            "unbound provider evidence cost binding is invalid"
                        )
                    if not records:
                        run_uncertain = request_reserves[harness]
                        run_upper = 1
                        cost_disposition = (
                            "unbound_partial_or_uncertain_full_request_reserve"
                        )
            elif provider_disposition == "evidence_bound":
                if not process_started or not all(populated):
                    raise ValueError(
                        "bound provider evidence lacks an approved launcher"
                    )
                if not isinstance(evidence, Mapping):
                    raise ValueError("provider evidence coverage is incomplete")
                consumed_evidence_keys.add(key)
                state = evidence.get("state")
                records = evidence.get("records")
                evidence_harness = evidence.get("harness")
                if (
                    state not in {"ledger", "start_uncertain"}
                    or not isinstance(records, list)
                    or evidence_harness != harness
                ):
                    raise ValueError("provider evidence cost binding is invalid")
                if state == "start_uncertain":
                    if records:
                        raise ValueError(
                            "start-uncertain evidence contains ledger records"
                        )
                    run_uncertain = request_reserves[harness]
                    run_upper = 1
                    cost_disposition = "evidence_bound_start_uncertain_reserve"
            else:
                raise ValueError("provider remote-run disposition is unsupported")

            if records:
                if len(records) != 1 or not isinstance(
                    records[0], ProviderAttemptRecord
                ):
                    raise ValueError(
                        "provider ledger exceeds its one-attempt approval"
                    )
                record = records[0]
                plan = harness_plans[harness]
                launcher_started = _utc(
                    attempt["started_at_utc"],
                    "provider.launcher_started_at_utc",
                )
                launcher_finished = _utc(
                    attempt["finished_at_utc"],
                    "provider.launcher_finished_at_utc",
                )
                record_started = _utc(
                    record.started_at_utc,
                    "provider.record_started_at_utc",
                )
                record_ended = _utc(
                    record.ended_at_utc,
                    "provider.record_ended_at_utc",
                )
                if (
                    record.execution_backend != "modal"
                    or record.action_run_id != run_id
                    or record.harness != harness
                    or record.action != "one_opportunity_engineering_canary"
                    or record.api_endpoint != OFFICIAL_OPENAI_API_BASE
                    or record.model != TARGET_MODEL
                    or record.attempt_ordinal != 1
                    or record.generation_settings_sha256
                    != plan["generation_settings_sha256"]
                    or not (
                        launcher_started
                        <= record_started
                        <= record_ended
                        <= launcher_finished
                    )
                ):
                    raise ValueError("provider ledger differs from its approval")
                input_ceiling = plan["first_opportunity"][
                    "conservative_input_token_ceiling"
                ]
                output_ceiling = plan["request_settings"][
                    "max_completion_tokens"
                ]
                if (
                    record.input_tokens is not None
                    and record.input_tokens > input_ceiling
                ) or (
                    record.output_tokens is not None
                    and record.output_tokens > output_ceiling
                ):
                    raise ValueError(
                        "provider ledger exceeds approved token ceilings"
                    )
                record_request_ids = {
                    record.provider_request_id
                } - {None}
                record_response_ids = {
                    record.provider_response_id
                } - {None}
                if provider_request_ids.intersection(record_request_ids) or (
                    provider_response_ids.intersection(record_response_ids)
                ):
                    raise ValueError("provider request or response ID is reused")
                provider_request_ids.update(record_request_ids)
                provider_response_ids.update(record_response_ids)
                terminal_record_count += 1
                run_lower = run_upper = 1
                if record.status == "error":
                    failed_count += 1
                    run_failed = request_reserves[harness]
                    cost_disposition = (
                        "unbound_observed_failed_request_reserve"
                        if provider_disposition
                        == "start_unresolved_conservative"
                        else "evidence_bound_failed_request_reserve"
                    )
                else:
                    if (
                        record.provider_request_id is None
                        or record.provider_response_id is None
                        or record.usage_known is not True
                        or record.input_tokens is None
                        or record.output_tokens is None
                    ):
                        raise ValueError(
                            "successful provider ledger lacks exact response usage"
                        )
                    successful_count += 1
                    input_total += record.input_tokens
                    output_total += record.output_tokens
                    run_known = (
                        Decimal(record.input_tokens)
                        * input_rate
                        / Decimal(1_000_000)
                        + Decimal(record.output_tokens)
                        * output_rate
                        / Decimal(1_000_000)
                        + request_fee
                    )
                    cost_disposition = (
                        "unbound_observed_known_success_usage"
                        if provider_disposition
                        == "start_unresolved_conservative"
                        else "evidence_bound_known_success_usage"
                    )
            request_lower_bound += run_lower
            request_upper_bound += run_upper
            known_success += run_known
            failed_reserve += run_failed
            uncertain_reserve += run_uncertain
            launcher_known += run_known
            launcher_failed += run_failed
            launcher_uncertain += run_uncertain
            run_bounds.append(
                {
                    "launcher_attempt_id": attempt_id,
                    "run_id": run_id,
                    "harness": harness,
                    "cost_disposition": cost_disposition,
                    "provider_attempt_count_lower_bound": run_lower,
                    "provider_attempt_count_upper_bound": run_upper,
                    "known_success_usage_estimate_usd": format(run_known, "f"),
                    "failed_attempt_reserve_usd": format(run_failed, "f"),
                    "uncertain_request_start_reserve_usd": format(
                        run_uncertain, "f"
                    ),
                    "conservative_bound_usd": format(
                        run_known + run_failed + run_uncertain,
                        "f",
                    ),
                }
            )
        observed_bound = launcher_known + launcher_failed + launcher_uncertain
        if observed_bound > approved_cap:
            raise ValueError("provider conservative bound exceeds its launcher cap")
        launcher_bounds.append(
            {
                "launcher_attempt_id": attempt_id,
                "action": attempt["action"],
                "modal_cli_process_started": process_started,
                "remote_execution_state": attempt["remote_execution_state"],
                "provider_cost_approved": attempt["provider_cost_approved"],
                "provider_cost_cap_usd": (
                    format(approved_cap, "f") if all(populated) else None
                ),
                "source_bound_approval_ceiling_usd": format(source_bound, "f"),
                "known_success_usage_estimate_usd": format(launcher_known, "f"),
                "failed_attempt_reserve_usd": format(launcher_failed, "f"),
                "uncertain_request_start_reserve_usd": format(
                    launcher_uncertain, "f"
                ),
                "conservative_observed_bound_usd": format(observed_bound, "f"),
                "approval_plan": approval_binding,
                "price_basis": price_binding,
            }
        )

    if set(provider_evidence) != consumed_evidence_keys:
        raise ValueError("provider evidence includes omitted or invented remote runs")
    conservative_bound = known_success + failed_reserve + uncertain_reserve
    if conservative_bound > approved_cap_total:
        raise ValueError("provider conservative bound exceeds approved caps")
    return {
        "accounting_label": accounting_label,
        "provider_launcher_attempt_count": len(provider_attempts),
        "provider_terminal_attempt_record_count": terminal_record_count,
        "provider_attempt_count_lower_bound": request_lower_bound,
        "provider_attempt_count_upper_bound": request_upper_bound,
        "successful_provider_attempt_count": successful_count,
        "failed_provider_attempt_count": failed_count,
        "input_tokens": input_total,
        "output_tokens": output_total,
        "total_tokens": input_total + output_total,
        "known_success_usage_estimate_usd": format(known_success, "f"),
        "failed_attempt_reserve_usd": format(failed_reserve, "f"),
        "uncertain_request_start_reserve_usd": format(uncertain_reserve, "f"),
        "conservative_provider_spend_bound_usd": format(
            conservative_bound, "f"
        ),
        "approved_provider_cap_total_usd": format(approved_cap_total, "f"),
        "provider_request_ids": sorted(provider_request_ids),
        "provider_response_ids": sorted(provider_response_ids),
        "run_cost_dispositions": run_bounds,
        "launcher_approval_bounds": launcher_bounds,
    }


def _validate_prior_quarantine_accounting_payload(
    root: Path,
    logical: str,
    payload: object,
    accounting_raw: bytes,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("prior quarantine accounting must be a JSON object")
    raw_sha256 = hashlib.sha256(accounting_raw).hexdigest()
    if set(payload) != _PRIOR_QUARANTINE_ACCOUNTING_FIELDS:
        raise ValueError("prior quarantine accounting has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalPriorCohortQuarantineAccounting"
        or payload["schema_version"] != "1.1"
    ):
        raise ValueError("prior quarantine accounting contract drifted")
    identity = _cohort_identity_from_payload(payload, field="prior_quarantine")
    if logical != modal_prior_quarantine_accounting_path(identity).as_posix():
        raise ValueError("prior quarantine accounting path is not canonical")
    recorded_at = _utc(
        payload["recorded_at_utc"],
        "prior_quarantine.recorded_at_utc",
    )
    observed_now = datetime.now(UTC)
    if recorded_at > observed_now + MODAL_PRICE_BASIS_FUTURE_SKEW:
        raise ValueError(
            "prior quarantine accounting timestamp is too far in the future"
        )
    _exact_bool(payload["validated"], "prior_quarantine.validated")
    if payload["accepted_contexts"] != []:
        raise ValueError("prior quarantine cohort may not retain accepted contexts")
    for field in (
        "active_app_count",
        "active_container_count",
        "active_endpoint_count",
    ):
        if _exact_int(payload[field], f"prior_quarantine.{field}") != 0:
            raise ValueError("prior quarantine cohort retains an active resource")
    subtotal = _decimal_text(
        payload["app_compute_subtotal_usd"],
        "prior_quarantine.app_compute_subtotal_usd",
    )
    journal, attempts = _cohort_action_journal(root, identity)
    _assert_attempts_contained_for_seal(attempts, field="prior_quarantine")
    if not exact_json_equal(payload["action_journal"], journal):
        raise ValueError("prior quarantine action journal changed")
    reservation_bindings = sorted(
        {
            record["path"]: {
                **record,
                "size_bytes": len(
                    _read_regular_file_bytes(
                        _contained_path(
                            root,
                            record["path"],
                            "reservation",
                            kind="file",
                        ),
                        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
                    )
                ),
            }
            for attempt in attempts
            for record in attempt["remote_run_reservations"]
        }.values(),
        key=lambda item: item["path"],
    )
    if not exact_json_equal(payload["remote_run_reservations"], reservation_bindings):
        raise ValueError("prior quarantine reservation roster changed")
    expected_attempts = [
        {
            "attempt_id": attempt["attempt_id"],
            "action": attempt["action"],
            "status": attempt["status"],
            "concrete_remote_run_ids": attempt["concrete_remote_run_ids"],
            "disposition": "quarantined",
        }
        for attempt in sorted(attempts, key=lambda item: item["attempt_id"])
    ]
    if not exact_json_equal(payload["attempt_dispositions"], expected_attempts):
        raise ValueError("prior quarantine attempt dispositions are incomplete")

    attempt_by_id = {attempt["attempt_id"]: attempt for attempt in attempts}
    remote_run_dispositions = _validate_prior_remote_run_dispositions(
        root,
        identity,
        attempts,
        payload["remote_run_dispositions"],
    )
    manifest_logical = _text(
        payload["snapshot_capture_manifest_path"],
        "prior_quarantine.snapshot_capture_manifest_path",
    )
    manifest_path = _contained_path(
        root,
        manifest_logical,
        "prior_quarantine.snapshot_capture_manifest_path",
        kind="file",
    )
    manifest_raw = _read_regular_file_bytes(
        manifest_path,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        required_mode=0o600,
    )
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != _sha256(
            payload["snapshot_capture_manifest_sha256"],
            "prior_quarantine.snapshot_capture_manifest_sha256",
        )
        or len(manifest_raw)
        != _exact_int(
            payload["snapshot_capture_manifest_size_bytes"],
            "prior_quarantine.snapshot_capture_manifest_size_bytes",
        )
    ):
        raise ValueError("prior quarantine snapshot manifest bytes changed")
    raw_manifest = json.loads(
        _decode_utf8(manifest_raw, manifest_path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(raw_manifest, dict):
        raise ValueError("prior quarantine snapshot manifest is not an object")
    pseudo_roster = {
        "snapshot_capture_manifest_path": manifest_logical,
        "snapshot_capture_manifest_sha256": hashlib.sha256(
            manifest_raw
        ).hexdigest(),
        "billing_window_start_utc": raw_manifest.get(
            "billing_window_start_utc"
        ),
        "billing_window_end_utc": raw_manifest.get("billing_window_end_utc"),
        "snapshot_captured_at_utc": raw_manifest.get("finished_at_utc"),
    }
    snapshot_manifest, _manifest_path, _manifest_sha256, snapshot_rows = (
        _load_cleanup_snapshot_capture(root, pseudo_roster, identity)
    )
    app_list_captured_at = _prior_snapshot_captured_at(
        snapshot_manifest,
        "app_list",
    )
    volume_list_captured_at = _prior_snapshot_captured_at(
        snapshot_manifest,
        "volume_list",
    )
    run_directory_list_captured_at = _prior_snapshot_captured_at(
        snapshot_manifest,
        "run_directory_list",
    )
    billing_start, billing_end = _prior_billing_window(snapshot_manifest)
    _validate_prior_snapshot_artifact_volume(
        snapshot_rows["volume_list"],
        captured_at=volume_list_captured_at,
        recorded_at=recorded_at,
        observed_now=observed_now,
        missing_is_incomplete=False,
    )
    _validate_prior_billing_rows(
        snapshot_rows["billing_report"],
        billing_start=billing_start,
        billing_end=billing_end,
    )
    snapshot_finished = _utc(
        snapshot_manifest["finished_at_utc"],
        "prior_quarantine.snapshot_finished_at_utc",
    )
    if recorded_at < snapshot_finished:
        raise ValueError(
            "prior quarantine accounting predates its bound snapshot completion"
        )
    _assert_attempts_finished_before_snapshot(
        attempts,
        snapshot_manifest,
        field="prior_quarantine",
    )
    _validate_prior_attempt_billing_window(
        attempts,
        billing_start=billing_start,
        billing_end=billing_end,
    )

    remote_executions = payload["remote_executions"]
    if not isinstance(remote_executions, list):
        raise ValueError("prior quarantine remote executions must be a sorted list")
    execution_keys: list[tuple[str, str]] = []
    execution_contexts: dict[tuple[str, str], ExecutionContextV1] = {}
    for index, record in enumerate(remote_executions):
        if not isinstance(record, dict) or set(record) != (
            _PRIOR_REMOTE_EXECUTION_FIELDS
        ):
            raise ValueError("prior quarantine remote execution schema drifted")
        attempt_id = _attempt_id(
            record["attempt_id"],
            f"prior_quarantine.remote_executions[{index}].attempt_id",
        )
        run_id = validate_run_id(record["run_id"])
        attempt = attempt_by_id.get(attempt_id)
        if attempt is None or run_id not in attempt["concrete_remote_run_ids"]:
            raise ValueError("prior remote execution is not journal-bound")
        context_logical = _text(
            record["execution_context_path"],
            f"prior_quarantine.remote_executions[{index}].path",
        )
        context_path = _contained_path(
            root,
            context_logical,
            "prior_quarantine.execution_context_path",
            kind="file",
        )
        context_raw = _read_regular_file_bytes(
            context_path,
            maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        )
        if (
            hashlib.sha256(context_raw).hexdigest()
            != _sha256(
                record["execution_context_sha256"],
                "prior_quarantine.execution_context_sha256",
            )
            or len(context_raw)
            != _exact_int(
                record["execution_context_size_bytes"],
                "prior_quarantine.execution_context_size_bytes",
            )
        ):
            raise ValueError("prior remote execution context bytes changed")
        context_payload = json.loads(
            _decode_utf8(context_raw, context_path),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        try:
            context = ExecutionContextV1.from_dict(context_payload)
        except (TypeError, ValueError) as error:
            raise ValueError("prior remote execution context is invalid") from error
        _validate_prior_execution_context_identity(
            context,
            identity=identity,
            attempt=attempt,
            run_id=run_id,
        )
        key = (attempt_id, run_id)
        execution_keys.append(key)
        execution_contexts[key] = context
    if execution_keys != sorted(set(execution_keys)):
        raise ValueError("prior remote executions must be sorted and unique")
    expected_execution_keys = sorted(
        key
        for key, disposition in remote_run_dispositions.items()
        if disposition["execution_disposition"] == "remote_execution_bound"
    )
    if execution_keys != expected_execution_keys:
        raise ValueError("prior remote executions do not cover every remote run")

    app_rows = {
        _text(row["app_id"], "prior_snapshot.app_id"): row
        for row in snapshot_rows["app_list"]
    }
    if len(app_rows) != len(snapshot_rows["app_list"]):
        raise ValueError("prior snapshot repeats an App ID")
    app_to_attempt: dict[str, str] = {}
    context_apps_by_attempt: dict[str, set[str]] = {}
    for (attempt_id, _run_id), context in execution_contexts.items():
        app_id = context.modal_app_id
        assert app_id is not None
        context_apps_by_attempt.setdefault(attempt_id, set()).add(app_id)
    declared_apps_by_attempt = {
        attempt["attempt_id"]: set(
            remote_run_dispositions[
                (attempt["attempt_id"], attempt["concrete_remote_run_ids"][0])
            ]["snapshot_app_ids"]
        )
        for attempt in attempts
    }
    for attempt_id, context_app_ids in context_apps_by_attempt.items():
        if declared_apps_by_attempt[attempt_id] != context_app_ids:
            raise ValueError(
                "prior execution contexts differ from declared snapshot App IDs"
            )
    for attempt_id, app_ids in declared_apps_by_attempt.items():
        for app_id in app_ids:
            previous = app_to_attempt.setdefault(app_id, attempt_id)
            if previous != attempt_id:
                raise ValueError("prior App ID is shared across launcher attempts")
    observed_migration_app_ids: set[str] = set()
    for app_id, row in app_rows.items():
        if row["description"] != APP_NAME:
            continue
        observed_migration_app_ids.add(app_id)
        if row["state"] != "stopped" or row["tasks"] != "0":
            raise ValueError("migration App in prior snapshot is not stopped cleanly")
    expected_lifecycles: list[dict[str, str]] = []
    lifecycle_times: dict[str, tuple[datetime, datetime]] = {}
    for app_id, attempt_id in sorted(app_to_attempt.items()):
        row = app_rows.get(app_id)
        if row is None or row["state"] != "stopped" or row["tasks"] != "0":
            raise ValueError("prior migration App is not stopped cleanly")
        created = _raw_timestamp_utc(
            row["created_at"],
            "prior_snapshot.app.created_at",
            naive_utc=False,
        )
        stopped = _raw_timestamp_utc(
            row["stopped_at"],
            "prior_snapshot.app.stopped_at",
            naive_utc=False,
        )
        attempt = attempt_by_id[attempt_id]
        started = _utc(attempt["started_at_utc"], "prior_attempt.started_at_utc")
        finished = _utc(
            attempt["finished_at_utc"],
            "prior_attempt.finished_at_utc",
        )
        _validate_prior_app_lifecycle_time_bounds(
            created=created,
            stopped=stopped,
            captured_at=app_list_captured_at,
            recorded_at=recorded_at,
            observed_now=observed_now,
        )
        if not billing_start <= created <= stopped <= billing_end:
            raise ValueError("prior App lifecycle falls outside the billing window")
        if (
            stopped < created
            or created < started - MODAL_APP_LIFECYCLE_CLOCK_TOLERANCE
            or stopped > finished + MODAL_APP_LIFECYCLE_CLOCK_TOLERANCE
        ):
            raise ValueError("prior App lifecycle is not contained by its attempt")
        lifecycle_times[app_id] = (created, stopped)
        expected_lifecycles.append(
            {
                "attempt_id": attempt_id,
                "app_id": app_id,
                "created_at_utc": _utc_z(created),
                "stopped_at_utc": _utc_z(stopped),
            }
        )
    lifecycles = payload["app_lifecycles"]
    if not isinstance(lifecycles, list) or any(
        not isinstance(item, dict) or set(item) != _PRIOR_APP_LIFECYCLE_FIELDS
        for item in lifecycles
    ) or not exact_json_equal(lifecycles, expected_lifecycles):
        raise ValueError("prior App lifecycle roster changed")

    active_container_count = sum(
        row["app_id"] in app_to_attempt or row["app_name"] == APP_NAME
        for row in snapshot_rows["container_list"]
    )
    active_endpoint_count = sum(
        APP_NAME in row["name"] for row in snapshot_rows["endpoint_list"]
    )
    if active_container_count or active_endpoint_count:
        raise ValueError("prior quarantine snapshot retains an active resource")

    expected_billing: list[dict[str, Any]] = []
    observed_migration_billing_rows: dict[str, dict[str, Any]] = {}
    derived_subtotal = Decimal("0")
    for row in snapshot_rows["billing_report"]:
        app_id = _text(row["object_id"], "prior_billing.object_id")
        if (
            row["description"] == APP_NAME
            and row["environment"] == MODAL_ENVIRONMENT
        ):
            _raw_timestamp_utc(
                row["interval_start"],
                "prior_billing.interval_start",
                naive_utc=False,
            )
            _decimal_text(row["cost"], "prior_billing.cost")
            row_sha256 = canonical_sha256(row)
            previous_row = observed_migration_billing_rows.setdefault(
                row_sha256, row
            )
            if not exact_json_equal(previous_row, row):
                raise ValueError("migration billing row digest is ambiguous")
        if app_id not in app_to_attempt:
            continue
        if row["description"] != APP_NAME:
            raise ValueError("owned prior billing row has the wrong description")
        if row["environment"] != MODAL_ENVIRONMENT:
            raise ValueError("prior billing row uses the wrong environment")
        interval = _raw_timestamp_utc(
            row["interval_start"],
            "prior_billing.interval_start",
            naive_utc=False,
        )
        cost = _decimal_text(row["cost"], "prior_billing.cost")
        created, stopped = lifecycle_times[app_id]
        attempt_id = app_to_attempt[app_id]
        attempt = attempt_by_id[attempt_id]
        started = _utc(attempt["started_at_utc"], "prior_attempt.started_at_utc")
        finished = _utc(
            attempt["finished_at_utc"],
            "prior_attempt.finished_at_utc",
        )
        interval_end = interval + timedelta(hours=1)
        if cost > 0 and (
            interval > stopped
            or interval_end <= created
            or interval > finished
            or interval_end <= started
        ):
            raise ValueError("prior positive billing row is outside its lifecycle")
        derived_subtotal += cost
        expected_billing.append(
            {
                "attempt_id": attempt_id,
                "app_id": app_id,
                "row_sha256": canonical_sha256(row),
                "row": row,
            }
        )
    expected_billing.sort(
        key=lambda item: (item["attempt_id"], item["row_sha256"])
    )
    selected_billing = payload["selected_billing_rows"]
    if not isinstance(selected_billing, list) or any(
        not isinstance(item, dict) or set(item) != _PRIOR_BILLING_ROW_FIELDS
        for item in selected_billing
    ) or not exact_json_equal(selected_billing, expected_billing):
        raise ValueError("prior selected billing rows changed")
    if subtotal != derived_subtotal:
        raise ValueError("prior App compute subtotal does not reconcile")

    measured_by_attempt = {
        attempt_id: Decimal("0") for attempt_id in attempt_by_id
    }
    for record in expected_billing:
        measured_by_attempt[record["attempt_id"]] += _decimal_text(
            record["row"]["cost"],
            "prior_billing.cost",
        )
    unresolved_attempt_ids = {
        attempt_id
        for (attempt_id, _run_id), disposition in (
            remote_run_dispositions.items()
        )
        if disposition["execution_disposition"]
        == "may_have_started_unresolved_quarantined"
    }
    expected_modal_exposure = _derive_modal_compute_exposure(
        attempts,
        measured_by_attempt=measured_by_attempt,
        unresolved_attempt_ids=unresolved_attempt_ids,
        accounting_label=(
            "prior_quarantined_measured_billing_plus_unresolved_or_lagged_"
            "compute_reserve_not_a_platform_hard_bound"
        ),
    )
    if not exact_json_equal(
        payload["modal_compute_exposure"], expected_modal_exposure
    ):
        raise ValueError("prior Modal compute exposure changed")

    provider_evidence = payload["provider_attempt_evidence"]
    if not isinstance(provider_evidence, list):
        raise ValueError("prior provider evidence must be a sorted list")
    provider_keys: list[tuple[str, str]] = []
    provider_cost_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    provider_request_ids: set[str] = set()
    provider_response_ids: set[str] = set()
    bound_provider_paths: set[str] = set()
    for _index, record in enumerate(provider_evidence):
        if not isinstance(record, dict) or set(record) != (
            _PRIOR_PROVIDER_EVIDENCE_FIELDS
        ):
            raise ValueError("prior provider evidence schema drifted")
        attempt_id = _attempt_id(record["attempt_id"], "prior_provider.attempt_id")
        run_id = validate_run_id(record["run_id"])
        attempt = attempt_by_id.get(attempt_id)
        if (
            attempt is None
            or attempt["action"] not in {"canary", "canaries"}
            or run_id not in attempt["concrete_remote_run_ids"]
        ):
            raise ValueError("prior provider evidence is not journal-bound")
        context = execution_contexts[(attempt_id, run_id)]
        state = _text(record["state"], "prior_provider.state")
        evidence_logical = _text(
            record["evidence_path"],
            "prior_provider.evidence_path",
        )
        expected_parent = f"outputs/development/modal_downloads/{run_id}/controller"
        expected_name = (
            "provider_attempts.jsonl"
            if state == "ledger"
            else "provider_request_start_uncertain.json"
            if state == "start_uncertain"
            else None
        )
        if expected_name is None or evidence_logical != (
            f"{expected_parent}/{expected_name}"
        ):
            raise ValueError("prior provider evidence path is not canonical")
        evidence_path = _contained_path(
            root,
            evidence_logical,
            "prior_provider.evidence_path",
            kind="file",
        )
        if evidence_logical in bound_provider_paths:
            raise ValueError("prior provider evidence path is reused")
        bound_provider_paths.add(evidence_logical)
        evidence_raw = _read_regular_file_bytes(
            evidence_path,
            maximum_bytes=_MAX_PROVIDER_LEDGER_BYTES,
        )
        evidence_sha256 = _sha256(
            record["evidence_sha256"],
            "prior_provider.evidence_sha256",
        )
        if (
            hashlib.sha256(evidence_raw).hexdigest() != evidence_sha256
            or len(evidence_raw)
            != _exact_int(
                record["evidence_size_bytes"],
                "prior_provider.evidence_size_bytes",
            )
        ):
            raise ValueError("prior provider evidence bytes changed")
        if state == "ledger":
            records = _strict_provider_ledger(
                evidence_path,
                expected_sha256=evidence_sha256,
            )
            if any(
                item.execution_backend != "modal"
                or item.action_run_id != run_id
                or item.modal_call_id != context.modal_call_id
                or item.harness != _provider_harness_for_run(attempt, run_id)
                or item.action != "one_opportunity_engineering_canary"
                for item in records
            ) or [item.attempt_ordinal for item in records] != list(
                range(1, len(records) + 1)
            ):
                raise ValueError("prior provider ledger identity changed")
            launcher_started = _utc(
                attempt["started_at_utc"],
                "prior_provider.launcher_started_at_utc",
            )
            launcher_finished = _utc(
                attempt["finished_at_utc"],
                "prior_provider.launcher_finished_at_utc",
            )
            app_created, app_stopped = lifecycle_times[
                context.modal_app_id or ""
            ]
            for item in records:
                provider_started = _utc(
                    item.started_at_utc,
                    "prior_provider.started_at_utc",
                )
                provider_ended = _utc(
                    item.ended_at_utc,
                    "prior_provider.ended_at_utc",
                )
                if not (
                    launcher_started
                    <= provider_started
                    <= provider_ended
                    <= launcher_finished
                    and app_created
                    <= provider_started
                    <= provider_ended
                    <= app_stopped
                ):
                    raise ValueError(
                        "prior provider ledger lies outside launcher/App lifecycle"
                    )
            request_ids = sorted(
                item.provider_request_id
                for item in records
                if item.provider_request_id is not None
            )
            response_ids = sorted(
                item.provider_response_id
                for item in records
                if item.provider_response_id is not None
            )
            count = len(records)
            evidence_harness = _provider_harness_for_run(attempt, run_id)
        else:
            uncertainty = json.loads(
                _decode_utf8(evidence_raw, evidence_path),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            harness = (
                uncertainty.get("harness")
                if isinstance(uncertainty, dict)
                else None
            )
            if harness not in CANARY_ORDER:
                raise ValueError("prior provider uncertainty harness is invalid")
            _load_provider_start_uncertain_evidence(
                root,
                evidence_logical,
                evidence_sha256,
                harness=harness,
                run_id=run_id,
                expected_modal_call_id=context.modal_call_id,
            )
            request_ids = []
            response_ids = []
            count = 1
            records = []
            evidence_harness = harness
        if (
            _exact_int(record["provider_attempt_count"], "provider_attempt_count")
            != count
            or record["request_ids"] != request_ids
            or record["response_ids"] != response_ids
            or request_ids != sorted(set(request_ids))
            or response_ids != sorted(set(response_ids))
        ):
            raise ValueError("prior provider attempt accounting changed")
        if provider_request_ids.intersection(request_ids) or (
            provider_response_ids.intersection(response_ids)
        ):
            raise ValueError("prior provider IDs are duplicated")
        provider_request_ids.update(request_ids)
        provider_response_ids.update(response_ids)
        key = (attempt_id, run_id)
        provider_keys.append(key)
        provider_cost_evidence[key] = {
            "state": state,
            "records": records,
            "harness": evidence_harness,
        }
    if provider_keys != sorted(set(provider_keys)):
        raise ValueError("prior provider evidence must be sorted and unique")
    expected_provider_keys = sorted(
        key
        for key, disposition in remote_run_dispositions.items()
        if disposition["provider_disposition"] == "evidence_bound"
    )
    if provider_keys != expected_provider_keys:
        raise ValueError("prior provider evidence coverage is incomplete")

    unbound_provider_evidence = payload["unbound_provider_evidence"]
    if not isinstance(unbound_provider_evidence, list):
        raise ValueError("prior unbound provider evidence must be a sorted list")
    unbound_keys: list[tuple[str, str, str]] = []
    unbound_provider_paths: set[str] = set()
    unbound_cost_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(unbound_provider_evidence):
        if not isinstance(record, dict) or set(record) != (
            _PRIOR_UNBOUND_PROVIDER_EVIDENCE_FIELDS
        ):
            raise ValueError("prior unbound provider evidence schema drifted")
        attempt_id = _attempt_id(
            record["attempt_id"],
            f"prior_unbound_provider[{index}].attempt_id",
        )
        run_id = validate_run_id(record["run_id"])
        attempt = attempt_by_id.get(attempt_id)
        key = (attempt_id, run_id)
        disposition = remote_run_dispositions.get(key)
        if (
            attempt is None
            or attempt["action"] not in {"canary", "canaries"}
            or run_id not in attempt["concrete_remote_run_ids"]
            or disposition is None
            or disposition["provider_disposition"]
            != "start_unresolved_conservative"
        ):
            raise ValueError("unbound provider evidence is not unresolved-run-bound")
        evidence_kind = _text(
            record["evidence_kind"],
            f"prior_unbound_provider[{index}].evidence_kind",
        )
        expected_name = {
            "ledger": "provider_attempts.jsonl",
            "start_uncertain": "provider_request_start_uncertain.json",
        }.get(evidence_kind)
        evidence_logical = _text(
            record["evidence_path"],
            f"prior_unbound_provider[{index}].evidence_path",
        )
        expected_logical = (
            f"{_expected_download_path(run_id)}/controller/{expected_name}"
            if expected_name is not None
            else None
        )
        if expected_logical is None or evidence_logical != expected_logical:
            raise ValueError("unbound provider evidence path is not canonical")
        if (
            evidence_logical in unbound_provider_paths
            or evidence_logical in bound_provider_paths
        ):
            raise ValueError("provider evidence path is reused")
        unbound_provider_paths.add(evidence_logical)
        evidence_path = _contained_path(
            root,
            evidence_logical,
            "prior_unbound_provider.evidence_path",
            kind="file",
        )
        evidence_raw = _read_regular_file_bytes(
            evidence_path,
            maximum_bytes=_MAX_PROVIDER_LEDGER_BYTES,
        )
        evidence_sha256 = hashlib.sha256(evidence_raw).hexdigest()
        if (
            _sha256(
                record["evidence_sha256"],
                "prior_unbound_provider.evidence_sha256",
            )
            != evidence_sha256
            or _exact_int(
                record["evidence_size_bytes"],
                "prior_unbound_provider.evidence_size_bytes",
            )
            != len(evidence_raw)
        ):
            raise ValueError("unbound provider evidence bytes changed")

        records: list[ProviderAttemptRecord] = []
        request_ids: list[str] = []
        response_ids: list[str] = []
        lower_bound = 0
        if evidence_kind == "ledger":
            try:
                records = _lineage_provider_records(evidence_raw, evidence_path)
            except (TypeError, ValueError, json.JSONDecodeError):
                parse_disposition = "partial_unparseable"
                records = []
            else:
                if any(
                    item.execution_backend != "modal"
                    or item.action_run_id != run_id
                    or item.harness != _provider_harness_for_run(attempt, run_id)
                    or item.action != "one_opportunity_engineering_canary"
                    for item in records
                ) or [item.attempt_ordinal for item in records] != list(
                    range(1, len(records) + 1)
                ):
                    raise ValueError("unbound provider ledger identity changed")
                if len(records) > 1:
                    raise ValueError(
                        "unbound provider ledger exceeds one-request approval"
                    )
                parse_disposition = (
                    "valid_terminal_records" if records else "exact_empty"
                )
                lower_bound = len(records)
                request_ids = sorted(
                    item.provider_request_id
                    for item in records
                    if item.provider_request_id is not None
                )
                response_ids = sorted(
                    item.provider_response_id
                    for item in records
                    if item.provider_response_id is not None
                )
        else:
            try:
                _load_provider_start_uncertain_evidence(
                    root,
                    evidence_logical,
                    evidence_sha256,
                    harness=_provider_harness_for_run(attempt, run_id),
                    run_id=run_id,
                    expected_modal_call_id=None,
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                parse_disposition = "partial_unparseable"
            else:
                parse_disposition = "valid_start_uncertain"
        if (
            record["parse_disposition"] != parse_disposition
            or _exact_int(
                record["provider_attempt_count_lower_bound"],
                "prior_unbound_provider.provider_attempt_count_lower_bound",
            )
            != lower_bound
            or record["request_ids"] != request_ids
            or record["response_ids"] != response_ids
            or request_ids != sorted(set(request_ids))
            or response_ids != sorted(set(response_ids))
        ):
            raise ValueError("unbound provider evidence accounting changed")
        if provider_request_ids.intersection(request_ids) or (
            provider_response_ids.intersection(response_ids)
        ):
            raise ValueError("prior provider IDs are duplicated")
        provider_request_ids.update(request_ids)
        provider_response_ids.update(response_ids)
        aggregate = unbound_cost_evidence.setdefault(
            key,
            {
                "state": "unbound_observed",
                "records": [],
                "harness": _provider_harness_for_run(attempt, run_id),
                "parse_dispositions": [],
            },
        )
        if records:
            if aggregate["records"]:
                raise ValueError("unbound provider evidence double-counts a request")
            aggregate["records"] = records
        aggregate["parse_dispositions"].append(parse_disposition)
        unbound_keys.append((attempt_id, run_id, evidence_logical))
    if unbound_keys != sorted(set(unbound_keys)):
        raise ValueError("prior unbound provider evidence must be sorted and unique")

    observed_provider_paths: set[str] = set()
    for attempt in attempts:
        if attempt["action"] not in {"canary", "canaries"}:
            continue
        for run_id in attempt["concrete_remote_run_ids"]:
            key = (attempt["attempt_id"], run_id)
            present = {
                logical
                for logical in _prior_provider_evidence_candidates(run_id)
                if _path_has_any_entry(root, logical)
            }
            disposition = remote_run_dispositions[key]["provider_disposition"]
            if disposition == "definitely_not_started" and present:
                raise ValueError(
                    "definitely-not-started provider run has provider evidence"
                )
            observed_provider_paths.update(present)
    if observed_provider_paths != bound_provider_paths | unbound_provider_paths:
        raise ValueError("known provider evidence was omitted from accounting")
    provider_cost_evidence.update(unbound_cost_evidence)

    expected_provider_spend = _derive_journal_provider_spend_estimate(
        root,
        identity=identity,
        attempts=attempts,
        remote_run_dispositions=remote_run_dispositions,
        provider_evidence=provider_cost_evidence,
        accounting_label=(
            "prior_quarantined_known_usage_plus_failed_and_"
            "may_have_started_conservative_reserves_not_billed_cost"
        ),
    )
    if not exact_json_equal(
        payload["provider_spend_estimate"],
        expected_provider_spend,
    ):
        raise ValueError("prior provider spend estimate changed")

    run_directory_rows = _prior_run_directory_rows(
        snapshot_rows["run_directory_list"],
        captured_at=run_directory_list_captured_at,
        recorded_at=recorded_at,
        observed_now=observed_now,
    )
    _validate_owned_volume_run_start_times(
        snapshot_rows["run_directory_list"],
        attempts,
    )
    expected_volume_run_ids = {
        run_id
        for (_attempt_id_value, run_id), disposition in (
            remote_run_dispositions.items()
        )
        if disposition["volume_disposition"] == "present_bound"
    }
    all_journal_run_ids = {
        run_id
        for attempt in attempts
        for run_id in attempt["concrete_remote_run_ids"]
    }
    if set(run_directory_rows).intersection(
        all_journal_run_ids - expected_volume_run_ids
    ):
        raise ValueError(
            "prior no-resource disposition conflicts with a Volume /runs entry"
        )
    volume_dispositions = payload["volume_dispositions"]
    if not isinstance(volume_dispositions, list):
        raise ValueError("prior Volume dispositions must be a sorted list")
    volume_run_ids: list[str] = []
    artifact_paths: set[str] = set()
    succeeded_runs = {
        run_id
        for attempt in attempts
        if attempt["status"] == "succeeded"
        for run_id in attempt["concrete_remote_run_ids"]
    }
    for record in volume_dispositions:
        if not isinstance(record, dict) or set(record) != (
            _PRIOR_VOLUME_DISPOSITION_FIELDS
        ):
            raise ValueError("prior Volume disposition schema drifted")
        run_id = validate_run_id(record["run_id"])
        entry = run_directory_rows.get(run_id)
        if (
            entry is None
            or not exact_json_equal(record["entry"], entry)
            or record["entry_sha256"] != canonical_sha256(entry)
        ):
            raise ValueError("prior Volume entry changed")
        disposition = record["artifact_manifest_disposition"]
        if disposition == "bound":
            manifest_logical = _text(
                record["artifact_manifest_path"],
                "prior_volume.artifact_manifest_path",
            )
            manifest_path = _contained_path(
                root,
                manifest_logical,
                "prior_volume.artifact_manifest_path",
                kind="file",
            )
            raw = _read_regular_file_bytes(
                manifest_path,
                maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
            )
            if (
                hashlib.sha256(raw).hexdigest()
                != _sha256(
                    record["artifact_manifest_sha256"],
                    "prior_volume.artifact_manifest_sha256",
                )
                or len(raw)
                != _exact_int(
                    record["artifact_manifest_size_bytes"],
                    "prior_volume.artifact_manifest_size_bytes",
                )
            ):
                raise ValueError("prior artifact manifest bytes changed")
            raw_artifact = load_raw_artifact_manifest(manifest_path)
            if (
                raw_artifact.manifest.run_id != run_id
                or raw_artifact.manifest.image_source_sha256
                != identity.image_source_sha256
            ):
                raise ValueError("prior artifact manifest identity changed")
            if manifest_logical in artifact_paths:
                raise ValueError("prior artifact manifest path is reused")
            artifact_paths.add(manifest_logical)
        elif disposition == "unavailable_quarantined":
            if run_id in succeeded_runs or any(
                record[field] is not None
                for field in (
                    "artifact_manifest_path",
                    "artifact_manifest_sha256",
                    "artifact_manifest_size_bytes",
                )
            ):
                raise ValueError("prior missing artifact manifest is not quarantined")
        else:
            raise ValueError("prior artifact manifest disposition is unsupported")
        volume_run_ids.append(run_id)
    if (
        volume_run_ids != sorted(set(volume_run_ids))
        or set(volume_run_ids) != expected_volume_run_ids
    ):
        raise ValueError(
            "prior Volume dispositions differ from bound remote executions"
        )

    journal_price_candidates: list[
        tuple[Decimal, str, dict[str, Any], dict[str, Decimal]]
    ] = []
    seen_journal_price_bindings: set[tuple[str, str]] = set()
    for attempt in attempts:
        if (
            attempt["modal_price_basis_path"] is None
            and attempt["modal_price_basis_sha256"] is None
        ):
            continue
        price_logical = _text(
            attempt["modal_price_basis_path"],
            "prior_attempt.modal_price_basis_path",
        )
        price_sha256 = _sha256(
            attempt["modal_price_basis_sha256"],
            "prior_attempt.modal_price_basis_sha256",
        )
        if (price_logical, price_sha256) in seen_journal_price_bindings:
            continue
        seen_journal_price_bindings.add((price_logical, price_sha256))
        price_path = _contained_path(
            root,
            price_logical,
            "prior_attempt.modal_price_basis_path",
            kind="file",
        )
        price_raw = _read_regular_file_bytes(
            price_path,
            maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        )
        if hashlib.sha256(price_raw).hexdigest() != price_sha256:
            raise ValueError("prior journal Modal price-basis bytes changed")
        _price_payload, candidate_rates, _price_path = load_modal_price_basis(
            root,
            price_logical,
            expected_raw_sha256=price_sha256,
            expected_image_source_sha256=identity.image_source_sha256,
            require_freshness=False,
        )
        journal_price_candidates.append(
            (
                candidate_rates["volume"],
                price_logical,
                {
                    "path": price_logical,
                    "sha256": price_sha256,
                    "size_bytes": len(price_raw),
                },
                candidate_rates,
            )
        )
    if not journal_price_candidates:
        raise ValueError("prior journal has no Modal price-basis binding")
    highest_volume_rate = max(item[0] for item in journal_price_candidates)
    _selected_rate, _selected_path, price_binding, rates = min(
        (
            item
            for item in journal_price_candidates
            if item[0] == highest_volume_rate
        ),
        key=lambda item: item[1],
    )
    if not exact_json_equal(payload["modal_price_basis"], price_binding):
        raise ValueError(
            "prior retained-storage price basis is not the canonical "
            "highest-rate journal binding"
        )
    retained_count = len(volume_run_ids)
    bytes_per_run = MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES + MAX_ARTIFACT_MANIFEST_BYTES
    total_bytes = retained_count * bytes_per_run
    estimated_gib = Decimal(total_bytes) / Decimal(1024**3)
    estimated_monthly = estimated_gib * rates["volume"]
    expected_storage_estimate = {
        "retained_run_count": retained_count,
        "conservative_bytes_per_run": bytes_per_run,
        "conservative_total_bytes": total_bytes,
        "estimated_gib": format(estimated_gib, "f"),
        "volume_rate_usd_per_gib_month": format(rates["volume"], "f"),
        "estimated_monthly_usd": format(estimated_monthly, "f"),
        "basis": (
            "retained_run_count_times_per_run_artifact_download_and_manifest_caps; "
            "included_shared_volume_quota_not_subtracted"
        ),
    }
    estimate = payload["retained_storage_estimate"]
    if (
        not isinstance(estimate, dict)
        or set(estimate) != _RETAINED_STORAGE_ESTIMATE_FIELDS
        or not exact_json_equal(estimate, expected_storage_estimate)
    ):
        raise ValueError("prior retained-storage estimate changed")

    app_ids = set(app_to_attempt)
    call_ids = {
        context.modal_call_id for context in execution_contexts.values()
    }
    function_ids = {
        context.modal_function_id for context in execution_contexts.values()
    }
    image_ids = {context.modal_image_id for context in execution_contexts.values()}
    if len(call_ids) != len(execution_contexts):
        raise ValueError("prior remote call ID is reused")
    _global_bindings, global_reservations = _scan_global_remote_run_reservations(root)
    globally_reserved_run_ids = {
        record["payload"]["remote_run_id"]
        for record in global_reservations.values()
    }
    return payload, {
        "identity": identity,
        "raw_sha256": raw_sha256,
        "size_bytes": len(accounting_raw),
        "subtotal": subtotal,
        "attempt_ids": set(attempt_by_id),
        "run_ids": set(volume_run_ids),
        "app_ids": app_ids,
        "call_ids": call_ids,
        "function_ids": function_ids,
        "image_ids": image_ids,
        "provider_request_ids": provider_request_ids,
        "provider_response_ids": provider_response_ids,
        "provider_spend_bound": _decimal_text(
            expected_provider_spend["conservative_provider_spend_bound_usd"],
            "prior.provider_spend_bound",
        ),
        "modal_measured_billing": _decimal_text(
            expected_modal_exposure["measured_app_billing_usd"],
            "prior.modal_measured_billing",
        ),
        "modal_unresolved_reserve": _decimal_text(
            expected_modal_exposure["unresolved_compute_reserve_usd"],
            "prior.modal_unresolved_reserve",
        ),
        "modal_conservative_exposure": _decimal_text(
            expected_modal_exposure["conservative_compute_exposure_usd"],
            "prior.modal_conservative_exposure",
        ),
        "billing_row_keys": {
            record["row_sha256"] for record in expected_billing
        },
        "snapshot_finished_at": snapshot_finished,
        "observed_migration_app_ids": observed_migration_app_ids,
        "observed_migration_billing_rows": observed_migration_billing_rows,
        "observed_volume_run_ids": set(run_directory_rows).intersection(
            globally_reserved_run_ids
        ),
        "artifact_paths": artifact_paths,
    }


def _load_prior_quarantine_accounting(
    root: Path,
    logical: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _contained_path(
        root,
        logical,
        "prior_quarantine_accounting_path",
        kind="file",
    )
    accounting_raw = _read_regular_file_bytes(
        path,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        required_mode=0o600,
    )
    payload = json.loads(
        _decode_utf8(accounting_raw, path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    return _validate_prior_quarantine_accounting_payload(
        root,
        logical,
        payload,
        accounting_raw,
    )


def _exclusive_json_object_bytes(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    frozen = json_value(dict(value))
    if not isinstance(frozen, dict):
        raise TypeError("create-only JSON payload must be one object")
    encoded = (
        json.dumps(
            frozen,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_JSON_OBJECT_BYTES:
        raise ValueError("create-only JSON payload exceeds its size limit")
    return frozen, encoded


def _load_operator_json_input(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("operator input path must be lexically absolute")
    raw = _read_regular_file_bytes(
        candidate,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        required_mode=0o600,
    )
    payload = json.loads(
        _decode_utf8(raw, candidate),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError("operator input must contain one JSON object")
    return payload


def _scan_resolved_modal_global_action_journal(
    lock_descriptor: int,
) -> ModalGlobalJournalScan:
    """Return one complete, resolved journal scan under the caller's lock."""

    assert_modal_action_lock_identity(lock_descriptor)
    scan = scan_modal_global_action_journal(lock_descriptor=lock_descriptor)
    require_modal_global_action_journal_resolved(scan)
    assert_modal_action_lock_identity(lock_descriptor)
    return scan


def _lexically_absolute_operator_path(
    value: str | Path,
    *,
    field: str,
) -> Path:
    raw = os.fspath(value)
    candidate = Path(raw)
    if (
        not candidate.is_absolute()
        or os.path.normpath(raw) != raw
        or any(component in {"", ".", ".."} for component in candidate.parts[1:])
    ):
        raise ValueError(f"{field} must be a normalized, lexically absolute path")
    return candidate


def _prior_accounting_operator_output_path(
    root: Path,
    value: str | Path,
    *,
    field: str,
) -> Path:
    """Keep noncanonical operator material outside the live receipt tree."""

    candidate = _lexically_absolute_operator_path(value, field=field)
    project_root = Path(os.path.abspath(os.fspath(root)))
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            f"{field} must be inside the authenticated project root"
        ) from error
    safe = safe_relative_path(relative.as_posix())

    def key(component: str) -> str:
        return unicodedata.normalize("NFC", component).casefold()

    observed = tuple(key(component) for component in safe.parts)
    reserved = tuple(key(component) for component in MODAL_LIVE_COHORT_ROOT.parts)
    if observed[: len(reserved)] == reserved:
        raise ValueError(
            f"{field} may not enter the canonical live-cohort receipt namespace"
        )
    return candidate


def _absolute_project_file_logical(
    root: Path,
    value: str | Path,
    *,
    field: str,
) -> str:
    candidate = _lexically_absolute_operator_path(value, field=field)
    project_root = Path(os.path.abspath(os.fspath(root)))
    try:
        relative = candidate.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            f"{field} must be inside the authenticated project root"
        ) from error
    logical = safe_relative_path(relative.as_posix()).as_posix()
    contained = _contained_path(root, logical, field, kind="file")
    if contained != candidate:
        raise ValueError(f"{field} must name its exact canonical project path")
    return logical


def _prior_accounting_request_identity(
    payload: object,
) -> tuple[dict[str, Any], ModalLiveCohortIdentity]:
    if not isinstance(payload, Mapping):
        raise TypeError("prior quarantine accounting request must be a mapping")
    frozen = json_value(dict(payload))
    if not isinstance(frozen, dict):  # pragma: no cover - json_value guard
        raise TypeError("prior quarantine accounting request must be one object")
    if set(frozen) != _PRIOR_QUARANTINE_ACCOUNTING_REQUEST_FIELDS:
        raise ValueError(
            "prior quarantine accounting request has an invalid exact schema"
        )
    if (
        frozen["schema_name"]
        != "ModalPriorCohortQuarantineAccountingRequest"
        or frozen["schema_version"] != "1.0"
    ):
        raise ValueError("prior quarantine accounting request contract drifted")
    identity = _cohort_identity_from_payload(
        frozen,
        field="prior_quarantine_request",
    )
    _utc(
        frozen["recorded_at_utc"],
        "prior_quarantine_request.recorded_at_utc",
    )
    binding = frozen["snapshot_capture_manifest"]
    if not isinstance(binding, dict) or set(binding) != _FILE_BINDING_FIELDS:
        raise ValueError(
            "prior quarantine accounting request snapshot binding has an "
            "invalid exact schema"
        )
    _text(binding["path"], "prior_quarantine_request.snapshot.path")
    _sha256(binding["sha256"], "prior_quarantine_request.snapshot.sha256")
    _exact_int(
        binding["size_bytes"],
        "prior_quarantine_request.snapshot.size_bytes",
    )
    return frozen, identity


def _load_prior_accounting_selected_snapshot(
    root: Path,
    request: Mapping[str, Any],
    identity: ModalLiveCohortIdentity,
) -> tuple[
    dict[str, Any],
    Path,
    str,
    dict[str, list[dict[str, Any]]],
]:
    binding = request["snapshot_capture_manifest"]
    assert isinstance(binding, dict)
    logical = _text(binding["path"], "prior_quarantine_request.snapshot.path")
    manifest_path = _contained_path(
        root,
        logical,
        "prior_quarantine_request.snapshot.path",
        kind="file",
    )
    manifest_raw = _read_regular_file_bytes(
        manifest_path,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        required_mode=0o600,
    )
    if (
        hashlib.sha256(manifest_raw).hexdigest()
        != _sha256(
            binding["sha256"],
            "prior_quarantine_request.snapshot.sha256",
        )
        or len(manifest_raw)
        != _exact_int(
            binding["size_bytes"],
            "prior_quarantine_request.snapshot.size_bytes",
        )
    ):
        raise ValueError("selected cleanup snapshot binding changed")
    raw_manifest = json.loads(
        _decode_utf8(manifest_raw, manifest_path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(raw_manifest, dict):
        raise ValueError("selected cleanup snapshot manifest is not an object")
    pseudo_roster = {
        "snapshot_capture_manifest_path": logical,
        "snapshot_capture_manifest_sha256": hashlib.sha256(
            manifest_raw
        ).hexdigest(),
        "billing_window_start_utc": raw_manifest.get(
            "billing_window_start_utc"
        ),
        "billing_window_end_utc": raw_manifest.get("billing_window_end_utc"),
        "snapshot_captured_at_utc": raw_manifest.get("finished_at_utc"),
    }
    manifest, loaded_path, digest, rows = _load_cleanup_snapshot_capture(
        root,
        pseudo_roster,
        identity,
    )
    recorded_at = _utc(
        request["recorded_at_utc"],
        "prior_quarantine_request.recorded_at_utc",
    )
    snapshot_finished = _utc(
        manifest["finished_at_utc"],
        "prior_quarantine_request.snapshot.finished_at_utc",
    )
    if recorded_at < snapshot_finished:
        raise ValueError(
            "prior quarantine accounting request predates snapshot completion"
        )
    if recorded_at > datetime.now(UTC) + MODAL_PRICE_BASIS_FUTURE_SKEW:
        raise ValueError(
            "prior quarantine accounting request timestamp is too far in the future"
        )
    return manifest, loaded_path, digest, rows


def _prior_execution_context_paths(
    identity: ModalLiveCohortIdentity,
    attempt: Mapping[str, Any],
    run_id: str,
) -> tuple[str, ...]:
    paths = [f"{_expected_download_path(run_id)}/execution_context.json"]
    if attempt["action"] in {"download", "verify"}:
        paths.append(
            (
                modal_artifact_verifier_capture_directory_path(
                    identity,
                    validate_run_id(attempt["run_id"]),
                    run_id,
                    _attempt_id(attempt["attempt_id"], "prior.attempt_id"),
                )
                / "execution_context.json"
            ).as_posix()
        )
    return tuple(sorted(set(paths)))


def _prior_execution_artifact_uri(
    attempt: Mapping[str, Any],
    run_id: str,
) -> str:
    artifact_run_id = (
        validate_run_id(attempt["run_id"])
        if attempt["action"] in {"download", "verify"}
        else run_id
    )
    return volume_artifact_uri(artifact_run_id)


def _load_prior_execution_context(
    root: Path,
    identity: ModalLiveCohortIdentity,
    attempt: Mapping[str, Any],
    run_id: str,
) -> tuple[dict[str, Any], ExecutionContextV1] | None:
    observed = [
        logical
        for logical in _prior_execution_context_paths(identity, attempt, run_id)
        if _path_has_any_entry(root, logical)
    ]
    if len(observed) > 1:
        raise ValueError("prior remote execution context is ambiguous")
    if not observed:
        return None
    logical = observed[0]
    path = _contained_path(root, logical, "prior execution context", kind="file")
    raw = _read_regular_file_bytes(
        path,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
    )
    payload = json.loads(
        _decode_utf8(raw, path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    try:
        context = ExecutionContextV1.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("prior remote execution context is malformed") from error
    _validate_prior_execution_context_identity(
        context,
        identity=identity,
        attempt=attempt,
        run_id=run_id,
    )
    return (
        {
            "attempt_id": attempt["attempt_id"],
            "run_id": run_id,
            "execution_context_path": logical,
            "execution_context_sha256": hashlib.sha256(raw).hexdigest(),
            "execution_context_size_bytes": len(raw),
        },
        context,
    )


def _prior_provider_evidence_record(
    root: Path,
    *,
    attempt: Mapping[str, Any],
    run_id: str,
    context: ExecutionContextV1 | None,
    bound: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    observed = [
        logical
        for logical in _prior_provider_evidence_candidates(run_id)
        if _path_has_any_entry(root, logical)
    ]
    if len(observed) > 1:
        raise ValueError("prior provider evidence is ambiguous")
    if not observed:
        return None
    logical = observed[0]
    path = _contained_path(root, logical, "prior provider evidence", kind="file")
    raw = _read_regular_file_bytes(
        path,
        maximum_bytes=_MAX_PROVIDER_LEDGER_BYTES,
    )
    digest = hashlib.sha256(raw).hexdigest()
    harness = _provider_harness_for_run(attempt, run_id)
    if logical.endswith("provider_attempts.jsonl"):
        records = _lineage_provider_records(raw, path)
        if any(
            record.execution_backend != "modal"
            or record.action_run_id != run_id
            or record.harness != harness
            or record.action != "one_opportunity_engineering_canary"
            or (
                context is not None
                and record.modal_call_id != context.modal_call_id
            )
            for record in records
        ) or [record.attempt_ordinal for record in records] != list(
            range(1, len(records) + 1)
        ):
            raise ValueError("prior provider ledger identity is malformed")
        if not bound and len(records) > 1:
            raise ValueError(
                "unbound prior provider ledger exceeds one-request approval"
            )
        request_ids = sorted(
            record.provider_request_id
            for record in records
            if record.provider_request_id is not None
        )
        response_ids = sorted(
            record.provider_response_id
            for record in records
            if record.provider_response_id is not None
        )
        if bound:
            public = {
                "attempt_id": attempt["attempt_id"],
                "run_id": run_id,
                "state": "ledger",
                "evidence_path": logical,
                "evidence_sha256": digest,
                "evidence_size_bytes": len(raw),
                "provider_attempt_count": len(records),
                "request_ids": request_ids,
                "response_ids": response_ids,
            }
            cost = {"state": "ledger", "records": records, "harness": harness}
        else:
            public = {
                "attempt_id": attempt["attempt_id"],
                "run_id": run_id,
                "evidence_kind": "ledger",
                "evidence_path": logical,
                "evidence_sha256": digest,
                "evidence_size_bytes": len(raw),
                "parse_disposition": (
                    "valid_terminal_records" if records else "exact_empty"
                ),
                "provider_attempt_count_lower_bound": len(records),
                "request_ids": request_ids,
                "response_ids": response_ids,
            }
            cost = {
                "state": "unbound_observed",
                "records": records,
                "harness": harness,
                "parse_dispositions": [public["parse_disposition"]],
            }
        return public, cost

    _load_provider_start_uncertain_evidence(
        root,
        logical,
        digest,
        harness=harness,
        run_id=run_id,
        expected_modal_call_id=(context.modal_call_id if bound and context else None),
    )
    if bound:
        public = {
            "attempt_id": attempt["attempt_id"],
            "run_id": run_id,
            "state": "start_uncertain",
            "evidence_path": logical,
            "evidence_sha256": digest,
            "evidence_size_bytes": len(raw),
            "provider_attempt_count": 1,
            "request_ids": [],
            "response_ids": [],
        }
        cost = {"state": "start_uncertain", "records": [], "harness": harness}
    else:
        public = {
            "attempt_id": attempt["attempt_id"],
            "run_id": run_id,
            "evidence_kind": "start_uncertain",
            "evidence_path": logical,
            "evidence_sha256": digest,
            "evidence_size_bytes": len(raw),
            "parse_disposition": "valid_start_uncertain",
            "provider_attempt_count_lower_bound": 0,
            "request_ids": [],
            "response_ids": [],
        }
        cost = {
            "state": "unbound_observed",
            "records": [],
            "harness": harness,
            "parse_dispositions": ["valid_start_uncertain"],
        }
    return public, cost


def _prior_artifact_manifest_candidates(
    identity: ModalLiveCohortIdentity,
    attempt: Mapping[str, Any],
    run_id: str,
) -> tuple[str, ...]:
    roots = [PurePosixPath(_expected_download_path(run_id))]
    if attempt["action"] in {"download", "verify"}:
        roots.append(
            modal_artifact_verifier_capture_directory_path(
                identity,
                validate_run_id(attempt["run_id"]),
                run_id,
                _attempt_id(attempt["attempt_id"], "prior.attempt_id"),
            )
        )
    return tuple(
        sorted(
            {
                (directory / filename).as_posix()
                for directory in roots
                for filename in ARTIFACT_MANIFEST_FILENAMES
            }
        )
    )


def _prior_volume_disposition(
    root: Path,
    *,
    identity: ModalLiveCohortIdentity,
    attempt: Mapping[str, Any],
    run_id: str,
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    observed = [
        logical
        for logical in _prior_artifact_manifest_candidates(
            identity,
            attempt,
            run_id,
        )
        if _path_has_any_entry(root, logical)
    ]
    if len(observed) > 1:
        raise ValueError("prior artifact manifest selection is ambiguous")
    base: dict[str, Any] = {
        "run_id": run_id,
        "entry_sha256": canonical_sha256(entry),
        "entry": dict(entry),
    }
    if not observed:
        if attempt["status"] == "succeeded":
            raise _PriorQuarantineAccountingIncomplete(
                f"successful remote run {run_id} lacks its artifact manifest"
            )
        return {
            **base,
            "artifact_manifest_disposition": "unavailable_quarantined",
            "artifact_manifest_path": None,
            "artifact_manifest_sha256": None,
            "artifact_manifest_size_bytes": None,
        }
    logical = observed[0]
    path = _contained_path(root, logical, "prior artifact manifest", kind="file")
    raw = _read_regular_file_bytes(
        path,
        maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
    )
    raw_artifact = load_raw_artifact_manifest(path)
    if (
        raw_artifact.manifest.run_id != run_id
        or raw_artifact.manifest.image_source_sha256
        != identity.image_source_sha256
    ):
        raise ValueError("prior artifact manifest identity is malformed")
    return {
        **base,
        "artifact_manifest_disposition": "bound",
        "artifact_manifest_path": logical,
        "artifact_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "artifact_manifest_size_bytes": len(raw),
    }


def _derive_prior_quarantine_accounting_candidate(
    root: Path,
    request_payload: Mapping[str, Any],
) -> dict[str, Any]:
    request, identity = _prior_accounting_request_identity(request_payload)
    snapshot_manifest, _snapshot_path, snapshot_sha256, snapshot_rows = (
        _load_prior_accounting_selected_snapshot(root, request, identity)
    )
    recorded_at = _utc(
        request["recorded_at_utc"],
        "prior_quarantine.recorded_at_utc",
    )
    observed_now = datetime.now(UTC)
    app_list_captured_at = _prior_snapshot_captured_at(
        snapshot_manifest,
        "app_list",
    )
    volume_list_captured_at = _prior_snapshot_captured_at(
        snapshot_manifest,
        "volume_list",
    )
    run_directory_list_captured_at = _prior_snapshot_captured_at(
        snapshot_manifest,
        "run_directory_list",
    )
    billing_start, billing_end = _prior_billing_window(snapshot_manifest)
    _validate_prior_snapshot_artifact_volume(
        snapshot_rows["volume_list"],
        captured_at=volume_list_captured_at,
        recorded_at=recorded_at,
        observed_now=observed_now,
        missing_is_incomplete=True,
    )
    _validate_prior_billing_rows(
        snapshot_rows["billing_report"],
        billing_start=billing_start,
        billing_end=billing_end,
    )
    journal, attempts = _cohort_action_journal(root, identity)
    _assert_attempts_contained_for_seal(attempts, field="prior_quarantine")
    _assert_attempts_finished_before_snapshot(
        attempts,
        snapshot_manifest,
        field="prior_quarantine",
    )
    _validate_prior_attempt_billing_window(
        attempts,
        billing_start=billing_start,
        billing_end=billing_end,
    )
    attempt_by_id = {attempt["attempt_id"]: attempt for attempt in attempts}

    app_rows: dict[str, dict[str, Any]] = {}
    for row in snapshot_rows["app_list"]:
        app_id = _text(row["app_id"], "prior_snapshot.app_id")
        if app_id in app_rows:
            raise ValueError("prior snapshot repeats an App ID")
        app_rows[app_id] = row
    migration_apps = {
        app_id: row
        for app_id, row in app_rows.items()
        if row["description"] == APP_NAME
    }
    if any(
        row["state"] != "stopped" or row["tasks"] != "0"
        for row in migration_apps.values()
    ):
        raise ValueError("selected prior snapshot retains an active migration App")

    run_directory_rows = _prior_run_directory_rows(
        snapshot_rows["run_directory_list"],
        captured_at=run_directory_list_captured_at,
        recorded_at=recorded_at,
        observed_now=observed_now,
    )
    _validate_owned_volume_run_start_times(
        snapshot_rows["run_directory_list"],
        attempts,
    )

    execution_records: list[dict[str, Any]] = []
    execution_contexts: dict[tuple[str, str], ExecutionContextV1] = {}
    contexts_by_attempt: dict[str, set[str]] = {}
    provider_records: list[dict[str, Any]] = []
    unbound_provider_records: list[dict[str, Any]] = []
    provider_cost_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    provider_state: dict[tuple[str, str], str] = {}
    blockers: list[str] = []

    for attempt in attempts:
        attempt_id = attempt["attempt_id"]
        for raw_run_id in attempt["concrete_remote_run_ids"]:
            run_id = validate_run_id(raw_run_id)
            key = (attempt_id, run_id)
            context_result = _load_prior_execution_context(
                root,
                identity,
                attempt,
                run_id,
            )
            if not attempt["modal_cli_process_started"]:
                if context_result is not None:
                    raise ValueError(
                        "definitely-unstarted attempt has remote execution evidence"
                    )
            elif context_result is not None:
                record, context = context_result
                execution_records.append(record)
                execution_contexts[key] = context
                assert context.modal_app_id is not None
                contexts_by_attempt.setdefault(attempt_id, set()).add(
                    context.modal_app_id
                )
            else:
                non_context_evidence = [
                    logical
                    for logical in _prior_execution_evidence_candidates(
                        identity,
                        attempt,
                        run_id,
                    )
                    if logical
                    not in _prior_execution_context_paths(
                        identity,
                        attempt,
                        run_id,
                    )
                    and _path_has_any_entry(root, logical)
                ]
                if non_context_evidence:
                    blockers.append(
                        f"remote run {run_id} has execution evidence but lacks "
                        "one canonical execution context"
                    )
                elif attempt["status"] == "succeeded":
                    blockers.append(
                        f"successful remote run {run_id} lacks its execution context"
                    )

            provider_paths = [
                logical
                for logical in _prior_provider_evidence_candidates(run_id)
                if _path_has_any_entry(root, logical)
            ]
            if attempt["action"] not in {"canary", "canaries"}:
                if provider_paths:
                    raise ValueError(
                        "non-provider remote run has provider request evidence"
                    )
                provider_state[key] = "not_applicable"
                continue
            if not attempt["modal_cli_process_started"]:
                if provider_paths:
                    raise ValueError(
                        "definitely-unstarted provider run has provider evidence"
                    )
                provider_state[key] = "definitely_not_started"
                continue
            bound = context_result is not None
            provider = _prior_provider_evidence_record(
                root,
                attempt=attempt,
                run_id=run_id,
                context=(context_result[1] if context_result is not None else None),
                bound=bound,
            )
            if provider is None:
                provider_state[key] = "start_unresolved_conservative"
            else:
                public, cost = provider
                provider_cost_evidence[key] = cost
                if bound:
                    provider_state[key] = "evidence_bound"
                    provider_records.append(public)
                else:
                    provider_state[key] = "start_unresolved_conservative"
                    unbound_provider_records.append(public)

    app_to_attempt: dict[str, str] = {}
    app_lifecycles: list[dict[str, str]] = []
    lifecycle_times: dict[str, tuple[datetime, datetime]] = {}
    for attempt_id, app_ids in sorted(contexts_by_attempt.items()):
        attempt = attempt_by_id[attempt_id]
        for app_id in sorted(app_ids):
            previous = app_to_attempt.setdefault(app_id, attempt_id)
            if previous != attempt_id:
                raise ValueError("prior execution App is shared across attempts")
            row = app_rows.get(app_id)
            if row is None:
                blockers.append(
                    f"execution App {app_id} is absent from the selected snapshot"
                )
                continue
            if row["state"] != "stopped" or row["tasks"] != "0":
                raise ValueError("prior execution App did not stop cleanly")
            created = _raw_timestamp_utc(
                row["created_at"],
                "prior_snapshot.app.created_at",
                naive_utc=False,
            )
            stopped = _raw_timestamp_utc(
                row["stopped_at"],
                "prior_snapshot.app.stopped_at",
                naive_utc=False,
            )
            started = _utc(attempt["started_at_utc"], "prior_attempt.started_at")
            finished = _utc(
                attempt["finished_at_utc"],
                "prior_attempt.finished_at",
            )
            _validate_prior_app_lifecycle_time_bounds(
                created=created,
                stopped=stopped,
                captured_at=app_list_captured_at,
                recorded_at=recorded_at,
                observed_now=observed_now,
            )
            if not billing_start <= created <= stopped <= billing_end:
                raise ValueError("prior App lifecycle falls outside the billing window")
            if (
                stopped < created
                or created < started - MODAL_APP_LIFECYCLE_CLOCK_TOLERANCE
                or stopped > finished + MODAL_APP_LIFECYCLE_CLOCK_TOLERANCE
            ):
                raise ValueError("prior App lifecycle is not contained by its attempt")
            lifecycle_times[app_id] = (created, stopped)
            app_lifecycles.append(
                {
                    "attempt_id": attempt_id,
                    "app_id": app_id,
                    "created_at_utc": _utc_z(created),
                    "stopped_at_utc": _utc_z(stopped),
                }
            )
    app_lifecycles.sort(key=lambda item: item["app_id"])

    cohort_app_ids = set(app_to_attempt)
    for app_id in sorted(cohort_app_ids - set(migration_apps)):
        blockers.append(
            f"cohort execution App {app_id} is not identified as {APP_NAME}"
        )

    if blockers:
        raise _PriorQuarantineAccountingIncomplete(*sorted(set(blockers)))

    remote_dispositions: list[dict[str, Any]] = []
    unresolved_attempt_ids: set[str] = set()
    volume_dispositions: list[dict[str, Any]] = []
    for attempt in attempts:
        attempt_id = attempt["attempt_id"]
        snapshot_app_ids = sorted(contexts_by_attempt.get(attempt_id, set()))
        for raw_run_id in attempt["concrete_remote_run_ids"]:
            run_id = validate_run_id(raw_run_id)
            key = (attempt_id, run_id)
            context_bound = key in execution_contexts
            volume_entry = run_directory_rows.get(run_id)
            if not attempt["modal_cli_process_started"]:
                if volume_entry is not None or snapshot_app_ids:
                    raise ValueError(
                        "definitely-unstarted attempt has snapshot resources"
                    )
                execution_disposition = "definitely_not_started"
                snapshot_disposition = "no_remote_resources_observed"
                volume_disposition = "absent"
            elif context_bound:
                execution_disposition = "remote_execution_bound"
                if attempt["status"] == "succeeded":
                    if volume_entry is None:
                        raise _PriorQuarantineAccountingIncomplete(
                            f"successful remote run {run_id} is absent from "
                            "Volume snapshot"
                        )
                    snapshot_disposition = "app_volume_and_billing_bound"
                    volume_disposition = "present_bound"
                else:
                    snapshot_disposition = "stopped_resources_bound"
                    volume_disposition = (
                        "present_bound"
                        if volume_entry is not None
                        else "absent_after_failure"
                    )
            else:
                execution_disposition = (
                    "may_have_started_unresolved_quarantined"
                )
                unresolved_attempt_ids.add(attempt_id)
                if volume_entry is None and not snapshot_app_ids:
                    snapshot_disposition = "no_remote_resources_observed"
                    volume_disposition = "absent"
                else:
                    snapshot_disposition = "stopped_resources_bound"
                    volume_disposition = (
                        "present_bound"
                        if volume_entry is not None
                        else "absent_after_failure"
                    )
            remote_dispositions.append(
                {
                    "attempt_id": attempt_id,
                    "run_id": run_id,
                    "execution_disposition": execution_disposition,
                    "provider_disposition": provider_state[key],
                    "snapshot_disposition": snapshot_disposition,
                    "snapshot_app_ids": snapshot_app_ids,
                    "volume_disposition": volume_disposition,
                }
            )
            if volume_entry is not None:
                volume_dispositions.append(
                    _prior_volume_disposition(
                        root,
                        identity=identity,
                        attempt=attempt,
                        run_id=run_id,
                        entry=volume_entry,
                    )
                )

    selected_billing: list[dict[str, Any]] = []
    measured_by_attempt = {
        attempt_id: Decimal("0") for attempt_id in attempt_by_id
    }
    observed_billing_digests: set[str] = set()
    for row in snapshot_rows["billing_report"]:
        app_id = _text(row["object_id"], "prior_billing.object_id")
        attempt_id = app_to_attempt.get(app_id)
        if attempt_id is None:
            continue
        if row["description"] != APP_NAME:
            raise ValueError("owned prior billing row has the wrong description")
        if row["environment"] != MODAL_ENVIRONMENT:
            raise ValueError("prior billing row uses the wrong environment")
        interval = _raw_timestamp_utc(
            row["interval_start"],
            "prior_billing.interval_start",
            naive_utc=False,
        )
        cost = _decimal_text(row["cost"], "prior_billing.cost")
        created, stopped = lifecycle_times[app_id]
        attempt = attempt_by_id[attempt_id]
        started = _utc(attempt["started_at_utc"], "prior_attempt.started_at")
        finished = _utc(attempt["finished_at_utc"], "prior_attempt.finished_at")
        interval_end = interval + timedelta(hours=1)
        if cost > 0 and (
            interval > stopped
            or interval_end <= created
            or interval > finished
            or interval_end <= started
        ):
            raise ValueError("prior positive billing row is outside its lifecycle")
        row_sha256 = canonical_sha256(row)
        if row_sha256 in observed_billing_digests:
            raise ValueError("prior selected billing row is ambiguous")
        observed_billing_digests.add(row_sha256)
        measured_by_attempt[attempt_id] += cost
        selected_billing.append(
            {
                "attempt_id": attempt_id,
                "app_id": app_id,
                "row_sha256": row_sha256,
                "row": row,
            }
        )
    selected_billing.sort(key=lambda item: (item["attempt_id"], item["row_sha256"]))

    if any(
        row["app_id"] in app_to_attempt or row["app_name"] == APP_NAME
        for row in snapshot_rows["container_list"]
    ):
        raise ValueError("selected prior snapshot retains an active container")
    if any(APP_NAME in row["name"] for row in snapshot_rows["endpoint_list"]):
        raise ValueError("selected prior snapshot retains an active endpoint")

    reservation_bindings = sorted(
        {
            record["path"]: {
                **record,
                "size_bytes": len(
                    _read_regular_file_bytes(
                        _contained_path(
                            root,
                            record["path"],
                            "prior reservation",
                            kind="file",
                        ),
                        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
                    )
                ),
            }
            for attempt in attempts
            for record in attempt["remote_run_reservations"]
        }.values(),
        key=lambda item: item["path"],
    )
    _global_bindings, global_records = _scan_global_remote_run_reservations(root)
    selected_global = {
        logical: record
        for logical, record in global_records.items()
        if _cohort_identity_from_payload(
            record["payload"],
            field="global_remote_run_reservation",
        )
        == identity
    }
    expected_reservations = {
        record["path"]: record for record in reservation_bindings
    }
    if set(selected_global) != set(expected_reservations):
        raise ValueError(
            "selected cohort journal and global reservation namespace differ"
        )
    for logical, expected in expected_reservations.items():
        observed = selected_global[logical]
        if (
            observed["binding"]["sha256"] != expected["sha256"]
            or observed["binding"]["size_bytes"] != expected["size_bytes"]
            or observed["payload"]["owner_attempt_id"] not in attempt_by_id
        ):
            raise ValueError(
                "selected cohort global reservation ownership changed"
            )
    disposition_by_key = {
        (record["attempt_id"], record["run_id"]): record
        for record in remote_dispositions
    }
    provider_spend = _derive_journal_provider_spend_estimate(
        root,
        identity=identity,
        attempts=attempts,
        remote_run_dispositions=disposition_by_key,
        provider_evidence=provider_cost_evidence,
        accounting_label=(
            "prior_quarantined_known_usage_plus_failed_and_"
            "may_have_started_conservative_reserves_not_billed_cost"
        ),
    )
    modal_exposure = _derive_modal_compute_exposure(
        attempts,
        measured_by_attempt=measured_by_attempt,
        unresolved_attempt_ids=unresolved_attempt_ids,
        accounting_label=(
            "prior_quarantined_measured_billing_plus_unresolved_or_lagged_"
            "compute_reserve_not_a_platform_hard_bound"
        ),
    )

    journal_price_candidates: list[
        tuple[Decimal, str, dict[str, Any], dict[str, Decimal]]
    ] = []
    seen_price_bindings: set[tuple[str, str]] = set()
    for attempt in attempts:
        if attempt["modal_price_basis_path"] is None:
            continue
        logical = _text(
            attempt["modal_price_basis_path"],
            "prior_attempt.modal_price_basis_path",
        )
        digest = _sha256(
            attempt["modal_price_basis_sha256"],
            "prior_attempt.modal_price_basis_sha256",
        )
        if (logical, digest) in seen_price_bindings:
            continue
        seen_price_bindings.add((logical, digest))
        price_path = _contained_path(
            root,
            logical,
            "prior_attempt.modal_price_basis_path",
            kind="file",
        )
        price_raw = _read_regular_file_bytes(
            price_path,
            maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        )
        if hashlib.sha256(price_raw).hexdigest() != digest:
            raise ValueError("prior journal Modal price-basis bytes changed")
        _price, rates, _loaded = load_modal_price_basis(
            root,
            logical,
            expected_raw_sha256=digest,
            expected_image_source_sha256=identity.image_source_sha256,
            require_freshness=False,
        )
        journal_price_candidates.append(
            (
                rates["volume"],
                logical,
                {"path": logical, "sha256": digest, "size_bytes": len(price_raw)},
                rates,
            )
        )
    if not journal_price_candidates:
        raise _PriorQuarantineAccountingIncomplete(
            "cohort action journal lacks a Modal price-basis binding"
        )
    highest_rate = max(item[0] for item in journal_price_candidates)
    _rate, _logical, price_binding, rates = min(
        (item for item in journal_price_candidates if item[0] == highest_rate),
        key=lambda item: item[1],
    )
    retained_count = len(volume_dispositions)
    bytes_per_run = MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES + MAX_ARTIFACT_MANIFEST_BYTES
    total_bytes = retained_count * bytes_per_run
    estimated_gib = Decimal(total_bytes) / Decimal(1024**3)

    candidate: dict[str, Any] = {
        "schema_name": "ModalPriorCohortQuarantineAccounting",
        "schema_version": "1.1",
        **modal_cohort_identity_dict(identity),
        "recorded_at_utc": request["recorded_at_utc"],
        "action_journal": journal,
        "remote_run_reservations": reservation_bindings,
        "attempt_dispositions": [
            {
                "attempt_id": attempt["attempt_id"],
                "action": attempt["action"],
                "status": attempt["status"],
                "concrete_remote_run_ids": attempt["concrete_remote_run_ids"],
                "disposition": "quarantined",
            }
            for attempt in sorted(attempts, key=lambda item: item["attempt_id"])
        ],
        "remote_run_dispositions": sorted(
            remote_dispositions,
            key=lambda item: (item["attempt_id"], item["run_id"]),
        ),
        "remote_executions": sorted(
            execution_records,
            key=lambda item: (item["attempt_id"], item["run_id"]),
        ),
        "provider_attempt_evidence": sorted(
            provider_records,
            key=lambda item: (item["attempt_id"], item["run_id"]),
        ),
        "unbound_provider_evidence": sorted(
            unbound_provider_records,
            key=lambda item: (
                item["attempt_id"],
                item["run_id"],
                item["evidence_path"],
            ),
        ),
        "provider_spend_estimate": provider_spend,
        "modal_compute_exposure": modal_exposure,
        "snapshot_capture_manifest_path": request[
            "snapshot_capture_manifest"
        ]["path"],
        "snapshot_capture_manifest_sha256": snapshot_sha256,
        "snapshot_capture_manifest_size_bytes": request[
            "snapshot_capture_manifest"
        ]["size_bytes"],
        "app_lifecycles": app_lifecycles,
        "selected_billing_rows": selected_billing,
        "app_compute_subtotal_usd": format(
            sum(measured_by_attempt.values(), Decimal("0")),
            "f",
        ),
        "volume_dispositions": sorted(
            volume_dispositions,
            key=lambda item: item["run_id"],
        ),
        "modal_price_basis": price_binding,
        "active_app_count": 0,
        "active_container_count": 0,
        "active_endpoint_count": 0,
        "accepted_contexts": [],
        "retained_storage_estimate": {
            "retained_run_count": retained_count,
            "conservative_bytes_per_run": bytes_per_run,
            "conservative_total_bytes": total_bytes,
            "estimated_gib": format(estimated_gib, "f"),
            "volume_rate_usd_per_gib_month": format(rates["volume"], "f"),
            "estimated_monthly_usd": format(
                estimated_gib * rates["volume"],
                "f",
            ),
            "basis": (
                "retained_run_count_times_per_run_artifact_download_and_"
                "manifest_caps; included_shared_volume_quota_not_subtracted"
            ),
        },
        "validated": True,
    }
    frozen, encoded = _exclusive_json_object_bytes(candidate)
    logical = modal_prior_quarantine_accounting_path(identity).as_posix()
    validated, _metadata = _validate_prior_quarantine_accounting_payload(
        root,
        logical,
        frozen,
        encoded,
    )
    if not exact_json_equal(validated, frozen):
        raise ValueError("derived prior quarantine accounting candidate changed")
    return validated


def create_prior_quarantine_accounting_template(
    *,
    source_tree_sha256_value: str,
    image_source_sha256: str,
    cohort_id: str,
    recorded_at_utc: str,
    snapshot_capture_manifest_path: str | Path,
    output_path: str | Path,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create one minimal, immutable prior-accounting operator request."""

    project_root = Path(root)
    selected_snapshot = _lexically_absolute_operator_path(
        snapshot_capture_manifest_path,
        field="snapshot capture manifest path",
    )
    output = _prior_accounting_operator_output_path(
        project_root,
        output_path,
        field="prior quarantine accounting template output path",
    )
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=_sha256(
            source_tree_sha256_value,
            "source_tree_sha256",
        ),
        image_source_sha256=_sha256(
            image_source_sha256,
            "image_source_sha256",
        ),
        cohort_id=validate_run_id(cohort_id),
    )
    _utc(recorded_at_utc, "recorded_at_utc")
    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        logical = _absolute_project_file_logical(
            project_root,
            selected_snapshot,
            field="snapshot capture manifest path",
        )
        raw = _read_regular_file_bytes(
            selected_snapshot,
            maximum_bytes=_MAX_JSON_OBJECT_BYTES,
            required_mode=0o600,
        )
        request: dict[str, Any] = {
            "schema_name": "ModalPriorCohortQuarantineAccountingRequest",
            "schema_version": "1.0",
            **modal_cohort_identity_dict(identity),
            "recorded_at_utc": recorded_at_utc,
            "snapshot_capture_manifest": {
                "path": logical,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            },
        }
        frozen, _selected_identity = _prior_accounting_request_identity(request)
        _load_prior_accounting_selected_snapshot(
            project_root,
            frozen,
            identity,
        )
        assert_modal_action_lock_identity(lock_descriptor)
        create_json_exclusive(output, frozen)
        assert_modal_action_lock_identity(lock_descriptor)
        persisted = _load_operator_json_input(output)
        if not exact_json_equal(persisted, frozen):
            raise ValueError("persisted prior accounting request changed")
        _load_prior_accounting_selected_snapshot(
            project_root,
            persisted,
            identity,
        )
        assert_modal_action_lock_identity(lock_descriptor)
        return persisted
    finally:
        release_modal_action_lock(lock_descriptor)


def inspect_prior_quarantine_accounting(
    *,
    request: Mapping[str, Any],
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Read repository evidence and return a candidate or explicit blockers."""

    frozen, identity = _prior_accounting_request_identity(request)
    project_root = Path(root)
    canonical = Path(os.path.abspath(os.fspath(project_root))).joinpath(
        *modal_prior_quarantine_accounting_path(identity).parts
    )
    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        try:
            candidate = _derive_prior_quarantine_accounting_candidate(
                project_root,
                frozen,
            )
        except _PriorQuarantineAccountingIncomplete as error:
            candidate = None
            blockers = [
                {"code": "incomplete_repository_evidence", "message": message}
                for message in error.messages
            ]
        except FileNotFoundError as error:
            candidate = None
            blockers = [
                {
                    "code": "missing_repository_evidence",
                    "message": str(error),
                }
            ]
        else:
            blockers = []
        if (candidate is None) is not bool(blockers):
            raise AssertionError("prior accounting inspection result is inconsistent")
        inspection = {
            "schema_name": "ModalPriorCohortQuarantineAccountingInspection",
            "schema_version": "1.0",
            "request": frozen,
            "canonical_receipt_path": canonical.as_posix(),
            "candidate": candidate,
            "blockers": blockers,
        }
        if set(inspection) != _PRIOR_QUARANTINE_ACCOUNTING_INSPECTION_FIELDS:
            raise AssertionError("prior accounting inspection schema drifted")
        assert_modal_action_lock_identity(lock_descriptor)
        return inspection
    finally:
        release_modal_action_lock(lock_descriptor)


def scaffold_prior_quarantine_accounting(
    *,
    request: Mapping[str, Any],
    output_path: str | Path,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create a full validator-clean candidate without publishing its receipt."""

    frozen_request, identity = _prior_accounting_request_identity(request)
    project_root = Path(root)
    output = _prior_accounting_operator_output_path(
        project_root,
        output_path,
        field="prior quarantine accounting scaffold output path",
    )
    canonical_logical = modal_prior_quarantine_accounting_path(identity).as_posix()
    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        try:
            candidate = _derive_prior_quarantine_accounting_candidate(
                project_root,
                frozen_request,
            )
        except (FileNotFoundError, _PriorQuarantineAccountingIncomplete) as error:
            raise ValueError(
                f"prior quarantine accounting scaffold is blocked: {error}"
            ) from error
        assert_modal_action_lock_identity(lock_descriptor)
        repeated = _derive_prior_quarantine_accounting_candidate(
            project_root,
            frozen_request,
        )
        if not exact_json_equal(candidate, repeated):
            raise ValueError("prior accounting source evidence changed during scaffold")
        assert_modal_action_lock_identity(lock_descriptor)
        create_json_exclusive(output, candidate)
        assert_modal_action_lock_identity(lock_descriptor)
        persisted = _load_operator_json_input(output)
        if not exact_json_equal(persisted, candidate):
            raise ValueError("persisted prior accounting scaffold changed")
        persisted_frozen, persisted_raw = _exclusive_json_object_bytes(persisted)
        validated, _metadata = _validate_prior_quarantine_accounting_payload(
            project_root,
            canonical_logical,
            persisted_frozen,
            persisted_raw,
        )
        if not exact_json_equal(validated, candidate):
            raise ValueError(
                "persisted prior accounting scaffold is not validator-clean"
            )
        final_candidate = _derive_prior_quarantine_accounting_candidate(
            project_root,
            frozen_request,
        )
        if not exact_json_equal(final_candidate, candidate):
            raise ValueError("prior accounting source evidence changed after scaffold")
        assert_modal_action_lock_identity(lock_descriptor)
        return persisted
    finally:
        release_modal_action_lock(lock_descriptor)


def create_prior_quarantine_accounting(
    *,
    payload: Mapping[str, Any],
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Validate, publish, and revalidate one prior-cohort accounting seal.

    The caller supplies explicit evidence selections; the validator derives the
    journal, reservation, execution, provider, lifecycle, billing, Volume, and
    retained-storage claims from those bound files.  Invalid operator input is
    rejected before the canonical create-only path is published.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("prior quarantine accounting payload must be a mapping")
    project_root = Path(root)
    frozen, encoded = _exclusive_json_object_bytes(payload)

    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        _scan_resolved_modal_global_action_journal(lock_descriptor)
        identity = _cohort_identity_from_payload(
            frozen,
            field="prior_quarantine",
        )
        logical = modal_prior_quarantine_accounting_path(identity).as_posix()
        validated, _metadata = _validate_prior_quarantine_accounting_payload(
            project_root,
            logical,
            frozen,
            encoded,
        )
        if not exact_json_equal(validated, frozen):
            raise ValueError("prior quarantine accounting prevalidation changed")
        output = project_root.resolve().joinpath(*Path(logical).parts)
        assert_modal_action_lock_identity(lock_descriptor)
        create_json_exclusive(output, frozen)
        assert_modal_action_lock_identity(lock_descriptor)
        persisted, _metadata = _load_prior_quarantine_accounting(
            project_root,
            logical,
        )
        if not exact_json_equal(persisted, frozen):
            raise ValueError("persisted prior quarantine accounting changed")
        assert_modal_action_lock_identity(lock_descriptor)
        return persisted
    finally:
        release_modal_action_lock(lock_descriptor)


def _lineage_bound_bytes(
    root: Path,
    logical: str,
    *,
    field: str,
    maximum_bytes: int = _MAX_JSON_OBJECT_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    path = _contained_path(root, logical, field, kind="file")
    raw = _read_regular_file_bytes(path, maximum_bytes=maximum_bytes)
    return raw, {
        "path": logical,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def _lineage_bound_json(
    root: Path,
    logical: str,
    *,
    field: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, binding = _lineage_bound_bytes(root, logical, field=field)
    path = root.resolve().joinpath(*PurePosixPath(logical).parts)
    payload = json.loads(
        _decode_utf8(raw, path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return payload, binding


def _lineage_downloaded_execution(
    root: Path,
    *,
    attempt_id: str,
    run_id: str,
    action: str,
    harness: str | None,
    image_source_sha256: str,
) -> tuple[ExecutionContextV1, dict[str, Any], dict[str, Any], Any]:
    logical_root = _expected_download_path(run_id)
    run_root = _contained_path(
        root,
        logical_root,
        "lineage.downloaded_run",
        kind="directory",
    )
    raw_manifest = _select_downloaded_raw_manifest(run_root)
    manifest = raw_manifest.manifest
    if (
        manifest.run_id != run_id
        or manifest.image_source_sha256 != image_source_sha256
    ):
        raise ValueError("lineage artifact manifest identity changed")
    verification = verify_artifact_manifest(run_root, manifest)
    if verification.get("verified") is not True:
        raise ValueError("lineage artifact manifest did not verify")

    context_logical = f"{logical_root}/execution_context.json"
    context_raw, context_binding = _lineage_bound_bytes(
        root,
        context_logical,
        field="lineage.execution_context",
    )
    context_item = next(
        (
            item
            for item in manifest.files
            if item.relative_path == "execution_context.json"
        ),
        None,
    )
    if (
        context_item is None
        or context_item.sha256 != hashlib.sha256(context_raw).hexdigest()
        or context_item.size_bytes != len(context_raw)
    ):
        raise ValueError("lineage execution context is not manifest-bound")
    context_payload = json.loads(
        _decode_utf8(context_raw, run_root / "execution_context.json"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    try:
        context = ExecutionContextV1.from_dict(context_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("lineage execution context is invalid") from error
    if action in _ORDINARY_ACTION_FUNCTIONS:
        expected_function = _ORDINARY_ACTION_FUNCTIONS[action]
    elif action == "canary":
        expected_function = f"canary_{harness}"
    elif action == "canaries":
        expected_function = context.function_name
        if expected_function not in {
            f"canary_{selected}" for selected in CANARY_ORDER
        }:
            raise ValueError("lineage aggregate child function is invalid")
    else:
        raise ValueError("lineage downloaded execution has an unsupported action")
    if (
        context.execution_backend != "modal"
        or context.run_id != run_id
        or context.app_name != APP_NAME
        or context.function_name != expected_function
        or context.image_source_sha256 != image_source_sha256
        or context.artifact_uri != volume_artifact_uri(run_id)
        or context.modal_app_id is None
        or context.modal_function_id is None
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise ValueError("lineage remote execution identity is incomplete")

    manifest_logical = f"{logical_root}/{raw_manifest.filename}"
    manifest_binding = {
        "attempt_id": attempt_id,
        "run_id": run_id,
        "path": manifest_logical,
        "sha256": raw_manifest.raw_sha256,
        "size_bytes": raw_manifest.raw_size_bytes,
        "canonical_manifest_sha256": manifest.manifest_sha256,
    }
    execution = {
        "attempt_id": attempt_id,
        "run_id": run_id,
        "action": action,
        "evidence_kind": "downloaded_execution_context",
        "evidence": context_binding,
        "execution_context": context.to_dict(),
    }
    return context, execution, manifest_binding, raw_manifest


def _lineage_context_only_execution(
    root: Path,
    *,
    attempt_id: str,
    run_id: str,
    action: str,
    harness: str | None,
    image_source_sha256: str,
) -> tuple[ExecutionContextV1, dict[str, Any]]:
    """Bind a failed remote execution whose manifest was never finalized."""

    logical = f"{_expected_download_path(run_id)}/execution_context.json"
    raw, binding = _lineage_bound_bytes(
        root,
        logical,
        field="lineage.context_only_execution",
    )
    path = root.resolve().joinpath(*PurePosixPath(logical).parts)
    try:
        payload = json.loads(
            _decode_utf8(raw, path),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        context = ExecutionContextV1.from_dict(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("lineage context-only execution is invalid") from error
    if action in _ORDINARY_ACTION_FUNCTIONS:
        expected_function = _ORDINARY_ACTION_FUNCTIONS[action]
    elif action == "canary":
        expected_function = f"canary_{harness}"
    elif action == "canaries":
        expected_function = context.function_name
        if expected_function not in {
            f"canary_{selected}" for selected in CANARY_ORDER
        }:
            raise ValueError("lineage aggregate child function is invalid")
    else:
        raise ValueError("context-only execution has an unsupported action")
    if (
        context.execution_backend != "modal"
        or context.run_id != run_id
        or context.app_name != APP_NAME
        or context.function_name != expected_function
        or context.image_source_sha256 != image_source_sha256
        or context.artifact_uri != volume_artifact_uri(run_id)
        or context.modal_app_id is None
        or context.modal_function_id is None
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise ValueError("lineage context-only remote identity is incomplete")
    return context, {
        "attempt_id": attempt_id,
        "run_id": run_id,
        "action": action,
        "evidence_kind": "downloaded_execution_context_without_artifact_manifest",
        "evidence": binding,
        "execution_context": context.to_dict(),
    }


def _lineage_provider_records(
    raw: bytes,
    path: Path,
) -> list[ProviderAttemptRecord]:
    text = _decode_utf8(raw, path)
    if text == "":
        return []
    if not text.endswith("\n"):
        raise ValueError("lineage provider ledger is truncated")
    records: list[ProviderAttemptRecord] = []
    for line in text.splitlines():
        if not line:
            raise ValueError("lineage provider ledger has a blank record")
        value = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if not isinstance(value, dict):
            raise ValueError("lineage provider ledger record is not an object")
        records.append(ProviderAttemptRecord.from_dict(value))
    return records


def _lineage_verifier_execution(
    root: Path,
    *,
    identity: ModalLiveCohortIdentity,
    attempt: Mapping[str, Any],
    source_context: ExecutionContextV1,
    source_raw_manifest: Any,
) -> tuple[ExecutionContextV1, dict[str, Any], dict[str, Any] | None]:
    """Bind one verifier execution to its canonical or recovered evidence.

    Normal verifier calls are captured from the launcher's stdout as a
    canonical verification receipt.  If the local launcher loses that stdout
    (or the remote verifier fails), cleanup recovery instead preserves an
    exact four-file Volume capture.  Lineage must understand both forms or it
    silently excludes precisely the paid executions recovery is meant to
    preserve.
    """

    attempt_id = _attempt_id(attempt["attempt_id"], "lineage.attempt_id")
    source_run_id = validate_run_id(attempt["run_id"])
    verifier_run_id = validate_run_id(attempt["verifier_run_id"])
    canonical_logical = _remote_verification_logical(
        identity,
        source_run_id,
        verifier_run_id,
        attempt_id,
    )
    capture_root = modal_artifact_verifier_capture_directory_path(
        identity,
        source_run_id,
        verifier_run_id,
        attempt_id,
    )
    success_logical = (capture_root / "artifact_verification_result.json").as_posix()
    failure_logical = (
        capture_root / "artifact_verification_failure.json"
    ).as_posix()
    candidates = [
        ("remote_verification_receipt", canonical_logical),
        ("volume_success_capture", success_logical),
        ("volume_failure_capture", failure_logical),
    ]
    existing = [
        (kind, logical)
        for kind, logical in candidates
        if root.resolve().joinpath(*PurePosixPath(logical).parts).is_file()
    ]
    if len(existing) != 1:
        raise ValueError(
            "lineage verifier requires exactly one immutable execution capture"
        )
    evidence_kind, logical = existing[0]
    payload, binding = _lineage_bound_json(
        root,
        logical,
        field="lineage.verifier_execution",
    )
    artifact_binding: dict[str, Any] | None = None
    if evidence_kind in {"remote_verification_receipt", "volume_success_capture"}:
        verification = _validate_remote_verification(
            payload,
            source_run_id=source_run_id,
            verifier_run_id=verifier_run_id,
            raw_manifest=source_raw_manifest,
            source_execution_context=source_context,
        )
        context = verification.verifier_execution_context
        if attempt["status"] != "succeeded":
            raise ValueError("lineage successful verifier capture has failed status")
        if evidence_kind == "volume_success_capture":
            record = {
                "source_run_id": source_run_id,
                "verifier_run_id": verifier_run_id,
                "attempt_id": attempt_id,
                "remote_verification_path": logical,
                "remote_verification_sha256": binding["sha256"],
                "verifier_execution_context": context.to_dict(),
            }
            captured, canonical_manifest_sha256 = _successful_verifier_capture(
                root,
                record,
                identity=identity,
            )
            if captured != verification:
                raise ValueError("lineage successful verifier capture changed")
            manifest_logical = (capture_root / "artifact_manifest.json").as_posix()
            manifest_raw, manifest_binding = _lineage_bound_bytes(
                root,
                manifest_logical,
                field="lineage.verifier_artifact_manifest",
            )
            artifact_binding = {
                "attempt_id": attempt_id,
                "run_id": verifier_run_id,
                "path": manifest_logical,
                "sha256": hashlib.sha256(manifest_raw).hexdigest(),
                "size_bytes": len(manifest_raw),
                "canonical_manifest_sha256": canonical_manifest_sha256,
            }
    else:
        if attempt["status"] == "succeeded":
            raise ValueError("lineage failed verifier capture has successful status")
        try:
            declared_context = ExecutionContextV1.from_dict(
                payload["verifier_execution_context"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("lineage failed verifier context is invalid") from error
        record = {
            "source_run_id": source_run_id,
            "verifier_run_id": verifier_run_id,
            "attempt_id": attempt_id,
            "failure_receipt_path": logical,
            "failure_receipt_sha256": binding["sha256"],
            "failure_execution_context": declared_context.to_dict(),
        }
        context, canonical_manifest_sha256 = _failed_verifier_capture(
            root,
            record,
            identity=identity,
        )
        if (
            context.image_source_sha256 != source_context.image_source_sha256
            or context.modal_image_id != source_context.modal_image_id
            or context.artifact_uri != volume_artifact_uri(source_run_id)
        ):
            raise ValueError("lineage failed verifier used another source image")
        manifest_logical = (capture_root / "artifact_manifest.json").as_posix()
        manifest_raw, manifest_binding = _lineage_bound_bytes(
            root,
            manifest_logical,
            field="lineage.verifier_artifact_manifest",
        )
        artifact_binding = {
            "attempt_id": attempt_id,
            "run_id": verifier_run_id,
            "path": manifest_logical,
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "size_bytes": len(manifest_raw),
            "canonical_manifest_sha256": canonical_manifest_sha256,
        }

    if (
        context.image_source_sha256 != identity.image_source_sha256
        or context.modal_app_id is None
        or context.modal_function_id is None
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise ValueError("lineage verifier execution identity is incomplete")
    return (
        context,
        {
            "attempt_id": attempt_id,
            "run_id": verifier_run_id,
            "action": attempt["action"],
            "evidence_kind": evidence_kind,
            "evidence": binding,
            "execution_context": context.to_dict(),
        },
        artifact_binding,
    )


def _derive_final_lineage_evidence(
    root: Path,
    *,
    identity: ModalLiveCohortIdentity,
    journal: Mapping[str, list[dict[str, Any]]],
    attempts: list[dict[str, Any]],
    accepted_primary_runs: Mapping[str, str],
    accepted_attempt_ids: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    _assert_attempts_contained_for_seal(attempts, field="final_lineage")
    attempt_by_id = {record["attempt_id"]: record for record in attempts}
    expected_acceptance = {
        "cuda_environment": ("cuda-environment", None),
        "offline_smoke": ("offline-smoke", None),
        "candidate_smoke": ("candidate-smoke", None),
        "resume_attempt": ("checkpoint-resume", None),
        **{
            f"canary_{harness}": ("canary", harness)
            for harness in CANARY_ORDER
        },
    }
    for label, (action, harness) in expected_acceptance.items():
        attempt = attempt_by_id.get(accepted_attempt_ids[label])
        if (
            attempt is None
            or attempt["status"] != "succeeded"
            or attempt["action"] != action
            or attempt["harness"] != harness
            or attempt["run_id"] != accepted_primary_runs[label]
            or attempt["concrete_remote_run_ids"]
            != [accepted_primary_runs[label]]
        ):
            raise ValueError("lineage final accepted run disposition changed")

    run_disposition_map: dict[tuple[str, str], dict[str, Any]] = {}
    for attempt in sorted(attempts, key=lambda item: item["attempt_id"]):
        for run_id in attempt["concrete_remote_run_ids"]:
            run_disposition_map[(attempt["attempt_id"], run_id)] = {
                "attempt_id": attempt["attempt_id"],
                "action": attempt["action"],
                "status": attempt["status"],
                "failure_kind": attempt["failure_kind"],
                "run_id": run_id,
                "modal_cli_process_started": attempt[
                    "modal_cli_process_started"
                ],
                "remote_execution_state": attempt["remote_execution_state"],
                "execution_disposition": (
                    None
                    if attempt["modal_cli_process_started"]
                    else "definitely_not_started"
                ),
                "provider_disposition": (
                    "definitely_not_started"
                    if not attempt["modal_cli_process_started"]
                    and attempt["action"] in {"canary", "canaries"}
                    else "not_applicable"
                    if attempt["action"] not in {"canary", "canaries"}
                    else None
                ),
            }
    execution_contexts: dict[tuple[str, str], ExecutionContextV1] = {}
    executions: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    downloaded: dict[str, tuple[ExecutionContextV1, Any]] = {}

    for attempt in sorted(attempts, key=lambda item: item["attempt_id"]):
        if attempt["action"] in {"download", "verify"}:
            continue
        for run_id in attempt["concrete_remote_run_ids"]:
            key = (attempt["attempt_id"], run_id)
            context_logical = (
                f"{_expected_download_path(run_id)}/execution_context.json"
            )
            manifest_present = any(
                _path_has_any_entry(
                    root,
                    f"{_expected_download_path(run_id)}/{filename}",
                )
                for filename in ARTIFACT_MANIFEST_FILENAMES
            )
            context_present = _path_has_any_entry(root, context_logical)
            if not attempt["modal_cli_process_started"]:
                if context_present or manifest_present:
                    raise ValueError(
                        "definitely-not-started final run has execution evidence"
                    )
                continue
            if context_present and manifest_present:
                context, execution, artifact, raw_manifest = (
                    _lineage_downloaded_execution(
                        root,
                        attempt_id=attempt["attempt_id"],
                        run_id=run_id,
                        action=attempt["action"],
                        harness=attempt["harness"],
                        image_source_sha256=identity.image_source_sha256,
                    )
                )
                downloaded[run_id] = (context, raw_manifest)
                artifacts.append(artifact)
            elif context_present and not manifest_present:
                if attempt["status"] == "succeeded":
                    raise ValueError(
                        "successful final execution lacks its artifact manifest"
                    )
                context, execution = _lineage_context_only_execution(
                    root,
                    attempt_id=attempt["attempt_id"],
                    run_id=run_id,
                    action=attempt["action"],
                    harness=attempt["harness"],
                    image_source_sha256=identity.image_source_sha256,
                )
            elif manifest_present:
                raise ValueError(
                    "final artifact manifest lacks its execution context"
                )
            else:
                if attempt["status"] == "succeeded":
                    raise ValueError("successful final execution lacks evidence")
                run_disposition_map[key]["execution_disposition"] = (
                    "may_have_started_unresolved_quarantined"
                )
                continue
            execution_contexts[key] = context
            executions.append(execution)
            run_disposition_map[key]["execution_disposition"] = (
                "remote_execution_bound"
            )

    for attempt in sorted(attempts, key=lambda item: item["attempt_id"]):
        if attempt["action"] not in {"download", "verify"}:
            continue
        source_run_id = validate_run_id(attempt["run_id"])
        verifier_run_id = validate_run_id(attempt["verifier_run_id"])
        key = (attempt["attempt_id"], verifier_run_id)
        evidence_present = any(
            _path_has_any_entry(root, logical)
            for logical in _prior_execution_evidence_candidates(
                identity, attempt, verifier_run_id
            )
        )
        if not attempt["modal_cli_process_started"]:
            if evidence_present:
                raise ValueError(
                    "definitely-not-started verifier has execution evidence"
                )
            continue
        if not evidence_present:
            if attempt["status"] == "succeeded":
                raise ValueError("successful final verifier lacks evidence")
            run_disposition_map[key]["execution_disposition"] = (
                "may_have_started_unresolved_quarantined"
            )
            continue
        source = downloaded.get(source_run_id)
        if source is None:
            raise ValueError("lineage verifier source lacks a bound final execution")
        context, execution, artifact = _lineage_verifier_execution(
            root,
            identity=identity,
            attempt=attempt,
            source_context=source[0],
            source_raw_manifest=source[1],
        )
        execution_contexts[key] = context
        executions.append(execution)
        run_disposition_map[key]["execution_disposition"] = (
            "remote_execution_bound"
        )
        if artifact is not None:
            artifacts.append(artifact)

    expected_execution_keys = {
        key
        for key, disposition in run_disposition_map.items()
        if disposition["execution_disposition"] == "remote_execution_bound"
    }
    if set(execution_contexts) != expected_execution_keys:
        raise ValueError("lineage remote execution evidence is journal-incomplete")

    app_owner: dict[str, str] = {}
    call_ids: set[str] = set()
    function_ids: set[str] = set()
    image_ids: set[str] = set()
    for (attempt_id, _run_id), context in execution_contexts.items():
        app_id = context.modal_app_id or ""
        owner = app_owner.setdefault(app_id, attempt_id)
        if owner != attempt_id:
            raise ValueError("lineage final App ID is shared across attempts")
        call_id = context.modal_call_id or ""
        if call_id in call_ids:
            raise ValueError("lineage final call ID is reused")
        call_ids.add(call_id)
        function_ids.add(context.modal_function_id or "")
        image_ids.add(context.modal_image_id or "")

    provider_evidence: list[dict[str, Any]] = []
    provider_cost_evidence: dict[tuple[str, str], dict[str, Any]] = {}
    request_ids: set[str] = set()
    response_ids: set[str] = set()
    for attempt in sorted(attempts, key=lambda item: item["attempt_id"]):
        if attempt["action"] not in {"canary", "canaries"}:
            continue
        for run_id in attempt["concrete_remote_run_ids"]:
            key = (attempt["attempt_id"], run_id)
            context = execution_contexts.get(key)
            harness = _provider_harness_for_run(attempt, run_id)
            if harness not in CANARY_ORDER:
                raise ValueError("lineage provider harness is invalid")
            controller = f"{_expected_download_path(run_id)}/controller"
            ledger_logical = f"{controller}/provider_attempts.jsonl"
            uncertain_logical = (
                f"{controller}/provider_request_start_uncertain.json"
            )
            ledger_path = _contained_path(
                root,
                ledger_logical,
                "lineage.provider_ledger",
                kind="optional",
            )
            uncertain_path = _contained_path(
                root,
                uncertain_logical,
                "lineage.provider_uncertainty",
                kind="optional",
            )
            if not attempt["modal_cli_process_started"]:
                if ledger_path.exists() or uncertain_path.exists():
                    raise ValueError(
                        "definitely-not-started provider run has provider evidence"
                    )
                continue

            if context is None:
                ledger_binding: dict[str, Any] | None = None
                uncertainty_binding: dict[str, Any] | None = None
                records: list[ProviderAttemptRecord] = []
                parse_dispositions: list[str] = []
                if ledger_path.is_file():
                    raw, ledger_binding = _lineage_bound_bytes(
                        root,
                        ledger_logical,
                        field="lineage.unbound_provider_ledger",
                        maximum_bytes=_MAX_PROVIDER_LEDGER_BYTES,
                    )
                    try:
                        records = _lineage_provider_records(raw, ledger_path)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        records = []
                        parse_dispositions.append("partial_unparseable")
                    else:
                        if any(
                            record.execution_backend != "modal"
                            or record.action_run_id != run_id
                            or record.harness != harness
                            or record.action
                            != "one_opportunity_engineering_canary"
                            for record in records
                        ) or [record.attempt_ordinal for record in records] != list(
                            range(1, len(records) + 1)
                        ):
                            raise ValueError(
                                "lineage unbound provider ledger identity changed"
                            )
                        if len(records) > 1:
                            raise ValueError(
                                "lineage unbound provider ledger exceeds approval"
                            )
                        parse_dispositions.append(
                            "valid_terminal_records"
                            if records
                            else "exact_empty"
                        )
                if uncertain_path.is_file():
                    raw, uncertainty_binding = _lineage_bound_bytes(
                        root,
                        uncertain_logical,
                        field="lineage.unbound_provider_uncertainty",
                        maximum_bytes=_MAX_PROVIDER_LEDGER_BYTES,
                    )
                    try:
                        _load_provider_start_uncertain_evidence(
                            root,
                            uncertain_logical,
                            hashlib.sha256(raw).hexdigest(),
                            harness=harness,
                            run_id=run_id,
                            expected_modal_call_id=None,
                        )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        parse_dispositions.append("partial_unparseable")
                    else:
                        parse_dispositions.append("valid_start_uncertain")
                selected_requests = sorted(
                    record.provider_request_id
                    for record in records
                    if record.provider_request_id is not None
                )
                selected_responses = sorted(
                    record.provider_response_id
                    for record in records
                    if record.provider_response_id is not None
                )
                if (
                    len(selected_requests) != len(set(selected_requests))
                    or len(selected_responses) != len(set(selected_responses))
                    or request_ids.intersection(selected_requests)
                    or response_ids.intersection(selected_responses)
                ):
                    raise ValueError("lineage final provider IDs are reused")
                request_ids.update(selected_requests)
                response_ids.update(selected_responses)
                if ledger_binding is not None or uncertainty_binding is not None:
                    provider_evidence.append(
                        {
                            "attempt_id": attempt["attempt_id"],
                            "run_id": run_id,
                            "harness": harness,
                            "binding_state": "unbound_observed",
                            "ledger": ledger_binding,
                            "uncertainty": uncertainty_binding,
                            "parse_dispositions": parse_dispositions,
                            "provider_attempt_count": len(records),
                            "request_ids": selected_requests,
                            "response_ids": selected_responses,
                        }
                    )
                    provider_cost_evidence[key] = {
                        "state": "unbound_observed",
                        "records": records,
                        "harness": harness,
                        "parse_dispositions": parse_dispositions,
                    }
                run_disposition_map[key]["provider_disposition"] = (
                    "start_unresolved_conservative"
                )
                continue

            ledger_binding: dict[str, Any] | None = None
            records: list[ProviderAttemptRecord] = []
            if ledger_path.is_file():
                raw, ledger_binding = _lineage_bound_bytes(
                    root,
                    ledger_logical,
                    field="lineage.provider_ledger",
                    maximum_bytes=_MAX_PROVIDER_LEDGER_BYTES,
                )
                records = _lineage_provider_records(raw, ledger_path)
            uncertainty_binding: dict[str, Any] | None = None
            if uncertain_path.is_file():
                uncertainty, uncertainty_binding = _lineage_bound_json(
                    root,
                    uncertain_logical,
                    field="lineage.provider_uncertainty",
                )
                if (
                    set(uncertainty) != _PROVIDER_START_UNCERTAIN_FIELDS
                    or uncertainty["schema_name"]
                    != "ProviderRequestStartUncertainEvidence"
                    or uncertainty["schema_version"] != "1.0"
                    or uncertainty["harness"] != harness
                    or uncertainty["action"]
                    != "one_opportunity_engineering_canary"
                    or uncertainty["execution_backend"] != "modal"
                    or uncertainty["action_run_id"] != run_id
                    or uncertainty["modal_call_id"] != context.modal_call_id
                    or uncertainty["api_endpoint"] != OFFICIAL_OPENAI_API_BASE
                    or uncertainty["model"] != TARGET_MODEL
                    or uncertainty["provider_attempt_count_lower_bound"] != 0
                    or uncertainty["provider_attempt_count_upper_bound"] != 1
                    or uncertainty["provider_request_started"] != "unknown"
                    or uncertainty["provider_attempt_ledger_state"]
                    != ("present" if ledger_binding is not None else "missing")
                    or uncertainty["billing_treatment"]
                    != "reserve_one_full_approved_request"
                    or uncertainty["reason"]
                    != "controller_terminated_without_terminal_attempt_record"
                ):
                    raise ValueError("lineage provider uncertainty is invalid")
            if ledger_binding is None and uncertainty_binding is None:
                raise ValueError(
                    "lineage cannot infer zero provider starts without immutable "
                    "attempt or uncertainty evidence"
                )
            if not records and uncertainty_binding is None:
                raise ValueError("lineage provider ledger is exactly empty")
            if any(
                record.execution_backend != "modal"
                or record.action_run_id != run_id
                or record.modal_call_id != context.modal_call_id
                or record.harness != harness
                or record.action != "one_opportunity_engineering_canary"
                for record in records
            ) or [record.attempt_ordinal for record in records] != list(
                range(1, len(records) + 1)
            ):
                raise ValueError("lineage provider ledger identity changed")
            selected_requests = sorted(
                record.provider_request_id
                for record in records
                if record.provider_request_id is not None
            )
            selected_responses = sorted(
                record.provider_response_id
                for record in records
                if record.provider_response_id is not None
            )
            if (
                len(selected_requests) != len(set(selected_requests))
                or len(selected_responses) != len(set(selected_responses))
                or request_ids.intersection(selected_requests)
                or response_ids.intersection(selected_responses)
            ):
                raise ValueError("lineage final provider IDs are reused")
            request_ids.update(selected_requests)
            response_ids.update(selected_responses)
            provider_evidence.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "run_id": run_id,
                    "harness": harness,
                    "binding_state": "execution_context_bound",
                    "ledger": ledger_binding,
                    "uncertainty": uncertainty_binding,
                    "parse_dispositions": (
                        ["valid_terminal_records"]
                        if records
                        else ["valid_start_uncertain"]
                    ),
                    "provider_attempt_count": len(records),
                    "request_ids": selected_requests,
                    "response_ids": selected_responses,
                }
            )
            provider_cost_evidence[(attempt["attempt_id"], run_id)] = {
                "state": (
                    "start_uncertain"
                    if uncertainty_binding is not None and not records
                    else "ledger"
                ),
                "records": records,
                "harness": harness,
            }
            run_disposition_map[key]["provider_disposition"] = "evidence_bound"

    executions.sort(key=lambda item: (item["attempt_id"], item["run_id"]))
    artifacts.sort(key=lambda item: (item["attempt_id"], item["run_id"]))
    provider_evidence.sort(
        key=lambda item: (item["attempt_id"], item["run_id"])
    )
    if any(
        disposition["execution_disposition"] is None
        or disposition["provider_disposition"] is None
        for disposition in run_disposition_map.values()
    ):
        raise ValueError("final run dispositions are incomplete")
    run_dispositions = [
        run_disposition_map[key] for key in sorted(run_disposition_map)
    ]
    provider_remote_dispositions = {
        key: {
            "provider_disposition": disposition["provider_disposition"]
        }
        for key, disposition in run_disposition_map.items()
        if disposition["action"] in {"canary", "canaries"}
    }
    provider_spend_estimate = _derive_journal_provider_spend_estimate(
        root,
        identity=identity,
        attempts=attempts,
        remote_run_dispositions=provider_remote_dispositions,
        provider_evidence=provider_cost_evidence,
        accounting_label=(
            "final_journal_known_usage_plus_failed_and_uncertain_"
            "reserves_not_billed_cost"
        ),
    )
    artifact_paths = {record["path"] for record in artifacts}
    if len(artifact_paths) != len(artifacts):
        raise ValueError("lineage final artifact manifest path is reused")
    metadata = {
        "app_ids": set(app_owner),
        "call_ids": call_ids,
        "function_ids": function_ids,
        "image_ids": image_ids,
        "provider_request_ids": request_ids,
        "provider_response_ids": response_ids,
        "artifact_paths": artifact_paths,
    }
    evidence = {
        "run_dispositions": run_dispositions,
        "aggregate_receipts": list(journal["aggregate_receipts"]),
        "remote_executions": executions,
        "remote_object_ids": {
            "app_ids": sorted(metadata["app_ids"]),
            "function_ids": sorted(metadata["function_ids"]),
            "call_ids": sorted(metadata["call_ids"]),
            "image_ids": sorted(metadata["image_ids"]),
        },
        "provider_attempt_evidence": provider_evidence,
        "provider_spend_estimate": provider_spend_estimate,
        "artifact_manifests": artifacts,
    }
    return evidence, metadata


def _derive_migration_lineage_claims(
    root: Path,
    *,
    final_identity: ModalLiveCohortIdentity,
    accepted_primary_runs: Mapping[str, str],
    accepted_attempt_ids: Mapping[str, str],
    prior_quarantine_accounting_paths: list[str],
) -> dict[str, Any]:
    if set(accepted_primary_runs) != set(_PRIMARY_LABELS) or set(
        accepted_attempt_ids
    ) != set(_PRIMARY_LABELS):
        raise ValueError("lineage final accepted roster is not exact")
    final_runs = {
        label: validate_run_id(accepted_primary_runs[label])
        for label in _PRIMARY_LABELS
    }
    final_attempts = {
        label: _attempt_id(
            accepted_attempt_ids[label],
            f"accepted_attempt_ids.{label}",
        )
        for label in _PRIMARY_LABELS
    }
    if len(set(final_runs.values())) != 8 or len(set(final_attempts.values())) != 8:
        raise ValueError("lineage final accepted run and attempt IDs must be unique")

    identities = _discover_cohort_journal_identities(root)
    if final_identity not in identities:
        raise ValueError("lineage final cohort has no complete action journal")
    if len(identities) != len(set(identities)):
        raise ValueError("lineage cohort identities are duplicated")
    journal_by_identity: dict[
        ModalLiveCohortIdentity,
        tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]],
    ] = {
        identity: _cohort_action_journal(root, identity) for identity in identities
    }
    final_journal, final_records = journal_by_identity[final_identity]
    terminal_attempt_ids = {record["attempt_id"] for record in final_records}
    terminal_run_ids = {
        run_id
        for record in final_records
        for run_id in record["concrete_remote_run_ids"]
    }
    earliest_final_attempt_started = min(
        _utc(record["started_at_utc"], "final_attempt.started_at_utc")
        for record in final_records
    )
    if not set(final_attempts.values()) <= terminal_attempt_ids:
        raise ValueError("lineage accepted attempts are absent from the final journal")
    if not set(final_runs.values()) <= {
        record["run_id"] for record in final_records if record["run_id"] is not None
    } | terminal_run_ids:
        raise ValueError("lineage accepted runs are absent from the final journal")
    final_evidence, final_metadata = _derive_final_lineage_evidence(
        root,
        identity=final_identity,
        journal=final_journal,
        attempts=final_records,
        accepted_primary_runs=final_runs,
        accepted_attempt_ids=final_attempts,
    )

    prior_records: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for logical in sorted(prior_quarantine_accounting_paths):
        payload, metadata = _load_prior_quarantine_accounting(root, logical)
        prior_records.append((payload, metadata, logical))
    prior_identities = [record[1]["identity"] for record in prior_records]
    expected_prior_identities = sorted(
        (identity for identity in identities if identity != final_identity),
        key=lambda item: (
            item.source_tree_sha256,
            item.image_source_sha256,
            item.cohort_id,
        ),
    )
    if sorted(
        prior_identities,
        key=lambda item: (
            item.source_tree_sha256,
            item.image_source_sha256,
            item.cohort_id,
        ),
    ) != expected_prior_identities:
        raise ValueError("lineage omits or invents a prior live cohort")

    global_bindings, global_records = _scan_global_remote_run_reservations(root)
    referenced_reservations: dict[str, tuple[str, ModalLiveCohortIdentity]] = {}
    attempt_ids: set[str] = set()
    remote_run_ids: set[str] = set()
    for identity, (_journal, records) in journal_by_identity.items():
        for record in records:
            attempt_id = record["attempt_id"]
            if attempt_id in attempt_ids:
                raise ValueError("migration attempt ID is reused across cohorts")
            attempt_ids.add(attempt_id)
            for reservation in record["remote_run_reservations"]:
                run_id = reservation["run_id"]
                if run_id in remote_run_ids:
                    raise ValueError("migration remote run ID is reused")
                remote_run_ids.add(run_id)
                referenced_reservations[reservation["path"]] = (attempt_id, identity)
    if set(referenced_reservations) != set(global_records):
        raise ValueError("global reservation namespace is not journal-complete")
    for logical, (attempt_id, identity) in referenced_reservations.items():
        reservation = global_records[logical]["payload"]
        if (
            reservation["owner_attempt_id"] != attempt_id
            or _cohort_identity_from_payload(reservation) != identity
        ):
            raise ValueError("global reservation owner differs from its journal")

    def reservations_for(identity: ModalLiveCohortIdentity) -> list[dict[str, Any]]:
        return [
            global_records[path]["binding"]
            for path, (_attempt_id_value, selected) in sorted(
                referenced_reservations.items()
            )
            if selected == identity
        ]

    prior_entries = []
    prior_total = Decimal("0")
    prior_provider_total = Decimal("0")
    prior_modal_measured_total = Decimal("0")
    prior_modal_reserve_total = Decimal("0")
    prior_modal_conservative_total = Decimal("0")
    retained_estimates = []
    prior_unique: dict[str, set[str]] = {
        "app_ids": set(),
        "call_ids": set(),
        "function_ids": set(),
        "provider_request_ids": set(),
        "provider_response_ids": set(),
        "artifact_paths": set(),
    }
    image_source_by_id = {
        image_id: final_identity.image_source_sha256
        for image_id in final_metadata["image_ids"]
    }
    prior_billing_row_keys: set[str] = set()
    prior_app_owners: dict[str, ModalLiveCohortIdentity] = {}
    prior_billing_owners: dict[
        str, tuple[str, ModalLiveCohortIdentity]
    ] = {}
    prior_billing_accounting_owners: dict[
        tuple[str, str, datetime, str],
        tuple[str, ModalLiveCohortIdentity, str],
    ] = {}
    prior_run_owners: dict[str, ModalLiveCohortIdentity] = {}
    observed_migration_app_ids: set[str] = set()
    observed_app_cohorts: dict[str, set[ModalLiveCohortIdentity]] = {}
    observed_migration_billing_rows: dict[str, dict[str, Any]] = {}
    observed_billing_cohorts: dict[str, set[ModalLiveCohortIdentity]] = {}
    observed_billing_accounting_rows: dict[
        tuple[str, str, datetime, str], tuple[str, dict[str, Any]]
    ] = {}
    observed_billing_accounting_cohorts: dict[
        tuple[str, str, datetime, str], set[ModalLiveCohortIdentity]
    ] = {}
    observed_volume_run_ids: set[str] = set()
    observed_run_cohorts: dict[str, set[ModalLiveCohortIdentity]] = {}
    for payload, metadata, logical in prior_records:
        identity = metadata["identity"]
        if metadata["snapshot_finished_at"] >= earliest_final_attempt_started:
            raise ValueError(
                "prior snapshot does not finish before the final cohort starts"
            )
        prior_total += metadata["subtotal"]
        retained_estimates.append(payload["retained_storage_estimate"])
        for field, observed in prior_unique.items():
            selected = metadata[field]
            if final_metadata[field].intersection(selected):
                raise ValueError(
                    f"migration final and prior {field} are reused"
                )
            if observed.intersection(selected):
                raise ValueError(f"migration prior {field} are reused across cohorts")
            observed.update(selected)
        for image_id in metadata["image_ids"]:
            previous_source = image_source_by_id.setdefault(
                image_id,
                identity.image_source_sha256,
            )
            if previous_source != identity.image_source_sha256:
                raise ValueError(
                    "Modal image ID maps to conflicting image source digests"
                )
        prior_provider_total += metadata["provider_spend_bound"]
        prior_modal_measured_total += metadata["modal_measured_billing"]
        prior_modal_reserve_total += metadata["modal_unresolved_reserve"]
        prior_modal_conservative_total += metadata[
            "modal_conservative_exposure"
        ]
        if prior_billing_row_keys.intersection(metadata["billing_row_keys"]):
            raise ValueError(
                "migration billing row is owned by multiple prior cohorts"
            )
        prior_billing_row_keys.update(metadata["billing_row_keys"])
        for app_id in metadata["app_ids"]:
            previous_owner = prior_app_owners.setdefault(app_id, identity)
            if previous_owner != identity:
                raise ValueError("migration prior App is owned by multiple cohorts")
        for run_id in metadata["run_ids"]:
            previous_owner = prior_run_owners.setdefault(run_id, identity)
            if previous_owner != identity:
                raise ValueError("migration prior Volume run has multiple owners")
        selected_billing = {
            record["row_sha256"]: (record["app_id"], record["row"])
            for record in payload["selected_billing_rows"]
        }
        if set(selected_billing) != metadata["billing_row_keys"]:
            raise ValueError("prior selected billing ownership metadata changed")
        for row_key, (app_id, row) in selected_billing.items():
            if prior_app_owners.get(app_id) != identity:
                raise ValueError("prior billing row differs from its App owner")
            previous_owner = prior_billing_owners.setdefault(
                row_key,
                (app_id, identity),
            )
            if previous_owner != (app_id, identity):
                raise ValueError(
                    "migration billing row is owned by multiple prior cohorts"
                )
            accounting_key = _modal_billing_accounting_key(
                row,
                field="prior_selected_billing",
            )
            previous_accounting_owner = prior_billing_accounting_owners.setdefault(
                accounting_key,
                (app_id, identity, row_key),
            )
            if previous_accounting_owner != (app_id, identity, row_key):
                raise ValueError(
                    "migration billing charge is owned by multiple prior cohorts"
                )
        observed_migration_app_ids.update(
            metadata["observed_migration_app_ids"]
        )
        for app_id in metadata["observed_migration_app_ids"]:
            observed_app_cohorts.setdefault(app_id, set()).add(identity)
        observed_volume_run_ids.update(metadata["observed_volume_run_ids"])
        for run_id in metadata["observed_volume_run_ids"]:
            observed_run_cohorts.setdefault(run_id, set()).add(identity)
        for row_key, row in metadata["observed_migration_billing_rows"].items():
            previous = observed_migration_billing_rows.setdefault(row_key, row)
            if not exact_json_equal(previous, row):
                raise ValueError("migration billing row digest is ambiguous")
            observed_billing_cohorts.setdefault(row_key, set()).add(identity)
            accounting_key = _modal_billing_accounting_key(
                row,
                field="prior_observed_billing",
            )
            previous_accounting = observed_billing_accounting_rows.setdefault(
                accounting_key,
                (row_key, row),
            )
            if previous_accounting[0] != row_key or not exact_json_equal(
                previous_accounting[1],
                row,
            ):
                raise ValueError(
                    "overlapping prior snapshots conflict on one billing charge"
                )
            observed_billing_accounting_cohorts.setdefault(
                accounting_key,
                set(),
            ).add(identity)
        prior_entries.append(
            {
                "identity": modal_cohort_identity_dict(identity),
                "disposition": "quarantined",
                "accounting_receipt": {
                    "path": logical,
                    "sha256": metadata["raw_sha256"],
                    "size_bytes": metadata["size_bytes"],
                },
                "action_journal": journal_by_identity[identity][0],
                "remote_run_reservations": reservations_for(identity),
                "provider_spend_estimate": payload["provider_spend_estimate"],
                "modal_compute_exposure": payload["modal_compute_exposure"],
            }
        )
    if not observed_migration_app_ids <= set(prior_app_owners):
        raise ValueError("migration snapshot contains an unknown stopped App")
    if any(
        prior_app_owners[app_id] not in observed_app_cohorts[app_id]
        for app_id in observed_migration_app_ids
    ):
        raise ValueError("prior App is absent from its owning cohort snapshot")
    for row_key, row in observed_migration_billing_rows.items():
        app_id = _text(row["object_id"], "migration_billing.object_id")
        app_owner = prior_app_owners.get(app_id)
        if app_owner is None:
            raise ValueError("migration snapshot contains unknown App billing")
        if (
            _text(row["description"], "migration_billing.description")
            != APP_NAME
            or _text(row["environment"], "migration_billing.environment")
            != MODAL_ENVIRONMENT
        ):
            raise ValueError("migration billing row has the wrong scope")
        cost = _decimal_text(row["cost"], "migration_billing.cost")
        accounting_key = _modal_billing_accounting_key(
            row,
            field="migration_billing",
        )
        if cost > 0:
            if prior_billing_owners.get(row_key) != (app_id, app_owner):
                raise ValueError(
                    "migration billing row is not selected by its owning prior cohort"
                )
            if app_owner not in observed_billing_cohorts[row_key]:
                raise ValueError(
                    "migration billing row is absent from its owning cohort snapshot"
                )
            if prior_billing_accounting_owners.get(accounting_key) != (
                app_id,
                app_owner,
                row_key,
            ):
                raise ValueError(
                    "migration billing charge is not selected by its App owner"
                )
            if app_owner not in observed_billing_accounting_cohorts[accounting_key]:
                raise ValueError(
                    "migration billing charge is absent from its owner snapshot"
                )
        elif row_key in prior_billing_owners:
            if prior_billing_owners[row_key] != (app_id, app_owner):
                raise ValueError(
                    "migration zero-cost billing row differs from its App owner"
                )
            if app_owner not in observed_billing_cohorts[row_key]:
                raise ValueError(
                    "selected zero-cost billing row is absent from its owner snapshot"
                )
    if not observed_volume_run_ids <= set(prior_run_owners):
        raise ValueError("migration snapshot contains an unknown Volume run")
    if any(
        prior_run_owners[run_id] not in observed_run_cohorts[run_id]
        for run_id in observed_volume_run_ids
    ):
        raise ValueError("prior Volume run is absent from its owning cohort snapshot")
    final_provider_total = _decimal_text(
        final_evidence["provider_spend_estimate"][
            "conservative_provider_spend_bound_usd"
        ],
        "lineage.final_provider_spend_bound_usd",
    )
    return {
        "selected_final": {
            "identity": modal_cohort_identity_dict(final_identity),
            "accepted_primary_runs": final_runs,
            "accepted_attempt_ids": final_attempts,
            "action_journal": final_journal,
            "remote_run_reservations": reservations_for(final_identity),
            **final_evidence,
        },
        "prior_quarantined_cohorts": prior_entries,
        "global_remote_run_reservations": global_bindings,
        "legacy_superseded_usage": {
            "run_id": _SUPERSEDED_RUN_ID,
            "amount_usd": format(_SUPERSEDED_USAGE_USD, "f"),
            "accounting_basis": (
                "preserved_legacy_measurement_excluded_from_all_cohort_snapshots"
            ),
        },
        "prior_app_compute_total_usd": format(prior_total, "f"),
        "final_provider_spend_bound_usd": format(final_provider_total, "f"),
        "prior_provider_spend_bound_usd": format(prior_provider_total, "f"),
        "migration_provider_spend_bound_usd": format(
            final_provider_total + prior_provider_total,
            "f",
        ),
        "prior_modal_measured_app_billing_usd": format(
            prior_modal_measured_total, "f"
        ),
        "prior_modal_unresolved_compute_reserve_usd": format(
            prior_modal_reserve_total, "f"
        ),
        "prior_modal_conservative_exposure_usd": format(
            prior_modal_conservative_total, "f"
        ),
        "retained_storage_estimate": {
            "prior_cohort_estimates": retained_estimates,
            "final_cohort_included": False,
            "basis": (
                "prior_quarantine_receipts_only; final retained storage is "
                "reported by the cleanup receipt"
            ),
        },
        "global_uniqueness_validated": True,
    }


def create_modal_migration_lineage(
    *,
    final_identity: ModalLiveCohortIdentity,
    accepted_primary_runs: Mapping[str, str],
    accepted_attempt_ids: Mapping[str, str],
    prior_quarantine_accounting_paths: list[str] | None = None,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create the immutable final-cohort lineage seal."""

    project_root = Path(root)
    prior_paths = list(prior_quarantine_accounting_paths or [])
    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        initial_scan = _scan_resolved_modal_global_action_journal(
            lock_descriptor
        )
        if any(cohort.sealed for cohort in initial_scan.cohorts):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal already exists"
            )
        derived = _derive_migration_lineage_claims(
            project_root,
            final_identity=final_identity,
            accepted_primary_runs=accepted_primary_runs,
            accepted_attempt_ids=accepted_attempt_ids,
            prior_quarantine_accounting_paths=prior_paths,
        )
        rejection_logical = modal_global_launch_rejection_seal_path().as_posix()
        rejection_output = project_root.resolve().joinpath(
            *modal_global_launch_rejection_seal_path().parts
        )
        if initial_scan.global_rejection_seal is None:
            rejection_payload = build_modal_global_launch_rejection_seal_payload(
                initial_scan,
                recorded_at_utc=_utc_z(datetime.now(UTC)),
            )
            assert_modal_action_lock_identity(lock_descriptor)
            create_json_exclusive(rejection_output, rejection_payload)
        else:
            if initial_scan.global_rejection_seal.binding.path != rejection_logical:
                raise ModalActionJournalIntegrityError(
                    "global launch-rejection seal path is not canonical"
                )
            rejection_payload = json_value(
                dict(initial_scan.global_rejection_seal.payload)
            )
            if not isinstance(rejection_payload, dict):  # pragma: no cover
                raise ModalActionJournalIntegrityError(
                    "global launch-rejection seal is not an object"
                )

        sealed_scan = _scan_resolved_modal_global_action_journal(
            lock_descriptor
        )
        if any(cohort.sealed for cohort in sealed_scan.cohorts):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal appeared before publication"
            )
        observed_rejection = sealed_scan.global_rejection_seal
        if (
            observed_rejection is None
            or observed_rejection.binding.path != rejection_logical
            or not exact_json_equal(
                observed_rejection.payload,
                rejection_payload,
            )
        ):
            raise ModalActionJournalIntegrityError(
                "global launch-rejection seal changed before lineage publication"
            )
        repeated = _derive_migration_lineage_claims(
            project_root,
            final_identity=final_identity,
            accepted_primary_runs=accepted_primary_runs,
            accepted_attempt_ids=accepted_attempt_ids,
            prior_quarantine_accounting_paths=prior_paths,
        )
        if not exact_json_equal(repeated, derived):
            raise ModalActionJournalIntegrityError(
                "migration lineage claims changed while sealing launches"
            )
        rejection_recorded_at = _utc(
            rejection_payload["recorded_at_utc"],
            "global_launch_rejection_seal.recorded_at_utc",
        )
        payload = {
            "schema_name": "ModalMigrationLineage",
            "schema_version": "1.1",
            "recorded_at_utc": _utc_z(
                max(datetime.now(UTC), rejection_recorded_at)
            ),
            **derived,
            "validated": True,
        }
        output = project_root.resolve().joinpath(
            *modal_migration_lineage_path(final_identity).parts
        )
        assert_modal_action_lock_identity(lock_descriptor)
        create_json_exclusive(output, payload)
        final_scan = _scan_resolved_modal_global_action_journal(
            lock_descriptor
        )
        observed_rejection = final_scan.global_rejection_seal
        sealed_cohorts = [
            cohort for cohort in final_scan.cohorts if cohort.sealed
        ]
        if (
            observed_rejection is None
            or observed_rejection.binding.path != rejection_logical
            or not exact_json_equal(
                observed_rejection.payload,
                rejection_payload,
            )
            or len(sealed_cohorts) != 1
            or sealed_cohorts[0].identity != final_identity
            or sealed_cohorts[0].migration_terminal_seal is None
            or sealed_cohorts[0].migration_terminal_seal.binding.path
            != modal_migration_lineage_path(final_identity).as_posix()
            or not exact_json_equal(
                sealed_cohorts[0].migration_terminal_seal.payload,
                payload,
            )
        ):
            raise ModalActionJournalIntegrityError(
                "global journal does not contain exactly the expected terminal seals"
            )
        persisted, _raw_sha256 = _load_object_with_sha256(output)
        rederived = _derive_migration_lineage_claims(
            project_root,
            final_identity=final_identity,
            accepted_primary_runs=accepted_primary_runs,
            accepted_attempt_ids=accepted_attempt_ids,
            prior_quarantine_accounting_paths=prior_paths,
        )
        if not exact_json_equal(persisted, payload) or not all(
            exact_json_equal(persisted[field], expected)
            for field, expected in rederived.items()
        ):
            raise ModalActionJournalIntegrityError(
                "persisted migration lineage changed after creation"
            )
        assert_modal_action_lock_identity(lock_descriptor)
        return persisted
    finally:
        release_modal_action_lock(lock_descriptor)


def create_modal_migration_lineage_from_input(
    *,
    payload: Mapping[str, Any],
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Validate one reviewed lineage-selection input and publish its seal."""

    if not isinstance(payload, Mapping):
        raise TypeError("migration lineage input must be a mapping")
    frozen, _encoded = _exclusive_json_object_bytes(payload)
    if set(frozen) != _MIGRATION_LINEAGE_INPUT_FIELDS:
        raise ValueError("migration lineage input has an invalid exact schema")
    if (
        frozen["schema_name"] != "ModalMigrationLineageInput"
        or frozen["schema_version"] != "1.0"
    ):
        raise ValueError("migration lineage input contract drifted")
    identity = _cohort_identity_from_payload(
        frozen,
        field="migration_lineage_input",
    )
    accepted_runs = frozen["accepted_primary_runs"]
    accepted_attempts = frozen["accepted_attempt_ids"]
    if not isinstance(accepted_runs, dict) or set(accepted_runs) != set(
        _PRIMARY_LABELS
    ):
        raise ValueError("migration lineage input accepted runs are not exact")
    if not isinstance(accepted_attempts, dict) or set(accepted_attempts) != set(
        _PRIMARY_LABELS
    ):
        raise ValueError("migration lineage input accepted attempts are not exact")
    for label in _PRIMARY_LABELS:
        validate_run_id(accepted_runs[label])
        _attempt_id(
            accepted_attempts[label],
            f"migration_lineage_input.accepted_attempt_ids.{label}",
        )
    prior_paths = frozen["prior_quarantine_accounting_paths"]
    if (
        not isinstance(prior_paths, list)
        or any(not isinstance(item, str) for item in prior_paths)
        or prior_paths != sorted(set(prior_paths))
    ):
        raise ValueError(
            "migration lineage input prior accounting paths must be sorted and unique"
        )
    for logical in prior_paths:
        safe_relative_path(logical)
    return create_modal_migration_lineage(
        final_identity=identity,
        accepted_primary_runs=accepted_runs,
        accepted_attempt_ids=accepted_attempts,
        prior_quarantine_accounting_paths=prior_paths,
        root=root,
    )


def _load_migration_lineage(
    root: Path,
    roster: Mapping[str, Any],
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, Any], Path, str]:
    logical = _text(roster["migration_lineage_path"], "migration_lineage_path")
    expected_path = modal_migration_lineage_path(identity).as_posix()
    if logical != expected_path:
        raise ValueError("migration lineage path is not canonical")
    path = _contained_path(root, logical, "migration_lineage_path", kind="file")
    payload, raw_sha256 = _load_object_with_sha256(path)
    if raw_sha256 != _sha256(
        roster["migration_lineage_sha256"],
        "migration_lineage_sha256",
    ):
        raise ValueError("migration lineage raw digest changed")
    if set(payload) != _MIGRATION_LINEAGE_FIELDS:
        raise ValueError("migration lineage has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalMigrationLineage"
        or payload["schema_version"] != "1.1"
    ):
        raise ValueError("migration lineage contract drifted")
    _utc(payload["recorded_at_utc"], "migration_lineage.recorded_at_utc")
    _exact_bool(payload["validated"], "migration_lineage.validated")
    selected = payload["selected_final"]
    if not isinstance(selected, dict) or set(selected) != (
        _LINEAGE_SELECTED_FINAL_FIELDS
    ):
        raise ValueError("migration lineage final selection schema drifted")
    prior = payload["prior_quarantined_cohorts"]
    if not isinstance(prior, list) or any(
        not isinstance(item, dict) or set(item) != _LINEAGE_PRIOR_FIELDS
        for item in prior
    ):
        raise ValueError("migration lineage prior cohort schema drifted")
    prior_paths = [item["accounting_receipt"]["path"] for item in prior]
    expected = _derive_migration_lineage_claims(
        root,
        final_identity=identity,
        accepted_primary_runs=roster["accepted_primary_runs"],
        accepted_attempt_ids=roster["accepted_attempt_ids"],
        prior_quarantine_accounting_paths=prior_paths,
    )
    for field, expected_value in expected.items():
        if not exact_json_equal(payload[field], expected_value):
            raise ValueError(f"migration lineage field {field} changed")
    return payload, path, raw_sha256


def _load_price_basis(
    root: Path,
    logical: str,
) -> tuple[dict[str, Any], Path, str]:
    safe_relative_path(logical)
    path = _contained_path(root, logical, "provider_price_basis_path", kind="file")
    payload, raw_sha256 = _load_object_with_sha256(path)
    if set(payload) != _PRICE_BASIS_FIELDS:
        raise ValueError("provider price basis has an invalid exact schema")
    if (
        payload["schema_name"] != "ProviderPriceBasis"
        or payload["schema_version"] != "1.0"
        or payload["model"] != TARGET_MODEL
    ):
        raise ValueError("provider price basis has the wrong contract")
    source_url = _text(payload["official_source_url"], "official_source_url")
    if re.fullmatch(r"https://(?:platform\.)?openai\.com/[^\s]*", source_url) is None:
        raise ValueError("provider price basis must cite an official OpenAI HTTPS URL")
    _utc(payload["retrieved_at_utc"], "price_basis.retrieved_at_utc")
    input_rate = _decimal_text(
        payload["uncached_input_usd_per_million_tokens"],
        "price_basis.uncached_input_usd_per_million_tokens",
    )
    output_rate = _decimal_text(
        payload["output_usd_per_million_tokens"],
        "price_basis.output_usd_per_million_tokens",
    )
    _decimal_text(
        payload["per_request_fee_usd"],
        "price_basis.per_request_fee_usd",
    )
    if input_rate <= 0 or output_rate <= 0:
        raise ValueError("provider token rates must be strictly positive")
    return payload, path, raw_sha256


def _load_provider_approval_plan(
    root: Path,
    logical: object,
    *,
    expected_approval_sha256: object,
    expected_image_source_sha256: object,
    expected_identity: ModalLiveCohortIdentity,
    expected_preflight_binding: Mapping[str, str],
    expected_evolution_spec: str | None = None,
) -> tuple[dict[str, Any], Path]:
    path = _contained_path(root, logical, "provider_approval_plan_path", kind="file")
    plan = _load_object(path)
    if plan.get("schema_name") == "EvolutionProviderApprovalPlan":
        approval_sha256 = verify_evolution_approval_plan(plan)
        if approval_sha256 != _sha256(
            expected_approval_sha256, "approval_plan_sha256"
        ):
            raise ValueError("evolution approval digest differs from the intent")
        if expected_evolution_spec is None:
            raise ValueError("evolution approval lacks its intent specification")
        spec = EvolutionRunSpec.parse(expected_evolution_spec)
        if (
            plan.get("schema_version") != "1.0"
            or plan.get("action") != EVOLUTION_ACTION
            or plan.get("evolution_spec") != spec.token
            or plan.get("source_tree_sha256")
            != expected_identity.source_tree_sha256
            or plan.get("image_source_sha256")
            != _sha256(
                expected_image_source_sha256,
                "provider_plan.image_source_sha256",
            )
            or plan.get("cohort_id") != expected_identity.cohort_id
            or plan.get("candidate_resume_preflight_receipt")
            != {
                "path": expected_preflight_binding["path"],
                "sha256": expected_preflight_binding["sha256"],
            }
            or plan.get("provider_calls_started") != 0
            or plan.get("modal_calls_started") != 0
            or plan.get("openai_clients_initialized") != 0
        ):
            raise ValueError("evolution approval identity is invalid")
        try:
            expected = build_evolution_approval_plan(
                root,
                source_tree_sha256=expected_identity.source_tree_sha256,
                cohort_id=expected_identity.cohort_id,
                candidate_resume_preflight_receipt_path=(
                    expected_preflight_binding["path"]
                ),
                candidate_resume_preflight_receipt_sha256=(
                    expected_preflight_binding["sha256"]
                ),
                evolution_spec=spec.token,
            )
        except ValueError:
            expected = None
        if expected is not None and not exact_json_equal(plan, expected):
            raise ValueError("evolution approval differs from current source")
        return plan, path
    if plan.get("schema_name") == "OpenEvolve60ProviderApprovalPlan":
        approval_sha256 = verify_openevolve_60_approval_plan(plan)
        if approval_sha256 != _sha256(
            expected_approval_sha256,
            "approval_plan_sha256",
        ):
            raise ValueError("OpenEvolve 60 approval digest differs from the intent")
        if (
            plan.get("schema_version") != "1.0"
            or plan.get("action") != OPENEVOLVE_60_ACTION
            or plan.get("source_tree_sha256")
            != expected_identity.source_tree_sha256
            or plan.get("image_source_sha256")
            != _sha256(
                expected_image_source_sha256,
                "provider_plan.image_source_sha256",
            )
            or plan.get("cohort_id") != expected_identity.cohort_id
            or plan.get("candidate_resume_preflight_receipt")
            != {
                "path": expected_preflight_binding["path"],
                "sha256": expected_preflight_binding["sha256"],
            }
            or plan.get("provider_calls_started") != 0
            or plan.get("modal_calls_started") != 0
            or plan.get("openai_clients_initialized") != 0
        ):
            raise ValueError("OpenEvolve 60 approval identity is invalid")
        try:
            expected = build_openevolve_60_approval_plan(
                root,
                source_tree_sha256=expected_identity.source_tree_sha256,
                cohort_id=expected_identity.cohort_id,
                candidate_resume_preflight_receipt_path=(
                    expected_preflight_binding["path"]
                ),
                candidate_resume_preflight_receipt_sha256=(
                    expected_preflight_binding["sha256"]
                ),
            )
        except ValueError:
            expected = None
        if expected is not None and not exact_json_equal(plan, expected):
            raise ValueError("OpenEvolve 60 approval differs from current source")
        return plan, path
    approval_sha256 = verify_provider_canary_approval_plan(plan)
    if approval_sha256 != _sha256(
        expected_approval_sha256,
        "approval_plan_sha256",
    ):
        raise ValueError("provider approval plan digest differs from the intent")
    expected_image = _sha256(
        expected_image_source_sha256,
        "provider_plan.image_source_sha256",
    )
    if (
        plan.get("schema_name") != "ProviderCanaryApprovalPlan"
        or plan.get("schema_version") != "1.2"
        or plan.get("source_tree_sha256") != expected_identity.source_tree_sha256
        or plan.get("image_source_sha256") != expected_image
        or plan.get("cohort_id") != expected_identity.cohort_id
        or plan.get("candidate_resume_preflight_receipt")
        != {
            "path": expected_preflight_binding["path"],
            "sha256": expected_preflight_binding["sha256"],
        }
        or plan.get("harness_order") != list(CANARY_ORDER)
        or plan.get("provider_calls_started") != 0
        or type(plan.get("provider_calls_started")) is not int
        or plan.get("modal_calls_started") != 0
        or type(plan.get("modal_calls_started")) is not int
        or plan.get("openai_clients_initialized") != 0
        or type(plan.get("openai_clients_initialized")) is not int
        or plan.get("claim_scope") != "cost_free_pre_request_approval_only"
        or plan.get("provider")
        != {
            "identity": "openai_official",
            "api_mode": API_MODE,
            "api_endpoint": OFFICIAL_OPENAI_API_BASE,
            "model": TARGET_MODEL,
        }
    ):
        raise ValueError("provider approval plan identity is invalid")
    harnesses = plan.get("harnesses")
    if not isinstance(harnesses, list) or len(harnesses) != len(CANARY_ORDER):
        raise ValueError("provider approval plan harness roster is invalid")
    by_harness: dict[str, dict[str, Any]] = {}
    for item in harnesses:
        if not isinstance(item, dict) or item.get("harness") not in CANARY_ORDER:
            raise ValueError("provider approval plan harness entry is invalid")
        harness = item["harness"]
        first = item.get("first_opportunity")
        settings = item.get("request_settings")
        if (
            harness in by_harness
            or item.get("api_endpoint") != OFFICIAL_OPENAI_API_BASE
            or item.get("model") != TARGET_MODEL
            or item.get("maximum_attempts") != 1
            or type(item.get("maximum_attempts")) is not int
            or not isinstance(first, dict)
            or not isinstance(settings, dict)
        ):
            raise ValueError("provider approval plan harness contract is invalid")
        _exact_int(
            first.get("conservative_input_token_ceiling"),
            f"provider_plan.{harness}.input_ceiling",
            minimum=1,
        )
        _exact_int(
            settings.get("max_completion_tokens"),
            f"provider_plan.{harness}.completion_ceiling",
            minimum=1,
        )
        by_harness[harness] = item
    if list(by_harness) != list(CANARY_ORDER):
        raise ValueError("provider approval plan harness order changed")
    totals = plan.get("totals")
    expected_totals = {
        "harness_count": len(CANARY_ORDER),
        "maximum_requests": len(CANARY_ORDER),
        "conservative_input_token_ceiling": sum(
            item["first_opportunity"]["conservative_input_token_ceiling"]
            for item in harnesses
        ),
        "requested_completion_token_ceiling": sum(
            item["request_settings"]["max_completion_tokens"]
            for item in harnesses
        ),
    }
    if not isinstance(totals, dict) or not exact_json_equal(
        totals, expected_totals
    ):
        raise ValueError("provider approval plan totals do not reconcile")
    try:
        expected = build_provider_canary_approval_plan(
            root,
            source_tree_sha256=expected_identity.source_tree_sha256,
            cohort_id=expected_identity.cohort_id,
            candidate_resume_preflight_receipt_path=(
                expected_preflight_binding["path"]
            ),
            candidate_resume_preflight_receipt_sha256=(
                expected_preflight_binding["sha256"]
            ),
        )
    except ValueError:
        # Unit fixtures use isolated roots. The real project root is fully
        # recomputed; fixtures still receive exact schema/self-hash validation.
        expected = None
    if expected is not None and not exact_json_equal(plan, expected):
        raise ValueError("provider approval plan differs from current source")
    return plan, path


def _failed_verifier_capture(
    root: Path,
    record: Mapping[str, Any],
    *,
    identity: ModalLiveCohortIdentity,
) -> tuple[ExecutionContextV1, str]:
    """Validate the immutable four-file capture for one failed paid verifier."""

    source_run_id = validate_run_id(record["source_run_id"])
    verifier_run_id = validate_run_id(record["verifier_run_id"])
    attempt_id = _attempt_id(
        record.get("verifier_attempt_id", record.get("attempt_id")),
        "verifier_attempt_id",
    )
    expected_logical = (
        modal_artifact_verifier_capture_directory_path(
            identity,
            source_run_id,
            verifier_run_id,
            attempt_id,
        )
        / "artifact_verification_failure.json"
    ).as_posix()
    if record["failure_receipt_path"] != expected_logical:
        raise ValueError("failed verifier receipt path drifted")
    failure_path = _contained_path(
        root,
        expected_logical,
        "additional_artifact_verifier.failure_receipt_path",
        kind="file",
    )
    _sha256(
        record["failure_receipt_sha256"],
        "additional_artifact_verifier.failure_receipt_sha256",
    )
    if record["failure_receipt_sha256"] != _sha256_file(failure_path):
        raise ValueError("failed verifier receipt digest changed")
    capture_root = failure_path.parent
    entries = tuple(capture_root.iterdir())
    if any(item.is_symlink() for item in entries):
        raise ValueError("failed verifier capture may not contain symbolic links")
    if {item.name for item in entries} != set(
        _FAILED_VERIFIER_REMOTE_RECEIPT_ROSTER
    ) or any(not item.is_file() for item in entries):
        raise ValueError("failed verifier capture differs from its exact file roster")

    failure = _load_object(failure_path)
    if set(failure) != _FAILED_VERIFIER_RECEIPT_FIELDS:
        raise ValueError("failed verifier receipt has an invalid exact schema")
    if (
        failure["schema_name"] != "ModalArtifactVerificationFailure"
        or failure["schema_version"] != "1.0"
        or failure["source_run_id"] != source_run_id
        or failure["verifier_run_id"] != verifier_run_id
        or failure["message"]
        != "artifact verification failed; details suppressed"
        or not isinstance(failure["error_type"], str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", failure["error_type"])
        is None
    ):
        raise ValueError("failed verifier receipt identity or sanitization drifted")
    try:
        context = ExecutionContextV1.from_dict(failure["verifier_execution_context"])
        declared_context = ExecutionContextV1.from_dict(
            record["failure_execution_context"]
        )
        context_file = ExecutionContextV1.from_dict(
            _load_object(capture_root / "execution_context.json")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("failed verifier lacks an exact execution context") from error
    if (
        context != declared_context
        or context != context_file
        or context.execution_backend != "modal"
        or context.run_id != verifier_run_id
        or context.app_name != APP_NAME
        or context.function_name != "artifact_verify"
        or context.artifact_uri != volume_artifact_uri(source_run_id)
        or context.modal_app_id is None
        or context.modal_function_id is None
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise ValueError("failed verifier execution identity is incomplete or mixed")
    image_manifest = _image_source_manifest(
        capture_root / "image_source_manifest.json"
    )
    if image_manifest.manifest_sha256 != context.image_source_sha256:
        raise ValueError("failed verifier context differs from its image manifest")
    raw_manifest = load_raw_artifact_manifest(capture_root / "artifact_manifest.json")
    if (
        raw_manifest.filename != "artifact_manifest.json"
        or raw_manifest.manifest.run_id != verifier_run_id
        or raw_manifest.manifest.image_source_sha256 != context.image_source_sha256
        or {item.relative_path for item in raw_manifest.manifest.files}
        != {
            "artifact_verification_failure.json",
            "execution_context.json",
            "image_source_manifest.json",
        }
    ):
        raise ValueError("failed verifier artifact manifest is not identity-bound")
    verification = verify_artifact_manifest(capture_root, raw_manifest.manifest)
    if verification != {
        "run_id": verifier_run_id,
        "file_count": 3,
        "manifest_sha256": raw_manifest.manifest.manifest_sha256,
        "verified": True,
    }:
        raise ValueError("failed verifier capture did not verify exactly")
    return context, raw_manifest.manifest.manifest_sha256


def _successful_verifier_capture(
    root: Path,
    record: Mapping[str, Any],
    *,
    identity: ModalLiveCohortIdentity,
) -> tuple[ArtifactVerificationV1, str]:
    """Validate an exact four-file Volume capture after local launcher failure."""

    source_run_id = validate_run_id(record["source_run_id"])
    verifier_run_id = validate_run_id(record["verifier_run_id"])
    attempt_id = _attempt_id(
        record.get("verifier_attempt_id", record.get("attempt_id")),
        "verifier_attempt_id",
    )
    expected_logical = (
        modal_artifact_verifier_capture_directory_path(
            identity,
            source_run_id,
            verifier_run_id,
            attempt_id,
        )
        / "artifact_verification_result.json"
    ).as_posix()
    if record["remote_verification_path"] != expected_logical:
        raise ValueError("captured successful verifier result path drifted")
    result_path = _contained_path(
        root,
        expected_logical,
        "additional_artifact_verifier.remote_verification_path",
        kind="file",
    )
    _sha256(
        record["remote_verification_sha256"],
        "additional_artifact_verifier.remote_verification_sha256",
    )
    if record["remote_verification_sha256"] != _sha256_file(result_path):
        raise ValueError("captured successful verifier result digest changed")
    capture_root = result_path.parent
    entries = tuple(capture_root.iterdir())
    if any(item.is_symlink() for item in entries):
        raise ValueError("successful verifier capture may not contain symbolic links")
    if {item.name for item in entries} != set(
        _VERIFIER_REMOTE_RECEIPT_ROSTER
    ) or any(not item.is_file() for item in entries):
        raise ValueError(
            "successful verifier capture differs from its exact file roster"
        )
    try:
        verification = ArtifactVerificationV1.from_dict(_load_object(result_path))
        declared_context = ExecutionContextV1.from_dict(
            record["verifier_execution_context"]
        )
        context_file = ExecutionContextV1.from_dict(
            _load_object(capture_root / "execution_context.json")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("successful verifier capture context is invalid") from error
    context = verification.verifier_execution_context
    if (
        verification.source_run_id != source_run_id
        or verification.verifier_run_id != verifier_run_id
        or verification.verified is not True
        or context != declared_context
        or context != context_file
        or context.execution_backend != "modal"
        or context.run_id != verifier_run_id
        or context.app_name != APP_NAME
        or context.function_name != "artifact_verify"
        or context.artifact_uri != volume_artifact_uri(source_run_id)
        or context.modal_app_id is None
        or context.modal_function_id is None
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise ValueError("successful verifier capture is not execution-bound")
    image_manifest = _image_source_manifest(
        capture_root / "image_source_manifest.json"
    )
    if image_manifest.manifest_sha256 != context.image_source_sha256:
        raise ValueError("successful verifier context differs from its image manifest")
    raw_manifest = load_raw_artifact_manifest(capture_root / "artifact_manifest.json")
    if (
        raw_manifest.filename != "artifact_manifest.json"
        or raw_manifest.manifest.run_id != verifier_run_id
        or raw_manifest.manifest.image_source_sha256 != context.image_source_sha256
        or {item.relative_path for item in raw_manifest.manifest.files}
        != {
            "artifact_verification_result.json",
            "execution_context.json",
            "image_source_manifest.json",
        }
    ):
        raise ValueError("successful verifier capture manifest is not identity-bound")
    checked = verify_artifact_manifest(capture_root, raw_manifest.manifest)
    if checked["verified"] is not True or checked["file_count"] != 3:
        raise ValueError("successful verifier capture did not verify exactly")
    return verification, raw_manifest.manifest.manifest_sha256


def _load_provider_start_uncertain_evidence(
    root: Path,
    logical: object,
    expected_sha256: object,
    *,
    harness: str,
    run_id: str,
    expected_modal_call_id: str | None = None,
) -> tuple[dict[str, Any], Path]:
    expected_logical = (
        f"{_expected_download_path(run_id)}/controller/"
        "provider_request_start_uncertain.json"
    )
    if logical != expected_logical:
        raise ValueError("provider start-uncertainty evidence path drifted")
    path = _contained_path(
        root,
        logical,
        "provider_start_uncertain_evidence_path",
        kind="file",
    )
    evidence, raw_sha256 = _load_object_with_sha256(path)
    if raw_sha256 != _sha256(
        expected_sha256,
        "provider_start_uncertain_evidence_sha256",
    ):
        raise ValueError("provider start-uncertainty evidence digest changed")
    if set(evidence) != _PROVIDER_START_UNCERTAIN_FIELDS:
        raise ValueError(
            "provider start-uncertainty evidence has an invalid exact schema"
        )
    lower_bound = _exact_int(
        evidence["provider_attempt_count_lower_bound"],
        "provider_start_uncertain.provider_attempt_count_lower_bound",
    )
    upper_bound = _exact_int(
        evidence["provider_attempt_count_upper_bound"],
        "provider_start_uncertain.provider_attempt_count_upper_bound",
    )
    modal_call_id = _text(
        evidence["modal_call_id"],
        "provider_start_uncertain.modal_call_id",
    )
    if (
        evidence["schema_name"] != "ProviderRequestStartUncertainEvidence"
        or evidence["schema_version"] != "1.0"
        or evidence["harness"] != harness
        or evidence["action"] != "one_opportunity_engineering_canary"
        or evidence["execution_backend"] != "modal"
        or evidence["action_run_id"] != run_id
        or evidence["api_endpoint"] != OFFICIAL_OPENAI_API_BASE
        or evidence["model"] != TARGET_MODEL
        or lower_bound != 0
        or upper_bound != 1
        or evidence["provider_request_started"] != "unknown"
        or evidence["provider_attempt_ledger_state"] not in {"missing", "present"}
        or evidence["billing_treatment"]
        != "reserve_one_full_approved_request"
        or evidence["reason"]
        != "controller_terminated_without_terminal_attempt_record"
        or (
            expected_modal_call_id is not None
            and modal_call_id != expected_modal_call_id
        )
    ):
        raise ValueError("provider start-uncertainty evidence is invalid")
    return evidence, path


def _provider_ledger_capture(
    root: Path,
    logical: str,
    expected_sha256: object,
    *,
    outcome: str,
) -> tuple[Path, str]:
    """Bind a provider ledger as present, or as typed uncertainty-only missing."""

    path = _contained_path(
        root,
        logical,
        "provider_canary_outcome.provider_attempt_ledger_path",
        kind="optional",
    )
    if path.exists() and not path.is_file():
        raise ValueError("provider attempt ledger is not a regular file")
    if path.is_file():
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != _sha256(
            expected_sha256,
            "provider_canary_outcome.provider_attempt_ledger_sha256",
        ):
            raise ValueError("provider canary outcome ledger changed")
        return path, "present"
    if outcome != "provider_request_start_uncertain":
        raise ValueError(
            "unresolved_provider_request_state: provider attempt ledger is "
            "missing; readiness cannot infer zero provider starts"
        )
    if expected_sha256 is not None:
        raise ValueError("missing uncertain provider ledger must have a null digest")
    return path, "missing"


def _validate_cohort_roster_payload(
    root: Path,
    logical: str,
    roster: object,
) -> tuple[dict[str, Any], Path]:
    relative = safe_relative_path(logical)
    path = root.resolve().joinpath(*relative.parts)
    if not isinstance(roster, dict):
        raise ValueError("migration cohort roster must be one JSON object")
    if set(roster) != _COHORT_ROSTER_FIELDS:
        raise ValueError("migration cohort roster has an invalid exact schema")
    if (
        roster["schema_name"] != "ModalMigrationCohortRoster"
        or roster["schema_version"] != "4.0"
    ):
        raise ValueError("migration cohort roster has the wrong contract")
    identity = _cohort_identity_from_payload(roster, field="cohort_roster")
    cleanup_run_id = validate_run_id(roster["cleanup_run_id"])
    expected_logical = modal_cohort_roster_path(identity).as_posix()
    if logical != expected_logical:
        raise ValueError("migration cohort roster is not at its frozen path")

    components = roster["component_receipts"]
    if not isinstance(components, dict) or set(components) != set(
        _ROSTER_COMPONENT_KEYS
    ):
        raise ValueError("cohort component receipt roster is not canonical")
    for gate, binding in components.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise ValueError(f"cohort component {gate} has an invalid exact schema")
        component_logical = _text(binding["path"], f"component_receipts.{gate}.path")
        component_sha256 = _sha256(
            binding["sha256"], f"component_receipts.{gate}.sha256"
        )
        if gate == "candidate_resume_preflight_validated":
            expected_parent = (
                modal_live_cohort_root(identity)
                / "components"
                / "candidate_resume_preflight_receipts"
                / "v2.0"
            )
            relative = safe_relative_path(component_logical)
            if (
                relative.parent != expected_parent
                or re.fullmatch(r"[0-9a-f]{64}\.json", relative.name) is None
            ):
                raise ValueError("cohort preflight component path is not canonical")
        else:
            expected_component = modal_component_receipt_path(
                identity, gate
            ).as_posix()
            if component_logical != expected_component:
                raise ValueError(f"cohort component {gate} path drifted")
        component_path = _contained_path(
            root,
            component_logical,
            f"component_receipts.{gate}.path",
            kind="file",
        )
        component_payload, observed_sha256 = _load_object_with_sha256(
            component_path
        )
        if observed_sha256 != component_sha256:
            raise ValueError(f"cohort component {gate} raw digest changed")
        _assert_identity_matches(
            component_payload,
            identity,
            field=f"component_receipts.{gate}",
        )
        expected_contract = (
            {
                "schema_name": "ModalOfflineSmokeValidationReceipt",
                "schema_version": "2.0",
            }
            if gate == "modal_offline_smoke_validated"
            else {
                "schema_name": "CandidateResumePreflightReceipt",
                "schema_version": "2.0",
            }
            if gate == "candidate_resume_preflight_validated"
            else MODAL_READINESS_RECEIPT_CONTRACTS[gate]["receipt_contract"]
        )
        if any(
            component_payload.get(field) != value
            for field, value in expected_contract.items()
        ):
            raise ValueError(f"cohort component {gate} contract drifted")

    accepted = roster["accepted_primary_runs"]
    if not isinstance(accepted, dict) or set(accepted) != set(_PRIMARY_LABELS):
        raise ValueError("cohort accepted-primary roster must use the exact keys")
    accepted_runs = [validate_run_id(value) for value in accepted.values()]
    if len(set(accepted_runs)) != 8:
        raise ValueError("cohort roster requires eight distinct accepted run IDs")
    if cleanup_run_id != accepted["cuda_environment"]:
        raise ValueError("cleanup run ID must be the accepted CUDA environment run")
    selector_logical = _text(
        roster["provider_canary_selector_path"],
        "provider_canary_selector_path",
    )
    selector_path = _contained_path(
        root,
        selector_logical,
        "provider_canary_selector_path",
        kind="file",
    )
    _sha256(
        roster["provider_canary_selector_sha256"],
        "provider_canary_selector_sha256",
    )
    if roster["provider_canary_selector_sha256"] != _sha256_file(selector_path):
        raise ValueError("provider canary selector digest changed")
    selector, _ = load_modal_canary_selector(
        selector_path,
        project_root=root,
        expected_identity=identity,
    )
    if any(
        selector["runs"][harness]["run_id"] != accepted[f"canary_{harness}"]
        for harness in CANARY_ORDER
    ):
        raise ValueError("provider canary selector differs from accepted primary runs")

    accepted_attempts = roster["accepted_attempt_ids"]
    if (
        not isinstance(accepted_attempts, dict)
        or set(accepted_attempts) != set(_PRIMARY_LABELS)
    ):
        raise ValueError("cohort accepted-attempt roster must use the exact keys")
    for value in accepted_attempts.values():
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
            raise ValueError("cohort accepted attempt ID is invalid")
    lineage, _lineage_path, _lineage_sha256 = _load_migration_lineage(
        root, roster, identity
    )

    verifiers = roster["artifact_verifiers"]
    if not isinstance(verifiers, dict) or set(verifiers) != set(_PRIMARY_LABELS):
        raise ValueError("cohort artifact-verifier roster must use the exact keys")
    verifier_run_ids: list[str] = []
    verifier_attempt_ids: list[str] = []
    for label, record in verifiers.items():
        if not isinstance(record, dict) or set(record) != _VERIFIER_ROSTER_FIELDS:
            raise ValueError(f"artifact verifier {label} has an invalid exact schema")
        source = validate_run_id(record["source_run_id"])
        verifier = validate_run_id(record["verifier_run_id"])
        attempt_id = _text(
            record["attempt_id"], f"artifact_verifiers.{label}.attempt_id"
        )
        if (
            record["source_label"] != label
            or source != accepted[label]
            or source == verifier
        ):
            raise ValueError(f"artifact verifier {label} is not source-run bound")
        if re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None:
            raise ValueError(f"artifact verifier {label} attempt ID is invalid")
        remote_logical = _remote_verification_logical(
            identity,
            source,
            verifier,
            attempt_id,
        )
        if record["remote_verification_path"] != remote_logical:
            raise ValueError(f"artifact verifier {label} capture path drifted")
        remote_path = _contained_path(
            root,
            remote_logical,
            f"artifact_verifiers.{label}.remote_verification_path",
            kind="file",
        )
        _sha256(
            record["remote_verification_sha256"],
            f"artifact_verifiers.{label}.remote_verification_sha256",
        )
        if record["remote_verification_sha256"] != _sha256_file(remote_path):
            raise ValueError(f"artifact verifier {label} capture digest changed")
        if record["expected_remote_receipt_roster"] != list(
            _VERIFIER_REMOTE_RECEIPT_ROSTER
        ):
            raise ValueError(f"artifact verifier {label} remote receipt roster drifted")
        try:
            captured = ArtifactVerificationV1.from_dict(_load_object(remote_path))
            context = ExecutionContextV1.from_dict(
                record["verifier_execution_context"]
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"artifact verifier {label} context is invalid") from error
        if (
            captured.source_run_id != source
            or captured.verifier_run_id != verifier
            or captured.verifier_execution_context != context
            or context.run_id != verifier
            or context.function_name != "artifact_verify"
            or context.app_name != APP_NAME
            or context.artifact_uri != volume_artifact_uri(source)
            or context.modal_app_id is None
            or context.modal_function_id is None
            or context.modal_call_id is None
            or context.modal_image_id is None
        ):
            raise ValueError(
                f"artifact verifier {label} capture is not execution-bound"
            )
        verifier_run_ids.append(verifier)
        verifier_attempt_ids.append(attempt_id)
    if len(set(verifier_run_ids)) != 8 or len(set(verifier_attempt_ids)) != 8:
        raise ValueError("artifact verifier run and attempt IDs must be unique")
    if set(verifier_run_ids).intersection(accepted_runs):
        raise ValueError("primary and verifier run IDs must be disjoint")

    additional_verifiers = roster["additional_artifact_verifiers"]
    if not isinstance(additional_verifiers, list):
        raise ValueError("additional artifact verifiers must be a sorted list")
    additional_keys: list[tuple[str, str, str]] = []
    for index, record in enumerate(additional_verifiers):
        if not isinstance(record, dict) or set(record) != _ADDITIONAL_VERIFIER_FIELDS:
            raise ValueError(
                f"additional_artifact_verifiers[{index}] has an invalid exact schema"
            )
        source = validate_run_id(record["source_run_id"])
        verifier = validate_run_id(record["verifier_run_id"])
        attempt_id = _text(
            record["attempt_id"],
            f"additional_artifact_verifiers[{index}].attempt_id",
        )
        status = _text(
            record["status"],
            f"additional_artifact_verifiers[{index}].status",
        )
        if (
            source == verifier
            or re.fullmatch(r"[0-9a-f]{32}", attempt_id) is None
            or status not in _ACTION_STATUSES
        ):
            raise ValueError("additional artifact verifier identity is invalid")
        billing_ids = _sorted_unique_text(
            record["billing_object_ids"],
            f"additional_artifact_verifiers[{index}].billing_object_ids",
        )
        remote_outcome = _text(
            record["remote_verifier_outcome"],
            f"additional_artifact_verifiers[{index}].remote_verifier_outcome",
        )
        evidence_kind = _text(
            record["remote_evidence_kind"],
            f"additional_artifact_verifiers[{index}].remote_evidence_kind",
        )
        if remote_outcome not in {"success", "failure", "unresolved"}:
            raise ValueError("additional verifier remote outcome is unsupported")
        recovery_attempt_id = record["recovery_verifier_attempt_id"]
        if remote_outcome == "failure":
            recovery_attempt_id = _text(
                recovery_attempt_id,
                (
                    f"additional_artifact_verifiers[{index}]."
                    "recovery_verifier_attempt_id"
                ),
            )
            if re.fullmatch(r"[0-9a-f]{32}", recovery_attempt_id) is None:
                raise ValueError("failed verifier recovery attempt ID is invalid")
        elif recovery_attempt_id is not None:
            raise ValueError("non-failed verifier invents a recovery attempt")
        if status == "succeeded" and remote_outcome != "success":
            raise ValueError("successful launcher lacks successful remote verifier")
        if remote_outcome == "success":
            if not billing_ids:
                raise ValueError(
                    "successful additional verifier lacks cost attribution"
                )
            if record["expected_remote_receipt_roster"] != list(
                _VERIFIER_REMOTE_RECEIPT_ROSTER
            ):
                raise ValueError("successful additional verifier roster drifted")
            if any(
                record[field] is not None
                for field in (
                    "failure_receipt_path",
                    "failure_receipt_sha256",
                    "failure_execution_context",
                )
            ):
                raise ValueError(
                    "successful additional verifier invents failure evidence"
                )
            if evidence_kind == "round_trip_success":
                remote_logical = _remote_verification_logical(
                    identity,
                    source,
                    verifier,
                    attempt_id,
                )
                if record["remote_verification_path"] != remote_logical:
                    raise ValueError("additional verifier capture path drifted")
                remote_path = _contained_path(
                    root,
                    remote_logical,
                    "additional_artifact_verifier.remote_verification_path",
                    kind="file",
                )
                _sha256(
                    record["remote_verification_sha256"],
                    "additional_artifact_verifier.remote_verification_sha256",
                )
                if record["remote_verification_sha256"] != _sha256_file(remote_path):
                    raise ValueError("additional verifier capture digest changed")
                try:
                    captured = ArtifactVerificationV1.from_dict(
                        _load_object(remote_path)
                    )
                    context = ExecutionContextV1.from_dict(
                        record["verifier_execution_context"]
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "additional verifier context is invalid"
                    ) from error
                if (
                    captured.source_run_id != source
                    or captured.verifier_run_id != verifier
                    or captured.verifier_execution_context != context
                    or context.run_id != verifier
                    or context.function_name != "artifact_verify"
                    or context.app_name != APP_NAME
                    or context.artifact_uri != volume_artifact_uri(source)
                ):
                    raise ValueError(
                        "additional verifier capture is not execution-bound"
                    )
            elif evidence_kind == "volume_success_capture":
                _successful_verifier_capture(root, record, identity=identity)
            else:
                raise ValueError("successful remote verifier evidence kind is invalid")
        elif remote_outcome == "failure":
            if status == "succeeded":
                raise ValueError("successful launcher cannot contain remote failure")
            if any(
                record[field] is not None
                for field in (
                    "remote_verification_path",
                    "remote_verification_sha256",
                    "verifier_execution_context",
                )
            ):
                raise ValueError("failed additional verifier invents success evidence")
            if evidence_kind == "volume_failure_capture":
                if not billing_ids:
                    raise ValueError("captured failed verifier lacks cost attribution")
                if record["expected_remote_receipt_roster"] != list(
                    _FAILED_VERIFIER_REMOTE_RECEIPT_ROSTER
                ):
                    raise ValueError("failed additional verifier roster drifted")
                _failed_verifier_capture(root, record, identity=identity)
            else:
                raise ValueError("failed additional verifier evidence kind is invalid")
        else:
            if status == "succeeded" or evidence_kind != "unresolved_remote_identity":
                raise ValueError("unresolved verifier launcher state is invalid")
            if billing_ids:
                raise ValueError("unresolved verifier invents billed object identity")
            if record["expected_remote_receipt_roster"] != [] or any(
                record[field] is not None
                for field in (
                    "remote_verification_path",
                    "remote_verification_sha256",
                    "verifier_execution_context",
                    "failure_receipt_path",
                    "failure_receipt_sha256",
                    "failure_execution_context",
                )
            ):
                raise ValueError("unresolved verifier invents remote capture evidence")
        additional_keys.append((source, verifier, attempt_id))
        verifier_run_ids.append(verifier)
        verifier_attempt_ids.append(attempt_id)
    if additional_keys != sorted(set(additional_keys)):
        raise ValueError("additional artifact verifiers must be sorted and unique")
    if len(verifier_run_ids) != len(set(verifier_run_ids)) or len(
        verifier_attempt_ids
    ) != len(set(verifier_attempt_ids)):
        raise ValueError("artifact verifier IDs are duplicated across cohort maps")
    successful_verifiers_by_attempt = {
        record["attempt_id"]: record for record in verifiers.values()
    } | {
        record["attempt_id"]: record
        for record in additional_verifiers
        if record["remote_verifier_outcome"] == "success"
    }
    for record in additional_verifiers:
        if record["remote_verifier_outcome"] != "failure":
            continue
        recovery = successful_verifiers_by_attempt.get(
            record["recovery_verifier_attempt_id"]
        )
        if (
            recovery is None
            or recovery["source_run_id"] != record["source_run_id"]
            or recovery["verifier_run_id"] == record["verifier_run_id"]
        ):
            raise ValueError(
                "failed paid verifier lacks a source-bound successful verifier retry"
            )

    receipt_paths = _sorted_unique_text(
        roster["action_attempt_receipts"], "action_attempt_receipts"
    )
    attempt_directory = modal_action_attempt_directory(identity)
    for item in receipt_paths:
        relative = safe_relative_path(item)
        if (
            relative.parent != attempt_directory
            or re.fullmatch(r"[0-9a-f]{32}\.json", relative.name) is None
        ):
            raise ValueError(
                "action attempt receipt path is outside the frozen directory"
            )
    receipt_attempt_ids = {PurePosixPath(item).stem for item in receipt_paths}
    intent_paths = _sorted_unique_text(
        roster["action_intent_receipts"], "action_intent_receipts"
    )
    for item in intent_paths:
        relative = safe_relative_path(item)
        if (
            relative.parent != attempt_directory
            or re.fullmatch(r"[0-9a-f]{32}\.intent\.json", relative.name) is None
        ):
            raise ValueError("action intent path is outside the frozen directory")
    aggregate_paths = _sorted_unique_text(
        roster["provider_canary_aggregate_outcome_receipts"],
        "provider_canary_aggregate_outcome_receipts",
    )
    for item in aggregate_paths:
        relative = safe_relative_path(item)
        if (
            relative.parent != attempt_directory
            or re.fullmatch(r"[0-9a-f]{32}\.aggregate\.json", relative.name)
            is None
        ):
            raise ValueError(
                "provider aggregate outcome path is outside the frozen directory"
            )

    provider_outcomes = roster["provider_canary_outcomes"]
    if not isinstance(provider_outcomes, list):
        raise ValueError("provider canary outcomes must be a CANARY_ORDER list")
    outcome_keys: list[tuple[int, str, str]] = []
    accepted_outcome_harnesses: list[str] = []
    accepted_verifier_by_attempt = {
        record["attempt_id"]: record for record in verifiers.values()
    }
    additional_verifier_by_attempt = {
        record["attempt_id"]: record for record in additional_verifiers
    }
    for index, record in enumerate(provider_outcomes):
        if not isinstance(record, dict) or set(record) != (
            _PROVIDER_CANARY_OUTCOME_FIELDS
        ):
            raise ValueError(
                f"provider_canary_outcomes[{index}] has an invalid exact schema"
            )
        harness = record["harness"]
        if harness not in CANARY_ORDER:
            raise ValueError("provider canary outcome harness is not frozen")
        attempt_id = _text(
            record["launcher_attempt_id"],
            f"provider_canary_outcomes[{index}].launcher_attempt_id",
        )
        run_id = validate_run_id(record["concrete_run_id"])
        outcome = _text(
            record["outcome"],
            f"provider_canary_outcomes[{index}].outcome",
        )
        if outcome not in {
            "accepted",
            "failed",
            "completed_unaccepted",
            "provider_request_start_uncertain",
        }:
            raise ValueError("provider canary outcome disposition is unsupported")
        receipt_logical = modal_action_terminal_receipt_path(
            identity, attempt_id
        ).as_posix()
        if (
            record["launcher_attempt_receipt_path"] != receipt_logical
            or receipt_logical not in receipt_paths
        ):
            raise ValueError("provider canary outcome launcher receipt path drifted")
        receipt_path = _contained_path(
            root,
            receipt_logical,
            "provider_canary_outcome.launcher_attempt_receipt_path",
            kind="file",
        )
        _sha256(
            record["launcher_attempt_receipt_sha256"],
            "provider_canary_outcome.launcher_attempt_receipt_sha256",
        )
        if record["launcher_attempt_receipt_sha256"] != _sha256_file(receipt_path):
            raise ValueError("provider canary outcome launcher receipt changed")
        ledger_logical = (
            f"{_expected_download_path(run_id)}/controller/"
            "provider_attempts.jsonl"
        )
        if record["provider_attempt_ledger_path"] != ledger_logical:
            raise ValueError("provider canary outcome ledger path drifted")
        _ledger_path, ledger_state = _provider_ledger_capture(
            root,
            ledger_logical,
            record["provider_attempt_ledger_sha256"],
            outcome=outcome,
        )
        uncertain_path_value = record["provider_start_uncertain_evidence_path"]
        uncertain_sha_value = record["provider_start_uncertain_evidence_sha256"]
        legacy_zero_logical = (
            f"{_expected_download_path(run_id)}/controller/"
            "provider_request_not_started.json"
        )
        uncertain_logical = (
            f"{_expected_download_path(run_id)}/controller/"
            "provider_request_start_uncertain.json"
        )
        legacy_zero_candidate = _contained_path(
            root,
            legacy_zero_logical,
            "legacy_provider_zero_attempt_evidence_path",
            kind="optional",
        )
        uncertain_candidate = _contained_path(
            root,
            uncertain_logical,
            "provider_start_uncertain_evidence_path",
            kind="optional",
        )
        if legacy_zero_candidate.exists():
            raise ValueError(
                "legacy provider zero-attempt evidence is forbidden in the final cohort"
            )
        if outcome == "provider_request_start_uncertain":
            uncertain, _uncertain_path = _load_provider_start_uncertain_evidence(
                root,
                uncertain_path_value,
                uncertain_sha_value,
                harness=harness,
                run_id=run_id,
            )
            if uncertain["provider_attempt_ledger_state"] != ledger_state:
                raise ValueError(
                    "provider start-uncertainty evidence misstates ledger presence"
                )
        elif uncertain_path_value is not None or uncertain_sha_value is not None:
            raise ValueError("provider outcome invents request-state evidence")
        if uncertain_candidate.exists() is not (
            outcome == "provider_request_start_uncertain"
        ):
            raise ValueError(
                "provider request-state evidence differs from its cohort outcome"
            )
        verifier_link = record["artifact_verifier"]
        if not isinstance(verifier_link, dict) or set(verifier_link) != (
            _OUTCOME_VERIFIER_FIELDS
        ):
            raise ValueError("provider canary outcome verifier link is invalid")
        verifier_attempt_id = _text(
            verifier_link["attempt_id"],
            "provider_canary_outcome.artifact_verifier.attempt_id",
        )
        bound_verifier = accepted_verifier_by_attempt.get(
            verifier_attempt_id
        ) or additional_verifier_by_attempt.get(verifier_attempt_id)
        if bound_verifier is None or any(
            verifier_link[field] != bound_verifier[field]
            for field in (
                "verifier_run_id",
                "remote_verification_path",
                "remote_verification_sha256",
            )
        ):
            raise ValueError("provider canary outcome verifier link is swappable")
        if bound_verifier["source_run_id"] != run_id:
            raise ValueError("provider canary outcome verifier used another source")
        if (
            bound_verifier in additional_verifiers
            and bound_verifier["remote_verifier_outcome"] != "success"
        ):
            raise ValueError("provider canary outcome used a failed verifier")
        label = f"canary_{harness}"
        if outcome == "accepted":
            if run_id != accepted[label] or bound_verifier not in verifiers.values():
                raise ValueError("accepted provider outcome is not final-roster bound")
            accepted_outcome_harnesses.append(harness)
        elif run_id == accepted[label]:
            raise ValueError("unaccepted provider outcome appears in accepted roster")
        outcome_keys.append((CANARY_ORDER.index(harness), attempt_id, run_id))
    if outcome_keys != sorted(set(outcome_keys)):
        raise ValueError("provider canary outcomes must be unique in CANARY_ORDER")
    if accepted_outcome_harnesses != list(CANARY_ORDER):
        raise ValueError(
            "provider outcomes require exactly one accepted run per harness"
        )

    classifications = roster["attempt_classifications"]
    if not isinstance(classifications, list):
        raise ValueError("attempt classifications must be a sorted list")
    classification_ids: list[str] = []
    for index, item in enumerate(classifications):
        if not isinstance(item, dict) or set(item) != _ATTEMPT_CLASSIFICATION_FIELDS:
            raise ValueError(
                f"attempt_classifications[{index}] has an invalid exact schema"
            )
        attempt_id = _text(
            item["attempt_id"], f"attempt_classifications[{index}].attempt_id"
        )
        roles = _sorted_unique_text(
            item["roles"], f"attempt_classifications[{index}].roles"
        )
        if not roles or not set(roles) <= _ATTEMPT_ROLES:
            raise ValueError("attempt classification contains unsupported roles")
        classification_ids.append(attempt_id)
    if classification_ids != sorted(receipt_attempt_ids):
        raise ValueError(
            "attempt classifications must cover every final-cohort receipt"
        )

    attributions = roster["billing_attributions"]
    if not isinstance(attributions, list):
        raise ValueError("billing attributions must be a sorted list")
    attribution_ids: list[str] = []
    attributed_object_ids: set[str] = set()
    for index, item in enumerate(attributions):
        if not isinstance(item, dict) or set(item) != _BILLING_ATTRIBUTION_FIELDS:
            raise ValueError(
                f"billing_attributions[{index}] has an invalid exact schema"
            )
        attempt_id = _text(
            item["attempt_id"], f"billing_attributions[{index}].attempt_id"
        )
        disposition = item["disposition"]
        if disposition not in {"billed", "no_remote_start", "start_uncertain"}:
            raise ValueError("billing attribution disposition is unsupported")
        object_ids = _sorted_unique_text(
            item["object_ids"], f"billing_attributions[{index}].object_ids"
        )
        if disposition == "billed" and not object_ids:
            raise ValueError("billing attribution disposition and object IDs disagree")
        if disposition == "no_remote_start" and object_ids:
            raise ValueError("no-start billing attribution invents object IDs")
        overlap = attributed_object_ids.intersection(object_ids)
        if overlap:
            raise ValueError("billing object IDs are attributed more than once")
        attributed_object_ids.update(object_ids)
        attribution_ids.append(attempt_id)
    if attribution_ids != sorted(receipt_attempt_ids):
        raise ValueError("billing attributions must cover every final-cohort receipt")

    links = roster["recovery_links"]
    if not isinstance(links, list):
        raise ValueError("recovery links must be a sorted list")
    link_keys: list[tuple[str, str]] = []
    for index, item in enumerate(links):
        if not isinstance(item, dict) or set(item) != _RECOVERY_LINK_FIELDS:
            raise ValueError(f"recovery_links[{index}] has an invalid exact schema")
        failed = _text(
            item["failed_attempt_id"], f"recovery_links[{index}].failed_attempt_id"
        )
        recovery = _text(
            item["recovery_attempt_id"], f"recovery_links[{index}].recovery_attempt_id"
        )
        recovered = _sorted_unique_text(
            item["recovered_run_ids"],
            f"recovery_links[{index}].recovered_run_ids",
            run_ids=True,
        )
        if (
            failed == recovery
            or not recovered
            or not set(recovered) <= set(accepted_runs)
        ):
            raise ValueError("recovery link is not bound to accepted run IDs")
        link_keys.append((failed, recovery))
    if link_keys != sorted(set(link_keys)):
        raise ValueError("recovery links must be sorted and unique")

    for field in (
        "declared_failed_run_ids",
        "declared_quarantined_run_ids",
        "declared_recovery_run_ids",
    ):
        _sorted_unique_text(roster[field], field, run_ids=True)
    failed_provider_runs = sorted(
        record["concrete_run_id"]
        for record in provider_outcomes
        if record["outcome"]
        in {
            "failed",
            "provider_request_start_uncertain",
        }
    )
    quarantined_provider_runs = sorted(
        record["concrete_run_id"]
        for record in provider_outcomes
        if record["outcome"] != "accepted"
    )
    if not set(failed_provider_runs) <= set(roster["declared_failed_run_ids"]):
        raise ValueError("provider outcomes hide failed concrete run IDs")
    if not set(quarantined_provider_runs) <= set(
        roster["declared_quarantined_run_ids"]
    ):
        raise ValueError("unaccepted provider runs must remain quarantined")
    outcome_verifier_attempts = {
        record["artifact_verifier"]["attempt_id"] for record in provider_outcomes
    }
    required_outcome_verifiers = {
        record["attempt_id"]
        for record in additional_verifiers
        if record["remote_verifier_outcome"] == "success"
        and record["source_run_id"]
        in {outcome["concrete_run_id"] for outcome in provider_outcomes}
        and record["source_run_id"] not in accepted_runs
    }
    if required_outcome_verifiers - outcome_verifier_attempts:
        raise ValueError("additional provider verifier is not outcome-bound")
    start = _utc(roster["billing_window_start_utc"], "billing_window_start_utc")
    end = _utc(roster["billing_window_end_utc"], "billing_window_end_utc")
    captured = _utc(roster["snapshot_captured_at_utc"], "snapshot_captured_at_utc")
    if (
        start.minute
        or start.second
        or start.microsecond
        or end.minute
        or end.second
        or end.microsecond
        or end <= start
        or captured < end
    ):
        raise ValueError("cohort billing window is not a completed hourly interval")
    snapshot_manifest, _manifest_path, _manifest_sha256, snapshot_rows = (
        _load_cleanup_snapshot_capture(root, roster, identity)
    )
    _journal, frozen_attempts = _cohort_action_journal(root, identity)
    _assert_attempts_contained_for_seal(frozen_attempts, field="final_cohort")
    _assert_attempts_finished_before_snapshot(
        frozen_attempts,
        snapshot_manifest,
        field="final_cohort",
    )
    terminal_dispositions = _validate_prior_remote_run_dispositions(
        root,
        identity,
        frozen_attempts,
        roster["terminal_run_dispositions"],
    )
    lineage_dispositions = {
        (record["attempt_id"], record["run_id"]): record
        for record in lineage["selected_final"]["run_dispositions"]
    }
    if set(lineage_dispositions) != set(terminal_dispositions):
        raise ValueError("terminal dispositions differ from migration lineage")
    for key, disposition in terminal_dispositions.items():
        lineage_disposition = lineage_dispositions[key]
        if any(
            lineage_disposition[field] != disposition[field]
            for field in ("execution_disposition", "provider_disposition")
        ):
            raise ValueError("terminal disposition changed after lineage sealing")
    attribution_by_attempt = {
        item["attempt_id"]: item for item in attributions
    }
    for (attempt_id, _run_id), disposition in terminal_dispositions.items():
        attribution = attribution_by_attempt.get(attempt_id)
        if attribution is None:
            raise ValueError(
                "cohort roster does not classify every Modal action terminal "
                "disposition"
            )
        if disposition["snapshot_app_ids"] != attribution["object_ids"]:
            raise ValueError(
                "terminal snapshot App IDs differ from billing attribution"
            )
    snapshot_app_rows = {
        row["app_id"]: row
        for row in snapshot_rows["app_list"]
        if row["description"] == APP_NAME
    }
    for disposition in terminal_dispositions.values():
        for app_id in disposition["snapshot_app_ids"]:
            row = snapshot_app_rows.get(app_id)
            if row is None or row["state"] != "stopped" or row["tasks"] != "0":
                raise ValueError("terminal disposition App did not stop cleanly")
    snapshot_volume_run_ids = {
        validate_run_id(
            PurePosixPath(
                row["filename"].removeprefix("/").removesuffix("/")
            ).parts[1]
        )
        for row in snapshot_rows["run_directory_list"]
        if len(
            PurePosixPath(
                row["filename"].removeprefix("/").removesuffix("/")
            ).parts
        )
        == 2
        and PurePosixPath(
            row["filename"].removeprefix("/").removesuffix("/")
        ).parts[0]
        == "runs"
    }
    for (_attempt_id_value, run_id), disposition in terminal_dispositions.items():
        present = run_id in snapshot_volume_run_ids
        if present is not (disposition["volume_disposition"] == "present_bound"):
            raise ValueError(
                "terminal Volume disposition differs from the cleanup snapshot"
            )
    superseded = roster["superseded_usage"]
    if not isinstance(superseded, dict) or set(superseded) != {
        "run_id",
        "amount_usd",
        "accounting_basis",
    }:
        raise ValueError("superseded usage has an invalid exact schema")
    if (
        superseded["run_id"] != _SUPERSEDED_RUN_ID
        or _decimal_text(superseded["amount_usd"], "superseded_usage.amount_usd")
        != _SUPERSEDED_USAGE_USD
        or superseded["accounting_basis"]
        != "preserved_prior_measurement_excluded_from_cohort_billing_snapshot"
    ):
        raise ValueError("superseded migration usage was not preserved exactly")
    if superseded["run_id"] in accepted_runs:
        raise ValueError("superseded run may not enter the final accepted roster")
    price_logical = _text(
        roster["provider_price_basis_path"], "provider_price_basis_path"
    )
    expected_price_path = modal_provider_price_basis_path(identity).as_posix()
    if price_logical != expected_price_path:
        raise ValueError("provider price basis is not at its frozen cohort path")
    _load_price_basis(root, price_logical)
    return roster, path


def _load_cohort_roster(root: Path, logical: str) -> tuple[dict[str, Any], Path]:
    path = _contained_path(root, logical, "cohort_roster_path", kind="file")
    raw = _read_regular_file_bytes(path, maximum_bytes=_MAX_JSON_OBJECT_BYTES)
    roster = json.loads(
        _decode_utf8(raw, path),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    return _validate_cohort_roster_payload(root, logical, roster)


def create_modal_cohort_roster(
    *,
    payload: Mapping[str, Any],
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Prevalidate and publish one immutable final-cohort roster."""

    if not isinstance(payload, Mapping):
        raise TypeError("Modal cohort roster payload must be a mapping")
    frozen, _encoded = _exclusive_json_object_bytes(payload)
    identity = _cohort_identity_from_payload(frozen, field="cohort_roster")
    logical = modal_cohort_roster_path(identity).as_posix()
    project_root = Path(root)
    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        _scan_resolved_modal_global_action_journal(lock_descriptor)
        validated, output = _validate_cohort_roster_payload(
            project_root,
            logical,
            frozen,
        )
        if not exact_json_equal(validated, frozen):
            raise ValueError("Modal cohort roster prevalidation changed")
        assert_modal_action_lock_identity(lock_descriptor)
        create_json_exclusive(output, frozen)
        assert_modal_action_lock_identity(lock_descriptor)
        persisted, persisted_path = _load_cohort_roster(project_root, logical)
        if persisted_path != output or not exact_json_equal(persisted, frozen):
            raise ValueError("persisted Modal cohort roster changed")
        assert_modal_action_lock_identity(lock_descriptor)
        return persisted
    finally:
        release_modal_action_lock(lock_descriptor)


def _image_source_manifest(path: Path) -> ImageSourceManifestV1:
    payload = _load_object(path)
    expected_fields = {
        "schema_name",
        "schema_version",
        "recipe_version",
        "python_version",
        "uv_version",
        "modal_version",
        "dependency_lock_sha256",
        "files",
    }
    if set(payload) != expected_fields:
        raise ValueError("image source manifest has an invalid exact schema")
    if payload["schema_name"] != ImageSourceManifestV1.SCHEMA_NAME:
        raise ValueError("image source manifest has the wrong schema")
    if payload["schema_version"] != ImageSourceManifestV1.SCHEMA_VERSION:
        raise ValueError("image source manifest has an unsupported version")
    raw_files = payload["files"]
    if not isinstance(raw_files, list):
        raise ValueError("image source manifest files must be a list")
    files = []
    for index, entry in enumerate(raw_files):
        if not isinstance(entry, dict) or set(entry) != SourceFileV1.FIELDS:
            raise ValueError(f"image source file {index} has an invalid exact schema")
        relative_path = entry["relative_path"]
        sha256 = entry["sha256"]
        if not isinstance(relative_path, str) or not isinstance(sha256, str):
            raise ValueError("image source file path and digest must be text")
        files.append(
            SourceFileV1(
                relative_path=relative_path,
                sha256=sha256,
                size_bytes=_exact_int(
                    entry["size_bytes"], f"image source file {index} size"
                ),
            )
        )
    for field, expected in {
        "recipe_version": IMAGE_RECIPE_VERSION,
        "python_version": PYTHON_VERSION,
        "uv_version": UV_VERSION,
        "modal_version": MODAL_VERSION,
    }.items():
        if payload[field] != expected:
            raise ValueError(f"image source manifest {field} differs from the recipe")
    manifest = ImageSourceManifestV1(
        dependency_lock_sha256=_sha256(
            payload["dependency_lock_sha256"], "dependency_lock_sha256"
        ),
        files=tuple(files),
        recipe_version=payload["recipe_version"],
        python_version=payload["python_version"],
        uv_version=payload["uv_version"],
        modal_version=payload["modal_version"],
    )
    if manifest.to_dict() != payload:
        raise ValueError("image source manifest contains coerced field types")
    if canonical_sha256(payload) != manifest.manifest_sha256:
        raise ValueError("image source manifest canonical digest is inconsistent")
    return manifest


def _detached_call_policy(
    root: Path,
    run_root: Path,
    expected_image_source_sha256: str,
) -> tuple[str, str, str]:
    logical = "modal_app.py"
    image_manifest = _image_source_manifest(run_root / "image_source_manifest.json")
    if image_manifest.manifest_sha256 != expected_image_source_sha256:
        raise ValueError("downloaded image source manifest digest is not run-bound")
    source_entries = [
        item for item in image_manifest.files if item.relative_path == logical
    ]
    if len(source_entries) != 1:
        raise ValueError("bound image source must contain exactly one modal_app.py")
    source_entry = source_entries[0]
    source_path = _contained_path(
        root, logical, "detached_call_policy_source_path", kind="file"
    )
    source_raw = _read_regular_file_bytes(
        source_path,
        maximum_bytes=_MAX_JSON_OBJECT_BYTES,
    )
    source_sha256 = hashlib.sha256(source_raw).hexdigest()
    if (
        source_entry.sha256 != source_sha256
        or source_entry.size_bytes != len(source_raw)
    ):
        raise ValueError("current modal_app.py differs from the bound image source")
    source = _decode_utf8(source_raw, source_path)
    tree = ast.parse(source, filename=logical)
    if any(
        isinstance(node, ast.Attribute) and node.attr == "spawn"
        for node in ast.walk(tree)
    ):
        raise ValueError("Modal execution source contains a detached spawn path")
    if "invoke_synchronously" not in source:
        raise ValueError("Modal execution source lacks synchronous invocation")
    manifest_path = _contained_path(
        root,
        "experiment_manifest.yaml",
        "detached_call_policy_manifest_path",
        kind="file",
    )
    # Avoid importing the Modal SDK or a second configuration parser here; the
    # frozen manifest text is also checked structurally by the readiness audit.
    manifest_text = _decode_utf8(
        _read_regular_file_bytes(
            manifest_path,
            maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        ),
        manifest_path,
    )
    if "detached_calls: false" not in manifest_text:
        raise ValueError("experiment manifest does not prohibit detached calls")
    return logical, source_sha256, image_manifest.manifest_sha256


def _bound_source_text(
    root: Path,
    image_manifest: ImageSourceManifestV1,
    logical: str,
) -> tuple[str, str]:
    entries = [item for item in image_manifest.files if item.relative_path == logical]
    if len(entries) != 1:
        raise ValueError(f"bound image source must contain exactly one {logical}")
    path = _contained_path(root, logical, f"bound_source.{logical}", kind="file")
    raw = _read_regular_file_bytes(path, maximum_bytes=_MAX_JSON_OBJECT_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    entry = entries[0]
    if entry.sha256 != digest or entry.size_bytes != len(raw):
        raise ValueError(f"current {logical} differs from the bound image source")
    return _decode_utf8(raw, path), digest


def _artifact_verifier_network_policy(
    root: Path,
    run_root: Path,
    expected_image_source_sha256: str,
) -> dict[str, Any]:
    """Prove the verifier's no-secret/no-network decorator from bound source."""

    image_manifest = _image_source_manifest(run_root / "image_source_manifest.json")
    if image_manifest.manifest_sha256 != expected_image_source_sha256:
        raise ValueError("verifier policy image manifest is not execution-bound")
    modal_source, modal_sha256 = _bound_source_text(
        root, image_manifest, "modal_app.py"
    )
    boundary_source, boundary_sha256 = _bound_source_text(
        root, image_manifest, "modal_boundary.py"
    )

    boundary_tree = ast.parse(boundary_source, filename="modal_boundary.py")
    verifier_specs: list[ast.Call] = []
    for node in ast.walk(boundary_tree):
        value: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "FUNCTION_SPECS"
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "FUNCTION_SPECS"
        ):
            value = node.value
        if not isinstance(value, ast.Dict):
            continue
        for key, item in zip(value.keys, value.values, strict=True):
            if (
                isinstance(key, ast.Constant)
                and key.value == "artifact_verify"
                and isinstance(item, ast.Call)
            ):
                verifier_specs.append(item)
    if len(verifier_specs) != 1:
        raise ValueError("bound FUNCTION_SPECS lacks one artifact verifier spec")
    verifier_spec = verifier_specs[0]
    positional_secret = (
        verifier_spec.args[2]
        if len(verifier_spec.args) >= 3
        else None
    )
    keyword_secrets = [
        keyword.value
        for keyword in verifier_spec.keywords
        if keyword.arg == "provider_secret"
    ]
    secret_values = [
        value
        for value in ([positional_secret] if positional_secret is not None else [])
        + keyword_secrets
        if isinstance(value, ast.Constant)
    ]
    if (
        not isinstance(verifier_spec.func, ast.Name)
        or verifier_spec.func.id != "FunctionSpec"
        or len(verifier_spec.args) < 1
        or not isinstance(verifier_spec.args[0], ast.Constant)
        or verifier_spec.args[0].value != "artifact_verify"
        or len(secret_values) != 1
        or secret_values[0].value is not False
    ):
        raise ValueError(
            "artifact verifier provider-secret policy is not statically false"
        )

    modal_tree = ast.parse(modal_source, filename="modal_app.py")
    option_functions = [
        node
        for node in modal_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_function_options"
    ]
    if len(option_functions) != 1:
        raise ValueError("bound modal source lacks one _function_options definition")
    option_function = option_functions[0]
    spec_binding = any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "spec"
            for target in node.targets
        )
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "FUNCTION_SPECS"
        for node in ast.walk(option_function)
    )
    block_network_values: list[ast.AST] = []
    for node in ast.walk(option_function):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "block_network":
                block_network_values.append(value)
    exact_network_derivation = (
        len(block_network_values) == 1
        and isinstance(block_network_values[0], ast.UnaryOp)
        and isinstance(block_network_values[0].op, ast.Not)
        and isinstance(block_network_values[0].operand, ast.Attribute)
        and isinstance(block_network_values[0].operand.value, ast.Name)
        and block_network_values[0].operand.value.id == "spec"
        and block_network_values[0].operand.attr == "provider_secret"
    )
    verifier_functions = [
        node
        for node in ast.walk(modal_tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "artifact_verify"
    ]
    registered_with_options = False
    if len(verifier_functions) == 1:
        registered_with_options = any(
            isinstance(nested, ast.Call)
            and isinstance(nested.func, ast.Name)
            and nested.func.id == "_function_options"
            and len(nested.args) == 1
            and isinstance(nested.args[0], ast.Constant)
            and nested.args[0].value == "artifact_verify"
            for decorator in verifier_functions[0].decorator_list
            for nested in ast.walk(decorator)
        )
    if not spec_binding or not exact_network_derivation or not registered_with_options:
        raise ValueError(
            "artifact verifier block-network policy is not statically bound"
        )
    return {
        "function_name": "artifact_verify",
        "provider_secret": False,
        "block_network": True,
        "proof_kind": "bound_image_source_ast",
        "sources": [
            {"path": "modal_app.py", "sha256": modal_sha256},
            {"path": "modal_boundary.py", "sha256": boundary_sha256},
        ],
    }


def _contained_path(
    root: Path,
    logical: object,
    field: str,
    *,
    kind: str,
) -> Path:
    if not isinstance(logical, str):
        raise ValueError(f"{field} must be a project-relative path")
    try:
        relative = safe_relative_path(logical)
    except ValueError as error:
        raise ValueError(
            f"{field} must be normalized, project-relative, and non-traversing"
        ) from error
    raw_root = Path(root)
    if raw_root.is_symlink():
        raise ValueError("project root may not be a symbolic link")
    resolved_root = raw_root.resolve()
    path = resolved_root.joinpath(*relative.parts)
    cursor = resolved_root
    for component in relative.parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"{field} may not traverse symbolic links")
    if kind == "file" and not path.is_file():
        raise FileNotFoundError(path)
    if kind == "directory" and not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _expected_download_path(run_id: str) -> str:
    return f"outputs/development/modal_downloads/{validate_run_id(run_id)}"


def _select_downloaded_raw_manifest(run_root: Path):
    candidates = []
    for filename in ARTIFACT_MANIFEST_FILENAMES:
        path = run_root / filename
        if path.is_symlink():
            raise ValueError("downloaded artifact manifest may not be a symlink")
        if path.is_file():
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("download must contain exactly one selected raw manifest")
    return load_raw_artifact_manifest(candidates[0])


def _validate_finalized_ordinary_failure_run(
    run_root: Path,
    raw_manifest: Any,
    context: ExecutionContextV1,
) -> dict[str, Any]:
    """Validate the exact manifest-last failure shape emitted by an ordinary action."""

    specifications = {
        "cuda_environment": (
            "artifact_manifest.json",
            "remote_action_result.json",
            frozenset({"success", "error_type"}),
        ),
        "offline_smoke": (
            "artifact_manifest.json",
            "remote_action_result.json",
            frozenset({"success", "error_type"}),
        ),
        "candidate_smoke": (
            "artifact_manifest.checkpoint.json",
            "remote_action_result.json",
            frozenset({"success", "error_type"}),
        ),
        "checkpoint_resume": (
            "artifact_manifest.json",
            "resume_action_result.json",
            frozenset(
                {"success", "mode", "error_type", "source_run_id"}
            ),
        ),
    }
    specification = specifications.get(context.function_name)
    if specification is None:
        raise ValueError("source evidence recovery is not an ordinary Modal action")
    manifest_filename, result_filename, result_fields = specification
    expected_files = {
        manifest_filename,
        "execution_context.json",
        "image_source_manifest.json",
        "remote_action_failure.json",
        result_filename,
    }
    entries = tuple(run_root.iterdir())
    if (
        raw_manifest.filename != manifest_filename
        or {entry.name for entry in entries} != expected_files
        or any(entry.is_symlink() or not entry.is_file() for entry in entries)
    ):
        raise ValueError(
            "finalized ordinary failure differs from its exact artifact roster"
        )
    failure = _load_object(run_root / "remote_action_failure.json")
    if set(failure) != {"error_type", "message"}:
        raise ValueError("ordinary failure receipt has an invalid exact schema")
    error_type = failure["error_type"]
    if (
        not isinstance(error_type, str)
        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", error_type) is None
        or failure["message"] != "remote action failed; details suppressed"
    ):
        raise ValueError("ordinary failure receipt is not sanitized")
    result = _load_object(run_root / result_filename)
    if (
        set(result) != result_fields
        or result["success"] is not False
        or type(result["success"]) is not bool
        or result["error_type"] != error_type
    ):
        raise ValueError("ordinary failure result is not failure-bound")
    if context.function_name == "checkpoint_resume":
        if result["mode"] != "checkpoint_resume":
            raise ValueError("resume failure result has the wrong mode")
        validate_run_id(result["source_run_id"])
    for field, value in {
        "modal_app_id": context.modal_app_id,
        "modal_function_id": context.modal_function_id,
        "modal_call_id": context.modal_call_id,
        "modal_image_id": context.modal_image_id,
    }.items():
        _text(value, f"finalized_ordinary_failure.{field}")
    return result


def _inspect_downloaded_run_raw(
    root: Path,
    run_id: str,
    downloaded_run_path: object,
    *,
    allow_finalized_ordinary_failure: bool = False,
) -> tuple[Path, Any, dict[str, Any], ExecutionContextV1]:
    expected_path = _expected_download_path(run_id)
    if downloaded_run_path != expected_path:
        raise ValueError(
            "downloaded_run_path must use the frozen Modal download location"
        )
    run_root = _contained_path(
        root, downloaded_run_path, "downloaded_run_path", kind="directory"
    )
    if run_root.name != run_id:
        raise ValueError("download directory name differs from run_id")
    raw_manifest = _select_downloaded_raw_manifest(run_root)
    manifest = raw_manifest.manifest
    verification = verify_artifact_manifest(run_root, manifest)
    if manifest.run_id != run_id or verification.get("verified") is not True:
        raise ValueError("downloaded artifact manifest belongs to another run")
    context_payload = _load_object(run_root / "execution_context.json")
    context = ExecutionContextV1.from_dict(context_payload)
    if (
        context.execution_backend != "modal"
        or context.run_id != run_id
        or context.app_name != APP_NAME
        or context.artifact_uri != volume_artifact_uri(run_id)
        or context.image_source_sha256 != manifest.image_source_sha256
    ):
        raise ValueError("downloaded execution context differs from the Modal run")
    if context.function_name in PROVIDER_FREE_MODAL_FUNCTIONS:
        if allow_finalized_ordinary_failure:
            _validate_finalized_ordinary_failure_run(
                run_root,
                raw_manifest,
                context,
            )
        else:
            validate_provider_free_action_outer_roster(
                run_root,
                function_name=context.function_name,
            )
            validate_provider_free_network_denial_probe(
                _load_object(run_root / "provider_free_network_denial_probe.json"),
                expected_context=context,
            )
    return run_root, raw_manifest, verification, context


def _inspect_downloaded_run(
    root: Path,
    run_id: str,
    downloaded_run_path: object,
) -> tuple[Path, Any, dict[str, Any], ExecutionContextV1]:
    run_root, raw_manifest, verification, context = _inspect_downloaded_run_raw(
        root,
        run_id,
        downloaded_run_path,
    )
    return run_root, raw_manifest.manifest, verification, context


def _trailing_json_object(path: Path) -> dict[str, Any]:
    """Extract exactly one final JSON object from a saved CLI transcript."""

    try:
        raw = _read_regular_file_bytes(path, maximum_bytes=2 * 1024 * 1024)
        source = raw.decode("utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"verification transcript is missing: {path}"
        ) from error
    except UnicodeDecodeError as error:
        raise ValueError("verification transcript is not UTF-8") from error
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_json_keys)
    candidates: list[dict[str, Any]] = []
    for offset, character in enumerate(source):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(source, offset)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and not source[end:].strip():
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError(
            "verification transcript must end in exactly one unambiguous JSON object"
        )
    return candidates[0]


def _validate_remote_verification(
    payload: Mapping[str, Any],
    *,
    source_run_id: str,
    verifier_run_id: str,
    raw_manifest: Any,
    source_execution_context: ExecutionContextV1,
) -> ArtifactVerificationV1:
    try:
        verification = ArtifactVerificationV1.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("remote verification has an invalid exact schema") from error
    expected = {
        "source_run_id": validate_run_id(source_run_id),
        "verifier_run_id": validate_run_id(verifier_run_id),
        "manifest_filename": raw_manifest.filename,
        "raw_manifest_sha256": raw_manifest.raw_sha256,
        "raw_manifest_size_bytes": raw_manifest.raw_size_bytes,
        "canonical_manifest_sha256": raw_manifest.manifest.manifest_sha256,
        "file_count": len(raw_manifest.manifest.files),
        "verified": True,
    }
    observed = verification.to_dict()
    for field, expected_value in expected.items():
        if observed[field] != expected_value:
            raise ValueError(
                f"remote verification {field} differs from downloaded artifacts"
            )
    verifier_context = verification.verifier_execution_context
    if (
        verifier_context.image_source_sha256
        != raw_manifest.manifest.image_source_sha256
    ):
        raise ValueError(
            "remote verifier image source digest differs from downloaded artifacts"
        )
    if source_execution_context.modal_image_id is None:
        raise ValueError("downloaded source execution context lacks an image ID")
    if verifier_context.modal_image_id != source_execution_context.modal_image_id:
        raise ValueError(
            "remote verifier image ID differs from downloaded source execution"
        )
    return verification


def capture_remote_verification(
    *,
    source_run_id: str,
    verifier_run_id: str,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    transcript_path: str | Path,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Persist the final provider-free ``--action verify`` result exactly once."""

    project_root = Path(root)
    source_run = validate_run_id(source_run_id)
    verifier_run = validate_run_id(verifier_run_id)
    _, raw_manifest, _, source_context = _inspect_downloaded_run_raw(
        project_root,
        source_run,
        _expected_download_path(source_run),
    )
    captured = _trailing_json_object(Path(transcript_path).expanduser())
    verification = _validate_remote_verification(
        captured,
        source_run_id=source_run,
        verifier_run_id=verifier_run,
        raw_manifest=raw_manifest,
        source_execution_context=source_context,
    )
    logical = _remote_verification_logical(
        identity,
        source_run,
        verifier_run,
        attempt_id,
    )
    output = project_root.resolve().joinpath(*PurePosixPath(logical).parts)
    create_json_exclusive(output, verification.to_dict())
    persisted = _load_object(output)
    persisted_verification = _validate_remote_verification(
        persisted,
        source_run_id=source_run,
        verifier_run_id=verifier_run,
        raw_manifest=raw_manifest,
        source_execution_context=source_context,
    )
    if persisted_verification != verification:
        raise ValueError("persisted remote verification changed after creation")
    return persisted_verification.to_dict()


def _remote_verification_logical(
    identity: ModalLiveCohortIdentity,
    source_run_id: str,
    verifier_run_id: str,
    attempt_id: str,
) -> str:
    """Return an explicit source/verifier/attempt-bound verification path."""

    return modal_remote_verification_receipt_path(
        identity,
        source_run_id,
        verifier_run_id,
        _attempt_id(attempt_id, "verifier_attempt_id"),
    ).as_posix()


def _remote_download_evidence(
    root: Path,
    run_id: str,
    manifest: Any,
    *,
    identity: ModalLiveCohortIdentity,
    verifier_run_id: str,
    verifier_attempt_id: str,
) -> dict[str, Any]:
    logical = _remote_verification_logical(
        identity,
        run_id,
        verifier_run_id,
        verifier_attempt_id,
    )
    path = _contained_path(root, logical, "remote_verification_path", kind="file")
    payload = _load_object(path)
    _, raw_manifest, _, source_context = _inspect_downloaded_run_raw(
        root,
        run_id,
        _expected_download_path(run_id),
    )
    if raw_manifest.manifest != manifest:
        raise ValueError(f"downloaded manifest for {run_id} changed")
    raw_verifier_run_id = payload.get("verifier_run_id")
    if not isinstance(raw_verifier_run_id, str):
        raise ValueError(f"remote verifier capture for {run_id} is invalid")
    requested_verifier_run_id = validate_run_id(verifier_run_id)
    if raw_verifier_run_id != requested_verifier_run_id:
        raise ValueError(
            f"remote verifier capture for {run_id} differs from the explicit selection"
        )
    verification = _validate_remote_verification(
        payload,
        source_run_id=run_id,
        verifier_run_id=requested_verifier_run_id,
        raw_manifest=raw_manifest,
        source_execution_context=source_context,
    )
    return {
        "run_id": run_id,
        "verifier_run_id": verification.verifier_run_id,
        "verifier_attempt_id": _attempt_id(
            verifier_attempt_id, "verifier_attempt_id"
        ),
        "remote_verification_path": logical,
        "remote_verification_sha256": _sha256_file(path),
        "local_artifact_manifest_sha256": manifest.manifest_sha256,
        "files_verified": len(manifest.files),
    }


def _validate_cuda_payload(payload: Mapping[str, Any], *, root: Path) -> str:
    if set(payload) != _CUDA_FIELDS:
        raise ValueError("CUDA receipt has unexpected or missing fields")
    contract = MODAL_READINESS_RECEIPT_CONTRACTS["modal_cuda_environment_validated"][
        "receipt_contract"
    ]
    if (
        payload["schema_name"] != contract["schema_name"]
        or payload["schema_version"] != contract["schema_version"]
    ):
        raise ValueError("CUDA receipt has the wrong schema contract")
    _utc(payload["recorded_at_utc"], "recorded_at_utc")
    identity = _cohort_identity_from_payload(payload, field="cuda_receipt")
    run_id = validate_run_id(payload["run_id"])
    run_root, manifest, verification, context = _inspect_downloaded_run(
        root, run_id, payload["downloaded_run_path"]
    )
    if (
        payload["execution_backend"] != "modal"
        or payload["app_name"] != APP_NAME
        or payload["function_name"] != "cuda_environment"
        or payload["requested_gpu"] != GPU_TYPE
        or context.function_name != "cuda_environment"
        or payload["artifact_uri"] != volume_artifact_uri(run_id)
        or manifest.image_source_sha256 != identity.image_source_sha256
    ):
        raise ValueError("CUDA receipt execution identity is invalid")
    cuda_path = run_root / "cuda_environment.json"
    action_path = run_root / "remote_action_result.json"
    context_path = run_root / "execution_context.json"
    cuda = _load_object(cuda_path)
    action = _load_object(action_path)
    expected_cuda_fields = {
        "python",
        "platform",
        "torch",
        "git_version",
        "cuda_available",
        "cuda_device_count",
        "cuda_device_name",
        "cuda_compute_capability",
        "cuda_runtime",
        "cuda_driver",
        "cuda_total_memory_bytes",
        "accelerator_fingerprint",
        "execution_context",
    }
    if set(cuda) != expected_cuda_fields:
        raise ValueError("cuda_environment.json has an invalid exact schema")
    if set(action) != {"success", "mode", "observed_gpu"}:
        raise ValueError("CUDA environment action has an invalid exact schema")
    _exact_bool(action["success"], "cuda_environment_action.success")
    if action["mode"] != "cuda_environment":
        raise ValueError("CUDA environment action has the wrong mode")
    _git_version(cuda["git_version"], "cuda_environment.git_version")
    python_version = _text(cuda["python"], "cuda_environment.python")
    if python_version != PYTHON_VERSION and not python_version.startswith(
        f"{PYTHON_VERSION}."
    ):
        raise ValueError("CUDA environment Python differs from the frozen version")
    for field in ("platform", "torch", "cuda_runtime", "cuda_driver"):
        _text(cuda[field], f"cuda_environment.{field}")
    _exact_bool(cuda["cuda_available"], "cuda_environment.cuda_available")
    if _exact_int(cuda["cuda_device_count"], "cuda_device_count") != 1:
        raise ValueError("cuda_device_count must be exactly one")
    observed_name = cuda["cuda_device_name"]
    if (
        not isinstance(observed_name, str)
        or GPU_TYPE.lower() not in observed_name.lower()
    ):
        raise ValueError(f"observed GPU is not a {GPU_TYPE}")
    if cuda["execution_context"] != context.to_dict():
        raise ValueError("CUDA report execution context is not hash-linked")
    fingerprint = AcceleratorFingerprint.from_dict(
        cuda["accelerator_fingerprint"]
    ).validate_cuda(exact_gpu_count=1, require_driver=True)
    if fingerprint.gpu_name != observed_name:
        raise ValueError("CUDA fingerprint GPU name differs from the report")
    raw_capability = cuda["cuda_compute_capability"]
    if (
        not isinstance(raw_capability, list)
        or len(raw_capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_capability
        )
    ):
        raise ValueError("CUDA compute capability must be two exact integers")
    if raw_capability != [
        int(part) for part in fingerprint.compute_capability.split(".")
    ]:
        raise ValueError("CUDA compute capability differs from the fingerprint")
    for report_field, fingerprint_value in {
        "platform": fingerprint.host_platform,
        "torch": fingerprint.torch_version,
        "cuda_runtime": fingerprint.cuda_runtime,
        "cuda_driver": fingerprint.cuda_driver,
    }.items():
        if cuda[report_field] != fingerprint_value:
            raise ValueError(f"CUDA report {report_field} differs from the fingerprint")
    _exact_int(
        cuda["cuda_total_memory_bytes"],
        "cuda_environment.cuda_total_memory_bytes",
        minimum=1,
    )
    if _text(action["observed_gpu"], "cuda_environment_action.observed_gpu") != (
        observed_name
    ):
        raise ValueError("CUDA environment action GPU differs from the report")
    expected = {
        "observed_gpu_name": observed_name,
        "cuda_available": True,
        "cuda_device_count": 1,
        "image_source_sha256": manifest.image_source_sha256,
        "execution_context_sha256": _sha256_file(context_path),
        "cuda_environment_sha256": _sha256_file(cuda_path),
        "remote_action_result_sha256": _sha256_file(action_path),
        "artifact_manifest_sha256": manifest.manifest_sha256,
        "files_verified": verification["file_count"],
        "validated": True,
    }
    for field, expected_value in expected.items():
        observed = payload[field]
        if type(expected_value) is bool and type(observed) is not bool:
            raise ValueError(f"{field} must be boolean")
        if isinstance(expected_value, int) and not isinstance(expected_value, bool):
            _exact_int(observed, field)
        if observed != expected_value:
            raise ValueError(f"CUDA receipt field {field} differs from artifacts")
    for field in (
        "image_source_sha256",
        "execution_context_sha256",
        "cuda_environment_sha256",
        "remote_action_result_sha256",
        "artifact_manifest_sha256",
    ):
        _sha256(payload[field], field)
    return f"run={run_id} gpu={observed_name} files={verification['file_count']}"


def _validate_round_trip_payload(payload: Mapping[str, Any], *, root: Path) -> str:
    if set(payload) != _ROUND_TRIP_FIELDS:
        raise ValueError("artifact round-trip receipt has unexpected or missing fields")
    contract = MODAL_READINESS_RECEIPT_CONTRACTS["modal_artifact_round_trip_validated"][
        "receipt_contract"
    ]
    if (
        payload["schema_name"] != contract["schema_name"]
        or payload["schema_version"] != contract["schema_version"]
    ):
        raise ValueError("artifact round-trip receipt has the wrong schema contract")
    _utc(payload["recorded_at_utc"], "recorded_at_utc")
    identity = _cohort_identity_from_payload(
        payload, field="artifact_round_trip_receipt"
    )
    source_run_id = validate_run_id(payload["source_run_id"])
    verifier_run_id = validate_run_id(payload["verifier_run_id"])
    _, raw_manifest, verification, context = _inspect_downloaded_run_raw(
        root,
        source_run_id,
        payload["downloaded_run_path"],
    )
    if context.function_name != "candidate_smoke":
        raise ValueError("artifact round-trip must bind a candidate_smoke run")
    if raw_manifest.manifest.image_source_sha256 != identity.image_source_sha256:
        raise ValueError("artifact round-trip image differs from its cohort")
    expected_remote_path = _remote_verification_logical(
        identity,
        source_run_id,
        verifier_run_id,
        payload["verifier_attempt_id"],
    )
    if payload["remote_verification_path"] != expected_remote_path:
        raise ValueError("remote_verification_path differs from the frozen path")
    remote_path = _contained_path(
        root,
        payload["remote_verification_path"],
        "remote_verification_path",
        kind="file",
    )
    remote = _load_object(remote_path)
    remote_verification = _validate_remote_verification(
        remote,
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        raw_manifest=raw_manifest,
        source_execution_context=context,
    )
    manifest = raw_manifest.manifest
    expected = {
        "artifact_uri": volume_artifact_uri(source_run_id),
        "manifest_filename": raw_manifest.filename,
        "remote_verification_sha256": _sha256_file(remote_path),
        "verifier_execution_context_sha256": canonical_sha256(
            remote_verification.verifier_execution_context.to_dict()
        ),
        "remote_raw_manifest_sha256": remote_verification.raw_manifest_sha256,
        "remote_raw_manifest_size_bytes": (remote_verification.raw_manifest_size_bytes),
        "local_raw_manifest_sha256": raw_manifest.raw_sha256,
        "local_raw_manifest_size_bytes": raw_manifest.raw_size_bytes,
        "remote_canonical_manifest_sha256": (
            remote_verification.canonical_manifest_sha256
        ),
        "local_canonical_manifest_sha256": manifest.manifest_sha256,
        "files_verified": verification["file_count"],
        "remote_verification_completed": True,
        "local_verification_completed": True,
        "validated": True,
    }
    for field, expected_value in expected.items():
        observed = payload[field]
        if type(expected_value) is bool and type(observed) is not bool:
            raise ValueError(f"{field} must be boolean")
        if isinstance(expected_value, int) and not isinstance(expected_value, bool):
            _exact_int(observed, field)
        if observed != expected_value:
            raise ValueError(f"round-trip receipt field {field} is invalid")
    for field in (
        "remote_verification_sha256",
        "verifier_execution_context_sha256",
        "remote_raw_manifest_sha256",
        "local_raw_manifest_sha256",
        "remote_canonical_manifest_sha256",
        "local_canonical_manifest_sha256",
    ):
        _sha256(payload[field], field)
    return (
        f"source_run={source_run_id} verifier_run={verifier_run_id} "
        f"files={verification['file_count']} remote_and_local_verified"
    )


def _primary_executions_from_roster(
    root: Path,
    roster: Mapping[str, Any],
) -> dict[str, tuple[dict[str, Any], Path, Any, ExecutionContextV1]]:
    executions = {
        label: _execution_for_run(
            root,
            roster["accepted_primary_runs"][label],
            expected_function=function_name,
        )
        for label, function_name in _PRIMARY_FUNCTIONS.items()
    }
    contexts = [item[3] for item in executions.values()]
    if len(contexts) != 8:
        raise AssertionError("primary execution roster drifted")
    if len({item.run_id for item in contexts}) != 8:
        raise ValueError("primary execution run IDs are not unique")
    if len({item.modal_call_id for item in contexts}) != 8:
        raise ValueError("primary execution Modal call IDs are not unique")
    return executions


def _artifact_verifier_executions(
    root: Path,
    roster: Mapping[str, Any],
    primary: Mapping[str, tuple[dict[str, Any], Path, Any, ExecutionContextV1]],
) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    contexts: list[ExecutionContextV1] = []
    for label in _PRIMARY_LABELS:
        verifier = roster["artifact_verifiers"][label]
        source_run_id = verifier["source_run_id"]
        _, raw_manifest, _, source_context = _inspect_downloaded_run_raw(
            root, source_run_id, _expected_download_path(source_run_id)
        )
        logical = verifier["remote_verification_path"]
        path = _contained_path(root, logical, "remote_verification_path", kind="file")
        verification = _validate_remote_verification(
            _load_object(path),
            source_run_id=source_run_id,
            verifier_run_id=verifier["verifier_run_id"],
            raw_manifest=raw_manifest,
            source_execution_context=source_context,
        )
        context = verification.verifier_execution_context
        primary_context = primary[label][3]
        if (
            context.image_source_sha256 != primary_context.image_source_sha256
            or context.modal_image_id != primary_context.modal_image_id
            or context.to_dict() != verifier["verifier_execution_context"]
            or _sha256_file(path) != verifier["remote_verification_sha256"]
            or verifier["expected_remote_receipt_roster"]
            != list(_VERIFIER_REMOTE_RECEIPT_ROSTER)
        ):
            raise ValueError(f"artifact verifier {label} used a different image")
        contexts.append(context)
        evidence[label] = {
            "source_run_id": source_run_id,
            "verifier_run_id": verification.verifier_run_id,
            "attempt_id": verifier["attempt_id"],
            "execution_context": context.to_dict(),
            "remote_verification_path": logical,
            "remote_verification_sha256": _sha256_file(path),
            "expected_remote_receipt_roster": list(
                _VERIFIER_REMOTE_RECEIPT_ROSTER
            ),
        }
    if len({item.run_id for item in contexts}) != 8:
        raise ValueError("artifact verifier run IDs are not unique")
    if len({item.modal_call_id for item in contexts}) != 8:
        raise ValueError("artifact verifier Modal call IDs are not unique")
    return evidence


def _additional_artifact_verifier_executions(
    root: Path,
    roster: Mapping[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Revalidate every paid verifier outside the eight accepted-source map."""

    by_id = {record["attempt_id"]: record for record in attempts}
    identity = _cohort_identity_from_payload(roster, field="cohort_roster")
    attempt_directory = modal_action_attempt_directory(identity).as_posix()
    successful_verifiers_by_attempt = {
        record["attempt_id"]: record
        for record in roster["artifact_verifiers"].values()
    } | {
        record["attempt_id"]: record
        for record in roster["additional_artifact_verifiers"]
        if record["remote_verifier_outcome"] == "success"
    }
    evidence: list[dict[str, Any]] = []
    for record in roster["additional_artifact_verifiers"]:
        attempt = by_id.get(record["attempt_id"])
        if (
            attempt is None
            or attempt["action"] not in {"download", "verify"}
            or attempt["run_id"] != record["source_run_id"]
            or attempt["verifier_run_id"] != record["verifier_run_id"]
            or attempt["status"] != record["status"]
            or attempt["concrete_remote_run_ids"] != [record["verifier_run_id"]]
            or attempt["modal_cli_process_started"] is not True
        ):
            raise ValueError("additional verifier is not action-attempt bound")
        source_bindings = attempt["predecessor_receipts"][3:]
        if (
            len(source_bindings) not in {3, 4}
            or source_bindings[0]["gate"] != "source_action_intent"
            or source_bindings[1]["gate"] != "source_action_attempt_terminal"
            or source_bindings[2]["gate"] != "source_local_process_start"
        ):
            raise ValueError("additional verifier lacks its source action bindings")
        source_intent_match = re.fullmatch(
            rf"{re.escape(attempt_directory)}/([0-9a-f]{{32}})\.intent\.json",
            source_bindings[0]["path"],
        )
        if source_intent_match is None:
            raise ValueError("additional verifier source intent path is invalid")
        source_attempt_id = source_intent_match.group(1)
        if source_bindings[1]["path"] != modal_action_terminal_receipt_path(
            identity, source_attempt_id
        ).as_posix():
            raise ValueError("additional verifier source terminal path is swappable")
        if source_bindings[2]["path"] != modal_local_process_start_receipt_path(
            source_attempt_id
        ).as_posix():
            raise ValueError("additional verifier source marker path is swappable")
        source_attempt = by_id.get(source_attempt_id)
        if (
            source_attempt is None
            or record["source_run_id"]
            not in source_attempt["concrete_remote_run_ids"]
        ):
            raise ValueError("additional verifier source action is not in the cohort")
        source_action_evidence = {
            "attempt_id": source_attempt_id,
            "intent_path": source_bindings[0]["path"],
            "intent_sha256": source_bindings[0]["sha256"],
            "terminal_path": source_bindings[1]["path"],
            "terminal_sha256": source_bindings[1]["sha256"],
            "process_start_path": source_bindings[2]["path"],
            "process_start_sha256": source_bindings[2]["sha256"],
        }
        if record["remote_verifier_outcome"] == "unresolved":
            evidence.append(
                {
                    "source_run_id": record["source_run_id"],
                    "verifier_run_id": record["verifier_run_id"],
                    "attempt_id": record["attempt_id"],
                    "launcher_status": record["status"],
                    "remote_verifier_outcome": "unresolved",
                    "remote_evidence_kind": record["remote_evidence_kind"],
                    "billing_object_ids": record["billing_object_ids"],
                    "source_evidence_recovery": attempt[
                        "source_evidence_recovery"
                    ],
                    "source_action_evidence": source_action_evidence,
                    "source_execution_context": None,
                    "execution_context": None,
                    "remote_verification_path": None,
                    "remote_verification_sha256": None,
                    "failure_receipt_path": None,
                    "failure_receipt_sha256": None,
                    "failure_manifest_sha256": None,
                    "recovery_verifier_attempt_id": None,
                    "recovery_verifier_run_id": None,
                    "recovery_remote_verification_path": None,
                    "recovery_remote_verification_sha256": None,
                    "expected_remote_receipt_roster": [],
                }
            )
            continue
        ordinary_source_recovery = (
            attempt["source_evidence_recovery"]
            and source_attempt["action"] in _ORDINARY_ACTION_FUNCTIONS
        )
        if ordinary_source_recovery and (
            source_attempt["status"] == "succeeded"
            or source_attempt["process_group_closed"] is not True
            or record["remote_verifier_outcome"] != "success"
            or record["status"] != "succeeded"
        ):
            raise ValueError(
                "ordinary source recovery lacks a closed failure and "
                "successful verifier"
            )
        source_run_id = record["source_run_id"]
        _, raw_manifest, _, source_context = _inspect_downloaded_run_raw(
            root,
            source_run_id,
            _expected_download_path(source_run_id),
            allow_finalized_ordinary_failure=ordinary_source_recovery,
        )
        if ordinary_source_recovery:
            expected_function = _ORDINARY_ACTION_FUNCTIONS[source_attempt["action"]]
            if source_context.function_name != expected_function:
                raise ValueError("ordinary source recovery used another function")
            if source_attempt["action"] == "checkpoint-resume":
                failure_result = _load_object(
                    _contained_path(
                        root,
                        (
                            f"{_expected_download_path(source_run_id)}/"
                            "resume_action_result.json"
                        ),
                        "source_evidence_recovery.resume_action_result",
                        kind="file",
                    )
                )
                if failure_result["source_run_id"] != source_attempt["source_run_id"]:
                    raise ValueError("resume source recovery changed its source run")
        if record["remote_verifier_outcome"] == "failure":
            recovery_record = successful_verifiers_by_attempt.get(
                record["recovery_verifier_attempt_id"]
            )
            recovery_attempt = (
                by_id.get(recovery_record["attempt_id"])
                if recovery_record is not None
                else None
            )
            if (
                recovery_record is None
                or recovery_attempt is None
                or recovery_record["source_run_id"] != source_run_id
                or recovery_attempt["status"] != "succeeded"
                or recovery_attempt["action"] not in {"download", "verify"}
                or _utc(
                    recovery_attempt["started_at_utc"],
                    "recovery_verifier.started_at_utc",
                )
                < _utc(
                    attempt["finished_at_utc"],
                    "failed_verifier.finished_at_utc",
                )
            ):
                raise ValueError(
                    "failed paid verifier lacks a later successful verifier retry"
                )
            context, failure_manifest_sha256 = _failed_verifier_capture(
                root, record, identity=identity
            )
            if (
                context.image_source_sha256 != source_context.image_source_sha256
                or context.modal_image_id != source_context.modal_image_id
                or context.artifact_uri != volume_artifact_uri(source_run_id)
            ):
                raise ValueError("failed additional verifier used another image cohort")
            evidence.append(
                {
                    "source_run_id": source_run_id,
                    "verifier_run_id": record["verifier_run_id"],
                    "attempt_id": record["attempt_id"],
                    "launcher_status": record["status"],
                    "remote_verifier_outcome": "failure",
                    "remote_evidence_kind": record["remote_evidence_kind"],
                    "billing_object_ids": record["billing_object_ids"],
                    "source_evidence_recovery": attempt[
                        "source_evidence_recovery"
                    ],
                    "source_action_evidence": source_action_evidence,
                    "source_execution_context": source_context.to_dict(),
                    "execution_context": context.to_dict(),
                    "remote_verification_path": None,
                    "remote_verification_sha256": None,
                    "failure_receipt_path": record["failure_receipt_path"],
                    "failure_receipt_sha256": record["failure_receipt_sha256"],
                    "failure_manifest_sha256": failure_manifest_sha256,
                    "recovery_verifier_attempt_id": record[
                        "recovery_verifier_attempt_id"
                    ],
                    "recovery_verifier_run_id": recovery_record[
                        "verifier_run_id"
                    ],
                    "recovery_remote_verification_path": recovery_record[
                        "remote_verification_path"
                    ],
                    "recovery_remote_verification_sha256": recovery_record[
                        "remote_verification_sha256"
                    ],
                    "expected_remote_receipt_roster": list(
                        _FAILED_VERIFIER_REMOTE_RECEIPT_ROSTER
                    ),
                }
            )
            continue
        if record["remote_evidence_kind"] == "volume_success_capture":
            captured_verification, _ = _successful_verifier_capture(
                root, record, identity=identity
            )
            remote_path = _contained_path(
                root,
                record["remote_verification_path"],
                "additional_artifact_verifier.remote_verification_path",
                kind="file",
            )
            remote_payload = captured_verification.to_dict()
        else:
            remote_path = _contained_path(
                root,
                record["remote_verification_path"],
                "additional_artifact_verifier.remote_verification_path",
                kind="file",
            )
            remote_payload = _load_object(remote_path)
        verification = _validate_remote_verification(
            remote_payload,
            source_run_id=source_run_id,
            verifier_run_id=record["verifier_run_id"],
            raw_manifest=raw_manifest,
            source_execution_context=source_context,
        )
        context = verification.verifier_execution_context
        if (
            context.to_dict() != record["verifier_execution_context"]
            or _sha256_file(remote_path) != record["remote_verification_sha256"]
        ):
            raise ValueError("additional verifier evidence changed")
        evidence.append(
            {
                "source_run_id": source_run_id,
                "verifier_run_id": record["verifier_run_id"],
                "attempt_id": record["attempt_id"],
                "launcher_status": record["status"],
                "remote_verifier_outcome": "success",
                "remote_evidence_kind": record["remote_evidence_kind"],
                "billing_object_ids": record["billing_object_ids"],
                "source_evidence_recovery": attempt["source_evidence_recovery"],
                "source_action_evidence": source_action_evidence,
                "source_execution_context": source_context.to_dict(),
                "execution_context": context.to_dict(),
                "remote_verification_path": record["remote_verification_path"],
                "remote_verification_sha256": record[
                    "remote_verification_sha256"
                ],
                "failure_receipt_path": None,
                "failure_receipt_sha256": None,
                "failure_manifest_sha256": None,
                "recovery_verifier_attempt_id": None,
                "recovery_verifier_run_id": None,
                "recovery_remote_verification_path": None,
                "recovery_remote_verification_sha256": None,
                "expected_remote_receipt_roster": list(
                    _VERIFIER_REMOTE_RECEIPT_ROSTER
                ),
            }
        )
    return evidence


def _validate_attempt_cohort(
    root: Path,
    roster: Mapping[str, Any],
    attempts: list[dict[str, Any]],
    primary: Mapping[str, tuple[dict[str, Any], Path, Any, ExecutionContextV1]],
    verifiers: Mapping[str, dict[str, Any]],
    additional_verifiers: list[dict[str, Any]],
    aggregate_outcomes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {item["attempt_id"]: item for item in attempts}
    if len(by_id) != len(attempts):
        raise ValueError("final cohort contains duplicate attempt IDs")
    roles = {
        item["attempt_id"]: set(item["roles"])
        for item in roster["attempt_classifications"]
    }
    additional_verifiers_by_attempt = {
        item["attempt_id"]: item for item in additional_verifiers
    }
    if len(additional_verifiers_by_attempt) != len(additional_verifiers):
        raise ValueError("additional verifier execution evidence is duplicated")
    ordinary_source_recoveries: dict[str, dict[str, Any]] = {}
    provider_outcomes_by_run = {
        item["concrete_run_id"]: item for item in roster["provider_canary_outcomes"]
    }
    provider_outcomes_by_attempt: dict[str, list[Mapping[str, Any]]] = {}
    terminal_dispositions_by_attempt: dict[str, list[Mapping[str, Any]]] = {}
    for disposition in roster["terminal_run_dispositions"]:
        terminal_dispositions_by_attempt.setdefault(
            disposition["attempt_id"], []
        ).append(disposition)
    provider_aggregate_child_status_by_run: dict[str, str] = {}
    for outcome in roster["provider_canary_outcomes"]:
        provider_outcomes_by_attempt.setdefault(
            outcome["launcher_attempt_id"], []
        ).append(outcome)
    for attempt_id, aggregate in aggregate_outcomes.items():
        declared = provider_outcomes_by_attempt.get(attempt_id, [])
        if len(declared) != len(CANARY_ORDER):
            raise ValueError(
                "provider aggregate outcome lacks four cohort child outcomes"
            )
        declared_by_run = {item["concrete_run_id"]: item for item in declared}
        if len(declared_by_run) != len(CANARY_ORDER):
            raise ValueError("provider aggregate cohort outcomes are duplicated")
        for child in aggregate["outcomes"]:
            outcome = declared_by_run.get(child["run_id"])
            if outcome is None or outcome["harness"] != child["harness"]:
                raise ValueError("provider aggregate child is not cohort-bound")
            allowed = (
                {"accepted", "completed_unaccepted"}
                if child["status"] == "success"
                else {
                    "failed",
                    "provider_request_start_uncertain",
                }
            )
            if outcome["outcome"] not in allowed:
                raise ValueError(
                    "provider aggregate child status differs from its cohort outcome"
                )
            provider_aggregate_child_status_by_run[child["run_id"]] = child[
                "status"
            ]
    for attempt_id, receipt in by_id.items():
        classified = roles[attempt_id]
        failed = receipt["status"] != "succeeded"
        if ("failed" in classified) is not failed:
            raise ValueError("attempt failed role differs from its terminal status")
        if not failed and "quarantined" in classified:
            raise ValueError("successful attempts may not be quarantined")
        if (
            failed
            and receipt["modal_cli_process_started"]
            and "quarantined" not in classified
        ):
            raise ValueError("failed remote or uncertain attempts must be quarantined")
        if (
            receipt["action"] in {"download", "verify"}
            and not {
                "artifact_verifier",
                "validation_only",
            }
            <= classified
        ):
            raise ValueError("artifact verifier attempts must be validation-only")
        if receipt["source_evidence_recovery"]:
            if not {"artifact_verifier", "validation_only"} <= classified:
                raise ValueError(
                    "source-evidence recovery must remain validation-only"
                )
            if receipt["modal_cli_process_started"]:
                verifier_evidence = additional_verifiers_by_attempt.get(attempt_id)
                if verifier_evidence is None:
                    raise ValueError(
                        "source-evidence recovery lacks additional verifier evidence"
                    )
                source_action_evidence = verifier_evidence["source_action_evidence"]
                source_attempt_id = source_action_evidence["attempt_id"]
                source_terminal = by_id.get(source_attempt_id)
                if source_terminal is None:
                    raise ValueError(
                        "source-evidence recovery terminal is outside the cohort"
                    )
                if source_terminal.get("status") == "succeeded" or (
                    source_terminal.get("action") == "canaries"
                    and source_terminal.get("status") == "failed"
                    and source_terminal.get("returncode") == 2
                ):
                    raise ValueError(
                        "source-evidence recovery cannot target a complete source "
                        "terminal"
                    )
                outcome = provider_outcomes_by_run.get(receipt["run_id"])
                if source_terminal["action"] in _ORDINARY_ACTION_FUNCTIONS:
                    if (
                        verifier_evidence["remote_verifier_outcome"] != "success"
                        or verifier_evidence["launcher_status"] != "succeeded"
                        or source_terminal["process_group_closed"] is not True
                        or source_terminal["run_id"] != receipt["run_id"]
                        or "failed" not in roles[source_attempt_id]
                        or "quarantined" not in roles[source_attempt_id]
                        or source_attempt_id in ordinary_source_recoveries
                    ):
                        raise ValueError(
                            "ordinary source recovery is not a unique closed "
                            "failure capture"
                        )
                    ordinary_source_recoveries[source_attempt_id] = verifier_evidence
                elif outcome is None:
                    raise ValueError(
                        "source-evidence recovery lacks a typed provider outcome"
                    )
                elif (
                    receipt["status"] == "succeeded"
                    and outcome["artifact_verifier"]["attempt_id"]
                    != receipt["attempt_id"]
                ):
                    raise ValueError(
                        "successful source-evidence recovery is not outcome-bound"
                    )

    classified_additional_verifier_attempts = {
        outcome["artifact_verifier"]["attempt_id"]
        for outcome in roster["provider_canary_outcomes"]
    } | {
        evidence["attempt_id"]
        for evidence in ordinary_source_recoveries.values()
    }
    unexplained_successful_verifiers = {
        evidence["attempt_id"]
        for evidence in additional_verifiers
        if evidence["remote_verifier_outcome"] == "success"
        and evidence["source_run_id"] not in roster["accepted_primary_runs"].values()
    } - classified_additional_verifier_attempts
    if unexplained_successful_verifiers:
        raise ValueError(
            "additional successful verifier is neither provider-outcome nor "
            "ordinary-failure bound"
        )

    for attempt_id, outcomes in provider_outcomes_by_attempt.items():
        receipt = by_id.get(attempt_id)
        if (
            receipt is None
            or receipt["action"] not in {"canary", "canaries"}
            or receipt["modal_cli_process_started"] is not True
        ):
            raise ValueError("provider outcome launcher identity is invalid")
        for outcome in outcomes:
            run_id = outcome["concrete_run_id"]
            if run_id not in receipt["concrete_remote_run_ids"]:
                raise ValueError("provider outcome is outside its launcher run roster")
            if receipt["action"] == "canary" and (
                receipt["run_id"] != run_id
                or receipt["harness"] != outcome["harness"]
            ):
                raise ValueError("single provider outcome is not launcher-bound")
        if receipt["status"] == "succeeded":
            expected_count = len(CANARY_ORDER) if receipt["action"] == "canaries" else 1
            if len(outcomes) != expected_count or any(
                outcome["outcome"] != "accepted" for outcome in outcomes
            ):
                raise ValueError("successful provider launcher outcomes are incomplete")
        elif receipt["action"] == "canary" and len(outcomes) != 1:
            terminal_dispositions = terminal_dispositions_by_attempt.get(
                attempt_id, []
            )
            if len(terminal_dispositions) != 1 or terminal_dispositions[0][
                "provider_disposition"
            ] != "start_unresolved_conservative":
                raise ValueError(
                    "failed single provider launcher outcome is incomplete"
                )

    for label in _PRIMARY_LABELS:
        attempt_id = roster["accepted_attempt_ids"][label]
        receipt = by_id.get(attempt_id)
        if receipt is None:
            raise ValueError(f"accepted primary attempt {label} did not succeed")
        if "accepted_primary" not in roles[attempt_id]:
            raise ValueError(f"accepted primary attempt {label} is not classified")
        accepted_run_id = primary[label][3].run_id
        if label.startswith("canary_"):
            harness = label.removeprefix("canary_")
            if receipt["action"] == "canary":
                valid = (
                    receipt["run_id"] == accepted_run_id
                    and receipt["harness"] == harness
                )
            elif receipt["action"] == "canaries":
                valid = accepted_run_id == (
                    f"{receipt['run_id']}-{_CANARY_SUFFIXES[harness]}"
                )
            else:
                valid = False
            accepted_outcome = provider_outcomes_by_run.get(accepted_run_id)
            status_valid = receipt["status"] == "succeeded" or (
                accepted_outcome is not None
                and accepted_outcome["outcome"] == "accepted"
                and accepted_outcome["launcher_attempt_id"] == attempt_id
            )
        else:
            expected_actions = {
                "cuda_environment": "cuda-environment",
                "offline_smoke": "offline-smoke",
                "candidate_smoke": "candidate-smoke",
                "resume_attempt": "checkpoint-resume",
            }
            valid = (
                receipt["action"] == expected_actions[label]
                and receipt["run_id"] == accepted_run_id
            )
            if label == "resume_attempt":
                valid = (
                    valid
                    and receipt["source_run_id"] == primary["candidate_smoke"][3].run_id
                )
            status_valid = receipt["status"] == "succeeded"
        if not valid or not status_valid:
            raise ValueError(f"accepted primary attempt {label} is not run-bound")

    for label in _PRIMARY_LABELS:
        verifier = verifiers[label]
        receipt = by_id.get(verifier["attempt_id"])
        source_attempt_id = roster["accepted_attempt_ids"][label]
        identity = _cohort_identity_from_payload(roster, field="cohort_roster")
        expected_source_bindings = [
            (
                "source_action_intent",
                modal_action_intent_receipt_path(
                    identity, source_attempt_id
                ).as_posix(),
            ),
            (
                "source_action_attempt_terminal",
                modal_action_terminal_receipt_path(
                    identity, source_attempt_id
                ).as_posix(),
            ),
            (
                "source_local_process_start",
                modal_local_process_start_receipt_path(
                    source_attempt_id
                ).as_posix(),
            ),
        ]
        source_receipt = by_id.get(source_attempt_id)
        if source_receipt is not None and source_receipt["action"] == "canaries":
            expected_source_bindings.append(
                (
                    "provider_canary_aggregate_outcomes",
                    provider_canary_aggregate_outcome_receipt_path(
                        identity, source_attempt_id
                    ).as_posix(),
                )
            )
        observed_source_bindings = (
            [
                (record["gate"], record["path"])
                for record in receipt["predecessor_receipts"][3:]
            ]
            if receipt is not None
            else []
        )
        if (
            receipt is None
            or receipt["status"] != "succeeded"
            or receipt["action"] not in {"download", "verify"}
            or receipt["run_id"] != verifier["source_run_id"]
            or receipt["verifier_run_id"] != verifier["verifier_run_id"]
            or observed_source_bindings != expected_source_bindings
        ):
            raise ValueError(
                f"artifact verifier attempt {label} is not execution-bound"
            )

    paid_verifier_attempt_ids = {
        receipt["attempt_id"]
        for receipt in attempts
        if receipt["action"] in {"download", "verify"}
        and receipt["modal_cli_process_started"]
    }
    declared_paid_verifier_attempt_ids = {
        item["attempt_id"] for item in roster["artifact_verifiers"].values()
    } | {
        item["attempt_id"] for item in roster["additional_artifact_verifiers"]
    }
    if paid_verifier_attempt_ids != declared_paid_verifier_attempt_ids:
        raise ValueError("cohort hides or invents paid artifact verifier attempts")

    failed_run_ids = sorted(
        {
            run_id
            for receipt in attempts
            if receipt["status"] != "succeeded"
            and receipt["modal_cli_process_started"]
            and receipt["action"]
            not in {"canary", "canaries", "download", "verify"}
            for run_id in receipt["concrete_remote_run_ids"]
        }
        | {
            item["concrete_run_id"]
            for item in roster["provider_canary_outcomes"]
            if item["outcome"]
            in {
                "failed",
                "provider_request_start_uncertain",
            }
        }
        | {
            item["verifier_run_id"]
            for item in roster["additional_artifact_verifiers"]
            if item["remote_verifier_outcome"] == "failure"
        }
    )
    quarantined_run_ids = sorted(
        {
            run_id
            for receipt in attempts
            if "quarantined" in roles[receipt["attempt_id"]]
            and receipt["action"] not in {"canary", "canaries"}
            for run_id in receipt["concrete_remote_run_ids"]
        }
        | {
            item["concrete_run_id"]
            for item in roster["provider_canary_outcomes"]
            if item["outcome"] != "accepted"
        }
    )
    if failed_run_ids != roster["declared_failed_run_ids"]:
        raise ValueError("cohort roster hides or invents failed run IDs")
    if quarantined_run_ids != roster["declared_quarantined_run_ids"]:
        raise ValueError("cohort roster hides or invents quarantined run IDs")

    recovery_run_ids: set[str] = set()
    linked_recovery_attempts: set[str] = set()
    recovery_links_by_failed_attempt: dict[str, list[Mapping[str, Any]]] = {}
    for link in roster["recovery_links"]:
        failed = by_id.get(link["failed_attempt_id"])
        recovery = by_id.get(link["recovery_attempt_id"])
        if (
            failed is None
            or failed["status"] == "succeeded"
            or recovery is None
            or recovery["status"] != "succeeded"
            or "recovery" not in roles[recovery["attempt_id"]]
            or set(failed["concrete_remote_run_ids"]).intersection(
                recovery["concrete_remote_run_ids"]
            )
            or _utc(
                failed["finished_at_utc"], "failed_recovery.finished_at_utc"
            )
            > _utc(
                recovery["started_at_utc"], "recovery.started_at_utc"
            )
        ):
            raise ValueError("recovery link does not connect failure to success")
        for run_id in link["recovered_run_ids"]:
            labels = [
                label for label in _PRIMARY_LABELS if primary[label][3].run_id == run_id
            ]
            if (
                len(labels) != 1
                or roster["accepted_attempt_ids"][labels[0]] != (recovery["attempt_id"])
            ):
                raise ValueError("recovery link does not name its accepted execution")
        recovery_run_ids.update(link["recovered_run_ids"])
        linked_recovery_attempts.add(recovery["attempt_id"])
        recovery_links_by_failed_attempt.setdefault(
            link["failed_attempt_id"], []
        ).append(link)
    classified_recoveries = {
        attempt_id for attempt_id, value in roles.items() if "recovery" in value
    }
    if classified_recoveries != linked_recovery_attempts:
        raise ValueError("recovery classifications and links differ")
    for outcome in roster["provider_canary_outcomes"]:
        if outcome["outcome"] in {"accepted", "completed_unaccepted"}:
            if (
                outcome["outcome"] == "completed_unaccepted"
                and provider_aggregate_child_status_by_run.get(
                    outcome["concrete_run_id"]
                )
                != "success"
            ):
                raise ValueError(
                    "completed-unaccepted provider outcome is not a successful child"
                )
            accepted_run_id = primary[f"canary_{outcome['harness']}"][3].run_id
            if any(
                link["failed_attempt_id"] == outcome["launcher_attempt_id"]
                and accepted_run_id in link["recovered_run_ids"]
                for link in roster["recovery_links"]
            ):
                raise ValueError(
                    "successful provider child is not eligible for recovery"
                )
            continue
        accepted_run_id = primary[f"canary_{outcome['harness']}"][3].run_id
        if not any(
            link["failed_attempt_id"] == outcome["launcher_attempt_id"]
            and accepted_run_id in link["recovered_run_ids"]
            for link in roster["recovery_links"]
        ):
            raise ValueError(
                "unaccepted provider outcome lacks an explicit recovery link"
            )
    ordinary_recovery_evidence: list[dict[str, Any]] = []
    ordinary_label_by_action = {
        "cuda-environment": "cuda_environment",
        "offline-smoke": "offline_smoke",
        "candidate-smoke": "candidate_smoke",
        "checkpoint-resume": "resume_attempt",
    }
    for failed_attempt_id, verifier_evidence in ordinary_source_recoveries.items():
        failed = by_id[failed_attempt_id]
        label = ordinary_label_by_action[failed["action"]]
        accepted_run_id = primary[label][3].run_id
        matching_links = recovery_links_by_failed_attempt.get(failed_attempt_id, [])
        link = matching_links[0] if len(matching_links) == 1 else None
        if (
            link is None
            or link["recovered_run_ids"] != [accepted_run_id]
            or link["recovery_attempt_id"] != roster["accepted_attempt_ids"][label]
            or failed["run_id"] == accepted_run_id
        ):
            raise ValueError(
                "evidence-backed ordinary failure lacks its exact accepted replacement"
            )
        ordinary_recovery_evidence.append(
            {
                "failed_attempt_id": failed_attempt_id,
                "failed_action": failed["action"],
                "failed_run_id": failed["run_id"],
                "failed_execution_context": verifier_evidence[
                    "source_execution_context"
                ],
                "source_evidence_verifier_attempt_id": verifier_evidence[
                    "attempt_id"
                ],
                "source_evidence_verifier_run_id": verifier_evidence[
                    "verifier_run_id"
                ],
                "replacement_attempt_id": link["recovery_attempt_id"],
                "replacement_run_id": accepted_run_id,
            }
        )
    if sorted(recovery_run_ids) != roster["declared_recovery_run_ids"]:
        raise ValueError("cohort roster hides or invents recovery run IDs")
    return {
        "failed_run_ids": failed_run_ids,
        "quarantined_run_ids": quarantined_run_ids,
        "recovery_run_ids": sorted(recovery_run_ids),
        "validation_only_attempt_ids": sorted(
            attempt_id
            for attempt_id, value in roles.items()
            if "validation_only" in value
        ),
        "evidence_backed_failed_ordinary_executions": sorted(
            ordinary_recovery_evidence,
            key=lambda item: item["failed_attempt_id"],
        ),
        "recovery_links": roster["recovery_links"],
    }


def _strict_provider_ledger(
    path: Path,
    *,
    expected_sha256: object,
    allow_exact_empty: bool = False,
) -> list[ProviderAttemptRecord]:
    raw = _read_regular_file_bytes(
        path,
        maximum_bytes=_MAX_PROVIDER_LEDGER_BYTES,
    )
    if hashlib.sha256(raw).hexdigest() != _sha256(
        expected_sha256,
        "provider_attempt_ledger_sha256",
    ):
        raise ValueError("provider outcome ledger digest changed")
    text = _decode_utf8(raw, path)
    if allow_exact_empty and text == "":
        return []
    if not text or not text.endswith("\n"):
        raise ValueError("provider attempt ledger is empty or truncated")
    records: list[ProviderAttemptRecord] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise ValueError("provider attempt ledger contains a blank record")
        try:
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"provider attempt ledger line {line_number} is invalid JSON"
            ) from error
        if not isinstance(raw, dict):
            raise ValueError("provider attempt ledger record must be an object")
        records.append(ProviderAttemptRecord.from_dict(raw))
    return records


def _provider_spend_estimate(
    root: Path,
    roster: Mapping[str, Any],
    primary: Mapping[str, tuple[dict[str, Any], Path, Any, ExecutionContextV1]],
    attempts: list[dict[str, Any]],
    *,
    app_lifecycles: Mapping[str, tuple[datetime, datetime]] | None = None,
    attribution_roster: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    identity = _cohort_identity_from_payload(roster, field="cohort_roster")
    price_basis, price_path, price_basis_sha256 = _load_price_basis(
        root, roster["provider_price_basis_path"]
    )
    ledgers: list[dict[str, Any]] = []
    response_ids: list[str] = []
    request_ids: list[str] = []
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    failed_attempts: list[dict[str, Any]] = []
    uncertain_request_starts: list[dict[str, Any]] = []
    successful_attempt_count = 0
    failed_attempt_count = 0
    uncertain_request_start_count = 0
    accepted_successful_attempt_count = 0
    provider_terminal_attempt_record_count = 0
    input_rate = _decimal_text(
        price_basis["uncached_input_usd_per_million_tokens"], "input_rate"
    )
    output_rate = _decimal_text(
        price_basis["output_usd_per_million_tokens"], "output_rate"
    )
    request_fee = _decimal_text(price_basis["per_request_fee_usd"], "per_request_fee")
    attempt_by_id = {item["attempt_id"]: item for item in attempts}
    provider_launcher_ids = {
        item["launcher_attempt_id"] for item in roster["provider_canary_outcomes"]
    }
    launcher_approvals: dict[str, dict[str, Any]] = {}
    approval_bindings: set[tuple[str, str]] = set()
    for attempt_id in sorted(provider_launcher_ids):
        terminal = attempt_by_id.get(attempt_id)
        if (
            terminal is None
            or terminal["action"] not in {"canary", "canaries"}
            or terminal["modal_cli_process_started"] is not True
            or terminal["provider_cost_approved"] is not True
        ):
            raise ValueError("provider outcome lacks one approved launcher attempt")
        if terminal["provider_price_basis_path"] != roster[
            "provider_price_basis_path"
        ] or terminal["provider_price_basis_sha256"] != price_basis_sha256:
            raise ValueError("provider launcher differs from the cohort price basis")
        plan, plan_path = _load_provider_approval_plan(
            root,
            terminal["provider_approval_plan_path"],
            expected_approval_sha256=terminal["approval_plan_sha256"],
            expected_image_source_sha256=terminal[
                "approved_image_source_sha256"
            ],
            expected_identity=identity,
            expected_preflight_binding=_candidate_preflight_binding(
                terminal["predecessor_receipts"]
            ),
        )
        approval_bindings.add(
            (terminal["provider_approval_plan_path"], terminal["approval_plan_sha256"])
        )
        harness_by_name = {item["harness"]: item for item in plan["harnesses"]}
        selected = (
            list(harness_by_name.values())
            if terminal["action"] == "canaries"
            else [harness_by_name[terminal["harness"]]]
        )
        approved_input_ceiling = sum(
            item["first_opportunity"]["conservative_input_token_ceiling"]
            for item in selected
        )
        approved_completion_ceiling = sum(
            item["request_settings"]["max_completion_tokens"]
            for item in selected
        )
        approved_request_count = sum(item["maximum_attempts"] for item in selected)
        required_bound = (
            Decimal(approved_input_ceiling) * input_rate / Decimal(1_000_000)
            + Decimal(approved_completion_ceiling)
            * output_rate
            / Decimal(1_000_000)
            + Decimal(approved_request_count) * request_fee
        )
        approved_cap = _decimal_text(
            terminal["provider_cost_cap_usd"],
            "provider_cost_cap_usd",
        )
        if approved_cap < required_bound:
            raise ValueError(
                "provider launcher cap is below its source-bound approval ceiling"
            )
        launcher_approvals[attempt_id] = {
            "plan": plan,
            "plan_path": plan_path,
            "harnesses": harness_by_name,
            "approved_cap": approved_cap,
            "required_bound": required_bound,
            "known_success_estimate": Decimal("0"),
            "failed_attempt_reserve": Decimal("0"),
            "uncertain_request_start_reserve": Decimal("0"),
        }
    if len(approval_bindings) != 1:
        raise ValueError("provider cohort does not bind one approval plan")
    known_success_estimate = Decimal("0")
    failed_attempt_reserve = Decimal("0")
    uncertain_request_start_reserve = Decimal("0")
    for outcome in roster["provider_canary_outcomes"]:
        harness = outcome["harness"]
        label = f"canary_{harness}"
        run_id = outcome["concrete_run_id"]
        launcher_attempt_id = outcome["launcher_attempt_id"]
        approval = launcher_approvals[launcher_attempt_id]
        harness_plan = approval["harnesses"][harness]
        input_ceiling = harness_plan["first_opportunity"][
            "conservative_input_token_ceiling"
        ]
        completion_ceiling = harness_plan["request_settings"][
            "max_completion_tokens"
        ]
        full_approved_request_reserve = (
            Decimal(input_ceiling) * input_rate / Decimal(1_000_000)
            + Decimal(completion_ceiling)
            * output_rate
            / Decimal(1_000_000)
            + request_fee
        )
        logical = outcome["provider_attempt_ledger_path"]
        request_start_uncertain = (
            outcome["outcome"] == "provider_request_start_uncertain"
        )
        path, ledger_state = _provider_ledger_capture(
            root,
            logical,
            outcome["provider_attempt_ledger_sha256"],
            outcome=outcome["outcome"],
        )
        records = (
            _strict_provider_ledger(
                path,
                expected_sha256=outcome["provider_attempt_ledger_sha256"],
                allow_exact_empty=request_start_uncertain,
            )
            if ledger_state == "present"
            else []
        )
        if not records and not request_start_uncertain:
            raise ValueError("provider outcome ledger must contain an attempt")
        _, _, _, run_context = _inspect_downloaded_run(
            root,
            run_id,
            _expected_download_path(run_id),
        )
        if run_context.function_name != _PRIMARY_FUNCTIONS[label]:
            raise ValueError("provider outcome execution function is invalid")
        launcher = attempt_by_id[launcher_attempt_id]
        launcher_started = _utc(
            launcher["started_at_utc"], "provider_launcher.started_at_utc"
        )
        launcher_finished = _utc(
            launcher["finished_at_utc"], "provider_launcher.finished_at_utc"
        )
        provider_app_lifecycle: tuple[datetime, datetime] | None = None
        if app_lifecycles is not None or attribution_roster is not None:
            if app_lifecycles is None or attribution_roster is None:
                raise ValueError(
                    "provider lifecycle validation is only partially bound"
                )
            attribution = attribution_roster[launcher_attempt_id]
            if (
                run_context.modal_app_id is None
                or run_context.modal_app_id not in attribution["object_ids"]
            ):
                raise ValueError("provider ledger lacks its launcher App attribution")
            provider_app_lifecycle = app_lifecycles[run_context.modal_app_id]
        request_state_evidence: dict[str, str] | None = None
        request_count_lower_bound = len(records)
        request_count_upper_bound = len(records)
        request_start_state = "started"
        if request_start_uncertain:
            uncertain, _uncertain_path = (
                _load_provider_start_uncertain_evidence(
                    root,
                    outcome["provider_start_uncertain_evidence_path"],
                    outcome["provider_start_uncertain_evidence_sha256"],
                    harness=harness,
                    run_id=run_id,
                    expected_modal_call_id=run_context.modal_call_id,
                )
            )
            if uncertain["provider_attempt_ledger_state"] != ledger_state:
                raise ValueError(
                    "provider start-uncertainty evidence misstates ledger presence"
                )
            request_count_lower_bound = uncertain[
                "provider_attempt_count_lower_bound"
            ]
            request_count_upper_bound = uncertain[
                "provider_attempt_count_upper_bound"
            ]
            request_start_state = "unknown"
            request_state_evidence = {
                "kind": "provider_request_start_uncertain",
                "path": outcome["provider_start_uncertain_evidence_path"],
                "sha256": outcome["provider_start_uncertain_evidence_sha256"],
            }
            reserve = full_approved_request_reserve * Decimal(
                request_count_upper_bound
            )
            uncertain_request_start_count += 1
            uncertain_request_start_reserve += reserve
            approval["uncertain_request_start_reserve"] += reserve
            uncertain_request_starts.append(
                {
                    "harness": harness,
                    "run_id": run_id,
                    "launcher_attempt_id": launcher_attempt_id,
                    "provider_attempt_count_lower_bound": (
                        request_count_lower_bound
                    ),
                    "provider_attempt_count_upper_bound": (
                        request_count_upper_bound
                    ),
                    "provider_request_started": uncertain[
                        "provider_request_started"
                    ],
                    "provider_attempt_ledger_state": uncertain[
                        "provider_attempt_ledger_state"
                    ],
                    "billing_treatment": uncertain["billing_treatment"],
                    "conservative_input_token_ceiling": input_ceiling,
                    "requested_completion_token_ceiling": completion_ceiling,
                    "conservative_uncertain_request_reserve_usd": format(
                        reserve, "f"
                    ),
                    "evidence_path": outcome[
                        "provider_start_uncertain_evidence_path"
                    ],
                    "evidence_sha256": outcome[
                        "provider_start_uncertain_evidence_sha256"
                    ],
                }
            )
        for record in records:
            if (
                record.harness != harness
                or record.action != "one_opportunity_engineering_canary"
                or record.execution_backend != "modal"
                or record.action_run_id != run_id
                or record.modal_call_id != run_context.modal_call_id
                or record.api_endpoint != OFFICIAL_OPENAI_API_BASE
                or record.model != TARGET_MODEL
                or record.attempt_ordinal != 1
                or record.generation_settings_sha256
                != harness_plan["generation_settings_sha256"]
            ):
                raise ValueError(f"provider ledger {harness} is not execution-bound")
            record_started = _utc(
                record.started_at_utc,
                f"provider_ledger.{harness}.started_at_utc",
            )
            record_ended = _utc(
                record.ended_at_utc,
                f"provider_ledger.{harness}.ended_at_utc",
            )
            if not (
                launcher_started
                <= record_started
                <= record_ended
                <= launcher_finished
            ):
                raise ValueError(
                    f"provider ledger {harness} falls outside its launcher attempt"
                )
            if provider_app_lifecycle is not None and not (
                provider_app_lifecycle[0]
                <= record_started
                <= record_ended
                <= provider_app_lifecycle[1]
            ):
                raise ValueError(
                    f"provider ledger {harness} falls outside its App lifecycle"
                )
            if (
                record.input_tokens is not None
                and record.input_tokens > input_ceiling
            ) or (
                record.output_tokens is not None
                and record.output_tokens > completion_ceiling
            ):
                raise ValueError(
                    f"provider ledger {harness} exceeds its approved token ceilings"
                )
            provider_terminal_attempt_record_count += 1
            if record.provider_request_id is not None:
                request_ids.append(record.provider_request_id)
            if record.provider_response_id is not None:
                response_ids.append(record.provider_response_id)
            if record.status == "error":
                failed_attempt_count += 1
                reserve = full_approved_request_reserve
                failed_attempt_reserve += reserve
                approval["failed_attempt_reserve"] += reserve
                failed_attempts.append(
                    {
                        "harness": harness,
                        "run_id": run_id,
                        "attempt_ordinal": record.attempt_ordinal,
                        "provider_request_id": record.provider_request_id,
                        "error_class": record.error_class,
                        "conservative_input_token_ceiling": input_ceiling,
                        "requested_completion_token_ceiling": completion_ceiling,
                        "conservative_failed_attempt_reserve_usd": format(
                            reserve, "f"
                        ),
                    }
                )
                continue
            successful_attempt_count += 1
            if (
                record.provider_response_id is None
                or record.provider_request_id is None
                or record.usage_known is not True
                or record.input_tokens is None
                or record.output_tokens is None
                or record.total_tokens is None
            ):
                raise ValueError(
                    "successful provider ledger lacks IDs or response usage"
                )
            input_tokens += record.input_tokens
            output_tokens += record.output_tokens
            total_tokens += record.total_tokens
            success_estimate = (
                Decimal(record.input_tokens)
                * input_rate
                / Decimal(1_000_000)
                + Decimal(record.output_tokens)
                * output_rate
                / Decimal(1_000_000)
                + request_fee
            )
            known_success_estimate += success_estimate
            approval["known_success_estimate"] += success_estimate
        success_count = sum(record.status == "success" for record in records)
        error_count = sum(record.status == "error" for record in records)
        if outcome["outcome"] in {"accepted", "completed_unaccepted"}:
            if success_count != 1 or error_count != 0 or len(records) != 1:
                raise ValueError(
                    "each completed canary must have exactly one successful request"
                )
            if (
                outcome["outcome"] == "accepted"
                and run_id != primary[label][3].run_id
            ):
                raise ValueError(
                    "accepted provider outcome differs from primary roster"
                )
            accepted_successful_attempt_count += outcome["outcome"] == "accepted"
        elif outcome["outcome"] == "failed":
            if (
                len(records) != 1
                or success_count + error_count != 1
            ):
                raise ValueError(
                    "failed provider outcome lacks one terminal provider attempt"
                )
        elif request_start_uncertain:
            if records:
                raise ValueError(
                    "start-uncertain provider outcome contains ledger records"
                )
        else:
            raise ValueError("provider outcome accounting disposition is unsupported")
        ledgers.append(
            {
                "harness": harness,
                "run_id": run_id,
                "outcome": outcome["outcome"],
                "launcher_attempt_id": launcher_attempt_id,
                "path": logical,
                "sha256": (
                    _sha256_file(path) if ledger_state == "present" else None
                ),
                "provider_terminal_attempt_record_count": len(records),
                "provider_attempt_count_lower_bound": request_count_lower_bound,
                "provider_attempt_count_upper_bound": request_count_upper_bound,
                "provider_attempt_count": request_count_upper_bound,
                "provider_request_start_state": request_start_state,
                "request_state_evidence": request_state_evidence,
                "successful_provider_attempt_count": success_count,
                "failed_provider_attempt_count": error_count,
            }
        )
    if accepted_successful_attempt_count != 4:
        raise ValueError("provider cohort requires four accepted successful requests")
    if len(response_ids) != len(set(response_ids)) or len(request_ids) != len(
        set(request_ids)
    ):
        raise ValueError("provider request and response IDs must be unique")
    if total_tokens != input_tokens + output_tokens:
        raise ValueError("aggregate provider response usage does not reconcile")
    provider_attempt_count_lower_bound = provider_terminal_attempt_record_count
    provider_attempt_count_upper_bound = (
        provider_terminal_attempt_record_count + uncertain_request_start_count
    )
    conservative_bound = (
        known_success_estimate
        + failed_attempt_reserve
        + uncertain_request_start_reserve
    )
    approved_cap_total = sum(
        (
            item["approved_cap"]
            for item in launcher_approvals.values()
        ),
        Decimal("0"),
    )
    launcher_bounds: list[dict[str, Any]] = []
    for attempt_id in sorted(launcher_approvals):
        approval = launcher_approvals[attempt_id]
        observed_bound = (
            approval["known_success_estimate"]
            + approval["failed_attempt_reserve"]
            + approval["uncertain_request_start_reserve"]
        )
        if observed_bound > approval["approved_cap"]:
            raise ValueError(
                "provider usage estimate and request reserves exceed launcher cap"
            )
        launcher_bounds.append(
            {
                "launcher_attempt_id": attempt_id,
                "provider_cost_cap_usd": format(approval["approved_cap"], "f"),
                "source_bound_approval_ceiling_usd": format(
                    approval["required_bound"], "f"
                ),
                "known_success_usage_estimate_usd": format(
                    approval["known_success_estimate"], "f"
                ),
                "failed_attempt_reserve_usd": format(
                    approval["failed_attempt_reserve"], "f"
                ),
                "uncertain_request_start_reserve_usd": format(
                    approval["uncertain_request_start_reserve"], "f"
                ),
                "conservative_observed_bound_usd": format(observed_bound, "f"),
            }
        )
    if conservative_bound > approved_cap_total:
        raise ValueError("conservative provider spend bound exceeds approved caps")
    approval_path, approval_sha256 = next(iter(approval_bindings))
    approval_file = _contained_path(
        root,
        approval_path,
        "provider_approval_plan_path",
        kind="file",
    )
    return {
        "accounting_label": (
            "known_success_usage_estimate_plus_conservative_failed_and_"
            "uncertain_request_reserves_not_billed_cost"
        ),
        "provider_attempt_count": provider_attempt_count_upper_bound,
        "provider_terminal_attempt_record_count": (
            provider_terminal_attempt_record_count
        ),
        "provider_attempt_count_lower_bound": provider_attempt_count_lower_bound,
        "provider_attempt_count_upper_bound": provider_attempt_count_upper_bound,
        "successful_provider_attempt_count": successful_attempt_count,
        "accepted_successful_provider_attempt_count": (
            accepted_successful_attempt_count
        ),
        "failed_provider_attempt_count": failed_attempt_count,
        "failed_provider_attempts": failed_attempts,
        "provider_request_start_uncertain_count": uncertain_request_start_count,
        "provider_request_start_uncertainties": uncertain_request_starts,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "provider_request_ids": sorted(request_ids),
        "provider_response_ids": sorted(response_ids),
        "estimated_provider_usd": format(known_success_estimate, "f"),
        "known_success_usage_estimate_usd": format(
            known_success_estimate, "f"
        ),
        "failed_attempt_reserve_usd": format(failed_attempt_reserve, "f"),
        "uncertain_request_start_reserve_usd": format(
            uncertain_request_start_reserve, "f"
        ),
        "conservative_provider_spend_bound_usd": format(
            conservative_bound, "f"
        ),
        "approved_provider_cap_total_usd": format(approved_cap_total, "f"),
        "launcher_approval_bounds": launcher_bounds,
        "approval_plan": {
            "path": approval_path,
            "approval_plan_sha256": approval_sha256,
            "file_sha256": _sha256_file(approval_file),
        },
        "price_basis": {
            "path": roster["provider_price_basis_path"],
            "sha256": price_basis_sha256,
            "record": price_basis,
        },
        "ledgers": ledgers,
    }


def _provider_request_state_evidence_paths(
    provider_spend_estimate: Mapping[str, Any],
) -> tuple[str, ...]:
    paths: list[str] = []
    for index, ledger in enumerate(provider_spend_estimate["ledgers"]):
        evidence = ledger["request_state_evidence"]
        if evidence is None:
            continue
        if not isinstance(evidence, dict) or set(evidence) != {
            "kind",
            "path",
            "sha256",
        }:
            raise ValueError(
                f"provider ledger {index} request-state evidence schema changed"
            )
        if evidence["kind"] != "provider_request_start_uncertain":
            raise ValueError("provider request-state evidence kind is unsupported")
        path = _text(evidence["path"], f"provider_ledgers[{index}].evidence.path")
        safe_relative_path(path)
        _sha256(evidence["sha256"], f"provider_ledgers[{index}].evidence.sha256")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError("provider request-state evidence paths are duplicated")
    return tuple(sorted(paths))


def _derive_cleanup_claims(
    root: Path,
    cohort_roster_path: str,
    *,
    recorded_at_utc: str,
) -> dict[str, Any]:
    roster, roster_path = _load_cohort_roster(root, cohort_roster_path)
    identity = _cohort_identity_from_payload(roster, field="cohort_roster")
    recorded_at = _utc(recorded_at_utc, "recorded_at_utc")
    observed_now = datetime.now(UTC)
    if recorded_at < _utc(
        roster["snapshot_captured_at_utc"], "snapshot_captured_at_utc"
    ):
        raise ValueError("cleanup receipt predates its frozen snapshots")
    snapshot_manifest, snapshot_manifest_path, snapshot_manifest_sha256, rows = (
        _load_cleanup_snapshot_capture(root, roster, identity)
    )
    lineage, lineage_path, lineage_sha256 = _load_migration_lineage(
        root,
        roster,
        identity,
    )
    prior_accounting = [
        _load_prior_quarantine_accounting(
            root,
            item["accounting_receipt"]["path"],
        )
        for item in lineage["prior_quarantined_cohorts"]
    ]
    prior_app_lifecycles: dict[str, dict[str, str]] = {}
    prior_billing_row_keys: set[str] = set()
    prior_run_ids: set[str] = set()
    prior_app_ids: set[str] = set()
    prior_call_ids: set[str] = set()
    for prior_payload, prior_metadata in prior_accounting:
        for lifecycle in prior_payload["app_lifecycles"]:
            app_id = lifecycle["app_id"]
            if app_id in prior_app_lifecycles:
                raise ValueError("prior migration App ID is reused")
            prior_app_lifecycles[app_id] = lifecycle
        selected_prior_billing_keys = {
            item["row_sha256"]
            for item in prior_payload["selected_billing_rows"]
        }
        if prior_billing_row_keys.intersection(selected_prior_billing_keys):
            raise ValueError("prior migration billing row is multiply owned")
        prior_billing_row_keys.update(selected_prior_billing_keys)
        prior_run_ids.update(prior_metadata["run_ids"])
        prior_app_ids.update(prior_metadata["app_ids"])
        prior_call_ids.update(prior_metadata["call_ids"])
    primary = _primary_executions_from_roster(root, roster)
    primary_contexts = {label: item[3] for label, item in primary.items()}
    image_hashes = {item.image_source_sha256 for item in primary_contexts.values()}
    image_ids = {item.modal_image_id for item in primary_contexts.values()}
    if len(image_hashes) != 1 or None in image_ids or len(image_ids) != 1:
        raise ValueError("accepted primary executions do not share one image")
    verifier_evidence = _artifact_verifier_executions(root, roster, primary)
    verifier_contexts = {
        label: ExecutionContextV1.from_dict(record["execution_context"])
        for label, record in verifier_evidence.items()
    }
    attempts, attempt_evidence, aggregate_outcomes = _load_action_attempts(
        root,
        _cohort_identity_from_payload(roster, field="cohort_roster"),
        roster["action_attempt_receipts"],
        roster["action_intent_receipts"],
        roster["provider_canary_aggregate_outcome_receipts"],
    )
    additional_verifier_evidence = _additional_artifact_verifier_executions(
        root,
        roster,
        attempts,
    )
    final_contexts = [*primary_contexts.values(), *verifier_contexts.values()]
    final_contexts.extend(
        ExecutionContextV1.from_dict(record["execution_context"])
        for record in additional_verifier_evidence
        if record.get("execution_context") is not None
    )
    final_app_ids = {
        context.modal_app_id
        for context in final_contexts
        if context.modal_app_id is not None
    }
    final_call_ids = {
        context.modal_call_id
        for context in final_contexts
        if context.modal_call_id is not None
    }
    if final_app_ids.intersection(prior_app_ids) or final_call_ids.intersection(
        prior_call_ids
    ):
        raise ValueError("final and prior Modal execution IDs overlap")
    cohort = _validate_attempt_cohort(
        root,
        roster,
        attempts,
        primary,
        verifier_evidence,
        additional_verifier_evidence,
        aggregate_outcomes,
    )

    for name, id_field in (
        ("app_list", "app_id"),
        ("container_list", "container_id"),
        ("endpoint_list", "endpoint_id"),
        ("volume_list", "name"),
    ):
        identifiers = [
            _text(row[id_field], f"{name}[{index}].{id_field}")
            for index, row in enumerate(rows[name])
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"{name} contains duplicate {id_field} values")

    by_attempt = {item["attempt_id"]: item for item in attempts}
    attribution_roster = {
        item["attempt_id"]: item for item in roster["billing_attributions"]
    }
    attributed_app_ids = {
        object_id
        for item in attribution_roster.values()
        for object_id in item["object_ids"]
    }
    for attempt_id, attribution in attribution_roster.items():
        receipt = by_attempt[attempt_id]
        if attribution["disposition"] == "no_remote_start":
            if receipt["modal_cli_process_started"]:
                raise ValueError(
                    "started Modal process may not claim no remote start"
                )
        elif attribution["disposition"] == "start_uncertain":
            if not receipt["modal_cli_process_started"]:
                raise ValueError(
                    "unstarted Modal process may not claim uncertain start"
                )
        elif not receipt["modal_cli_process_started"]:
            raise ValueError("unstarted Modal process may not claim billed execution")
    for label, context in primary_contexts.items():
        attempt_id = roster["accepted_attempt_ids"][label]
        if context.modal_app_id not in attribution_roster[attempt_id]["object_ids"]:
            raise ValueError(f"primary execution {label} lacks billing attribution")
    for label, context in verifier_contexts.items():
        attempt_id = roster["artifact_verifiers"][label]["attempt_id"]
        if context.modal_app_id not in attribution_roster[attempt_id]["object_ids"]:
            raise ValueError(f"artifact verifier {label} lacks billing attribution")
    for record in additional_verifier_evidence:
        attempt_id = record["attempt_id"]
        if attribution_roster[attempt_id]["object_ids"] != record[
            "billing_object_ids"
        ]:
            raise ValueError("additional verifier cost attribution is swappable")
        if record["execution_context"] is not None:
            context = ExecutionContextV1.from_dict(record["execution_context"])
            if context.modal_app_id not in record["billing_object_ids"]:
                raise ValueError(
                    "additional verifier lacks its execution App attribution"
                )
    for outcome in roster["provider_canary_outcomes"]:
        label = f"canary_{outcome['harness']}"
        if outcome["outcome"] == "accepted":
            context = primary_contexts[label]
        else:
            _, _, _, context = _inspect_downloaded_run(
                root,
                outcome["concrete_run_id"],
                _expected_download_path(outcome["concrete_run_id"]),
            )
        if context.modal_app_id in prior_app_ids or context.modal_call_id in (
            prior_call_ids
        ):
            raise ValueError("final and prior provider execution IDs overlap")
        attribution = attribution_roster[outcome["launcher_attempt_id"]]
        if (
            attribution["disposition"] != "billed"
            or context.modal_app_id is None
            or context.modal_app_id not in attribution["object_ids"]
        ):
            raise ValueError(
                "provider outcome lacks its execution App billing attribution"
            )

    known_states = {
        "deployed",
        "ephemeral (detached)",
        "disabled",
        "ephemeral",
        "initializing...",
        "stopped",
        "stopping...",
    }
    app_rows: dict[str, dict[str, Any]] = {}
    app_lifecycles: dict[str, tuple[datetime, datetime]] = {}
    active_app_count = 0
    billing_start = _utc(roster["billing_window_start_utc"], "billing_window_start_utc")
    billing_end = _utc(roster["billing_window_end_utc"], "billing_window_end_utc")
    for index, row in enumerate(rows["app_list"]):
        app_id = _text(row["app_id"], f"app_list[{index}].app_id")
        description = _text(row["description"], f"app_list[{index}].description")
        state = _text(row["state"], f"app_list[{index}].state")
        if state not in known_states:
            raise ValueError(f"app_list[{index}].state is unsupported")
        tasks_text = _text(row["tasks"], f"app_list[{index}].tasks")
        if not tasks_text.isdigit():
            raise ValueError(f"app_list[{index}].tasks must be a decimal integer")
        tasks = int(tasks_text)
        created = _raw_timestamp_utc(
            row["created_at"], f"app_list[{index}].created_at", naive_utc=False
        )
        stopped_raw = row["stopped_at"]
        stopped = (
            _raw_timestamp_utc(
                stopped_raw, f"app_list[{index}].stopped_at", naive_utc=False
            )
            if stopped_raw is not None
            else None
        )
        if state == "stopped" and stopped is None:
            raise ValueError("stopped migration app lacks stopped_at")
        if state not in {"disabled", "stopped"} and stopped is not None:
            raise ValueError("active app unexpectedly has stopped_at")
        if app_id in attributed_app_ids:
            if (
                description != APP_NAME
                or state != "stopped"
                or tasks != 0
                or stopped is None
                or stopped < created
                or not (billing_start <= created <= stopped <= billing_end)
            ):
                raise ValueError(
                    f"migration app {app_id} did not stop cleanly in-window"
                )
            app_rows[app_id] = row
            app_lifecycles[app_id] = (created, stopped)
        elif app_id in prior_app_ids:
            expected_lifecycle = prior_app_lifecycles[app_id]
            if (
                description != APP_NAME
                or state != "stopped"
                or tasks != 0
                or stopped is None
                or stopped < created
                or _utc_z(created) != expected_lifecycle["created_at_utc"]
                or _utc_z(stopped) != expected_lifecycle["stopped_at_utc"]
            ):
                raise ValueError(
                    f"prior migration app {app_id} differs from quarantine evidence"
                )
        if description == APP_NAME and app_id not in (
            attributed_app_ids | prior_app_ids
        ):
            raise ValueError("app snapshot contains an unattributed migration app")
        if description == APP_NAME and (
            state not in {"disabled", "stopped"} or tasks != 0
        ):
            active_app_count += 1
    if set(app_rows) != attributed_app_ids:
        raise ValueError("app snapshot does not cover every attributed migration app")
    if active_app_count:
        raise ValueError("active_app_count must be zero after cleanup")

    attempt_by_app_id: dict[str, str] = {}
    for attempt_id, attribution in attribution_roster.items():
        receipt = by_attempt[attempt_id]
        attempt_started = _utc(
            receipt["started_at_utc"], "attempt.started_at_utc"
        )
        attempt_finished = _utc(
            receipt["finished_at_utc"], "attempt.finished_at_utc"
        )
        for app_id in attribution["object_ids"]:
            previous = attempt_by_app_id.setdefault(app_id, attempt_id)
            if previous != attempt_id:
                raise ValueError("migration App is attributed to multiple attempts")
            created, stopped = app_lifecycles[app_id]
            if (
                created < attempt_started - MODAL_APP_LIFECYCLE_CLOCK_TOLERANCE
                or stopped
                > attempt_finished + MODAL_APP_LIFECYCLE_CLOCK_TOLERANCE
            ):
                raise ValueError(
                    "migration App lifecycle is not contained by its launcher attempt"
                )

    volume_run_inventory = _volume_run_directory_inventory(
        rows["run_directory_list"],
        roster,
        prior_quarantined_run_ids=prior_run_ids,
    )
    _validate_owned_volume_run_time_bounds(
        rows["run_directory_list"],
        owned_run_ids=set(volume_run_inventory["required_current_run_ids"])
        | set(volume_run_inventory["required_prior_quarantined_run_ids"])
        | {volume_run_inventory["superseded_run_id"]},
        captured_at=_utc(
            snapshot_manifest["snapshots"]["run_directory_list"][
                "captured_at_utc"
            ],
            "snapshot_capture.run_directory_list.captured_at_utc",
        ),
        recorded_at=recorded_at,
        observed_now=observed_now,
    )
    _validate_owned_volume_run_start_times(rows["run_directory_list"], attempts)

    container_ids: set[str] = set()
    active_container_count = 0
    for index, row in enumerate(rows["container_list"]):
        container_id = _text(
            row["container_id"], f"container_list[{index}].container_id"
        )
        if container_id in container_ids:
            raise ValueError("container list contains duplicate container IDs")
        container_ids.add(container_id)
        app_id = _text(row["app_id"], f"container_list[{index}].app_id")
        app_name = _text(row["app_name"], f"container_list[{index}].app_name")
        start_time = _text(row["start_time"], f"container_list[{index}].start_time")
        if start_time != "Pending":
            _raw_timestamp_utc(
                start_time, f"container_list[{index}].start_time", naive_utc=False
            )
        if app_id in attributed_app_ids | prior_app_ids or app_name == APP_NAME:
            active_container_count += 1
    if active_container_count:
        raise ValueError("active_container_count must be zero after cleanup")

    endpoint_ids: set[str] = set()
    active_endpoint_count = 0
    for index, row in enumerate(rows["endpoint_list"]):
        endpoint_id = _text(row["endpoint_id"], f"endpoint_list[{index}].endpoint_id")
        if endpoint_id in endpoint_ids:
            raise ValueError("endpoint list contains duplicate endpoint IDs")
        endpoint_ids.add(endpoint_id)
        name = _text(row["name"], f"endpoint_list[{index}].name")
        _text(row["status"], f"endpoint_list[{index}].status")
        _raw_timestamp_utc(
            row["created_at"], f"endpoint_list[{index}].created_at", naive_utc=False
        )
        _text(row["created_by"], f"endpoint_list[{index}].created_by")
        if APP_NAME in name:
            active_endpoint_count += 1
    if active_endpoint_count:
        raise ValueError("active_endpoint_count must be zero after cleanup")

    volume_list_captured_at = _utc(
        snapshot_manifest["snapshots"]["volume_list"]["captured_at_utc"],
        "snapshot_capture.volume_list.captured_at_utc",
    )
    _validate_snapshot_artifact_volume(
        rows["volume_list"],
        captured_at=volume_list_captured_at,
        recorded_at=recorded_at,
        observed_now=observed_now,
        field_prefix="volume_list",
        timestamp_field="artifact Volume creation",
        missing_message="volume snapshot must contain exactly one artifact Volume",
        duplicate_message="volume snapshot must contain exactly one artifact Volume",
        missing_is_incomplete=False,
    )

    billing_by_object: dict[str, list[tuple[str, Decimal]]] = {}
    billing_keys: set[str] = set()
    accounting_keys: set[tuple[Any, ...]] = set()
    observed_intervals: set[datetime] = set()
    for index, row in enumerate(rows["billing_report"]):
        object_id = _text(row["object_id"], f"billing_report[{index}].object_id")
        description = _text(
            row["description"], f"billing_report[{index}].description", allow_empty=True
        )
        environment = _text(
            row["environment"], f"billing_report[{index}].environment", allow_empty=True
        )
        interval = _raw_timestamp_utc(
            row["interval_start"],
            f"billing_report[{index}].interval_start",
            naive_utc=False,
        )
        if interval.minute or interval.second or interval.microsecond:
            raise ValueError("billing report is not hourly aligned")
        _text(row["resource"], f"billing_report[{index}].resource")
        cost = _decimal_text(row["cost"], f"billing_report[{index}].cost")
        row_key = canonical_sha256(row)
        accounting_key = _modal_billing_accounting_key(
            row,
            field=f"billing_report[{index}]",
        )
        if accounting_key in accounting_keys:
            raise ValueError("billing report contains a duplicate accounting row")
        accounting_keys.add(accounting_key)
        if row_key in billing_keys:
            raise ValueError("billing report contains a duplicate row")
        billing_keys.add(row_key)
        if (
            description == APP_NAME
            and environment == MODAL_ENVIRONMENT
            and object_id not in (attributed_app_ids | prior_app_ids)
        ):
            raise ValueError("billing report contains an unattributed migration row")
        if object_id in attributed_app_ids:
            if description != APP_NAME:
                raise ValueError(
                    "owned migration billing row has the wrong description"
                )
            if environment != MODAL_ENVIRONMENT:
                raise ValueError("migration billing row uses the wrong environment")
            if cost > 0:
                created, stopped = app_lifecycles[object_id]
                attempt = by_attempt[attempt_by_app_id[object_id]]
                attempt_started = _utc(
                    attempt["started_at_utc"], "attempt.started_at_utc"
                )
                attempt_finished = _utc(
                    attempt["finished_at_utc"], "attempt.finished_at_utc"
                )
                interval_end = interval + timedelta(hours=1)
                if not (interval <= stopped and interval_end > created):
                    raise ValueError(
                        "positive billing row does not overlap its App lifecycle"
                    )
                if not (
                    interval <= attempt_finished
                    and interval_end > attempt_started
                ):
                    raise ValueError(
                        "positive billing row does not overlap its launcher attempt"
                    )
        elif object_id in prior_app_ids:
            if description != APP_NAME:
                raise ValueError(
                    "owned prior migration billing row has the wrong description"
                )
            if environment != MODAL_ENVIRONMENT:
                raise ValueError(
                    "prior migration billing row uses the wrong environment"
                )
            if cost > 0 and row_key not in prior_billing_row_keys:
                raise ValueError(
                    "positive prior billing row is absent from quarantine accounting"
                )
        observed_intervals.add(interval)
        billing_by_object.setdefault(object_id, []).append((row_key, cost))
    if any(
        interval < billing_start or interval >= billing_end
        for interval in observed_intervals
    ):
        raise ValueError(
            "billing report contains a row outside the completed query window"
        )

    derived_attributions: list[dict[str, Any]] = []
    cohort_total = Decimal("0")
    assigned_row_keys: set[str] = set()
    for attempt_id in sorted(attribution_roster):
        attribution = attribution_roster[attempt_id]
        receipt = by_attempt[attempt_id]
        started = _utc(receipt["started_at_utc"], "attempt.started_at_utc")
        finished = _utc(receipt["finished_at_utc"], "attempt.finished_at_utc")
        if not (billing_start <= started <= finished <= billing_end):
            raise ValueError("action attempt falls outside the frozen billing window")
        attributed_rows = [
            item
            for object_id in attribution["object_ids"]
            for item in billing_by_object.get(object_id, [])
        ]
        keys = sorted(item[0] for item in attributed_rows)
        if assigned_row_keys.intersection(keys):
            raise ValueError("Modal billing rows were attributed more than once")
        assigned_row_keys.update(keys)
        total = sum((item[1] for item in attributed_rows), Decimal("0"))
        cap_text = receipt["modal_cost_cap_usd"]
        approved_cap = (
            _decimal_text(cap_text, "attempt.modal_cost_cap_usd")
            if cap_text is not None
            else None
        )
        estimate = receipt["modal_cost_estimate"]
        estimate_text = (
            estimate["action_estimate_usd"]
            if isinstance(estimate, dict)
            else None
        )
        attributed_within_cap = (
            total <= approved_cap if approved_cap is not None else total == 0
        )
        cohort_total += total
        derived_attributions.append(
            {
                "attempt_id": attempt_id,
                "disposition": attribution["disposition"],
                "object_ids": attribution["object_ids"],
                "billing_row_keys": keys,
                "billing_total_usd": format(total, "f"),
                "approved_modal_cost_cap_usd": cap_text,
                "preflight_modal_action_estimate_usd": estimate_text,
                "attributed_app_billing_within_approved_cap": (
                    attributed_within_cap
                ),
            }
        )
    unattributed_migration_rows = {
        row_key
        for object_id in attributed_app_ids
        for row_key, _ in billing_by_object.get(object_id, [])
    } - assigned_row_keys
    if unattributed_migration_rows:
        raise ValueError("migration billing rows remain unattributed")

    measured_by_attempt = {
        item["attempt_id"]: Decimal("0") for item in attempts
    }
    for record in derived_attributions:
        measured_by_attempt[record["attempt_id"]] = _decimal_text(
            record["billing_total_usd"],
            "cleanup.billing_total_usd",
        )
    unresolved_attempt_ids = {
        disposition["attempt_id"]
        for disposition in roster["terminal_run_dispositions"]
        if disposition["execution_disposition"]
        == "may_have_started_unresolved_quarantined"
    }
    final_modal_compute_exposure = _derive_modal_compute_exposure(
        attempts,
        measured_by_attempt=measured_by_attempt,
        unresolved_attempt_ids=unresolved_attempt_ids,
        accounting_label=(
            "final_cohort_measured_billing_plus_unresolved_or_lagged_compute_"
            "reserve_not_a_platform_hard_bound"
        ),
    )
    prior_modal_cohorts: list[dict[str, Any]] = []
    prior_modal_measured = Decimal("0")
    prior_modal_reserve = Decimal("0")
    prior_modal_conservative = Decimal("0")
    prior_lineage_by_path = {
        item["accounting_receipt"]["path"]: item
        for item in lineage["prior_quarantined_cohorts"]
    }
    for prior_payload, prior_metadata in prior_accounting:
        prior_path = modal_prior_quarantine_accounting_path(
            prior_metadata["identity"]
        ).as_posix()
        prior_entry = prior_lineage_by_path[prior_path]
        exposure = prior_payload["modal_compute_exposure"]
        if not exact_json_equal(prior_entry["modal_compute_exposure"], exposure):
            raise ValueError("prior Modal exposure differs from migration lineage")
        measured = _decimal_text(
            exposure["measured_app_billing_usd"],
            "prior_modal.measured_app_billing_usd",
        )
        reserve = _decimal_text(
            exposure["unresolved_compute_reserve_usd"],
            "prior_modal.unresolved_compute_reserve_usd",
        )
        conservative = _decimal_text(
            exposure["conservative_compute_exposure_usd"],
            "prior_modal.conservative_compute_exposure_usd",
        )
        prior_modal_measured += measured
        prior_modal_reserve += reserve
        prior_modal_conservative += conservative
        prior_modal_cohorts.append(
            {
                "identity": modal_cohort_identity_dict(
                    prior_metadata["identity"]
                ),
                "accounting_receipt": prior_entry["accounting_receipt"],
                "modal_compute_exposure": exposure,
            }
        )
    if (
        prior_modal_measured
        != _decimal_text(
            lineage["prior_modal_measured_app_billing_usd"],
            "lineage.prior_modal_measured_app_billing_usd",
        )
        or prior_modal_reserve
        != _decimal_text(
            lineage["prior_modal_unresolved_compute_reserve_usd"],
            "lineage.prior_modal_unresolved_compute_reserve_usd",
        )
        or prior_modal_conservative
        != _decimal_text(
            lineage["prior_modal_conservative_exposure_usd"],
            "lineage.prior_modal_conservative_exposure_usd",
        )
    ):
        raise ValueError("prior Modal exposure totals do not reconcile")
    final_modal_measured = _decimal_text(
        final_modal_compute_exposure["measured_app_billing_usd"],
        "final_modal.measured_app_billing_usd",
    )
    final_modal_reserve = _decimal_text(
        final_modal_compute_exposure["unresolved_compute_reserve_usd"],
        "final_modal.unresolved_compute_reserve_usd",
    )
    final_modal_conservative = _decimal_text(
        final_modal_compute_exposure["conservative_compute_exposure_usd"],
        "final_modal.conservative_compute_exposure_usd",
    )
    modal_compute_exposure = {
        "accounting_label": (
            "migration_measured_modal_billing_plus_unresolved_or_lagged_"
            "reserves_not_platform_hard_bounds"
        ),
        "final_cohort": final_modal_compute_exposure,
        "prior_quarantined_cohorts": prior_modal_cohorts,
        "migration_measured_app_billing_usd": format(
            final_modal_measured + prior_modal_measured,
            "f",
        ),
        "migration_unresolved_compute_reserve_usd": format(
            final_modal_reserve + prior_modal_reserve,
            "f",
        ),
        "migration_conservative_compute_exposure_usd": format(
            final_modal_conservative + prior_modal_conservative,
            "f",
        ),
        "local_authorization_is_platform_hard_bound": False,
    }

    cleanup_root = primary["cuda_environment"][1]
    cleanup_manifest = primary["cuda_environment"][2]
    policy_path, policy_sha256, bound_image_sha256 = _detached_call_policy(
        root, cleanup_root, cleanup_manifest.image_source_sha256
    )
    verifier_network_policy = _artifact_verifier_network_policy(
        root,
        cleanup_root,
        cleanup_manifest.image_source_sha256,
    )
    provider_estimate = _provider_spend_estimate(
        root,
        roster,
        primary,
        attempts,
        app_lifecycles=app_lifecycles,
        attribution_roster=attribution_roster,
    )
    journal_provider_estimate = lineage["selected_final"][
        "provider_spend_estimate"
    ]
    if not set(provider_estimate["provider_request_ids"]) <= set(
        journal_provider_estimate["provider_request_ids"]
    ) or not set(provider_estimate["provider_response_ids"]) <= set(
        journal_provider_estimate["provider_response_ids"]
    ):
        raise ValueError("detailed provider outcomes exceed journal evidence")
    for field in (
        "provider_launcher_attempt_count",
        "provider_terminal_attempt_record_count",
        "provider_attempt_count_lower_bound",
        "provider_attempt_count_upper_bound",
        "successful_provider_attempt_count",
        "failed_provider_attempt_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "provider_request_ids",
        "provider_response_ids",
        "known_success_usage_estimate_usd",
        "failed_attempt_reserve_usd",
        "uncertain_request_start_reserve_usd",
        "conservative_provider_spend_bound_usd",
        "approved_provider_cap_total_usd",
        "launcher_approval_bounds",
    ):
        provider_estimate[field] = journal_provider_estimate[field]
    provider_estimate["estimated_provider_usd"] = journal_provider_estimate[
        "known_success_usage_estimate_usd"
    ]
    provider_estimate["journal_provider_spend_estimate"] = (
        journal_provider_estimate
    )
    prior_request_ids = set().union(
        *(metadata["provider_request_ids"] for _payload, metadata in prior_accounting)
    ) if prior_accounting else set()
    prior_response_ids = set().union(
        *(metadata["provider_response_ids"] for _payload, metadata in prior_accounting)
    ) if prior_accounting else set()
    if prior_request_ids.intersection(
        provider_estimate["provider_request_ids"]
    ) or prior_response_ids.intersection(provider_estimate["provider_response_ids"]):
        raise ValueError("final and prior provider IDs overlap")
    final_provider_bound = _decimal_text(
        provider_estimate["conservative_provider_spend_bound_usd"],
        "cleanup.final_provider_spend_bound",
    )
    lineage_final_provider_bound = _decimal_text(
        lineage["final_provider_spend_bound_usd"],
        "lineage.final_provider_spend_bound_usd",
    )
    selected_final_provider_bound = _decimal_text(
        lineage["selected_final"]["provider_spend_estimate"][
            "conservative_provider_spend_bound_usd"
        ],
        "lineage.selected_final.provider_spend_bound",
    )
    if not (
        final_provider_bound
        == lineage_final_provider_bound
        == selected_final_provider_bound
    ):
        raise ValueError("final provider spend bound differs across migration seals")
    prior_provider_cohorts: list[dict[str, Any]] = []
    prior_provider_bound = Decimal("0")
    prior_lineage_by_path = {
        item["accounting_receipt"]["path"]: item
        for item in lineage["prior_quarantined_cohorts"]
    }
    for prior_payload, prior_metadata in prior_accounting:
        bound = _decimal_text(
            prior_payload["provider_spend_estimate"][
                "conservative_provider_spend_bound_usd"
            ],
            "prior.provider_spend_bound",
        )
        prior_provider_bound += bound
        prior_path = modal_prior_quarantine_accounting_path(
            prior_metadata["identity"]
        ).as_posix()
        prior_entry = prior_lineage_by_path[prior_path]
        if not exact_json_equal(
            prior_entry["provider_spend_estimate"],
            prior_payload["provider_spend_estimate"],
        ):
            raise ValueError("prior provider estimate differs from migration lineage")
        prior_provider_cohorts.append(
            {
                "identity": modal_cohort_identity_dict(prior_metadata["identity"]),
                "accounting_receipt": prior_entry["accounting_receipt"],
                "conservative_provider_spend_bound_usd": format(bound, "f"),
            }
        )
    lineage_prior_provider_bound = _decimal_text(
        lineage["prior_provider_spend_bound_usd"],
        "lineage.prior_provider_spend_bound_usd",
    )
    lineage_migration_provider_bound = _decimal_text(
        lineage["migration_provider_spend_bound_usd"],
        "lineage.migration_provider_spend_bound_usd",
    )
    if (
        prior_provider_bound != lineage_prior_provider_bound
        or final_provider_bound + prior_provider_bound
        != lineage_migration_provider_bound
    ):
        raise ValueError("migration provider spend bounds do not reconcile")
    migration_provider_spend_estimate = {
        "accounting_label": (
            "final_plus_all_prior_conservative_provider_bounds_not_billed_cost"
        ),
        "final_cohort_conservative_provider_spend_bound_usd": format(
            final_provider_bound,
            "f",
        ),
        "prior_quarantined_conservative_provider_spend_bound_usd": format(
            prior_provider_bound,
            "f",
        ),
        "migration_conservative_provider_spend_bound_usd": format(
            final_provider_bound + prior_provider_bound,
            "f",
        ),
        "prior_cohorts": prior_provider_cohorts,
    }

    modal_basis_by_path: dict[str, str] = {}
    for attempt in attempts:
        logical = attempt["modal_price_basis_path"]
        digest = attempt["modal_price_basis_sha256"]
        if logical is None or digest is None:
            continue
        previous = modal_basis_by_path.setdefault(logical, digest)
        if previous != digest:
            raise ValueError("Modal price basis path has mixed digests")
    if not modal_basis_by_path:
        raise ValueError("cleanup lacks a bound Modal storage price basis")
    modal_basis_bindings: list[dict[str, Any]] = []
    volume_rates: list[Decimal] = []
    for logical, digest in sorted(modal_basis_by_path.items()):
        _price, rates, price_path = load_modal_price_basis(
            root,
            logical,
            expected_raw_sha256=digest,
            expected_image_source_sha256=identity.image_source_sha256,
            require_freshness=False,
        )
        raw = _read_regular_file_bytes(
            price_path,
            maximum_bytes=_MAX_JSON_OBJECT_BYTES,
        )
        modal_basis_bindings.append(
            {
                "path": logical,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
        )
        volume_rates.append(rates["volume"])
    conservative_volume_rate = max(volume_rates)
    current_retained_count = (
        len(volume_run_inventory["required_current_run_ids"]) + 1
    )
    bytes_per_run = MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES + MAX_ARTIFACT_MANIFEST_BYTES
    current_retained_bytes = current_retained_count * bytes_per_run
    current_retained_gib = Decimal(current_retained_bytes) / Decimal(1024**3)
    current_monthly_estimate = current_retained_gib * conservative_volume_rate
    prior_retained_bytes = sum(
        payload["retained_storage_estimate"]["conservative_total_bytes"]
        for payload, _metadata in prior_accounting
    )
    prior_monthly_estimate = sum(
        (
            _decimal_text(
                payload["retained_storage_estimate"]["estimated_monthly_usd"],
                "prior_retained_storage.estimated_monthly_usd",
            )
            for payload, _metadata in prior_accounting
        ),
        Decimal("0"),
    )
    retained_storage_estimate = {
        "scope": "conservative_retained_Volume_storage_estimate_not_billed_cost",
        "current_and_legacy_retained_run_count": current_retained_count,
        "prior_quarantined_retained_run_count": len(prior_run_ids),
        "conservative_bytes_per_run": bytes_per_run,
        "conservative_total_bytes": current_retained_bytes + prior_retained_bytes,
        "current_conservative_volume_rate_usd_per_gib_month": format(
            conservative_volume_rate,
            "f",
        ),
        "estimated_monthly_usd": format(
            current_monthly_estimate + prior_monthly_estimate,
            "f",
        ),
        "modal_price_basis_bindings": modal_basis_bindings,
        "included_shared_volume_quota_subtracted": False,
        "basis": (
            "every retained current, prior, and legacy run times the per-run "
            "artifact-download-plus-manifest byte caps; not measured storage billing"
        ),
    }
    prior_total = _decimal_text(
        lineage["prior_app_compute_total_usd"],
        "migration_lineage.prior_app_compute_total_usd",
    )
    migration_total = cohort_total + prior_total + _SUPERSEDED_USAGE_USD
    return {
        "cohort_roster_path": cohort_roster_path,
        "cohort_roster_sha256": _sha256_file(roster_path),
        "modal_cli_version": MODAL_VERSION,
        "accepted_primary_executions": {
            label: context.to_dict() for label, context in primary_contexts.items()
        },
        "artifact_verifier_executions": verifier_evidence,
        "additional_artifact_verifier_executions": additional_verifier_evidence,
        "action_attempts": attempt_evidence,
        "final_accepted_roster": roster["accepted_primary_runs"],
        **cohort,
        "active_app_count": active_app_count,
        "active_container_count": active_container_count,
        "active_endpoint_count": active_endpoint_count,
        "volume_present": True,
        "task_function_call_inventory": _TASK_FUNCTION_CALL_INVENTORY,
        "direct_detached_call_inventory": _DIRECT_DETACHED_CALL_INVENTORY,
        "detached_calls_prohibited": True,
        "detached_call_policy_source_path": policy_path,
        "detached_call_policy_source_sha256": policy_sha256,
        "bound_image_source_sha256": bound_image_sha256,
        "artifact_verifier_network_policy": verifier_network_policy,
        "billing_window_start_utc": roster["billing_window_start_utc"],
        "billing_window_end_utc": roster["billing_window_end_utc"],
        "cohort_billing_total_usd": format(cohort_total, "f"),
        "superseded_usage_usd": format(_SUPERSEDED_USAGE_USD, "f"),
        "migration_total_usd": format(migration_total, "f"),
        "billing_scope": (
            "final_and_prior_app_attributed_compute_plus_preserved_legacy_usage"
        ),
        "billing_attributions": derived_attributions,
        "modal_compute_exposure": modal_compute_exposure,
        "provider_spend_estimate": provider_estimate,
        "migration_provider_spend_estimate": migration_provider_spend_estimate,
        "volume_run_directory_inventory": volume_run_inventory,
        "retained_storage_estimate": retained_storage_estimate,
        "snapshot_capture_manifest_path": (
            snapshot_manifest_path.relative_to(root.resolve()).as_posix()
        ),
        "snapshot_capture_manifest_sha256": snapshot_manifest_sha256,
        "migration_lineage_path": lineage_path.relative_to(
            root.resolve()
        ).as_posix(),
        "migration_lineage_sha256": lineage_sha256,
        "snapshots": snapshot_manifest["snapshots"],
    }


def _validate_cleanup_payload(payload: Mapping[str, Any], *, root: Path) -> str:
    if set(payload) != _CLEANUP_FIELDS:
        raise ValueError("resource cleanup receipt has unexpected or missing fields")
    contract = MODAL_READINESS_RECEIPT_CONTRACTS["modal_resource_cleanup_validated"][
        "receipt_contract"
    ]
    if (
        payload["schema_name"] != contract["schema_name"]
        or payload["schema_version"] != contract["schema_version"]
    ):
        raise ValueError("resource cleanup receipt has the wrong schema contract")
    recorded_at = _utc(payload["recorded_at_utc"], "recorded_at_utc")
    identity = _cohort_identity_from_payload(payload, field="resource_cleanup")
    if payload["app_name"] != APP_NAME or payload["volume_name"] != VOLUME_NAME:
        raise ValueError("resource cleanup receipt has the wrong Modal identity")
    cohort_roster_path = _text(payload["cohort_roster_path"], "cohort_roster_path")
    roster, roster_path = _load_cohort_roster(root, cohort_roster_path)
    _assert_identity_matches(roster, identity, field="resource_cleanup.roster")
    _sha256(payload["cohort_roster_sha256"], "cohort_roster_sha256")
    if payload["cohort_roster_sha256"] != _sha256_file(roster_path):
        raise ValueError("resource cleanup cohort roster digest mismatch")
    lineage, _lineage_path, _lineage_sha256 = _load_migration_lineage(
        root,
        roster,
        identity,
    )
    derived = _derive_cleanup_claims(
        root,
        cohort_roster_path,
        recorded_at_utc=_utc_z(recorded_at),
    )
    for field, expected_value in derived.items():
        observed = payload[field]
        if type(expected_value) is bool and type(observed) is not bool:
            raise ValueError(f"{field} must be boolean")
        if isinstance(expected_value, int) and not isinstance(expected_value, bool):
            _exact_int(observed, field)
        if not exact_json_equal(observed, expected_value):
            raise ValueError(
                f"resource cleanup field {field} differs from raw CLI snapshots"
            )
    for field in (
        "active_app_count",
        "active_container_count",
        "active_endpoint_count",
    ):
        if _exact_int(payload[field], field) != 0:
            raise ValueError(f"{field} must be zero after cleanup")
    _exact_bool(payload["volume_present"], "volume_present")
    _exact_bool(
        payload["detached_calls_prohibited"],
        "detached_calls_prohibited",
    )
    _exact_bool(payload["validated"], "validated")
    if (
        payload["task_function_call_inventory"] != _TASK_FUNCTION_CALL_INVENTORY
        or payload["direct_detached_call_inventory"] != _DIRECT_DETACHED_CALL_INVENTORY
    ):
        raise ValueError("cleanup receipt overclaims unavailable Modal CLI inventory")
    network_policy = payload["artifact_verifier_network_policy"]
    if (
        not isinstance(network_policy, dict)
        or set(network_policy) != _ARTIFACT_VERIFIER_NETWORK_POLICY_FIELDS
        or network_policy["function_name"] != "artifact_verify"
        or network_policy["provider_secret"] is not False
        or type(network_policy["provider_secret"]) is not bool
        or network_policy["block_network"] is not True
        or type(network_policy["block_network"]) is not bool
        or network_policy["proof_kind"] != "bound_image_source_ast"
        or not isinstance(network_policy["sources"], list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            for item in network_policy["sources"]
        )
        or [item["path"] for item in network_policy["sources"]]
        != ["modal_app.py", "modal_boundary.py"]
    ):
        raise ValueError("artifact verifier network policy has an invalid exact schema")
    for item in network_policy["sources"]:
        _sha256(item["sha256"], "artifact_verifier_network_policy.source.sha256")
    if payload["billing_scope"] != (
        "final_and_prior_app_attributed_compute_plus_preserved_legacy_usage"
    ):
        raise ValueError("resource cleanup receipt has the wrong billing scope")
    start = _utc(payload["billing_window_start_utc"], "billing_window_start_utc")
    end = _utc(payload["billing_window_end_utc"], "billing_window_end_utc")
    if end <= start:
        raise ValueError("billing window must contain at least one interval")
    cohort_total = _decimal_text(
        payload["cohort_billing_total_usd"], "cohort_billing_total_usd"
    )
    superseded_total = _decimal_text(
        payload["superseded_usage_usd"], "superseded_usage_usd"
    )
    migration_total = _decimal_text(
        payload["migration_total_usd"], "migration_total_usd"
    )
    prior_total = _decimal_text(
        lineage["prior_app_compute_total_usd"],
        "migration_lineage.prior_app_compute_total_usd",
    )
    if (
        superseded_total != _SUPERSEDED_USAGE_USD
        or migration_total != cohort_total + prior_total + superseded_total
    ):
        raise ValueError("migration billing totals do not reconcile")
    _sha256(
        payload["detached_call_policy_source_sha256"],
        "detached_call_policy_source_sha256",
    )
    if (
        not isinstance(payload["accepted_primary_executions"], dict)
        or set(payload["accepted_primary_executions"]) != set(_PRIMARY_LABELS)
        or not isinstance(payload["artifact_verifier_executions"], dict)
        or set(payload["artifact_verifier_executions"]) != set(_PRIMARY_LABELS)
        or not isinstance(payload["additional_artifact_verifier_executions"], list)
    ):
        raise ValueError(
            "cleanup receipt does not cover exactly eight primary/verifier contexts"
        )
    if (
        not isinstance(payload["action_attempts"], list)
        or not payload["action_attempts"]
    ):
        raise ValueError("cleanup receipt has no final-cohort action attempts")
    inventory = payload["volume_run_directory_inventory"]
    if not isinstance(inventory, dict) or set(inventory) != {
        "required_current_run_ids",
        "required_prior_quarantined_run_ids",
        "superseded_run_id",
        "observed_required_run_ids",
        "snapshot_entry_count",
        "unrelated_directory_count",
    }:
        raise ValueError("cleanup Volume /runs inventory has an invalid exact schema")
    return (
        f"runs=8 verifiers=8 attempts={len(payload['action_attempts'])} "
        f"resources=0 modal_usd={migration_total}"
    )


def _execution_for_run(
    root: Path,
    run_id: str,
    *,
    expected_function: str,
) -> tuple[dict[str, Any], Path, Any, ExecutionContextV1]:
    run_id = validate_run_id(run_id)
    run_root, manifest, _, context = _inspect_downloaded_run(
        root,
        run_id,
        _expected_download_path(run_id),
    )
    if context.function_name != expected_function:
        raise ValueError(
            f"migration run {run_id} has function {context.function_name!r}, "
            f"expected {expected_function!r}"
        )
    image_manifest = _image_source_manifest(run_root / "image_source_manifest.json")
    if image_manifest.manifest_sha256 != manifest.image_source_sha256:
        raise ValueError("migration execution image manifest is not artifact-bound")
    modal_ids = {
        "modal_app_id": context.modal_app_id,
        "modal_function_id": context.modal_function_id,
        "modal_call_id": context.modal_call_id,
        "modal_image_id": context.modal_image_id,
    }
    for field, value in modal_ids.items():
        _text(value, f"{run_id}.{field}")
    expected = {
        "run_id": run_id,
        "function_name": expected_function,
        **modal_ids,
        "image_source_sha256": manifest.image_source_sha256,
        "artifact_manifest_sha256": manifest.manifest_sha256,
    }
    return expected, run_root, manifest, context


def _frozen_image_source_binding(
    root: Path,
    executions: Mapping[str, tuple[Path, Any, ExecutionContextV1]],
) -> dict[str, Any]:
    if len(executions) != 8:
        raise ValueError("frozen image binding requires exactly eight executions")
    artifact_source_hashes = {
        manifest.image_source_sha256 for _, manifest, _ in executions.values()
    }
    if len(artifact_source_hashes) != 1:
        raise ValueError("migration bundle does not share one image source digest")
    image_source_sha256 = next(iter(artifact_source_hashes))
    _sha256(image_source_sha256, "migration image_source_sha256")

    modal_image_ids: list[str] = []
    for label, (_, manifest, context) in executions.items():
        if context.image_source_sha256 != manifest.image_source_sha256:
            raise ValueError(
                f"migration execution {label} context and artifact source differ"
            )
        image_id = context.modal_image_id
        if not isinstance(image_id, str) or not image_id:
            raise ValueError(
                f"migration execution {label} lacks a nonempty Modal image ID"
            )
        modal_image_ids.append(image_id)
    if len(set(modal_image_ids)) != 1:
        raise ValueError("migration bundle does not share one Modal image ID")
    modal_image_id = modal_image_ids[0]

    try:
        current_manifest = build_image_source_manifest(root)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"current image source manifest cannot be reconstructed: {error}"
        ) from error
    if current_manifest.manifest_sha256 != image_source_sha256:
        raise ValueError(
            "migration bundle image source differs from the current source tree"
        )
    for label, (run_root, manifest, context) in executions.items():
        downloaded_manifest = _image_source_manifest(
            run_root / "image_source_manifest.json"
        )
        if (
            downloaded_manifest.dependency_lock_sha256
            != current_manifest.dependency_lock_sha256
        ):
            raise ValueError(
                f"migration execution {label} dependency lock differs from "
                "current source"
            )
        if downloaded_manifest.to_dict() != current_manifest.to_dict():
            raise ValueError(
                f"migration execution {label} image source manifest differs from "
                "the current source tree"
            )
        if (
            manifest.image_source_sha256 != current_manifest.manifest_sha256
            or context.image_source_sha256 != current_manifest.manifest_sha256
        ):
            raise ValueError(
                f"migration execution {label} is not bound to the current image source"
            )
    return {
        "modal_image_id": modal_image_id,
        "image_source_sha256": current_manifest.manifest_sha256,
        "dependency_lock_sha256": current_manifest.dependency_lock_sha256,
        "image_source_file_count": len(current_manifest.files),
        "image_source_byte_count": sum(
            item.size_bytes for item in current_manifest.files
        ),
    }


def _required_artifact(
    root: Path,
    run_id: str,
    relative_path: str,
) -> dict[str, Any]:
    relative = safe_relative_path(relative_path)
    logical = f"{_expected_download_path(run_id)}/{relative.as_posix()}"
    path = _contained_path(root, logical, "required_artifact.path", kind="file")
    digest, size_bytes = _hash_regular_file_snapshot(path)
    return {
        "path": logical,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _required_project_artifact(root: Path, logical: str) -> dict[str, Any]:
    path = _contained_path(root, logical, "required_artifact.path", kind="file")
    digest, size_bytes = _hash_regular_file_snapshot(path)
    return {
        "path": logical,
        "sha256": digest,
        "size_bytes": size_bytes,
    }


def _additional_verifier_required_paths(
    records: list[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return every immutable receipt in additional verifier Volume captures."""

    paths: list[str] = []
    for record in records:
        evidence_kind = record["remote_evidence_kind"]
        if evidence_kind == "round_trip_success":
            paths.append(record["remote_verification_path"])
            continue
        if evidence_kind == "volume_success_capture":
            anchor = PurePosixPath(record["remote_verification_path"])
            roster = _VERIFIER_REMOTE_RECEIPT_ROSTER
        elif evidence_kind == "volume_failure_capture":
            anchor = PurePosixPath(record["failure_receipt_path"])
            roster = _FAILED_VERIFIER_REMOTE_RECEIPT_ROSTER
        else:  # pragma: no cover - cleanup rejects unresolved evidence first.
            raise ValueError("additional verifier lacks bundleable evidence")
        paths.extend((anchor.parent / filename).as_posix() for filename in roster)
    if len(paths) != len(set(paths)):
        raise ValueError("additional verifier bundle evidence paths are duplicated")
    return tuple(sorted(paths))


def _ordinary_failure_required_paths(
    root: Path,
    records: list[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return the exact downloaded provenance files for recovered ordinary failures."""

    paths: list[str] = []
    for record in records:
        run_id = validate_run_id(record["failed_run_id"])
        run_logical = _expected_download_path(run_id)
        run_root = _contained_path(
            root,
            run_logical,
            "ordinary_failure.downloaded_run",
            kind="directory",
        )
        raw_manifest = _select_downloaded_raw_manifest(run_root)
        context = ExecutionContextV1.from_dict(
            record["failed_execution_context"]
        )
        result_filename = (
            "resume_action_result.json"
            if context.function_name == "checkpoint_resume"
            else "remote_action_result.json"
        )
        paths.extend(
            f"{run_logical}/{filename}"
            for filename in (
                raw_manifest.filename,
                "execution_context.json",
                "image_source_manifest.json",
                "remote_action_failure.json",
                result_filename,
            )
        )
    if len(paths) != len(set(paths)):
        raise ValueError("ordinary failure bundle evidence paths are duplicated")
    return tuple(sorted(paths))


def _receipt_evidence(
    root: Path,
    gate_name: str,
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, str], dict[str, Any]]:
    logical = modal_component_receipt_path(identity, gate_name).as_posix()
    path = _contained_path(root, logical, f"{gate_name}.receipt", kind="file")
    validate_modal_readiness_receipt(gate_name, path, root=root)
    return {"path": logical, "sha256": _sha256_file(path)}, _load_object(path)


def _derive_offline_smoke_validation_claims(
    root: Path,
    *,
    run_id: str,
    identity: ModalLiveCohortIdentity,
    verifier_run_id: str,
    verifier_attempt_id: str,
) -> dict[str, Any]:
    """Rederive the local, provider-free proof for one downloaded smoke run."""

    run = validate_run_id(run_id)
    logical = _expected_download_path(run)
    run_root, raw_manifest, verification, context = _inspect_downloaded_run_raw(
        root,
        run,
        logical,
    )
    manifest = raw_manifest.manifest
    if (
        context.function_name != "offline_smoke"
        or context.modal_app_id is None
        or context.modal_function_id is None
        or context.modal_call_id is None
        or context.modal_image_id is None
    ):
        raise ValueError("offline smoke receipt lacks one concrete Modal execution")

    report = validate_downloaded_offline_bundle(run_root)
    study = report.get("study")
    modal_ids = report.get("modal_ids")
    if not isinstance(study, dict) or not isinstance(modal_ids, dict):
        raise ValueError("offline smoke validator returned an invalid report")
    if (
        report.get("schema_name") != "DownloadedModalOfflineStudyValidation"
        or report.get("schema_version") != "1.0"
        or report.get("verified") is not True
        or report.get("validation_mode") != "local_read_only_provider_free"
        or report.get("network_calls") != 0
        or report.get("provider_calls") != 0
        or report.get("modal_run_id") != run
        or report.get("modal_app_name") != APP_NAME
        or report.get("modal_function_name") != "offline_smoke"
        or report.get("artifact_uri") != context.artifact_uri
        or report.get("artifact_manifest_sha256") != manifest.manifest_sha256
        or report.get("artifact_file_count") != len(manifest.files)
        or report.get("artifact_bytes")
        != sum(item.size_bytes for item in manifest.files)
        or report.get("image_source_sha256") != manifest.image_source_sha256
        or modal_ids
        != {
            "app_id": context.modal_app_id,
            "function_id": context.modal_function_id,
            "call_id": context.modal_call_id,
            "image_id": context.modal_image_id,
        }
    ):
        raise ValueError("offline smoke validation is not execution-bound")
    image_manifest = _image_source_manifest(run_root / "image_source_manifest.json")
    if (
        report.get("dependency_lock_sha256")
        != image_manifest.dependency_lock_sha256
        or report.get("provider_free_network_denial_probe_sha256")
        != _sha256_file(run_root / "provider_free_network_denial_probe.json")
    ):
        raise ValueError("offline smoke validation differs from bound image evidence")
    validation_sha256 = _sha256(
        report.get("validation_sha256"),
        "offline_validation_sha256",
    )
    if not isinstance(study.get("study_id"), str) or not study["study_id"]:
        raise ValueError("offline smoke validation lacks a study ID")
    study_run_count = _exact_int(
        study.get("run_count"),
        "offline_study_run_count",
        minimum=1,
    )
    remote = _remote_download_evidence(
        root,
        run,
        manifest,
        identity=identity,
        verifier_run_id=verifier_run_id,
        verifier_attempt_id=verifier_attempt_id,
    )
    return {
        "run_id": run,
        "downloaded_run_path": logical,
        "execution_backend": context.execution_backend,
        "app_name": context.app_name,
        "function_name": context.function_name,
        "artifact_uri": context.artifact_uri,
        "modal_app_id": context.modal_app_id,
        "modal_function_id": context.modal_function_id,
        "modal_call_id": context.modal_call_id,
        "modal_image_id": context.modal_image_id,
        "image_source_sha256": manifest.image_source_sha256,
        "dependency_lock_sha256": image_manifest.dependency_lock_sha256,
        "execution_context_sha256": _sha256_file(
            run_root / "execution_context.json"
        ),
        "image_source_manifest_sha256": _sha256_file(
            run_root / "image_source_manifest.json"
        ),
        "provider_free_network_denial_probe_sha256": _sha256_file(
            run_root / "provider_free_network_denial_probe.json"
        ),
        "remote_action_result_sha256": _sha256_file(
            run_root / "remote_action_result.json"
        ),
        "manifest_filename": raw_manifest.filename,
        "raw_manifest_sha256": raw_manifest.raw_sha256,
        "raw_manifest_size_bytes": raw_manifest.raw_size_bytes,
        "artifact_manifest_sha256": manifest.manifest_sha256,
        "files_verified": verification["file_count"],
        "artifact_bytes": sum(item.size_bytes for item in manifest.files),
        "offline_validation_sha256": validation_sha256,
        "offline_study_id": study["study_id"],
        "offline_study_run_count": study_run_count,
        "offline_study_sha256": canonical_sha256(study),
        "remote_verifier_run_id": remote["verifier_run_id"],
        "remote_verifier_attempt_id": remote["verifier_attempt_id"],
        "remote_verification_path": remote["remote_verification_path"],
        "remote_verification_sha256": remote["remote_verification_sha256"],
        "validation_mode": "local_read_only_provider_free",
        "validation_network_calls": 0,
        "validation_provider_calls": 0,
        "validation_remote_calls_started": 0,
        "validation_training_runs_started": 0,
        "validated": True,
    }


def validate_offline_smoke_validation_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Reopen and fully rederive one cohort-scoped offline-smoke receipt."""

    project_root = Path(root)
    supplied = Path(receipt_path)
    payload = _load_object(supplied)
    if set(payload) != _OFFLINE_SMOKE_VALIDATION_FIELDS:
        raise ValueError(
            "offline smoke validation receipt has an invalid exact schema"
        )
    if (
        payload["schema_name"] != "ModalOfflineSmokeValidationReceipt"
        or payload["schema_version"] != "2.0"
    ):
        raise ValueError("offline smoke validation receipt has the wrong contract")
    _utc(payload["recorded_at_utc"], "offline_smoke.recorded_at_utc")
    identity = _cohort_identity_from_payload(payload, field="offline_smoke")
    expected_logical = modal_component_receipt_path(
        identity, "modal_offline_smoke_validated"
    ).as_posix()
    path = _contained_path(
        project_root,
        expected_logical,
        "offline_smoke_validation_receipt",
        kind="file",
    )
    if supplied.resolve() != path.resolve():
        raise ValueError("offline smoke receipt path differs from its cohort identity")
    derived = _derive_offline_smoke_validation_claims(
        project_root,
        run_id=payload["run_id"],
        identity=identity,
        verifier_run_id=payload["remote_verifier_run_id"],
        verifier_attempt_id=payload["remote_verifier_attempt_id"],
    )
    expected = {
        "schema_name": "ModalOfflineSmokeValidationReceipt",
        "schema_version": "2.0",
        **modal_cohort_identity_dict(identity),
        "recorded_at_utc": payload["recorded_at_utc"],
        **derived,
    }
    if payload != expected:
        raise ValueError(
            "offline smoke validation receipt differs from live artifacts"
        )
    return payload


def record_offline_smoke_validation(
    *,
    run_id: str,
    cohort_id: str,
    verifier_run_id: str,
    verifier_attempt_id: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create once, then reopen, the local offline-smoke predecessor receipt."""

    project_root = Path(root)
    run = validate_run_id(run_id)
    _run_root, raw_manifest, _verification, _context = _inspect_downloaded_run_raw(
        project_root,
        run,
        _expected_download_path(run),
    )
    identity = _identity_for_recording(
        project_root=project_root,
        image_source_sha256=raw_manifest.manifest.image_source_sha256,
        cohort_id=cohort_id,
    )
    payload = {
        "schema_name": "ModalOfflineSmokeValidationReceipt",
        "schema_version": "2.0",
        **modal_cohort_identity_dict(identity),
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **_derive_offline_smoke_validation_claims(
            project_root,
            run_id=run,
            identity=identity,
            verifier_run_id=verifier_run_id,
            verifier_attempt_id=verifier_attempt_id,
        ),
    }
    logical = modal_component_receipt_path(
        identity, "modal_offline_smoke_validated"
    )
    output = project_root.resolve().joinpath(*logical.parts)
    create_json_exclusive(output, payload)
    persisted = validate_offline_smoke_validation_receipt(
        output, root=project_root
    )
    if persisted != payload:
        raise ValueError("persisted offline smoke validation receipt changed")
    return persisted


def _offline_smoke_receipt_evidence(
    root: Path,
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, str], dict[str, Any]]:
    logical = modal_component_receipt_path(
        identity, "modal_offline_smoke_validated"
    ).as_posix()
    path = _contained_path(
        root,
        logical,
        "offline_smoke_validation_receipt",
        kind="file",
    )
    payload = validate_offline_smoke_validation_receipt(path, root=root)
    return (
        {
            "path": logical,
            "sha256": _sha256_file(path),
        },
        payload,
    )


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    converted = float(value)
    if converted != converted or converted in {float("inf"), float("-inf")}:
        raise ValueError(f"{field} must be finite")
    return converted


def _require_frozen_t4_fingerprint(
    summary: Mapping[str, Any],
    field: str,
) -> AcceleratorFingerprint:
    raw = summary.get("accelerator_fingerprint")
    if not isinstance(raw, dict):
        raise ValueError(f"{field} lacks an accelerator fingerprint")
    fingerprint = AcceleratorFingerprint.from_dict(raw).validate_cuda(
        exact_gpu_count=1,
        require_driver=True,
    )
    if (
        fingerprint.gpu_name is None
        or GPU_TYPE.lower() not in fingerprint.gpu_name.lower()
    ):
        raise ValueError(f"{field} did not run on the frozen {GPU_TYPE}")
    return fingerprint


def _validate_candidate_layer_a(
    root: Path,
    run_id: str,
    run_root: Path,
    context: ExecutionContextV1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    action = _load_object(run_root / "remote_action_result.json")
    command_fields = {
        "returncode",
        "stdout_sha256",
        "stdout_size_bytes",
        "stderr_sha256",
        "stderr_size_bytes",
    }
    if set(action) != {"success", "mode", *command_fields}:
        raise ValueError("candidate remote action has an invalid exact schema")
    if action["success"] is not True or action["mode"] != (
        "cuda_candidate_train_and_layer_a"
    ):
        raise ValueError("candidate remote action did not complete successfully")
    _command_result(
        {field: action[field] for field in command_fields},
        "candidate_remote_action",
    )
    training_relative = "candidate_smoke/seed_1"
    training_root = run_root / training_relative
    report = validate_existing_cuda_smoke(
        training_root,
        project_root=root,
        require_modal_context=True,
    )
    if (
        report.get("valid") is not True
        or report.get("artifact_self_consistent") is not True
        or report.get("parameters_changed") is not True
        or report.get("profile") != SMOKE_TRAIN_CUDA_V2.name
        or report.get("accelerator_kind") != "cuda"
        or report.get("training_started_by_validator") is not False
    ):
        raise ValueError("candidate CUDA smoke validation did not pass exactly")

    manifest_payload = _load_object(training_root / "training_manifest.json")
    _require_frozen_t4_fingerprint(
        _load_object(training_root / "training_summary.json"),
        "candidate CUDA smoke",
    )
    if manifest_payload.get("execution_context") != context.to_dict():
        raise ValueError("candidate training context differs from the outer Modal run")
    aggregate_path = run_root / "candidate_smoke" / "aggregate_retraining_report.json"
    aggregate = _load_object(aggregate_path)
    aggregate_fields = {
        "schema_name",
        "schema_version",
        "candidate_format",
        "candidate_source_hash",
        "candidate_graph_hash",
        "candidate_artifact_paths",
        "evaluation_plan",
        "profile",
        "profile_version",
        "profile_hash",
        "device",
        "sequential",
        "success_count",
        "layer_a_eligibility_rate",
        "mean_public_accuracy",
        "population_stddev_public_accuracy",
        "sealed_qualification_performed",
        "runs",
    }
    if set(aggregate) != aggregate_fields:
        raise ValueError("candidate Layer A aggregate has an invalid exact schema")
    for field, expected in {
        "schema_name": "AggregateRetrainingReport",
        "schema_version": "2.0",
        "candidate_format": "architecture_ir",
        "candidate_artifact_paths": ["seed_1/candidate_graph.json"],
        "profile": SMOKE_TRAIN_CUDA_V2.name,
        "profile_version": SMOKE_TRAIN_CUDA_V2.version,
        "profile_hash": SMOKE_TRAIN_CUDA_V2.profile_hash,
        "device": "cuda",
        "sequential": True,
        "success_count": 1,
        "sealed_qualification_performed": False,
    }.items():
        observed = aggregate[field]
        if type(expected) is bool and type(observed) is not bool:
            raise ValueError(f"candidate aggregate {field} must be boolean")
        if observed != expected:
            raise ValueError(f"candidate aggregate {field} differs from the smoke")
    if _exact_int(aggregate["success_count"], "candidate success_count") != 1:
        raise ValueError("candidate aggregate success_count must be exactly one")
    candidate_path = training_root / "candidate_graph.json"
    candidate_sha256 = _sha256_file(candidate_path)
    trusted_sha256 = _sha256_file(root / "common" / "initial_candidate.ir.json")
    if aggregate["candidate_source_hash"] != trusted_sha256:
        raise ValueError("candidate aggregate is not bound to the trusted source")
    if candidate_sha256 != trusted_sha256:
        raise ValueError("stored candidate bytes differ from the trusted source")
    _sha256(aggregate["candidate_graph_hash"], "candidate_graph_hash")
    if (
        manifest_payload.get("candidate_graph_hash")
        != aggregate["candidate_graph_hash"]
    ):
        raise ValueError("candidate graph semantic hash differs across artifacts")

    raw_plan = aggregate["evaluation_plan"]
    plan_fields = {
        "profile_name",
        "profile_version",
        "profile_hash",
        "layer",
        "case_count",
        "case_source_id",
        "case_source_sha256",
        "scientific",
        "synthetic",
        "controller_visible",
        "sealed",
        "pi_decision_record_id",
        "plan_hash",
    }
    if not isinstance(raw_plan, dict) or set(raw_plan) != plan_fields:
        raise ValueError("candidate Layer A plan has an invalid exact schema")
    plan_values = dict(raw_plan)
    plan_hash = plan_values.pop("plan_hash")
    try:
        plan_values["layer"] = EvaluationLayer(plan_values["layer"])
        plan = EvaluationPlan(**plan_values)
        plan.validate()
    except (TypeError, ValueError) as error:
        raise ValueError("candidate Layer A plan is invalid") from error
    if (
        plan.plan_hash != plan_hash
        or plan.profile_name != "smoke_eval_v1"
        or plan.layer is not EvaluationLayer.SEARCH
        or plan.case_count != 24
        or plan.case_source_id != PUBLIC_LAYER_A_SOURCE_ID
        or plan.case_source_sha256 != PUBLIC_LAYER_A_SOURCE_SHA256
        or plan.scientific is not False
        or plan.synthetic is not True
        or plan.controller_visible is not True
        or plan.sealed is not False
        or plan.pi_decision_record_id is not None
    ):
        raise ValueError("candidate Layer A plan differs from the frozen smoke plan")

    runs = aggregate["runs"]
    if not isinstance(runs, list) or len(runs) != 1:
        raise ValueError("candidate Layer A aggregate must contain exactly one seed")
    run = runs[0]
    run_fields = {
        "seed",
        "success",
        "public_accuracy",
        "eligible_for_parent",
        "failure_stage",
        "evaluation_record",
    }
    if not isinstance(run, dict) or set(run) != run_fields:
        raise ValueError("candidate Layer A run has an invalid exact schema")
    if (
        _exact_int(run["seed"], "candidate Layer A seed") != 1
        or type(run["success"]) is not bool
        or run["success"] is not True
    ):
        raise ValueError("candidate Layer A seed did not execute successfully")
    if type(run["eligible_for_parent"]) is not bool:
        raise ValueError("candidate Layer A eligibility must be boolean")
    raw_evaluation = run["evaluation_record"]
    if not isinstance(raw_evaluation, dict):
        raise ValueError("candidate Layer A record is missing")
    evaluation = search_evaluation_from_dict(raw_evaluation)
    expected_eligible = evaluation.public_accuracy >= 0.99
    expected_failure_stage = "" if expected_eligible else "public_accuracy"
    if (
        evaluation.envelope.study_id != "independent-retraining"
        or evaluation.envelope.block_id != "retraining"
        or evaluation.envelope.run_id != "retraining-seed-1"
        or evaluation.envelope.condition_id != "retraining"
        or evaluation.candidate_id != f"candidate-{candidate_sha256}"
        or evaluation.execution_ok is not True
        or evaluation.transformer_valid is not True
        or evaluation.infrastructure_failure is not False
        or evaluation.search_score != evaluation.public_accuracy
        or evaluation.eligible_for_parent is not expected_eligible
        or evaluation.failure_stage != expected_failure_stage
        or run["success"] is not evaluation.execution_ok
        or run["eligible_for_parent"] is not evaluation.eligible_for_parent
        or run["failure_stage"] != evaluation.failure_stage
        or _finite_number(run["public_accuracy"], "public_accuracy")
        != evaluation.public_accuracy
    ):
        raise ValueError("candidate Layer A record is not aggregate-bound")
    if evaluation.runtime_validity_artifact is None:
        raise ValueError("candidate Layer A lacks runtime-validity evidence")
    runtime_reference = evaluation.runtime_validity_artifact
    runtime_relative = safe_relative_path(runtime_reference.relative_path)
    if len(runtime_relative.parts) != 1:
        raise ValueError("runtime-validity evidence path is not training-root local")
    runtime_path = training_root.joinpath(*runtime_relative.parts)
    if _sha256_file(runtime_path) != runtime_reference.sha256:
        raise ValueError("runtime-validity evidence hash differs from Layer A")
    runtime = _load_object(runtime_path)
    if (
        runtime.get("candidate_artifact_hash") != candidate_sha256
        or runtime.get("candidate_graph_hash") != aggregate["candidate_graph_hash"]
        or runtime.get("training_profile_hash") != SMOKE_TRAIN_CUDA_V2.profile_hash
        or runtime.get("evaluation_plan_hash") != plan.plan_hash
        or runtime.get("requested_device") != "cuda"
        or not isinstance(runtime.get("runtime_evidence"), dict)
        or runtime["runtime_evidence"].get("passed") is not True
        or not isinstance(runtime.get("post_execution_decision"), dict)
        or runtime["post_execution_decision"].get("allowed") is not True
    ):
        raise ValueError("candidate runtime-validity evidence is not Layer A-bound")
    if evaluation.public_artifacts:
        raise ValueError("smoke Layer A unexpectedly exposed extra public artifacts")
    accuracy = evaluation.public_accuracy
    if (
        _finite_number(aggregate["mean_public_accuracy"], "mean_public_accuracy")
        != accuracy
        or _finite_number(
            aggregate["population_stddev_public_accuracy"],
            "population_stddev_public_accuracy",
        )
        != 0.0
        or _finite_number(
            aggregate["layer_a_eligibility_rate"],
            "layer_a_eligibility_rate",
        )
        != (1.0 if evaluation.eligible_for_parent else 0.0)
    ):
        raise ValueError("candidate Layer A aggregate statistics were rewritten")

    normalized_training = dict(report)
    normalized_training["output_dir"] = (
        f"{_expected_download_path(run_id)}/{training_relative}"
    )
    evidence = {
        "training_validation_sha256": canonical_sha256(normalized_training),
        "layer_a_record_sha256": canonical_sha256(raw_evaluation),
        "layer_a_plan_sha256": plan.plan_hash,
    }
    required = [
        _required_artifact(root, run_id, f"candidate_smoke/{name}")
        for name in ("aggregate_retraining_report.json",)
    ]
    required.append(_required_artifact(root, run_id, "remote_action_result.json"))
    required.extend(
        _required_artifact(root, run_id, f"{training_relative}/{name}")
        for name in (
            "training_summary.json",
            "training_manifest.json",
            "candidate_graph.json",
            "best_checkpoint.pt",
            "partial_resume_checkpoint.pt",
            "training_events.jsonl",
            runtime_relative.as_posix(),
        )
    )
    return evidence, required


def _manifest_entry_named(manifest: Any, name: str) -> Any:
    matches = [
        item
        for item in manifest.files
        if PurePosixPath(item.relative_path).name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"artifact manifest must contain exactly one {name}")
    return matches[0]


def _link_or_copy(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"resume revalidation input is unsafe: {source.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


_RESUME_PROGRESSION_TRAINING_FILES = (
    "best_checkpoint.pt",
    "candidate_graph.json",
    "latest_resume_checkpoint.pt",
    "partial_resume_checkpoint.pt",
    "rng_restore_attestation.json",
    "training_events.jsonl",
    "training_manifest.json",
    "training_summary.json",
)
_RESUME_PROGRESSION_ROOT_FILES = (
    "image_source_manifest.json",
    "resume_execution_context.json",
)


def _stage_resume_progression_inputs(
    *,
    attempt_root: Path,
    staging_parent: Path,
    training_relative: str,
) -> tuple[Path, Path]:
    """Copy the exact persisted resume verifier vocabulary into a fresh root."""

    staging_root = staging_parent / attempt_root.name
    for filename in _RESUME_PROGRESSION_ROOT_FILES:
        _link_or_copy(attempt_root / filename, staging_root / filename)
    training_root = attempt_root / training_relative
    staged_training = staging_root / training_relative
    for filename in _RESUME_PROGRESSION_TRAINING_FILES:
        _link_or_copy(training_root / filename, staged_training / filename)
    return staging_root, staged_training


def _command_result(value: object, field: str) -> None:
    expected_fields = {
        "returncode",
        "stdout_sha256",
        "stdout_size_bytes",
        "stderr_sha256",
        "stderr_size_bytes",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError(f"{field} has an invalid exact command-result schema")
    if _exact_int(value["returncode"], f"{field}.returncode") != 0:
        raise ValueError(f"{field} did not finish successfully")
    for digest in ("stdout_sha256", "stderr_sha256"):
        _sha256(value[digest], f"{field}.{digest}")
    for size in ("stdout_size_bytes", "stderr_size_bytes"):
        _exact_int(value[size], f"{field}.{size}")


def _revalidate_resume_attempt(
    root: Path,
    source_run_id: str,
    source_root: Path,
    source_manifest: Any,
    attempt_run_id: str,
    attempt_root: Path,
    attempt_manifest: Any,
    attempt_context: ExecutionContextV1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    binding = _load_object(attempt_root / "resume_source_binding.json")
    binding_fields = {
        "schema_name",
        "schema_version",
        "source_run_id",
        "resume_attempt_run_id",
        "source_manifest_sha256",
        "source_image_sha256",
        "source_training_relative_path",
        "partial_optimizer_step",
        "partial_examples_processed",
        "retained_event_count",
        "source_artifacts",
        "attempt_artifacts",
    }
    if set(binding) != binding_fields:
        raise ValueError("resume source binding has an invalid exact schema")
    training_relative = "candidate_smoke/seed_1"
    for field, expected in {
        "schema_name": "ResumeSourceBinding",
        "schema_version": "1.0",
        "source_run_id": source_run_id,
        "resume_attempt_run_id": attempt_run_id,
        "source_manifest_sha256": source_manifest.manifest_sha256,
        "source_image_sha256": source_manifest.image_source_sha256,
        "source_training_relative_path": training_relative,
        "partial_optimizer_step": SMOKE_TRAIN_CUDA_V2.checkpoint_interval,
        "partial_examples_processed": (
            SMOKE_TRAIN_CUDA_V2.checkpoint_interval
            * SMOKE_TRAIN_CUDA_V2.global_batch_size
        ),
        "retained_event_count": SMOKE_TRAIN_CUDA_V2.checkpoint_interval,
    }.items():
        if binding[field] != expected:
            raise ValueError(f"resume source binding {field} differs")
    source_entries = {
        label: _manifest_entry_named(source_manifest, filename)
        for label, filename in {
            "candidate": "candidate_graph.json",
            "partial_resume_checkpoint": "partial_resume_checkpoint.pt",
            "training_events": "training_events.jsonl",
        }.items()
    }
    source_records = binding["source_artifacts"]
    if (
        not isinstance(source_records, dict)
        or set(source_records) != set(source_entries)
        or any(
            source_records[label] != entry.to_dict()
            for label, entry in source_entries.items()
        )
    ):
        raise ValueError("resume source artifact binding was rewritten")

    training_root = attempt_root / training_relative
    candidate = training_root / "candidate_graph.json"
    partial = training_root / "partial_resume_checkpoint.pt"
    events = training_root / "training_events.jsonl"
    _require_frozen_t4_fingerprint(
        _load_object(training_root / "training_summary.json"),
        "resume CUDA smoke",
    )
    if _sha256_file(candidate) != source_entries["candidate"].sha256:
        raise ValueError("resume attempt candidate differs from its source")
    if _sha256_file(partial) != source_entries["partial_resume_checkpoint"].sha256:
        raise ValueError("resume attempt partial checkpoint differs from its source")
    raw_lines = events.read_bytes().splitlines(keepends=True)
    retained = SMOKE_TRAIN_CUDA_V2.checkpoint_interval
    if len(raw_lines) != SMOKE_TRAIN_CUDA_V2.max_steps:
        raise ValueError("resume event log does not reach the smoke maximum")
    event_prefix_sha256 = hashlib.sha256(b"".join(raw_lines[:retained])).hexdigest()
    source_events = source_root / source_entries["training_events"].relative_path
    source_lines = source_events.read_bytes().splitlines(keepends=True)
    if len(source_lines) != SMOKE_TRAIN_CUDA_V2.max_steps:
        raise ValueError("source event log does not reach the smoke maximum")
    source_prefix_sha256 = hashlib.sha256(b"".join(source_lines[:retained])).hexdigest()
    if event_prefix_sha256 != source_prefix_sha256:
        raise ValueError("resume attempt event prefix differs from its source run")
    attempt_records = binding["attempt_artifacts"]
    expected_attempt_records = {
        "candidate_relative_path": f"{training_relative}/candidate_graph.json",
        "candidate_sha256": source_entries["candidate"].sha256,
        "partial_checkpoint_relative_path": (
            f"{training_relative}/partial_resume_checkpoint.pt"
        ),
        "partial_checkpoint_sha256": source_entries["partial_resume_checkpoint"].sha256,
        "initial_latest_checkpoint_relative_path": (
            f"{training_relative}/latest_resume_checkpoint.pt"
        ),
        "initial_latest_checkpoint_sha256": source_entries[
            "partial_resume_checkpoint"
        ].sha256,
        "event_prefix_relative_path": f"{training_relative}/training_events.jsonl",
        "event_prefix_sha256": event_prefix_sha256,
    }
    if attempt_records != expected_attempt_records:
        raise ValueError("resume attempt artifact binding was rewritten")

    resume_context = _load_object(attempt_root / "resume_execution_context.json")
    if resume_context != attempt_context.to_dict():
        raise ValueError("resume execution context differs from the attempt context")
    action = _load_object(attempt_root / "resume_action_result.json")
    action_fields = {
        "success",
        "mode",
        "source_run_id",
        "source_manifest_sha256",
        "resume_contract_probe",
        "resume_progression_verification",
        "returncode",
        "stdout_sha256",
        "stdout_size_bytes",
        "stderr_sha256",
        "stderr_size_bytes",
    }
    if set(action) != action_fields:
        raise ValueError("resume action result has an invalid exact schema")
    if (
        action["success"] is not True
        or action["mode"] != "checkpoint_resume"
        or action["source_run_id"] != source_run_id
        or action["source_manifest_sha256"] != source_manifest.manifest_sha256
    ):
        raise ValueError("resume action result is not source-bound")
    _command_result(action["resume_contract_probe"], "resume_contract_probe")
    _command_result(
        action["resume_progression_verification"],
        "resume_progression_verification",
    )
    _command_result(
        {
            field: action[field]
            for field in action_fields
            if field
            in {
                "returncode",
                "stdout_sha256",
                "stdout_size_bytes",
                "stderr_sha256",
                "stderr_size_bytes",
            }
        },
        "resume_training",
    )

    stored_contract = _load_object(attempt_root / "resume_contract_verification.json")
    stored_progression = _load_object(
        attempt_root / "resume_progression_verification.json"
    )
    with tempfile.TemporaryDirectory(prefix="modal-resume-revalidation-") as raw_temp:
        staging_root = Path(raw_temp)
        staged_training = staging_root / training_relative
        _link_or_copy(partial, staged_training / "latest_resume_checkpoint.pt")
        _link_or_copy(candidate, staged_training / "candidate_graph.json")
        recomputed_contract = verify_resume_contract(
            checkpoint_path=staged_training / "latest_resume_checkpoint.pt",
            candidate_path=staged_training / "candidate_graph.json",
            profile_name=SMOKE_TRAIN_CUDA_V2.name,
            run_seed=1,
            output_path=staging_root / "resume_contract_verification.json",
        )
        if recomputed_contract != stored_contract:
            raise ValueError("stored resume negative probes do not revalidate")

    progression_inputs = _RESUME_PROGRESSION_TRAINING_FILES
    with tempfile.TemporaryDirectory(
        prefix="modal-progression-revalidation-"
    ) as raw_temp:
        staging_root, staged_training = _stage_resume_progression_inputs(
            attempt_root=attempt_root,
            staging_parent=Path(raw_temp),
            training_relative=training_relative,
        )
        recomputed_progression = verify_resume_progression(
            artifact_root=staging_root,
            training_output_dir=staged_training,
            profile_name=SMOKE_TRAIN_CUDA_V2.name,
            run_seed=1,
            output_path=staging_root / "resume_progression_verification.json",
        )
        if recomputed_progression != stored_progression:
            raise ValueError("stored resume progression does not revalidate")

    evidence = {
        "source_binding_sha256": _sha256_file(
            attempt_root / "resume_source_binding.json"
        ),
        "negative_probe_sha256": _sha256_file(
            attempt_root / "resume_contract_verification.json"
        ),
        "progression_sha256": _sha256_file(
            attempt_root / "resume_progression_verification.json"
        ),
    }
    required = [
        _required_artifact(root, attempt_run_id, name)
        for name in (
            "resume_source_binding.json",
            "resume_execution_context.json",
            "image_source_manifest.json",
            "resume_contract_verification.json",
            "resume_progression_verification.json",
            "resume_action_result.json",
        )
    ]
    required.extend(
        _required_artifact(root, attempt_run_id, f"{training_relative}/{name}")
        for name in progression_inputs
    )
    # The full manifest was already checked by _inspect_downloaded_run; bind it
    # again here so a substituted resume manifest cannot reuse these artifacts.
    _sha256(attempt_manifest.manifest_sha256, "resume artifact manifest")
    return evidence, required


def _canary_prefix(run_ids: Mapping[str, str]) -> str | None:
    prefixes: set[str] = set()
    for harness in CANARY_ORDER:
        run_id = validate_run_id(run_ids[harness])
        suffix = _CANARY_SUFFIXES[harness]
        marker = f"-{suffix}"
        if not run_id.endswith(marker) or len(run_id) == len(marker):
            raise ValueError(f"canary run {run_id} lacks the terminal {marker} suffix")
        prefixes.add(run_id[: -len(marker)])
    return next(iter(prefixes)) if len(prefixes) == 1 else None


def _validate_selected_canaries(
    root: Path,
    run_ids: Mapping[str, str],
) -> dict[str, Any]:
    if set(run_ids) != set(CANARY_ORDER):
        raise ValueError("canary selection must cover the exact frozen roster")
    prefix = _canary_prefix(run_ids)
    download_parent = root / "outputs" / "development" / "modal_downloads"
    if prefix is not None:
        report = validate_downloaded_modal_canaries(download_parent / prefix)
    else:
        # Recovery may select one successful terminal-suffix attempt per
        # harness. Build a temporary hard-linked directory bundle so the same
        # strict public four-run validator rechecks that explicit selection.
        with tempfile.TemporaryDirectory(prefix="modal-canary-selection-") as raw_temp:
            selected = Path(raw_temp)

            def copy_file(source: str, destination: str) -> str:
                _link_or_copy(Path(source), Path(destination))
                return destination

            for harness in CANARY_ORDER:
                source = _contained_path(
                    root,
                    _expected_download_path(run_ids[harness]),
                    f"canary.{harness}",
                    kind="directory",
                )
                shutil.copytree(
                    source,
                    selected / run_ids[harness],
                    copy_function=copy_file,
                )
            report = validate_downloaded_modal_canaries(selected)
    if (
        report.get("valid") is not True
        or report.get("all_four_canaries_validated") is not True
        or report.get("provider_calls_started_by_validator") != 0
        or report.get("remote_calls_started_by_validator") != 0
        or report.get("training_runs_started_by_validator") != 0
    ):
        raise ValueError("selected provider canary bundle did not revalidate")
    reports = report.get("runs")
    if not isinstance(reports, list) or len(reports) != len(CANARY_ORDER):
        raise ValueError("canary validator returned an incomplete roster")
    by_harness = {
        item.get("harness"): item for item in reports if isinstance(item, dict)
    }
    if set(by_harness) != set(CANARY_ORDER):
        raise ValueError("canary validator returned a substituted roster")
    for harness in CANARY_ORDER:
        if by_harness[harness].get("run_id") != run_ids[harness]:
            raise ValueError(f"canary validator substituted {harness}")
    normalized = dict(report)
    normalized["download_root"] = "outputs/development/modal_downloads"
    return normalized


def _intent_predecessor_paths(
    root: Path,
    intent_bindings: list[Mapping[str, Any]],
) -> set[str]:
    paths: set[str] = set()
    for index, binding in enumerate(intent_bindings):
        if not isinstance(binding, Mapping):
            raise ValueError("bundle intent binding is invalid")
        logical = _text(binding.get("path"), f"bundle.intent[{index}].path")
        payload, observed = _lineage_bound_json(
            root,
            logical,
            field=f"bundle.intent[{index}]",
        )
        if (
            binding.get("sha256") != observed["sha256"]
            or binding.get("size_bytes") != observed["size_bytes"]
        ):
            raise ValueError("bundle intent binding bytes changed")
        predecessors = payload.get("predecessor_receipts")
        if not isinstance(predecessors, list):
            raise ValueError("bundle intent predecessor roster is invalid")
        for predecessor in predecessors:
            if (
                not isinstance(predecessor, dict)
                or set(predecessor) != {"gate", "path", "sha256"}
            ):
                raise ValueError("bundle intent predecessor binding is invalid")
            paths.add(_text(predecessor["path"], "bundle.predecessor.path"))
    return paths


def _journal_process_start_paths(
    root: Path,
    terminal_bindings: Sequence[Mapping[str, Any]],
) -> set[str]:
    paths: set[str] = set()
    for index, binding in enumerate(terminal_bindings):
        if not isinstance(binding, Mapping):
            raise ValueError("bundle terminal binding is invalid")
        logical = _text(binding.get("path"), f"bundle.terminal[{index}].path")
        payload, observed = _lineage_bound_json(
            root,
            logical,
            field=f"bundle.terminal[{index}]",
        )
        if (
            binding.get("sha256") != observed["sha256"]
            or binding.get("size_bytes") != observed["size_bytes"]
        ):
            raise ValueError("bundle terminal binding bytes changed")
        marker_logical = payload.get("local_process_start_receipt_path")
        marker_sha256 = payload.get("local_process_start_receipt_sha256")
        if marker_logical is None:
            if marker_sha256 is not None:
                raise ValueError("bundle terminal has a partial process-start binding")
            continue
        marker_path = _text(marker_logical, "bundle.process_start.path")
        if marker_sha256 is not None or _path_has_any_entry(root, marker_path):
            paths.add(marker_path)
    return paths


def _lineage_required_paths(
    root: Path,
    lineage: Mapping[str, Any],
) -> set[str]:
    paths: set[str] = set()
    final = lineage["selected_final"]
    final_journal = final["action_journal"]
    for field in ("intent_receipts", "terminal_receipts", "aggregate_receipts"):
        paths.update(record["path"] for record in final_journal[field])
    paths.update(record["path"] for record in final["remote_run_reservations"])
    paths.update(record["evidence"]["path"] for record in final["remote_executions"])
    paths.update(record["path"] for record in final["artifact_manifests"])
    for record in final["provider_attempt_evidence"]:
        for field in ("ledger", "uncertainty"):
            if record[field] is not None:
                paths.add(record[field]["path"])
    for launcher in final["provider_spend_estimate"][
        "launcher_approval_bounds"
    ]:
        for field in ("approval_plan", "price_basis"):
            binding = launcher[field]
            if binding is not None:
                paths.add(binding["path"])
    paths.update(
        _intent_predecessor_paths(root, final_journal["intent_receipts"])
    )
    paths.update(
        _journal_process_start_paths(root, final_journal["terminal_receipts"])
    )
    paths.update(
        record["path"] for record in lineage["global_remote_run_reservations"]
    )

    for prior in lineage["prior_quarantined_cohorts"]:
        accounting_logical = prior["accounting_receipt"]["path"]
        paths.add(accounting_logical)
        payload, _metadata = _load_prior_quarantine_accounting(
            root,
            accounting_logical,
        )
        journal = prior["action_journal"]
        for field in (
            "intent_receipts",
            "terminal_receipts",
            "aggregate_receipts",
        ):
            paths.update(record["path"] for record in journal[field])
        paths.update(record["path"] for record in prior["remote_run_reservations"])
        paths.update(_intent_predecessor_paths(root, journal["intent_receipts"]))
        paths.update(
            _journal_process_start_paths(root, journal["terminal_receipts"])
        )
        manifest_logical = payload["snapshot_capture_manifest_path"]
        paths.add(manifest_logical)
        manifest, _binding = _lineage_bound_json(
            root,
            manifest_logical,
            field="bundle.prior_snapshot_manifest",
        )
        snapshots = manifest.get("snapshots")
        if not isinstance(snapshots, dict) or set(snapshots) != set(
            _SNAPSHOT_NAMES
        ):
            raise ValueError("bundle prior snapshot manifest roster changed")
        paths.update(snapshots[name]["path"] for name in _SNAPSHOT_NAMES)
        paths.update(
            record["execution_context_path"]
            for record in payload["remote_executions"]
        )
        paths.update(
            record["evidence_path"]
            for record in payload["provider_attempt_evidence"]
        )
        paths.update(
            record["evidence_path"]
            for record in payload["unbound_provider_evidence"]
        )
        paths.update(
            record["artifact_manifest_path"]
            for record in payload["volume_dispositions"]
            if record["artifact_manifest_disposition"] == "bound"
        )
        paths.add(payload["modal_price_basis"]["path"])
        for launcher in payload["provider_spend_estimate"][
            "launcher_approval_bounds"
        ]:
            for field in ("approval_plan", "price_basis"):
                binding = launcher[field]
                if binding is not None:
                    paths.add(binding["path"])
    return paths


def _derive_migration_bundle_claims(
    root: Path,
    *,
    cohort_roster_path: str,
) -> dict[str, Any]:
    roster, roster_path = _load_cohort_roster(root, cohort_roster_path)
    identity = _cohort_identity_from_payload(roster, field="cohort_roster")
    lineage, lineage_path, lineage_sha256 = _load_migration_lineage(
        root,
        roster,
        identity,
    )
    lineage_digest, lineage_size = _hash_regular_file_snapshot(lineage_path)
    if lineage_digest != lineage_sha256:
        raise ValueError("migration bundle lineage bytes changed")
    lineage_binding = {
        "path": roster["migration_lineage_path"],
        "sha256": lineage_digest,
        "size_bytes": lineage_size,
    }
    accepted = roster["accepted_primary_runs"]
    canary_runs = {
        harness: validate_run_id(accepted[f"canary_{harness}"])
        for harness in CANARY_ORDER
    }
    _canary_prefix(canary_runs)

    env_claim, env_root, env_manifest, env_context = _execution_for_run(
        root,
        accepted["cuda_environment"],
        expected_function="cuda_environment",
    )
    candidate_claim, candidate_root, candidate_manifest, candidate_context = (
        _execution_for_run(
            root,
            accepted["candidate_smoke"],
            expected_function="candidate_smoke",
        )
    )
    resume_claim, resume_root, resume_manifest, resume_context = _execution_for_run(
        root,
        accepted["resume_attempt"],
        expected_function="checkpoint_resume",
    )
    offline_claim, offline_root, offline_manifest, offline_context = _execution_for_run(
        root,
        accepted["offline_smoke"],
        expected_function="offline_smoke",
    )
    canary_claims: dict[str, dict[str, Any]] = {}
    canary_contexts: list[ExecutionContextV1] = []
    canary_manifests: dict[str, Any] = {}
    canary_roots: dict[str, Path] = {}
    for harness in CANARY_ORDER:
        claim, run_root, manifest, context = _execution_for_run(
            root,
            canary_runs[harness],
            expected_function=f"canary_{harness}",
        )
        canary_claims[harness] = claim
        canary_contexts.append(context)
        canary_manifests[harness] = manifest
        canary_roots[harness] = run_root

    environment_action = _load_object(env_root / "remote_action_result.json")
    if set(environment_action) != {
        "success",
        "mode",
        "observed_gpu",
    }:
        raise ValueError("CUDA environment action has an invalid exact schema")
    if (
        environment_action["success"] is not True
        or environment_action["mode"] != "cuda_environment"
        or environment_action["observed_gpu"]
        != _load_object(env_root / "cuda_environment.json")["cuda_device_name"]
    ):
        raise ValueError("CUDA environment action did not complete successfully")
    # cuda_environment is in-process and therefore has no subprocess command
    # fields. Keep this local to distinguish it from every command action.

    primary_contexts = [
        env_context,
        offline_context,
        candidate_context,
        resume_context,
        *canary_contexts,
    ]
    if len(primary_contexts) != 8:
        raise AssertionError("migration bundle primary-context roster drifted")
    run_ids = [item.run_id for item in primary_contexts]
    if len(set(run_ids)) != 8:
        raise ValueError("migration bundle requires eight distinct primary run IDs")
    call_ids = [item.modal_call_id for item in primary_contexts]
    if len(set(call_ids)) != 8:
        raise ValueError("migration bundle requires eight unique Modal call IDs")
    image_executions = {
        "cuda_environment": (env_root, env_manifest, env_context),
        "offline_smoke": (offline_root, offline_manifest, offline_context),
        "candidate_smoke": (
            candidate_root,
            candidate_manifest,
            candidate_context,
        ),
        "resume_attempt": (resume_root, resume_manifest, resume_context),
        **{
            f"canary_{harness}": (
                canary_roots[harness],
                canary_manifests[harness],
                canary_contexts[index],
            )
            for index, harness in enumerate(CANARY_ORDER)
        },
    }
    image_binding = _frozen_image_source_binding(root, image_executions)
    image_source_sha256 = image_binding["image_source_sha256"]
    download_evidence = {
        "cuda_environment": _remote_download_evidence(
            root,
            env_context.run_id,
            env_manifest,
            identity=identity,
            verifier_run_id=roster["artifact_verifiers"]["cuda_environment"][
                "verifier_run_id"
            ],
            verifier_attempt_id=roster["artifact_verifiers"][
                "cuda_environment"
            ]["attempt_id"],
        ),
        "candidate_smoke": _remote_download_evidence(
            root,
            candidate_context.run_id,
            candidate_manifest,
            identity=identity,
            verifier_run_id=roster["artifact_verifiers"]["candidate_smoke"][
                "verifier_run_id"
            ],
            verifier_attempt_id=roster["artifact_verifiers"]["candidate_smoke"][
                "attempt_id"
            ],
        ),
        "resume_attempt": _remote_download_evidence(
            root,
            resume_context.run_id,
            resume_manifest,
            identity=identity,
            verifier_run_id=roster["artifact_verifiers"]["resume_attempt"][
                "verifier_run_id"
            ],
            verifier_attempt_id=roster["artifact_verifiers"]["resume_attempt"][
                "attempt_id"
            ],
        ),
        "offline_smoke": _remote_download_evidence(
            root,
            offline_context.run_id,
            offline_manifest,
            identity=identity,
            verifier_run_id=roster["artifact_verifiers"]["offline_smoke"][
                "verifier_run_id"
            ],
            verifier_attempt_id=roster["artifact_verifiers"]["offline_smoke"][
                "attempt_id"
            ],
        ),
        "canaries": {
            harness: _remote_download_evidence(
                root,
                canary_contexts[index].run_id,
                canary_manifests[harness],
                identity=identity,
                verifier_run_id=roster["artifact_verifiers"][
                    f"canary_{harness}"
                ]["verifier_run_id"],
                verifier_attempt_id=roster["artifact_verifiers"][
                    f"canary_{harness}"
                ]["attempt_id"],
            )
            for index, harness in enumerate(CANARY_ORDER)
        },
    }

    cuda_receipt, cuda_payload = _receipt_evidence(
        root, "modal_cuda_environment_validated", identity
    )
    if (
        cuda_payload["run_id"] != env_context.run_id
        or cuda_payload["artifact_manifest_sha256"] != env_manifest.manifest_sha256
        or cuda_payload["image_source_sha256"] != image_source_sha256
    ):
        raise ValueError("CUDA receipt is not migration-environment bound")
    offline_receipt, offline_payload = _offline_smoke_receipt_evidence(
        root, identity
    )
    if (
        offline_payload["run_id"] != offline_context.run_id
        or offline_payload["artifact_manifest_sha256"]
        != offline_manifest.manifest_sha256
        or offline_payload["image_source_sha256"] != image_source_sha256
    ):
        raise ValueError("offline receipt is not migration-offline-run bound")
    round_trip_receipt, round_trip_payload = _receipt_evidence(
        root, "modal_artifact_round_trip_validated", identity
    )
    if (
        round_trip_payload["source_run_id"] != candidate_context.run_id
        or round_trip_payload["local_canonical_manifest_sha256"]
        != candidate_manifest.manifest_sha256
    ):
        raise ValueError("artifact round-trip receipt is not candidate-smoke bound")
    preflight_binding = roster["component_receipts"][
        "candidate_resume_preflight_validated"
    ]
    preflight_path = _contained_path(
        root,
        preflight_binding["path"],
        "bundle.candidate_resume_preflight",
        kind="file",
    )
    preflight_payload = validate_candidate_resume_preflight_receipt(
        preflight_path,
        root=root,
    )
    _assert_identity_matches(
        preflight_payload,
        identity,
        field="bundle.candidate_resume_preflight",
    )
    preflight_receipt = _required_project_artifact(
        root,
        preflight_binding["path"],
    )
    if preflight_receipt["sha256"] != preflight_binding["sha256"]:
        raise ValueError("candidate-resume preflight component digest changed")

    candidate_evidence, candidate_required = _validate_candidate_layer_a(
        root,
        candidate_context.run_id,
        candidate_root,
        candidate_context,
    )
    resume_evidence, resume_required = _revalidate_resume_attempt(
        root,
        candidate_context.run_id,
        candidate_root,
        candidate_manifest,
        resume_context.run_id,
        resume_root,
        resume_manifest,
        resume_context,
    )

    canary_report = _validate_selected_canaries(root, canary_runs)
    report_by_harness = {item["harness"]: item for item in canary_report["runs"]}
    for harness in CANARY_ORDER:
        report = report_by_harness[harness]
        claim = canary_claims[harness]
        if (
            report["artifact_manifest_sha256"] != claim["artifact_manifest_sha256"]
            or report["modal_call_id"] != claim["modal_call_id"]
            or report["modal_image_id"] != claim["modal_image_id"]
            or report["image_source_sha256"] != image_source_sha256
        ):
            raise ValueError(f"canary report for {harness} is not execution-bound")

    offline_report = validate_downloaded_offline_bundle(offline_root)
    if (
        offline_report.get("verified") is not True
        or offline_report.get("network_calls") != 0
        or offline_report.get("provider_calls") != 0
        or offline_report.get("modal_run_id") != offline_context.run_id
        or offline_report.get("modal_function_name") != "offline_smoke"
        or offline_report.get("artifact_manifest_sha256")
        != offline_manifest.manifest_sha256
        or offline_report.get("image_source_sha256") != image_source_sha256
        or offline_report.get("dependency_lock_sha256")
        != image_binding["dependency_lock_sha256"]
    ):
        raise ValueError("downloaded offline reconstruction did not revalidate")

    cleanup_receipt, cleanup_payload = _receipt_evidence(
        root, "modal_resource_cleanup_validated", identity
    )
    if cleanup_payload["cohort_roster_path"] != cohort_roster_path:
        raise ValueError("cleanup receipt uses a different final cohort roster")
    migration_cleanup = _derive_cleanup_claims(
        root,
        cohort_roster_path,
        recorded_at_utc=cleanup_payload["recorded_at_utc"],
    )
    if (
        migration_cleanup["billing_scope"]
        != "final_and_prior_app_attributed_compute_plus_preserved_legacy_usage"
        or migration_cleanup["active_app_count"] != 0
        or migration_cleanup["active_container_count"] != 0
        or migration_cleanup["active_endpoint_count"] != 0
    ):
        raise ValueError("migration-wide cleanup claims are incomplete")
    migration_cleanup_evidence = {
        "billing_scope": migration_cleanup["billing_scope"],
        "billing_window_start_utc": migration_cleanup["billing_window_start_utc"],
        "billing_window_end_utc": migration_cleanup["billing_window_end_utc"],
        "cohort_billing_total_usd": migration_cleanup["cohort_billing_total_usd"],
        "superseded_usage_usd": migration_cleanup["superseded_usage_usd"],
        "migration_total_usd": migration_cleanup["migration_total_usd"],
        "retained_storage_estimate": migration_cleanup[
            "retained_storage_estimate"
        ],
        "snapshot_capture_manifest_path": migration_cleanup[
            "snapshot_capture_manifest_path"
        ],
        "snapshot_capture_manifest_sha256": migration_cleanup[
            "snapshot_capture_manifest_sha256"
        ],
        "migration_lineage_path": migration_cleanup["migration_lineage_path"],
        "migration_lineage_sha256": migration_cleanup[
            "migration_lineage_sha256"
        ],
        "active_app_count": migration_cleanup["active_app_count"],
        "active_container_count": migration_cleanup["active_container_count"],
        "active_endpoint_count": migration_cleanup["active_endpoint_count"],
        "primary_execution_context_count": len(primary_contexts),
        "artifact_verifier_context_count": len(
            migration_cleanup["artifact_verifier_executions"]
        ),
        "additional_artifact_verifier_context_count": len(
            migration_cleanup["additional_artifact_verifier_executions"]
        ),
        "action_attempt_count": len(migration_cleanup["action_attempts"]),
        "failed_run_ids": migration_cleanup["failed_run_ids"],
        "quarantined_run_ids": migration_cleanup["quarantined_run_ids"],
        "recovery_run_ids": migration_cleanup["recovery_run_ids"],
        "final_accepted_roster": migration_cleanup["final_accepted_roster"],
        "artifact_verifier_executions": migration_cleanup[
            "artifact_verifier_executions"
        ],
        "additional_artifact_verifier_executions": migration_cleanup[
            "additional_artifact_verifier_executions"
        ],
        "evidence_backed_failed_ordinary_executions": migration_cleanup[
            "evidence_backed_failed_ordinary_executions"
        ],
        "action_attempts": migration_cleanup["action_attempts"],
        "billing_attributions": migration_cleanup["billing_attributions"],
        "modal_compute_exposure": migration_cleanup[
            "modal_compute_exposure"
        ],
        "task_function_call_inventory": migration_cleanup[
            "task_function_call_inventory"
        ],
        "direct_detached_call_inventory": migration_cleanup[
            "direct_detached_call_inventory"
        ],
        "artifact_verifier_network_policy": migration_cleanup[
            "artifact_verifier_network_policy"
        ],
        "provider_spend_estimate": migration_cleanup["provider_spend_estimate"],
        "migration_provider_spend_estimate": migration_cleanup[
            "migration_provider_spend_estimate"
        ],
        "volume_run_directory_inventory": migration_cleanup[
            "volume_run_directory_inventory"
        ],
        "snapshot_sha256": {
            name: cleanup_payload["snapshots"][name]["sha256"]
            for name in _SNAPSHOT_NAMES
        },
    }

    required: list[dict[str, Any]] = []
    for run_id in run_ids:
        required.extend(
            _required_artifact(root, run_id, name)
            for name in ("execution_context.json", "image_source_manifest.json")
        )
    required.extend(candidate_required)
    required.extend(resume_required)
    required.extend(
        _required_artifact(root, env_context.run_id, name)
        for name in ("cuda_environment.json", "remote_action_result.json")
    )
    required.append(
        _required_artifact(root, offline_context.run_id, "remote_action_result.json")
    )
    required.extend(
        _required_artifact(
            root, canary_claims[harness]["run_id"], "remote_action_result.json"
        )
        for harness in CANARY_ORDER
    )
    modal_price_basis_paths = sorted(
        {
            record["terminal"]["receipt"]["modal_price_basis_path"]
            for record in migration_cleanup["action_attempts"]
            if record["terminal"]["receipt"]["modal_cost_estimate"] is not None
        }
    )
    required.extend(
        _required_project_artifact(root, logical)
        for logical in (
            cohort_roster_path,
            roster["provider_price_basis_path"],
            *modal_price_basis_paths,
            migration_cleanup["provider_spend_estimate"]["approval_plan"]["path"],
            *(binding["path"] for binding in roster["component_receipts"].values()),
            modal_component_receipt_path(
                identity,
                "modal_resource_cleanup_validated",
            ).as_posix(),
            roster["provider_canary_selector_path"],
            roster["snapshot_capture_manifest_path"],
            *(
                migration_cleanup["snapshots"][name]["path"]
                for name in _SNAPSHOT_NAMES
            ),
            roster["migration_lineage_path"],
            *roster["action_intent_receipts"],
            *roster["action_attempt_receipts"],
            *roster["provider_canary_aggregate_outcome_receipts"],
            *(
                migration_cleanup["artifact_verifier_executions"][label][
                    "remote_verification_path"
                ]
                for label in _PRIMARY_LABELS
            ),
            *_additional_verifier_required_paths(
                migration_cleanup["additional_artifact_verifier_executions"]
            ),
            *_ordinary_failure_required_paths(
                root,
                migration_cleanup[
                    "evidence_backed_failed_ordinary_executions"
                ],
            ),
            *(
                record["path"]
                for record in migration_cleanup["provider_spend_estimate"][
                    "ledgers"
                ]
                if record["sha256"] is not None
            ),
            *_provider_request_state_evidence_paths(
                migration_cleanup["provider_spend_estimate"]
            ),
            *_lineage_required_paths(root, lineage),
        )
    )
    required_paths = {item["path"] for item in required}
    selected_manifest_runs = {
        record["run_id"]
        for record in lineage["selected_final"]["artifact_manifests"]
        if record["run_id"] in set(run_ids)
    }
    if selected_manifest_runs != set(run_ids):
        raise ValueError(
            "migration bundle lacks all eight accepted raw artifact manifests"
        )
    required_paths.update(
        record["path"]
        for record in lineage["selected_final"]["artifact_manifests"]
        if record["run_id"] in set(run_ids)
    )
    required_by_path = {
        logical: _required_project_artifact(root, logical)
        for logical in sorted(required_paths)
    }

    return {
        "source_tree_sha256": identity.source_tree_sha256,
        "cohort_id": identity.cohort_id,
        "cohort_roster": {
            "path": cohort_roster_path,
            "sha256": _sha256_file(roster_path),
        },
        "executions": {
            "cuda_environment": env_claim,
            "candidate_smoke": candidate_claim,
            "resume_attempt": resume_claim,
            "offline_smoke": offline_claim,
            "canaries": canary_claims,
        },
        "image_source_sha256": image_source_sha256,
        "migration_lineage": lineage_binding,
        "evidence": {
            "frozen_image_source": image_binding,
            "cuda_environment_receipt": cuda_receipt,
            "offline_smoke_validation_receipt": offline_receipt,
            "artifact_round_trip_receipt": round_trip_receipt,
            "candidate_resume_preflight_receipt": preflight_receipt,
            "artifact_downloads": download_evidence,
            "candidate_smoke": candidate_evidence,
            "resume_attempt": resume_evidence,
            "provider_canaries": {
                "run_id_prefix": _canary_prefix(canary_runs),
                "selected_run_ids": canary_runs,
                "validation_sha256": canonical_sha256(canary_report),
            },
            "offline_reconstruction_reporting": {
                "validation_sha256": offline_report["validation_sha256"],
                "study_id": offline_report["study"]["study_id"],
                "run_count": offline_report["study"]["run_count"],
                "offline_smoke": offline_report["study"]["offline_smoke"],
            },
            "resource_cleanup_receipt": cleanup_receipt,
            "migration_cleanup": migration_cleanup_evidence,
        },
        "required_artifacts": [
            required_by_path[path] for path in sorted(required_by_path)
        ],
        "validated": True,
    }


def _validate_bundle_payload(payload: Mapping[str, Any], *, root: Path) -> str:
    if set(payload) != _BUNDLE_FIELDS:
        raise ValueError("migration bundle receipt has unexpected or missing fields")
    contract = MODAL_READINESS_RECEIPT_CONTRACTS[
        "modal_migration_validation_bundle_validated"
    ]["receipt_contract"]
    if (
        payload["schema_name"] != contract["schema_name"]
        or payload["schema_version"] != contract["schema_version"]
    ):
        raise ValueError("migration bundle receipt has the wrong schema contract")
    _utc(payload["recorded_at_utc"], "recorded_at_utc")
    identity = _cohort_identity_from_payload(payload, field="migration_bundle")
    executions = payload["executions"]
    if not isinstance(executions, dict) or set(executions) != _BUNDLE_EXECUTION_KEYS:
        raise ValueError("migration bundle executions have an invalid exact schema")
    for name in (
        "cuda_environment",
        "candidate_smoke",
        "resume_attempt",
        "offline_smoke",
    ):
        if not isinstance(executions[name], dict) or set(executions[name]) != (
            _EXECUTION_FIELDS
        ):
            raise ValueError(f"migration execution {name} has an invalid exact schema")
    canaries = executions["canaries"]
    if not isinstance(canaries, dict) or set(canaries) != set(CANARY_ORDER):
        raise ValueError("migration bundle does not contain exactly four canaries")
    for harness in CANARY_ORDER:
        if not isinstance(canaries[harness], dict) or set(canaries[harness]) != (
            _EXECUTION_FIELDS
        ):
            raise ValueError(f"migration canary {harness} has an invalid exact schema")

    cohort_roster = payload["cohort_roster"]
    if not isinstance(cohort_roster, dict) or set(cohort_roster) != {
        "path",
        "sha256",
    }:
        raise ValueError("migration bundle cohort roster has an invalid exact schema")
    roster, roster_path = _load_cohort_roster(
        root, _text(cohort_roster["path"], "cohort_roster.path")
    )
    _assert_identity_matches(roster, identity, field="migration_bundle.roster")
    _sha256(cohort_roster["sha256"], "cohort_roster.sha256")
    if cohort_roster["sha256"] != _sha256_file(roster_path):
        raise ValueError("migration bundle cohort roster digest mismatch")
    expected_run_ids = {
        "cuda_environment": executions["cuda_environment"]["run_id"],
        "candidate_smoke": executions["candidate_smoke"]["run_id"],
        "resume_attempt": executions["resume_attempt"]["run_id"],
        "offline_smoke": executions["offline_smoke"]["run_id"],
        **{
            f"canary_{harness}": canaries[harness]["run_id"] for harness in CANARY_ORDER
        },
    }
    if roster["accepted_primary_runs"] != expected_run_ids:
        raise ValueError("migration bundle execution roster differs from cohort roster")

    derived = _derive_migration_bundle_claims(
        root,
        cohort_roster_path=cohort_roster["path"],
    )
    for field in (
        "cohort_roster",
        "executions",
        "migration_lineage",
        "evidence",
        "required_artifacts",
    ):
        if not exact_json_equal(payload[field], derived[field]):
            raise ValueError(f"migration bundle {field} differs from live artifacts")
    _sha256(payload["image_source_sha256"], "image_source_sha256")
    if payload["image_source_sha256"] != derived["image_source_sha256"]:
        raise ValueError("migration bundle image source digest differs")
    required = payload["required_artifacts"]
    if not isinstance(required, list) or not required:
        raise ValueError("migration bundle required-artifact roster is empty")
    previous_path = ""
    for index, record in enumerate(required):
        if not isinstance(record, dict) or set(record) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise ValueError(f"required_artifacts[{index}] has an invalid exact schema")
        path = _text(record["path"], f"required_artifacts[{index}].path")
        safe_relative_path(path)
        if path <= previous_path:
            raise ValueError("required artifact paths must be sorted and unique")
        previous_path = path
        _sha256(record["sha256"], f"required_artifacts[{index}].sha256")
        _exact_int(record["size_bytes"], f"required_artifacts[{index}].size_bytes")
    _exact_bool(payload["validated"], "validated")
    if payload["validated"] != derived["validated"]:
        raise ValueError("migration bundle is not validated")
    return (
        "runs=8 canaries=4 smoke_layer_a=valid resume=valid offline=valid cleanup=valid"
    )


def validate_modal_readiness_receipt(
    gate_name: str,
    receipt_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> str:
    """Revalidate one local receipt and every artifact it commits to."""

    if gate_name not in MODAL_READINESS_RECEIPT_CONTRACTS:
        raise ValueError(f"unknown Modal readiness gate: {gate_name}")
    project_root = Path(root)
    supplied = Path(receipt_path)
    payload = _load_object(supplied)
    identity = _cohort_identity_from_payload(payload, field=f"{gate_name}.receipt")
    expected = modal_component_receipt_path(identity, gate_name).as_posix()
    expected_path = _contained_path(
        project_root, expected, "receipt_path", kind="file"
    )
    if supplied.resolve() != expected_path.resolve():
        raise ValueError("receipt path differs from its cohort identity")
    validators = {
        "modal_cuda_environment_validated": _validate_cuda_payload,
        "modal_artifact_round_trip_validated": _validate_round_trip_payload,
        "modal_resource_cleanup_validated": _validate_cleanup_payload,
        "modal_migration_validation_bundle_validated": _validate_bundle_payload,
    }
    return validators[gate_name](payload, root=project_root)


def validate_modal_readiness_gate_record(
    evidence_ledger: Mapping[str, Any],
    gate_name: str,
    *,
    root: str | Path = ROOT,
) -> str:
    """Validate one schema-v4, cohort-selected live readiness record."""

    if (
        evidence_ledger.get("schema_name") != "scientific_readiness_evidence"
        or evidence_ledger.get("schema_version") != "4"
    ):
        raise ValueError("Modal live receipt gates require readiness schema v4")
    expected = MODAL_READINESS_RECEIPT_CONTRACTS.get(gate_name)
    if expected is None:
        raise ValueError(f"unknown Modal readiness gate: {gate_name}")
    gates = evidence_ledger.get("gates")
    record = gates.get(gate_name) if isinstance(gates, dict) else None
    if not isinstance(record, dict) or set(record) != {
        "passed",
        "evidence",
        "receipt_path",
        "receipt_sha256",
        "selected_cohort_identity",
        "receipt_contract",
    }:
        raise ValueError(f"readiness gate {gate_name} has an invalid exact schema")
    if type(record["passed"]) is not bool:
        raise ValueError(f"readiness gate {gate_name}.passed must be boolean")
    if not isinstance(record["evidence"], str) or not record["evidence"].strip():
        raise ValueError(f"readiness gate {gate_name} lacks evidence")
    if record["receipt_contract"] != expected["receipt_contract"]:
        raise ValueError(f"readiness gate {gate_name} receipt contract drifted")
    logical_path = record["receipt_path"]
    if logical_path is None:
        if (
            record["passed"] is not False
            or record["receipt_sha256"] is not None
            or record["selected_cohort_identity"] is not None
        ):
            raise ValueError(
                f"readiness gate {gate_name} pending binding is inconsistent"
            )
        raise FileNotFoundError(f"readiness gate {gate_name} receipt path is pending")
    if record["passed"] is not True:
        raise ValueError(f"readiness gate {gate_name} is not passed")
    selected = record["selected_cohort_identity"]
    if not isinstance(selected, dict) or set(selected) != {
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
    }:
        raise ValueError(f"readiness gate {gate_name} cohort identity is invalid")
    identity = _cohort_identity_from_payload(
        selected, field=f"readiness.{gate_name}.selected_cohort_identity"
    )
    expected_path = modal_component_receipt_path(identity, gate_name).as_posix()
    if logical_path != expected_path:
        raise ValueError(f"readiness gate {gate_name} receipt path drifted")
    expected_sha256 = _sha256(
        record["receipt_sha256"], f"readiness.{gate_name}.receipt_sha256"
    )
    receipt = _contained_path(
        Path(root), logical_path, f"readiness.{gate_name}.receipt_path", kind="file"
    )
    _payload, raw_sha256 = _load_object_with_sha256(receipt)
    if raw_sha256 != expected_sha256:
        raise ValueError(f"readiness gate {gate_name} receipt digest changed")
    for other_gate in MODAL_READINESS_RECEIPT_CONTRACTS:
        other = gates.get(other_gate) if isinstance(gates, dict) else None
        if not isinstance(other, dict):
            continue
        other_identity = other.get("selected_cohort_identity")
        if other_identity is not None and not exact_json_equal(
            other_identity, selected
        ):
            raise ValueError("Modal readiness gates select mixed cohort identities")
    result = validate_modal_readiness_receipt(gate_name, receipt, root=root)
    payload = _load_object(receipt)
    _assert_identity_matches(payload, identity, field=f"readiness.{gate_name}")
    return result


def record_cuda_environment(
    *,
    run_id: str,
    cohort_id: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create the immutable CUDA receipt from an already-downloaded run."""

    project_root = Path(root)
    run = validate_run_id(run_id)
    logical = _expected_download_path(run)
    run_root, manifest, verification, context = _inspect_downloaded_run(
        project_root, run, logical
    )
    identity = _identity_for_recording(
        project_root=project_root,
        image_source_sha256=manifest.image_source_sha256,
        cohort_id=cohort_id,
    )
    cuda_path = run_root / "cuda_environment.json"
    action_path = run_root / "remote_action_result.json"
    cuda = _load_object(cuda_path)
    payload = {
        **MODAL_READINESS_RECEIPT_CONTRACTS["modal_cuda_environment_validated"][
            "receipt_contract"
        ],
        **modal_cohort_identity_dict(identity),
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "run_id": run,
        "downloaded_run_path": logical,
        "execution_backend": context.execution_backend,
        "app_name": context.app_name,
        "function_name": context.function_name,
        "requested_gpu": GPU_TYPE,
        "observed_gpu_name": cuda.get("cuda_device_name"),
        "cuda_available": cuda.get("cuda_available"),
        "cuda_device_count": cuda.get("cuda_device_count"),
        "artifact_uri": context.artifact_uri,
        "image_source_sha256": manifest.image_source_sha256,
        "execution_context_sha256": _sha256_file(run_root / "execution_context.json"),
        "cuda_environment_sha256": _sha256_file(cuda_path),
        "remote_action_result_sha256": _sha256_file(action_path),
        "artifact_manifest_sha256": manifest.manifest_sha256,
        "files_verified": verification["file_count"],
        "validated": True,
    }
    _validate_cuda_payload(payload, root=project_root)
    output = project_root.resolve().joinpath(
        *modal_component_receipt_path(
            identity, "modal_cuda_environment_validated"
        ).parts
    )
    create_json_exclusive(output, payload)
    persisted = _load_object(output)
    _validate_cuda_payload(persisted, root=project_root)
    if persisted != payload:
        raise ValueError("persisted CUDA receipt changed after creation")
    return persisted


def record_artifact_round_trip(
    *,
    source_run_id: str,
    verifier_run_id: str,
    verifier_attempt_id: str,
    cohort_id: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Bind saved remote verification to a reverified local download."""

    project_root = Path(root)
    source_run = validate_run_id(source_run_id)
    verifier_run = validate_run_id(verifier_run_id)
    logical = _expected_download_path(source_run)
    _, raw_manifest, verification, source_context = _inspect_downloaded_run_raw(
        project_root,
        source_run,
        logical,
    )
    identity = _identity_for_recording(
        project_root=project_root,
        image_source_sha256=raw_manifest.manifest.image_source_sha256,
        cohort_id=cohort_id,
    )
    remote_logical = _remote_verification_logical(
        identity,
        source_run,
        verifier_run,
        verifier_attempt_id,
    )
    remote = _contained_path(
        project_root, remote_logical, "remote_verification_path", kind="file"
    )
    remote_verification = _validate_remote_verification(
        _load_object(remote),
        source_run_id=source_run,
        verifier_run_id=verifier_run,
        raw_manifest=raw_manifest,
        source_execution_context=source_context,
    )
    payload = {
        **MODAL_READINESS_RECEIPT_CONTRACTS["modal_artifact_round_trip_validated"][
            "receipt_contract"
        ],
        **modal_cohort_identity_dict(identity),
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_run_id": source_run,
        "verifier_run_id": verifier_run,
        "verifier_attempt_id": _attempt_id(
            verifier_attempt_id, "verifier_attempt_id"
        ),
        "downloaded_run_path": logical,
        "artifact_uri": volume_artifact_uri(source_run),
        "manifest_filename": raw_manifest.filename,
        "remote_verification_path": remote_logical,
        "remote_verification_sha256": _sha256_file(remote),
        "verifier_execution_context_sha256": canonical_sha256(
            remote_verification.verifier_execution_context.to_dict()
        ),
        "remote_raw_manifest_sha256": remote_verification.raw_manifest_sha256,
        "remote_raw_manifest_size_bytes": (remote_verification.raw_manifest_size_bytes),
        "local_raw_manifest_sha256": raw_manifest.raw_sha256,
        "local_raw_manifest_size_bytes": raw_manifest.raw_size_bytes,
        "remote_canonical_manifest_sha256": (
            remote_verification.canonical_manifest_sha256
        ),
        "local_canonical_manifest_sha256": (raw_manifest.manifest.manifest_sha256),
        "files_verified": verification["file_count"],
        "remote_verification_completed": True,
        "local_verification_completed": True,
        "validated": True,
    }
    _validate_round_trip_payload(payload, root=project_root)
    output = project_root.resolve().joinpath(
        *modal_component_receipt_path(
            identity, "modal_artifact_round_trip_validated"
        ).parts
    )
    create_json_exclusive(output, payload)
    persisted = _load_object(output)
    _validate_round_trip_payload(persisted, root=project_root)
    if persisted != payload:
        raise ValueError("persisted round-trip receipt changed after creation")
    return persisted


def record_resource_cleanup(
    *,
    cohort_roster_path: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create the aggregate cleanup receipt from one explicit frozen cohort."""

    project_root = Path(root)
    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        _scan_resolved_modal_global_action_journal(lock_descriptor)
        roster, _ = _load_cohort_roster(project_root, cohort_roster_path)
        identity = _cohort_identity_from_payload(roster, field="cohort_roster")
        recorded_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        payload = {
            **MODAL_READINESS_RECEIPT_CONTRACTS[
                "modal_resource_cleanup_validated"
            ]["receipt_contract"],
            **modal_cohort_identity_dict(identity),
            "recorded_at_utc": recorded_at,
            "app_name": APP_NAME,
            "volume_name": VOLUME_NAME,
            **_derive_cleanup_claims(
                project_root,
                cohort_roster_path,
                recorded_at_utc=recorded_at,
            ),
            "validated": True,
        }
        assert_modal_action_lock_identity(lock_descriptor)
        _validate_cleanup_payload(payload, root=project_root)
        output = project_root.resolve().joinpath(
            *modal_component_receipt_path(
                identity, "modal_resource_cleanup_validated"
            ).parts
        )
        assert_modal_action_lock_identity(lock_descriptor)
        create_json_exclusive(output, payload)
        assert_modal_action_lock_identity(lock_descriptor)
        persisted = _load_object(output)
        _validate_cleanup_payload(persisted, root=project_root)
        if persisted != payload:
            raise ValueError("persisted cleanup receipt changed after creation")
        assert_modal_action_lock_identity(lock_descriptor)
        return persisted
    finally:
        release_modal_action_lock(lock_descriptor)


def create_modal_price_basis(
    *,
    expected_image_source_sha256: str,
    official_source_url: str,
    retrieved_at_utc: str,
    cpu_usd_per_core_second: str,
    memory_usd_per_gib_second: str,
    t4_usd_per_gpu_second: str,
    volume_storage_usd_per_gib_month: str,
    included_volume_storage_gib_per_month: str,
    download_transfer_pricing: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create one immutable, source-bound operator-supplied Modal rate record."""

    project_root = Path(root)
    expected_image = _sha256(
        expected_image_source_sha256,
        "expected_image_source_sha256",
    )
    current_image = build_image_source_manifest(project_root).manifest_sha256
    if current_image != expected_image:
        raise ValueError("current image source differs from the Modal price basis")
    payload = {
        "schema_name": "ModalPriceBasis",
        "schema_version": "1.0",
        "image_source_sha256": expected_image,
        "official_source_url": official_source_url,
        "retrieved_at_utc": retrieved_at_utc,
        "region": None,
        "cpu_usd_per_core_second": cpu_usd_per_core_second,
        "memory_usd_per_gib_second": memory_usd_per_gib_second,
        "t4_usd_per_gpu_second": t4_usd_per_gpu_second,
        "volume_storage_usd_per_gib_month": volume_storage_usd_per_gib_month,
        "included_volume_storage_gib_per_month": (
            included_volume_storage_gib_per_month
        ),
        "download_transfer_pricing": download_transfer_pricing,
    }
    validate_modal_price_basis_payload(
        payload,
        expected_image_source_sha256=expected_image,
        require_freshness=True,
    )
    logical = modal_price_basis_logical_path(
        expected_image,
        retrieved_at_utc,
    )
    output = project_root.resolve().joinpath(*logical.parts)
    create_json_exclusive(output, payload)
    persisted, _rates, persisted_path = load_modal_price_basis(
        project_root,
        logical.as_posix(),
        expected_raw_sha256=_sha256_file(output),
        expected_image_source_sha256=expected_image,
        require_freshness=True,
    )
    if persisted_path != output or persisted != payload:
        raise ValueError("persisted Modal price basis changed after creation")
    return persisted


def create_provider_price_basis(
    *,
    source_tree_sha256_value: str,
    image_source_sha256: str,
    cohort_id: str,
    official_source_url: str,
    retrieved_at_utc: str,
    uncached_input_usd_per_million_tokens: str,
    output_usd_per_million_tokens: str,
    per_request_fee_usd: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create one immutable operator-supplied provider price-basis record."""

    identity = ModalLiveCohortIdentity(
        source_tree_sha256=_sha256(
            source_tree_sha256_value, "source_tree_sha256"
        ),
        image_source_sha256=_sha256(
            image_source_sha256, "image_source_sha256"
        ),
        cohort_id=validate_run_id(cohort_id),
    )
    source_url = _text(official_source_url, "official_source_url")
    if re.fullmatch(r"https://(?:platform\.)?openai\.com/[^\s]*", source_url) is None:
        raise ValueError("provider price basis must cite an official OpenAI HTTPS URL")
    _utc(retrieved_at_utc, "retrieved_at_utc")
    for value, field in (
        (
            uncached_input_usd_per_million_tokens,
            "uncached_input_usd_per_million_tokens",
        ),
        (output_usd_per_million_tokens, "output_usd_per_million_tokens"),
        (per_request_fee_usd, "per_request_fee_usd"),
    ):
        _decimal_text(value, field)
    payload = {
        "schema_name": "ProviderPriceBasis",
        "schema_version": "1.0",
        "model": TARGET_MODEL,
        "official_source_url": source_url,
        "retrieved_at_utc": retrieved_at_utc,
        "uncached_input_usd_per_million_tokens": (
            uncached_input_usd_per_million_tokens
        ),
        "output_usd_per_million_tokens": output_usd_per_million_tokens,
        "per_request_fee_usd": per_request_fee_usd,
    }
    project_root = Path(root)
    logical = modal_provider_price_basis_path(identity).as_posix()
    output = project_root.resolve().joinpath(*PurePosixPath(logical).parts)
    create_json_exclusive(output, payload)
    loaded, _, _raw_sha256 = _load_price_basis(project_root, logical)
    if loaded != payload:
        raise ValueError("persisted provider price basis changed")
    return payload


def record_migration_validation_bundle(
    *,
    cohort_roster_path: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Record the one strict, independently revalidated migration bundle."""

    project_root = Path(root)
    lock_descriptor = acquire_modal_action_lock(project_root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        _scan_resolved_modal_global_action_journal(lock_descriptor)
        claims = _derive_migration_bundle_claims(
            project_root,
            cohort_roster_path=cohort_roster_path,
        )
        payload = {
            **MODAL_READINESS_RECEIPT_CONTRACTS[
                "modal_migration_validation_bundle_validated"
            ]["receipt_contract"],
            "recorded_at_utc": datetime.now(UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            **claims,
        }
        assert_modal_action_lock_identity(lock_descriptor)
        _validate_bundle_payload(payload, root=project_root)
        identity = _cohort_identity_from_payload(payload, field="migration_bundle")
        output = project_root.resolve().joinpath(
            *modal_component_receipt_path(
                identity, "modal_migration_validation_bundle_validated"
            ).parts
        )
        assert_modal_action_lock_identity(lock_descriptor)
        create_json_exclusive(output, payload)
        assert_modal_action_lock_identity(lock_descriptor)
        persisted = _load_object(output)
        _validate_bundle_payload(persisted, root=project_root)
        if persisted != payload:
            raise ValueError("persisted migration bundle changed after creation")
        assert_modal_action_lock_identity(lock_descriptor)
        return persisted
    finally:
        release_modal_action_lock(lock_descriptor)


def _preflight_image_binding(
    root: Path,
    executions: Mapping[str, tuple[Path, Any, ExecutionContextV1]],
) -> dict[str, Any]:
    if len(executions) != 4:
        raise ValueError("candidate-resume preflight requires four image bindings")
    current = build_image_source_manifest(root)
    image_ids: set[str] = set()
    for label, (run_root, manifest, context) in executions.items():
        downloaded = _image_source_manifest(run_root / "image_source_manifest.json")
        if (
            downloaded.to_dict() != current.to_dict()
            or manifest.image_source_sha256 != current.manifest_sha256
            or context.image_source_sha256 != current.manifest_sha256
        ):
            raise ValueError(f"preflight image binding differs for {label}")
        image_ids.add(_text(context.modal_image_id, f"{label}.modal_image_id"))
    if len(image_ids) != 1:
        raise ValueError("early runs do not share one Modal image ID")
    return {
        "image_source_sha256": current.manifest_sha256,
        "dependency_lock_sha256": current.dependency_lock_sha256,
        "modal_image_id": next(iter(image_ids)),
    }


def _derive_candidate_resume_preflight(
    *,
    environment_run_id: str,
    offline_run_id: str,
    candidate_run_id: str,
    resume_run_id: str,
    identity: ModalLiveCohortIdentity,
    verifier_bindings: Mapping[str, Mapping[str, str]],
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Revalidate four early Modal runs locally before provider spend."""

    project_root = Path(root)
    environment_run = validate_run_id(environment_run_id)
    offline_run = validate_run_id(offline_run_id)
    candidate_run = validate_run_id(candidate_run_id)
    resume_run = validate_run_id(resume_run_id)
    if len({environment_run, offline_run, candidate_run, resume_run}) != 4:
        raise ValueError("preflight requires four distinct early run IDs")
    verifier_labels = {
        "cuda_environment",
        "offline_smoke",
        "candidate_smoke",
        "resume_attempt",
    }
    if not isinstance(verifier_bindings, Mapping) or set(
        verifier_bindings
    ) != verifier_labels:
        raise ValueError("preflight verifier roster must use the exact labels")
    selected_verifiers: dict[str, dict[str, str]] = {}
    for label in verifier_labels:
        record = verifier_bindings[label]
        if not isinstance(record, Mapping) or set(record) != {
            "verifier_run_id",
            "verifier_attempt_id",
        }:
            raise ValueError(f"preflight verifier binding {label} is invalid")
        selected_verifiers[label] = {
            "verifier_run_id": validate_run_id(record["verifier_run_id"]),
            "verifier_attempt_id": _attempt_id(
                record["verifier_attempt_id"],
                f"preflight.{label}.verifier_attempt_id",
            ),
        }
    environment_claim, environment_root, environment_manifest, environment_context = (
        _execution_for_run(
            project_root,
            environment_run,
            expected_function="cuda_environment",
        )
    )
    offline_claim, offline_root, offline_manifest, offline_context = _execution_for_run(
        project_root,
        offline_run,
        expected_function="offline_smoke",
    )
    candidate_claim, candidate_root, candidate_manifest, candidate_context = (
        _execution_for_run(
            project_root,
            candidate_run,
            expected_function="candidate_smoke",
        )
    )
    resume_claim, resume_root, resume_manifest, resume_context = _execution_for_run(
        project_root,
        resume_run,
        expected_function="checkpoint_resume",
    )
    contexts = (
        environment_context,
        offline_context,
        candidate_context,
        resume_context,
    )
    if len({item.modal_call_id for item in contexts}) != 4:
        raise ValueError("preflight requires four unique Modal call IDs")
    manifests = (
        environment_manifest,
        offline_manifest,
        candidate_manifest,
        resume_manifest,
    )
    if len({item.image_source_sha256 for item in manifests}) != 1:
        raise ValueError("early runs do not share one image source")
    if next(iter({item.image_source_sha256 for item in manifests})) != (
        identity.image_source_sha256
    ):
        raise ValueError("preflight executions differ from the selected cohort image")
    candidate_evidence, candidate_required = _validate_candidate_layer_a(
        project_root,
        candidate_run,
        candidate_root,
        candidate_context,
    )
    resume_evidence, resume_required = _revalidate_resume_attempt(
        project_root,
        candidate_run,
        candidate_root,
        candidate_manifest,
        resume_run,
        resume_root,
        resume_manifest,
        resume_context,
    )
    cuda_receipt, cuda_payload = _receipt_evidence(
        project_root,
        "modal_cuda_environment_validated",
        identity,
    )
    if (
        cuda_payload["run_id"] != environment_run
        or cuda_payload["artifact_manifest_sha256"]
        != environment_manifest.manifest_sha256
    ):
        raise ValueError("preflight CUDA receipt is not environment-run bound")
    offline_receipt, offline_payload = _offline_smoke_receipt_evidence(
        project_root, identity
    )
    if (
        offline_payload["run_id"] != offline_run
        or offline_payload["artifact_manifest_sha256"]
        != offline_manifest.manifest_sha256
        or offline_payload["remote_verifier_run_id"]
        != selected_verifiers["offline_smoke"]["verifier_run_id"]
        or offline_payload["remote_verifier_attempt_id"]
        != selected_verifiers["offline_smoke"]["verifier_attempt_id"]
    ):
        raise ValueError("preflight offline receipt is not offline-run bound")
    round_trip_receipt, round_trip_payload = _receipt_evidence(
        project_root,
        "modal_artifact_round_trip_validated",
        identity,
    )
    if (
        round_trip_payload["source_run_id"] != candidate_run
        or round_trip_payload["local_canonical_manifest_sha256"]
        != candidate_manifest.manifest_sha256
        or round_trip_payload["verifier_run_id"]
        != selected_verifiers["candidate_smoke"]["verifier_run_id"]
        or round_trip_payload["verifier_attempt_id"]
        != selected_verifiers["candidate_smoke"]["verifier_attempt_id"]
    ):
        raise ValueError("preflight round-trip receipt is not candidate-run bound")
    offline_evidence = validate_downloaded_offline_bundle(offline_root)
    if (
        offline_evidence.get("verified") is not True
        or offline_evidence.get("network_calls") != 0
        or offline_evidence.get("provider_calls") != 0
        or offline_evidence.get("modal_run_id") != offline_run
        or offline_evidence.get("artifact_manifest_sha256")
        != offline_manifest.manifest_sha256
    ):
        raise ValueError("preflight offline reconstruction did not revalidate")
    downloads = {
        "cuda_environment": _remote_download_evidence(
            project_root,
            environment_run,
            environment_manifest,
            identity=identity,
            verifier_run_id=selected_verifiers["cuda_environment"][
                "verifier_run_id"
            ],
            verifier_attempt_id=selected_verifiers["cuda_environment"][
                "verifier_attempt_id"
            ],
        ),
        "offline_smoke": _remote_download_evidence(
            project_root,
            offline_run,
            offline_manifest,
            identity=identity,
            verifier_run_id=selected_verifiers["offline_smoke"][
                "verifier_run_id"
            ],
            verifier_attempt_id=selected_verifiers["offline_smoke"][
                "verifier_attempt_id"
            ],
        ),
        "candidate_smoke": _remote_download_evidence(
            project_root,
            candidate_run,
            candidate_manifest,
            identity=identity,
            verifier_run_id=selected_verifiers["candidate_smoke"][
                "verifier_run_id"
            ],
            verifier_attempt_id=selected_verifiers["candidate_smoke"][
                "verifier_attempt_id"
            ],
        ),
        "resume_attempt": _remote_download_evidence(
            project_root,
            resume_run,
            resume_manifest,
            identity=identity,
            verifier_run_id=selected_verifiers["resume_attempt"][
                "verifier_run_id"
            ],
            verifier_attempt_id=selected_verifiers["resume_attempt"][
                "verifier_attempt_id"
            ],
        ),
    }
    image_binding = _preflight_image_binding(
        project_root,
        {
            "cuda_environment": (
                environment_root,
                environment_manifest,
                environment_context,
            ),
            "offline_smoke": (offline_root, offline_manifest, offline_context),
            "candidate_smoke": (
                candidate_root,
                candidate_manifest,
                candidate_context,
            ),
            "resume_attempt": (resume_root, resume_manifest, resume_context),
        },
    )
    predecessor_receipts = {
        "modal_cuda_environment_validated": cuda_receipt,
        "modal_offline_smoke_validated": offline_receipt,
        "modal_artifact_round_trip_validated": round_trip_receipt,
    }
    execution_run_ids = {
        "cuda_environment": environment_run,
        "offline_smoke": offline_run,
        "candidate_smoke": candidate_run,
        "resume_attempt": resume_run,
    }
    binding_sha256 = modal_candidate_resume_preflight_binding_sha256(
        identity=identity,
        execution_run_ids=execution_run_ids,
        verifier_bindings=selected_verifiers,
        predecessor_receipts=predecessor_receipts,
    )
    return {
        "source_tree_sha256": identity.source_tree_sha256,
        "cohort_id": identity.cohort_id,
        "binding_sha256": binding_sha256,
        "verifier_bindings": selected_verifiers,
        "predecessor_receipts": predecessor_receipts,
        "validation_mode": "local_read_only_provider_free",
        "cuda_environment": environment_claim,
        "offline_smoke": offline_claim,
        "candidate_smoke": candidate_claim,
        "resume_attempt": resume_claim,
        **image_binding,
        "evidence": {
            "cuda_environment_receipt": cuda_receipt,
            "offline_smoke_validation_receipt": offline_receipt,
            "offline_reconstruction_reporting": {
                "validation_sha256": offline_evidence["validation_sha256"],
                "study_id": offline_evidence["study"]["study_id"],
                "run_count": offline_evidence["study"]["run_count"],
            },
            "artifact_round_trip_receipt": round_trip_receipt,
            "candidate_smoke": candidate_evidence,
            "resume_attempt": resume_evidence,
            "artifact_downloads": downloads,
        },
        "required_artifact_count": len(candidate_required) + len(resume_required),
        "remote_calls_started": 0,
        "provider_calls_started": 0,
        "training_runs_started": 0,
        "valid": True,
    }


def validate_candidate_resume_preflight_receipt(
    receipt_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Reopen and rederive one binding-addressed provider-stage preflight."""

    project_root = Path(root)
    supplied = Path(receipt_path)
    payload = _load_object(supplied)
    if (
        payload.get("schema_name") != "CandidateResumePreflightReceipt"
        or payload.get("schema_version") != "2.0"
    ):
        raise ValueError("candidate-resume preflight receipt has the wrong contract")
    _utc(payload.get("recorded_at_utc"), "preflight.recorded_at_utc")
    identity = _cohort_identity_from_payload(payload, field="candidate_preflight")
    binding_sha256 = _sha256(
        payload.get("binding_sha256"), "preflight.binding_sha256"
    )
    expected_logical = modal_candidate_resume_preflight_receipt_path(
        identity, binding_sha256
    ).as_posix()
    path = _contained_path(
        project_root,
        expected_logical,
        "candidate_resume_preflight_receipt",
        kind="file",
    )
    if supplied.resolve() != path.resolve():
        raise ValueError("candidate preflight path differs from its binding digest")
    executions = {
        key: payload.get(key)
        for key in (
            "cuda_environment",
            "offline_smoke",
            "candidate_smoke",
            "resume_attempt",
        )
    }
    if any(not isinstance(item, dict) for item in executions.values()):
        raise ValueError("candidate-resume preflight execution roster is invalid")
    verifier_bindings = payload.get("verifier_bindings")
    if not isinstance(verifier_bindings, dict) or set(verifier_bindings) != {
        "cuda_environment",
        "offline_smoke",
        "candidate_smoke",
        "resume_attempt",
    }:
        raise ValueError("candidate-resume preflight verifier roster is invalid")
    for _label, record in verifier_bindings.items():
        if not isinstance(record, dict) or set(record) != {
            "verifier_run_id",
            "verifier_attempt_id",
        }:
            raise ValueError("candidate-resume preflight verifier entry is invalid")
    derived = _derive_candidate_resume_preflight(
        environment_run_id=executions["cuda_environment"]["run_id"],
        offline_run_id=executions["offline_smoke"]["run_id"],
        candidate_run_id=executions["candidate_smoke"]["run_id"],
        resume_run_id=executions["resume_attempt"]["run_id"],
        identity=identity,
        verifier_bindings=verifier_bindings,
        root=project_root,
    )
    expected = {
        "schema_name": "CandidateResumePreflightReceipt",
        "schema_version": "2.0",
        "recorded_at_utc": payload["recorded_at_utc"],
        **derived,
    }
    if payload != expected:
        raise ValueError(
            "candidate-resume preflight receipt differs from live artifacts"
        )
    return payload


def validate_candidate_resume_preflight(
    *,
    environment_run_id: str,
    offline_run_id: str,
    candidate_run_id: str,
    resume_run_id: str,
    cohort_id: str,
    verifier_bindings: Mapping[str, Mapping[str, str]],
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create and reopen one binding-addressed provider-stage preflight."""

    project_root = Path(root)
    environment_run = validate_run_id(environment_run_id)
    _claim, _run_root, manifest, _context = _execution_for_run(
        project_root,
        environment_run,
        expected_function="cuda_environment",
    )
    identity = _identity_for_recording(
        project_root=project_root,
        image_source_sha256=manifest.image_source_sha256,
        cohort_id=cohort_id,
    )
    payload = {
        "schema_name": "CandidateResumePreflightReceipt",
        "schema_version": "2.0",
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        **_derive_candidate_resume_preflight(
            environment_run_id=environment_run_id,
            offline_run_id=offline_run_id,
            candidate_run_id=candidate_run_id,
            resume_run_id=resume_run_id,
            identity=identity,
            verifier_bindings=verifier_bindings,
            root=project_root,
        ),
    }
    binding_sha256 = _sha256(
        payload["binding_sha256"], "preflight.binding_sha256"
    )
    output = project_root.resolve().joinpath(
        *modal_candidate_resume_preflight_receipt_path(
            identity, binding_sha256
        ).parts
    )
    create_json_exclusive(output, payload)
    persisted = validate_candidate_resume_preflight_receipt(
        output, root=project_root
    )
    if persisted != payload:
        raise ValueError("persisted candidate-resume preflight receipt changed")
    return persisted


def validate_artifact_verifier_capture(
    *,
    source_run_id: str,
    verifier_run_id: str,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Validate verifier evidence fetched within an approved ``download`` action.

    The Volume read is not intrinsically cost-free: storage and transfer may be
    billed, so capture belongs inside the launcher's separately capped action.
    """

    project_root = Path(root)
    source = validate_run_id(source_run_id)
    verifier = validate_run_id(verifier_run_id)
    capture_logical = modal_artifact_verifier_capture_directory_path(
        identity,
        source,
        verifier,
        _attempt_id(attempt_id, "attempt_id"),
    ).as_posix()
    capture_root = _contained_path(
        project_root,
        capture_logical,
        "artifact_verifier_capture_directory",
        kind="directory",
    )
    result_path = capture_root / "artifact_verification_result.json"
    failure_path = capture_root / "artifact_verification_failure.json"
    if result_path.is_file() is failure_path.is_file():
        raise ValueError(
            "verifier capture requires exactly one result or failure receipt"
        )
    if result_path.is_file():
        payload = _load_object(result_path)
        context = ExecutionContextV1.from_dict(
            payload["verifier_execution_context"]
        )
        record = {
            "source_run_id": source,
            "verifier_run_id": verifier,
            "verifier_attempt_id": _attempt_id(attempt_id, "attempt_id"),
            "remote_verification_path": result_path.relative_to(
                project_root
            ).as_posix(),
            "remote_verification_sha256": _sha256_file(result_path),
            "verifier_execution_context": context.to_dict(),
        }
        _, manifest_sha256 = _successful_verifier_capture(
            project_root, record, identity=identity
        )
        outcome = "success"
        evidence_kind = "volume_success_capture"
        expected_roster = list(_VERIFIER_REMOTE_RECEIPT_ROSTER)
        success_fields = record
        failure_fields = {
            "failure_receipt_path": None,
            "failure_receipt_sha256": None,
            "failure_execution_context": None,
        }
    else:
        payload = _load_object(failure_path)
        context = ExecutionContextV1.from_dict(
            payload["verifier_execution_context"]
        )
        record = {
            "source_run_id": source,
            "verifier_run_id": verifier,
            "verifier_attempt_id": _attempt_id(attempt_id, "attempt_id"),
            "failure_receipt_path": failure_path.relative_to(
                project_root
            ).as_posix(),
            "failure_receipt_sha256": _sha256_file(failure_path),
            "failure_execution_context": context.to_dict(),
        }
        _, manifest_sha256 = _failed_verifier_capture(
            project_root, record, identity=identity
        )
        outcome = "failure"
        evidence_kind = "volume_failure_capture"
        expected_roster = list(_FAILED_VERIFIER_REMOTE_RECEIPT_ROSTER)
        success_fields = {
            "remote_verification_path": None,
            "remote_verification_sha256": None,
            "verifier_execution_context": None,
        }
        failure_fields = record
    return {
        "schema_name": "ModalArtifactVerifierCaptureValidation",
        "schema_version": "2.0",
        **modal_cohort_identity_dict(identity),
        "source_run_id": source,
        "verifier_run_id": verifier,
        "verifier_attempt_id": _attempt_id(attempt_id, "attempt_id"),
        "remote_verifier_outcome": outcome,
        "remote_evidence_kind": evidence_kind,
        **{
            key: value
            for key, value in success_fields.items()
            if key not in {
                "source_run_id",
                "verifier_run_id",
                "verifier_attempt_id",
            }
        },
        **{
            key: value
            for key, value in failure_fields.items()
            if key not in {
                "source_run_id",
                "verifier_run_id",
                "verifier_attempt_id",
            }
        },
        "expected_remote_receipt_roster": expected_roster,
        "capture_manifest_sha256": manifest_sha256,
        "validated": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record completed Modal readiness checks without provider calls."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    cuda = subparsers.add_parser("cuda-environment")
    cuda.add_argument("--run-id", required=True)
    cuda.add_argument("--cohort-id", required=True)
    offline = subparsers.add_parser("offline-smoke")
    offline.add_argument("--run-id", required=True)
    offline.add_argument("--cohort-id", required=True)
    offline.add_argument("--verifier-run-id", required=True)
    offline.add_argument("--verifier-attempt-id", required=True)
    round_trip = subparsers.add_parser("artifact-round-trip")
    round_trip.add_argument("--source-run-id", required=True)
    round_trip.add_argument("--verifier-run-id", required=True)
    round_trip.add_argument("--verifier-attempt-id", required=True)
    round_trip.add_argument("--cohort-id", required=True)
    capture = subparsers.add_parser("capture-remote-verification")
    capture.add_argument("--source-run-id", required=True)
    capture.add_argument("--verifier-run-id", required=True)
    capture.add_argument("--verifier-attempt-id", required=True)
    capture.add_argument("--source-tree-sha256", required=True)
    capture.add_argument("--image-source-sha256", required=True)
    capture.add_argument("--cohort-id", required=True)
    capture.add_argument("--transcript", required=True)
    verifier_capture = subparsers.add_parser("validate-verifier-capture")
    verifier_capture.add_argument("--source-run-id", required=True)
    verifier_capture.add_argument("--verifier-run-id", required=True)
    verifier_capture.add_argument("--verifier-attempt-id", required=True)
    verifier_capture.add_argument("--source-tree-sha256", required=True)
    verifier_capture.add_argument("--image-source-sha256", required=True)
    verifier_capture.add_argument("--cohort-id", required=True)
    modal_price = subparsers.add_parser("modal-price-basis")
    modal_price.add_argument("--expected-image-source-sha256", required=True)
    modal_price.add_argument("--official-source-url", required=True)
    modal_price.add_argument("--retrieved-at-utc", required=True)
    modal_price.add_argument("--cpu-usd-per-core-second", required=True)
    modal_price.add_argument("--memory-usd-per-gib-second", required=True)
    modal_price.add_argument("--t4-usd-per-gpu-second", required=True)
    modal_price.add_argument("--volume-storage-usd-per-gib-month", required=True)
    modal_price.add_argument(
        "--included-volume-storage-gib-per-month", required=True
    )
    modal_price.add_argument("--download-transfer-pricing", required=True)
    price_basis = subparsers.add_parser("provider-price-basis")
    price_basis.add_argument("--source-tree-sha256", required=True)
    price_basis.add_argument("--image-source-sha256", required=True)
    price_basis.add_argument("--cohort-id", required=True)
    price_basis.add_argument("--official-source-url", required=True)
    price_basis.add_argument("--retrieved-at-utc", required=True)
    price_basis.add_argument("--uncached-input-usd-per-million-tokens", required=True)
    price_basis.add_argument("--output-usd-per-million-tokens", required=True)
    price_basis.add_argument("--per-request-fee-usd", required=True)
    cleanup = subparsers.add_parser("resource-cleanup")
    cleanup.add_argument("--cohort-roster", required=True)
    prior_template = subparsers.add_parser(
        "prior-quarantine-accounting-template"
    )
    prior_template.add_argument("--source-tree-sha256", required=True)
    prior_template.add_argument("--image-source-sha256", required=True)
    prior_template.add_argument("--cohort-id", required=True)
    prior_template.add_argument("--recorded-at-utc", required=True)
    prior_template.add_argument("--snapshot-capture-manifest", required=True)
    prior_template.add_argument("--output", required=True)
    prior_inspect = subparsers.add_parser(
        "prior-quarantine-accounting-inspect"
    )
    prior_inspect.add_argument("--input", required=True)
    prior_scaffold = subparsers.add_parser(
        "prior-quarantine-accounting-scaffold"
    )
    prior_scaffold.add_argument("--input", required=True)
    prior_scaffold.add_argument("--output", required=True)
    prior_accounting = subparsers.add_parser("prior-quarantine-accounting")
    prior_accounting.add_argument("--input", required=True)
    lineage = subparsers.add_parser("migration-lineage")
    lineage.add_argument("--input", required=True)
    cohort_roster = subparsers.add_parser("cohort-roster")
    cohort_roster.add_argument("--input", required=True)
    preflight = subparsers.add_parser("candidate-resume-preflight")
    preflight.add_argument("--environment-run-id", required=True)
    preflight.add_argument("--offline-run-id", required=True)
    preflight.add_argument("--candidate-run-id", required=True)
    preflight.add_argument("--resume-run-id", required=True)
    preflight.add_argument("--environment-verifier-run-id", required=True)
    preflight.add_argument("--offline-verifier-run-id", required=True)
    preflight.add_argument("--candidate-verifier-run-id", required=True)
    preflight.add_argument("--resume-verifier-run-id", required=True)
    preflight.add_argument("--environment-verifier-attempt-id", required=True)
    preflight.add_argument("--offline-verifier-attempt-id", required=True)
    preflight.add_argument("--candidate-verifier-attempt-id", required=True)
    preflight.add_argument("--resume-verifier-attempt-id", required=True)
    preflight.add_argument("--cohort-id", required=True)
    bundle = subparsers.add_parser("migration-bundle")
    bundle.add_argument("--cohort-roster", required=True)
    arguments = parser.parse_args()
    if arguments.action == "cuda-environment":
        payload = record_cuda_environment(
            run_id=arguments.run_id,
            cohort_id=arguments.cohort_id,
        )
    elif arguments.action == "offline-smoke":
        payload = record_offline_smoke_validation(
            run_id=arguments.run_id,
            cohort_id=arguments.cohort_id,
            verifier_run_id=arguments.verifier_run_id,
            verifier_attempt_id=arguments.verifier_attempt_id,
        )
    elif arguments.action == "capture-remote-verification":
        payload = capture_remote_verification(
            source_run_id=arguments.source_run_id,
            verifier_run_id=arguments.verifier_run_id,
            identity=ModalLiveCohortIdentity(
                source_tree_sha256=arguments.source_tree_sha256,
                image_source_sha256=arguments.image_source_sha256,
                cohort_id=arguments.cohort_id,
            ),
            attempt_id=arguments.verifier_attempt_id,
            transcript_path=arguments.transcript,
        )
    elif arguments.action == "artifact-round-trip":
        payload = record_artifact_round_trip(
            source_run_id=arguments.source_run_id,
            verifier_run_id=arguments.verifier_run_id,
            verifier_attempt_id=arguments.verifier_attempt_id,
            cohort_id=arguments.cohort_id,
        )
    elif arguments.action == "validate-verifier-capture":
        payload = validate_artifact_verifier_capture(
            source_run_id=arguments.source_run_id,
            verifier_run_id=arguments.verifier_run_id,
            identity=ModalLiveCohortIdentity(
                source_tree_sha256=arguments.source_tree_sha256,
                image_source_sha256=arguments.image_source_sha256,
                cohort_id=arguments.cohort_id,
            ),
            attempt_id=arguments.verifier_attempt_id,
        )
    elif arguments.action == "modal-price-basis":
        payload = create_modal_price_basis(
            expected_image_source_sha256=(
                arguments.expected_image_source_sha256
            ),
            official_source_url=arguments.official_source_url,
            retrieved_at_utc=arguments.retrieved_at_utc,
            cpu_usd_per_core_second=arguments.cpu_usd_per_core_second,
            memory_usd_per_gib_second=arguments.memory_usd_per_gib_second,
            t4_usd_per_gpu_second=arguments.t4_usd_per_gpu_second,
            volume_storage_usd_per_gib_month=(
                arguments.volume_storage_usd_per_gib_month
            ),
            included_volume_storage_gib_per_month=(
                arguments.included_volume_storage_gib_per_month
            ),
            download_transfer_pricing=arguments.download_transfer_pricing,
        )
    elif arguments.action == "provider-price-basis":
        payload = create_provider_price_basis(
            source_tree_sha256_value=arguments.source_tree_sha256,
            image_source_sha256=arguments.image_source_sha256,
            cohort_id=arguments.cohort_id,
            official_source_url=arguments.official_source_url,
            retrieved_at_utc=arguments.retrieved_at_utc,
            uncached_input_usd_per_million_tokens=(
                arguments.uncached_input_usd_per_million_tokens
            ),
            output_usd_per_million_tokens=(arguments.output_usd_per_million_tokens),
            per_request_fee_usd=arguments.per_request_fee_usd,
        )
    elif arguments.action == "resource-cleanup":
        payload = record_resource_cleanup(cohort_roster_path=arguments.cohort_roster)
    elif arguments.action == "prior-quarantine-accounting-template":
        payload = create_prior_quarantine_accounting_template(
            source_tree_sha256_value=arguments.source_tree_sha256,
            image_source_sha256=arguments.image_source_sha256,
            cohort_id=arguments.cohort_id,
            recorded_at_utc=arguments.recorded_at_utc,
            snapshot_capture_manifest_path=(
                arguments.snapshot_capture_manifest
            ),
            output_path=arguments.output,
        )
    elif arguments.action == "prior-quarantine-accounting-inspect":
        payload = inspect_prior_quarantine_accounting(
            request=_load_operator_json_input(arguments.input),
        )
    elif arguments.action == "prior-quarantine-accounting-scaffold":
        payload = scaffold_prior_quarantine_accounting(
            request=_load_operator_json_input(arguments.input),
            output_path=arguments.output,
        )
    elif arguments.action == "prior-quarantine-accounting":
        payload = create_prior_quarantine_accounting(
            payload=_load_operator_json_input(arguments.input),
        )
    elif arguments.action == "migration-lineage":
        payload = create_modal_migration_lineage_from_input(
            payload=_load_operator_json_input(arguments.input),
        )
    elif arguments.action == "cohort-roster":
        payload = create_modal_cohort_roster(
            payload=_load_operator_json_input(arguments.input),
        )
    elif arguments.action == "candidate-resume-preflight":
        payload = validate_candidate_resume_preflight(
            environment_run_id=arguments.environment_run_id,
            offline_run_id=arguments.offline_run_id,
            candidate_run_id=arguments.candidate_run_id,
            resume_run_id=arguments.resume_run_id,
            cohort_id=arguments.cohort_id,
            verifier_bindings={
                "cuda_environment": {
                    "verifier_run_id": arguments.environment_verifier_run_id,
                    "verifier_attempt_id": (
                        arguments.environment_verifier_attempt_id
                    ),
                },
                "offline_smoke": {
                    "verifier_run_id": arguments.offline_verifier_run_id,
                    "verifier_attempt_id": arguments.offline_verifier_attempt_id,
                },
                "candidate_smoke": {
                    "verifier_run_id": arguments.candidate_verifier_run_id,
                    "verifier_attempt_id": (
                        arguments.candidate_verifier_attempt_id
                    ),
                },
                "resume_attempt": {
                    "verifier_run_id": arguments.resume_verifier_run_id,
                    "verifier_attempt_id": arguments.resume_verifier_attempt_id,
                },
            },
        )
    else:
        payload = record_migration_validation_bundle(
            cohort_roster_path=arguments.cohort_roster,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
