"""Cost-free discovery and classification of the local Modal action journal.

The scanner in this module is deliberately independent of both launcher and
readiness scripts.  It accepts only a lock descriptor already held for the
project, derives the protected project root from that descriptor, and reads
the complete local journal through descriptor-relative, no-follow operations.
It never imports Modal, performs provider calls, or mutates recovery state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from common.evolution_run import (
    EVOLUTION_ACTION,
    EVOLUTION_FUNCTION_NAME,
    EvolutionRunSpec,
)
from common.modal_action_lock import (
    assert_modal_action_lock_identity,
    held_modal_action_lock_project_root,
)
from common.provider_attempts import ProviderAttemptRecord
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    APP_NAME,
    ARTIFACT_MANIFEST_FILENAMES,
    CANARY_ORDER,
    IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
    IMAGE_BUILD_CPU_REQUEST_CORES,
    IMAGE_BUILD_MEMORY_REQUEST_MIB,
    IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MAX_MODAL_BILLING_WINDOW,
    MODAL_ACTIONS,
    MODAL_DOWNLOAD_OUTPUT_ROOT,
    MODAL_ENVIRONMENT_NAME,
    MODAL_LAUNCH_REJECTION_ROOT,
    MODAL_LIVE_COHORT_ROOT,
    MODAL_LOCAL_PROCESS_START_ROOT,
    MODAL_REMOTE_RUN_RESERVATION_ROOT,
    MODAL_VERSION,
    OPENEVOLVE_60_ACTION,
    VOLUME_NAME,
    ArtifactIntegrityError,
    ArtifactVerificationV1,
    ModalLiveCohortIdentity,
    canary_run_suffix,
    modal_action_attempt_directory,
    modal_action_host_containment_path,
    modal_action_intent_receipt_path,
    modal_action_recovery_directory,
    modal_action_recovery_intent_path,
    modal_action_recovery_resolution_path,
    modal_action_terminal_receipt_path,
    modal_artifact_verifier_capture_directory_path,
    modal_global_launch_rejection_seal_path,
    modal_launch_rejection_receipt_path,
    modal_live_cohort_root,
    modal_local_host_anchor_path,
    modal_local_process_start_receipt_path,
    modal_migration_lineage_path,
    modal_remote_run_reservation_path,
    modal_remote_verification_receipt_path,
    parse_artifact_manifest_bytes,
    safe_relative_path,
    validate_modal_action_identity,
    validate_provider_canary_aggregate_outcome_receipt,
    validate_run_id,
    volume_artifact_uri,
    function_spec,
)

_ATTEMPT_ID = re.compile(r"\A[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_INTENT_NAME = re.compile(r"\A([0-9a-f]{32})\.intent\.json\Z")
_TERMINAL_NAME = re.compile(r"\A([0-9a-f]{32})\.json\Z")
_AGGREGATE_NAME = re.compile(r"\A([0-9a-f]{32})\.aggregate\.json\Z")
_RECOVERY_NAMES: tuple[
    tuple[re.Pattern[str], Literal["intent", "host_containment", "resolution"]],
    ...,
] = (
    (
        re.compile(r"\A([0-9a-f]{32})\.intent\.v1\.0\.json\Z"),
        "intent",
    ),
    (
        re.compile(r"\A([0-9a-f]{32})\.host-containment\.v1\.0\.json\Z"),
        "host_containment",
    ),
    (
        re.compile(r"\A([0-9a-f]{32})\.resolution\.v1\.0\.json\Z"),
        "resolution",
    ),
)
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_PROVIDER_EVIDENCE_BYTES = 4 * 1024 * 1024
_MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS = 946_684_800_000_000
_MODAL_CLI_ORCHESTRATION_RESERVE_SECONDS = 300
_CANONICAL_DECIMAL = re.compile(r"\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_CANARY_SUFFIXES = {harness: canary_run_suffix(harness) for harness in CANARY_ORDER}
_LOCAL_ENGINEERING_FREEZE_GATES = (
    "local_unit_tested",
    "local_offline_smoke_tested",
    "local_engineering_freeze_validated",
)
_PROCESS_PROBE_RESULTS = frozenset(
    {
        "different_boot_session",
        "same_boot_process_group_absent",
        "same_boot_process_group_exists",
        "same_boot_process_identity_changed",
    }
)

RECOVERY_INTENT_SCHEMA_NAME = "ModalActionRecoveryIntent"
RECOVERY_HOST_CONTAINMENT_SCHEMA_NAME = "ModalActionRecoveryHostContainment"
RECOVERY_RESOLUTION_SCHEMA_NAME = "ModalActionRecoveryResolution"
RECOVERY_SCHEMA_VERSION = "1.0"
RECOVERY_REQUEST_SCHEMA_NAME = "ModalActionRecoveryRequest"
RECOVERY_REQUEST_SCHEMA_VERSION = "1.0"
RECOVERY_BRANCHES = frozenset(
    {"definitely_not_started", "may_have_started_contained"}
)
RECOVERY_SNAPSHOT_NAMES = (
    "app_list",
    "container_list",
    "endpoint_list",
    "volume_list",
    "run_directory_list",
    "billing_report",
)
RECOVERY_SNAPSHOT_MANIFEST_SCHEMA_NAME = "ModalCleanupSnapshotCaptureManifest"
RECOVERY_SNAPSHOT_MANIFEST_SCHEMA_VERSION = "1.0"
RECOVERY_SNAPSHOT_MANIFEST_FILENAME = "capture_manifest.v1.0.json"

_RECOVERY_BINDING_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_RECOVERY_IDENTITY_FIELDS = frozenset(
    {"source_tree_sha256", "image_source_sha256", "cohort_id"}
)
_RECOVERY_SOURCE_EVIDENCE_FIELDS = frozenset(
    {
        "action_intent",
        "action_terminal",
        "global_rejection",
        "remote_run_reservations",
        "process_marker",
        "aggregate_receipts",
    }
)
_RECOVERY_REPAIR_FIELDS = frozenset(
    {
        "basis",
        "initial_reservation_bindings",
        "published_reservation_bindings",
        "final_reservation_bindings",
    }
)
_RECOVERY_INTENT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempt_id",
        "recorded_at_utc",
        "identity",
        "branch",
        "action",
        "run_id",
        "request_binding",
        "source_evidence",
        "reservation_repair",
        "snapshot_manifest",
        "fresh_candidate_attempt_id",
        "quarantined",
        "eligible_for_final_acceptance",
        "fresh_attempt_required",
    }
)
_RECOVERY_PROCESS_IDENTITY_FIELDS = frozenset(
    {
        "process_id",
        "process_group_id",
        "session_id",
        "process_birth_identity_sha256",
    }
)
_RECOVERY_HOST_CONTAINMENT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempt_id",
        "recorded_at_utc",
        "identity",
        "branch",
        "request_binding",
        "source_evidence",
        "recovery_intent",
        "containment_basis",
        "original_boot_started_at_unix_microseconds",
        "original_boot_session_sha256",
        "current_boot_started_at_unix_microseconds",
        "current_boot_session_sha256",
        "process_probe_result",
        "process_identity",
        "marker_binding",
        "terminal_binding",
        "quarantined",
        "eligible_for_final_acceptance",
        "fresh_attempt_required",
    }
)
_RECOVERY_SNAPSHOT_EVIDENCE_FIELDS = frozenset(
    {
        "manifest",
        "capture_id",
        "snapshots",
        "billing_window_start_utc",
        "billing_window_end_utc",
        "target_volume_name",
        "target_volume_present",
        "active_app_ids",
        "active_container_ids",
        "active_endpoint_ids",
    }
)
_RECOVERY_MODAL_EXPOSURE_FIELDS = frozenset(
    {
        "basis",
        "measured_app_name_main_billing_usd",
        "unresolved_compute_reserve_usd",
        "conservative_app_name_main_billing_usd",
        "complete_hourly_window",
        "local_authorization_is_platform_hard_bound",
        "modal_api_requests_performed",
        "snapshot_requests_performed",
        "billing_requests_performed",
        "price_requests_performed",
    }
)
_RECOVERY_PROVIDER_EXPOSURE_FIELDS = frozenset(
    {
        "applicable",
        "basis",
        "ledger_bindings",
        "provider_price_basis_binding",
        "attempt_count",
        "success_count",
        "error_count",
        "usage_known_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "exact_usage_cost_usd",
        "frozen_provider_approval_bound_usd",
        "conservative_provider_exposure_usd",
        "provider_requests_performed",
        "price_requests_performed",
    }
)
_RECOVERY_OBJECT_SET_FIELDS = frozenset({"coverage", "ids"})
_RECOVERY_KNOWN_OBJECT_FIELDS = frozenset(
    {"app_ids", "function_ids", "call_ids", "image_ids"}
)
_RECOVERY_RUN_DIRECTORY_FIELDS = frozenset(
    {"run_id", "present", "disposition"}
)
_RECOVERY_RESOLUTION_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempt_id",
        "recorded_at_utc",
        "identity",
        "branch",
        "request_binding",
        "source_evidence",
        "recovery_intent",
        "host_containment",
        "final_reservation_bindings",
        "snapshot_evidence",
        "modal_exposure",
        "provider_exposure",
        "known_remote_objects",
        "run_directory_dispositions",
        "fresh_candidate_attempt_id",
        "quarantined",
        "eligible_for_final_acceptance",
        "fresh_attempt_required",
        "validated",
    }
)
_RECOVERY_REQUEST_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "attempt_id",
        "fresh_candidate_attempt_id",
        "expected_branch",
        "snapshot_manifest_path",
        "initial_reservation_bindings",
    }
)
_RECOVERY_SNAPSHOT_MANIFEST_FIELDS = frozenset(
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
_RECOVERY_SNAPSHOT_RECORD_FIELDS = frozenset(
    {"path", "sha256", "size_bytes", "argv", "captured_at_utc"}
)
_RECOVERY_PRICE_BASIS_FIELDS = frozenset(
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
_RECOVERY_RAW_SNAPSHOT_FIELDS = {
    "app_list": frozenset(
        {"app_id", "description", "state", "tasks", "created_at", "stopped_at"}
    ),
    "container_list": frozenset(
        {"container_id", "app_id", "app_name", "start_time"}
    ),
    "endpoint_list": frozenset(
        {"name", "endpoint_id", "status", "created_at", "created_by"}
    ),
    "volume_list": frozenset({"name", "created_at", "created_by"}),
    "run_directory_list": frozenset(
        {"filename", "type", "created_modified", "size"}
    ),
    "billing_report": frozenset(
        {
            "object_id",
            "description",
            "environment",
            "interval_start",
            "resource",
            "cost",
        }
    ),
}
_RECOVERY_MODAL_TIMESTAMP = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} UTC\Z"
)
_RECOVERY_MODAL_SIZE = re.compile(
    r"\A(?:(?:0|[1-9][0-9]*) B|"
    r"(?:0|[1-9][0-9]*)\.[0-9] (?:KiB|MiB|GiB|TiB|PiB|EiB|ZiB))\Z"
)

_RESERVATION_BINDING_FIELDS = frozenset({"run_id", "path", "sha256"})
_RESERVATION_FIELDS = frozenset(
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
_GLOBAL_LAUNCH_REJECTION_SEAL_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "recorded_at_utc",
        "rejection_receipts",
        "validated",
    }
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
_LINEAGE_JOURNAL_FIELDS = frozenset(
    {"intent_receipts", "terminal_receipts", "aggregate_receipts"}
)
_LINEAGE_FILE_BINDING_FIELDS = frozenset({"path", "sha256", "size_bytes"})
_LINEAGE_PRIMARY_ACTIONS: dict[str, tuple[str, str | None]] = {
    "cuda_environment": ("cuda-environment", None),
    "offline_smoke": ("offline-smoke", None),
    "candidate_smoke": ("candidate-smoke", None),
    "resume_attempt": ("checkpoint-resume", None),
    **{f"canary_{harness}": ("canary", harness) for harness in CANARY_ORDER},
}
_LINEAGE_RUN_DISPOSITION_FIELDS = frozenset(
    {
        "attempt_id",
        "action",
        "status",
        "failure_kind",
        "run_id",
        "modal_cli_process_started",
        "remote_execution_state",
        "execution_disposition",
        "provider_disposition",
    }
)
_LINEAGE_REMOTE_OBJECT_FIELDS = frozenset(
    {"app_ids", "function_ids", "call_ids", "image_ids"}
)
_LINEAGE_REMOTE_EXECUTION_FIELDS = frozenset(
    {
        "attempt_id",
        "run_id",
        "action",
        "evidence_kind",
        "evidence",
        "execution_context",
    }
)
_LINEAGE_REMOTE_EXECUTION_EVIDENCE_KINDS = frozenset(
    {
        "downloaded_execution_context",
        "downloaded_execution_context_without_artifact_manifest",
        "remote_verification_receipt",
        "volume_success_capture",
        "volume_failure_capture",
    }
)
_LINEAGE_ARTIFACT_MANIFEST_FIELDS = frozenset(
    {
        "attempt_id",
        "run_id",
        "path",
        "sha256",
        "size_bytes",
        "canonical_manifest_sha256",
    }
)
_LINEAGE_PROVIDER_EVIDENCE_FIELDS = frozenset(
    {
        "attempt_id",
        "run_id",
        "harness",
        "binding_state",
        "ledger",
        "uncertainty",
        "parse_dispositions",
        "provider_attempt_count",
        "request_ids",
        "response_ids",
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
_ORDINARY_ACTION_FUNCTIONS = {
    "cuda-environment": "cuda_environment",
    "offline-smoke": "offline_smoke",
    "candidate-smoke": "candidate_smoke",
    "checkpoint-resume": "checkpoint_resume",
    OPENEVOLVE_60_ACTION: "openevolve_generic_60",
    EVOLUTION_ACTION: EVOLUTION_FUNCTION_NAME,
}
_PROVIDER_ACTIONS = frozenset(
    {
        "canary",
        "canaries",
        "exploratory_c0c3_pilot",
        OPENEVOLVE_60_ACTION,
        EVOLUTION_ACTION,
    }
)
_LINEAGE_PROVIDER_SPEND_FIELDS = frozenset(
    {
        "accounting_label",
        "provider_launcher_attempt_count",
        "provider_terminal_attempt_record_count",
        "provider_attempt_count_lower_bound",
        "provider_attempt_count_upper_bound",
        "successful_provider_attempt_count",
        "failed_provider_attempt_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "known_success_usage_estimate_usd",
        "failed_attempt_reserve_usd",
        "uncertain_request_start_reserve_usd",
        "conservative_provider_spend_bound_usd",
        "approved_provider_cap_total_usd",
        "provider_request_ids",
        "provider_response_ids",
        "run_cost_dispositions",
        "launcher_approval_bounds",
    }
)
_LINEAGE_MODAL_EXPOSURE_FIELDS = frozenset(
    {
        "accounting_label",
        "measured_app_billing_usd",
        "unresolved_compute_reserve_usd",
        "conservative_compute_exposure_usd",
        "measured_over_local_authorization_cap_usd",
        "local_authorization_cap_breach_attempt_ids",
        "local_authorization_is_platform_hard_bound",
        "attempts",
    }
)
_LINEAGE_RETAINED_STORAGE_FIELDS = frozenset(
    {"prior_cohort_estimates", "final_cohort_included", "basis"}
)
_INTENT_TERMINAL_SHARED_FIELDS = frozenset(
    _ACTION_INTENT_FIELDS
    - {"schema_name", "schema_version", "created_at_utc", "attempt_id"}
)


class ModalActionJournalIntegrityError(ValueError):
    """Raised when the durable journal is malformed, unstable, or ambiguous."""


class ModalActionJournalBlockedError(RuntimeError):
    """Raised when a stable journal contains unresolved action state."""


@dataclass(frozen=True, slots=True)
class ModalJournalFileBinding:
    """Immutable raw-byte binding for one discovered journal file."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModalJournalRecord:
    """One securely read JSON object and its raw-byte binding."""

    binding: ModalJournalFileBinding
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModalRecoveryJournalRecord:
    """One discovered recovery file awaiting the frozen v1.0 schema contract."""

    attempt_id: str
    kind: Literal["intent", "host_containment", "resolution"]
    record: ModalJournalRecord


@dataclass(frozen=True, slots=True)
class ModalRemoteRunReservationSpec:
    """Canonical payload and binding for one global remote-run reservation."""

    binding: Mapping[str, str]
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModalCohortActionJournal:
    """All global-gate journal material discovered below one cohort leaf."""

    identity: ModalLiveCohortIdentity
    migration_terminal_seal: ModalJournalRecord | None
    intents: tuple[ModalJournalRecord, ...]
    terminals: tuple[ModalJournalRecord, ...]
    aggregates: tuple[ModalJournalRecord, ...]
    recoveries: tuple[ModalRecoveryJournalRecord, ...]

    @property
    def sealed(self) -> bool:
        return self.migration_terminal_seal is not None


@dataclass(frozen=True, slots=True)
class ModalActionJournalBlocker:
    """A well-formed but unresolved state that prevents another launch."""

    code: str
    attempt_id: str
    identity: ModalLiveCohortIdentity | None
    paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModalAttemptJournalState:
    """Cross-namespace state for one globally unique attempt identifier."""

    attempt_id: str
    identity: ModalLiveCohortIdentity | None
    disposition: Literal["closed", "rejected", "unresolved"]
    intent: ModalJournalRecord | None
    terminal: ModalJournalRecord | None
    rejection: ModalJournalRecord | None
    reservations: tuple[ModalJournalRecord, ...]
    process_marker: ModalJournalRecord | None
    recoveries: tuple[ModalRecoveryJournalRecord, ...]
    process_probe_result: str | None
    blocker_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModalGlobalJournalScan:
    """Stable, lock-bound snapshot of every Modal action journal namespace."""

    project_root: Path
    cohorts: tuple[ModalCohortActionJournal, ...]
    attempts: tuple[ModalAttemptJournalState, ...]
    reservations: tuple[ModalJournalRecord, ...]
    rejections: tuple[ModalJournalRecord, ...]
    global_rejection_seal: ModalJournalRecord | None
    process_markers: tuple[ModalJournalRecord, ...]
    blockers: tuple[ModalActionJournalBlocker, ...]

    @property
    def launch_clear(self) -> bool:
        return not self.blockers


ProcessProbe = Callable[[Path, str, str], str]


@dataclass(slots=True)
class _DirectoryWitness:
    parts: tuple[str, ...]
    descriptor: int
    identity: tuple[int, ...]
    names: tuple[str, ...]


@dataclass(slots=True)
class _FileWitness:
    parent_parts: tuple[str, ...]
    filename: str
    descriptor: int
    identity: tuple[int, ...]


def _attempt_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _ATTEMPT_ID.fullmatch(value) is None:
        raise ModalActionJournalIntegrityError(f"{field} is invalid")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ModalActionJournalIntegrityError(f"{field} is invalid")
    return value


def _utc(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ModalActionJournalIntegrityError(f"{field} is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ModalActionJournalIntegrityError(f"{field} is invalid") from error
    if parsed.tzinfo is None:
        raise ModalActionJournalIntegrityError(f"{field} lacks a timezone")
    return parsed


def _canonical_utc(value: object, field: str) -> datetime:
    parsed = _utc(value, field)
    canonical = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if value != canonical:
        raise ModalActionJournalIntegrityError(f"{field} is not canonical UTC text")
    return parsed


def _recovery_identity(
    value: object,
    *,
    field: str,
) -> ModalLiveCohortIdentity:
    if not isinstance(value, dict) or set(value) != _RECOVERY_IDENTITY_FIELDS:
        raise ModalActionJournalIntegrityError(
            f"{field} has an invalid exact schema"
        )
    try:
        return ModalLiveCohortIdentity(
            source_tree_sha256=_sha256(
                value["source_tree_sha256"],
                f"{field}.source_tree_sha256",
            ),
            image_source_sha256=_sha256(
                value["image_source_sha256"],
                f"{field}.image_source_sha256",
            ),
            cohort_id=validate_run_id(value["cohort_id"]),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ModalActionJournalIntegrityError):
            raise
        raise ModalActionJournalIntegrityError(
            f"{field} contains an invalid cohort identity"
        ) from error


def _recovery_binding(
    value: object,
    *,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RECOVERY_BINDING_FIELDS:
        raise ModalActionJournalIntegrityError(
            f"{field} has an invalid exact binding schema"
        )
    try:
        logical = safe_relative_path(value["path"])
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            f"{field}.path is not a safe project-relative path"
        ) from error
    if logical.as_posix() != value["path"]:
        raise ModalActionJournalIntegrityError(f"{field}.path is not canonical")
    digest = _sha256(value["sha256"], f"{field}.sha256")
    size = value["size_bytes"]
    if type(size) is not int or size <= 0 or size > _MAX_JSON_BYTES:
        raise ModalActionJournalIntegrityError(f"{field}.size_bytes is invalid")
    return {"path": logical.as_posix(), "sha256": digest, "size_bytes": size}


def _recovery_optional_binding(
    value: object,
    *,
    field: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return _recovery_binding(value, field=field)


def _recovery_binding_roster(
    value: object,
    *,
    field: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ModalActionJournalIntegrityError(f"{field} must be a list")
    records = [
        _recovery_binding(item, field=f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    paths = [item["path"] for item in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ModalActionJournalIntegrityError(
            f"{field} must be unique and path-sorted"
        )
    return records


def _validate_recovery_flags(payload: Mapping[str, Any], *, field: str) -> None:
    if (
        payload["quarantined"] is not True
        or payload["eligible_for_final_acceptance"] is not False
        or payload["fresh_attempt_required"] is not True
    ):
        raise ModalActionJournalIntegrityError(
            f"{field} does not enforce permanent quarantine and a fresh attempt"
        )


def _validate_recovery_source_evidence(
    value: object,
    *,
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RECOVERY_SOURCE_EVIDENCE_FIELDS:
        raise ModalActionJournalIntegrityError(
            f"{field} has an invalid exact schema"
        )
    return {
        "action_intent": _recovery_optional_binding(
            value["action_intent"], field=f"{field}.action_intent"
        ),
        "action_terminal": _recovery_optional_binding(
            value["action_terminal"], field=f"{field}.action_terminal"
        ),
        "global_rejection": _recovery_optional_binding(
            value["global_rejection"], field=f"{field}.global_rejection"
        ),
        "remote_run_reservations": _recovery_binding_roster(
            value["remote_run_reservations"],
            field=f"{field}.remote_run_reservations",
        ),
        "process_marker": _recovery_optional_binding(
            value["process_marker"], field=f"{field}.process_marker"
        ),
        "aggregate_receipts": _recovery_binding_roster(
            value["aggregate_receipts"], field=f"{field}.aggregate_receipts"
        ),
    }


def _validate_recovery_repair(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _RECOVERY_REPAIR_FIELDS:
        raise ModalActionJournalIntegrityError(
            f"{field} has an invalid exact schema"
        )
    basis = value["basis"]
    if basis not in {
        "not_applicable",
        "global_rejection_planned_roster",
        "canonical_reservation_inference",
    }:
        raise ModalActionJournalIntegrityError(f"{field}.basis is invalid")
    return {
        "basis": basis,
        "initial_reservation_bindings": _recovery_binding_roster(
            value["initial_reservation_bindings"],
            field=f"{field}.initial_reservation_bindings",
        ),
        "published_reservation_bindings": _recovery_binding_roster(
            value["published_reservation_bindings"],
            field=f"{field}.published_reservation_bindings",
        ),
        "final_reservation_bindings": _recovery_binding_roster(
            value["final_reservation_bindings"],
            field=f"{field}.final_reservation_bindings",
        ),
    }


def _validate_recovery_intent_core(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    identity: ModalLiveCohortIdentity,
) -> dict[str, Any]:
    field = "recovery_intent"
    if set(payload) != _RECOVERY_INTENT_FIELDS:
        raise ModalActionJournalIntegrityError(
            "Modal recovery intent has an invalid exact schema"
        )
    if (
        payload["schema_name"] != RECOVERY_INTENT_SCHEMA_NAME
        or payload["schema_version"] != RECOVERY_SCHEMA_VERSION
        or payload["attempt_id"] != attempt_id
        or payload["branch"] not in RECOVERY_BRANCHES
        or payload["action"] not in MODAL_ACTIONS
    ):
        raise ModalActionJournalIntegrityError(
            "Modal recovery intent has the wrong contract"
        )
    _attempt_id(attempt_id, f"{field}.attempt_id")
    _canonical_utc(payload["recorded_at_utc"], f"{field}.recorded_at_utc")
    if _recovery_identity(payload["identity"], field=f"{field}.identity") != identity:
        raise ModalActionJournalIntegrityError(
            "Modal recovery intent differs from its cohort path"
        )
    try:
        validate_run_id(payload["run_id"])
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            f"{field}.run_id is invalid"
        ) from error
    _recovery_binding(payload["request_binding"], field=f"{field}.request_binding")
    _validate_recovery_source_evidence(
        payload["source_evidence"], field=f"{field}.source_evidence"
    )
    _validate_recovery_repair(
        payload["reservation_repair"], field=f"{field}.reservation_repair"
    )
    _recovery_optional_binding(
        payload["snapshot_manifest"], field=f"{field}.snapshot_manifest"
    )
    candidate = _attempt_id(
        payload["fresh_candidate_attempt_id"],
        f"{field}.fresh_candidate_attempt_id",
    )
    if candidate == attempt_id:
        raise ModalActionJournalIntegrityError(
            "Modal recovery fresh candidate reuses the orphan attempt ID"
        )
    _validate_recovery_flags(payload, field=field)
    return dict(payload)


def _validate_recovery_host_core(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    identity: ModalLiveCohortIdentity,
) -> dict[str, Any]:
    field = "recovery_host_containment"
    if set(payload) != _RECOVERY_HOST_CONTAINMENT_FIELDS:
        raise ModalActionJournalIntegrityError(
            "Modal recovery host containment has an invalid exact schema"
        )
    if (
        payload["schema_name"] != RECOVERY_HOST_CONTAINMENT_SCHEMA_NAME
        or payload["schema_version"] != RECOVERY_SCHEMA_VERSION
        or payload["attempt_id"] != attempt_id
        or payload["branch"] not in RECOVERY_BRANCHES
    ):
        raise ModalActionJournalIntegrityError(
            "Modal recovery host containment has the wrong contract"
        )
    _canonical_utc(payload["recorded_at_utc"], f"{field}.recorded_at_utc")
    if _recovery_identity(payload["identity"], field=f"{field}.identity") != identity:
        raise ModalActionJournalIntegrityError(
            "Modal recovery host containment differs from its cohort path"
        )
    for name in ("request_binding", "recovery_intent"):
        _recovery_binding(payload[name], field=f"{field}.{name}")
    _validate_recovery_source_evidence(
        payload["source_evidence"], field=f"{field}.source_evidence"
    )
    for name in (
        "original_boot_started_at_unix_microseconds",
        "current_boot_started_at_unix_microseconds",
    ):
        value = payload[name]
        if value is not None and (
            type(value) is not int
            or value < _MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS
        ):
            raise ModalActionJournalIntegrityError(f"{field}.{name} is invalid")
    for name in ("original_boot_session_sha256", "current_boot_session_sha256"):
        value = payload[name]
        if value is not None:
            _sha256(value, f"{field}.{name}")
    if payload["process_probe_result"] not in {
        "not_applicable",
        "different_boot_session",
        "same_boot_process_group_absent",
    }:
        raise ModalActionJournalIntegrityError(
            f"{field}.process_probe_result is invalid"
        )
    process_identity = payload["process_identity"]
    if process_identity is not None:
        if (
            not isinstance(process_identity, dict)
            or set(process_identity) != _RECOVERY_PROCESS_IDENTITY_FIELDS
        ):
            raise ModalActionJournalIntegrityError(
                f"{field}.process_identity has an invalid exact schema"
            )
        process_id = process_identity["process_id"]
        if (
            type(process_id) is not int
            or process_id <= 0
            or process_identity["process_group_id"] != process_id
            or process_identity["session_id"] != process_id
        ):
            raise ModalActionJournalIntegrityError(
                f"{field}.process_identity is inconsistent"
            )
        _sha256(
            process_identity["process_birth_identity_sha256"],
            f"{field}.process_identity.process_birth_identity_sha256",
        )
    _recovery_optional_binding(
        payload["marker_binding"], field=f"{field}.marker_binding"
    )
    _recovery_optional_binding(
        payload["terminal_binding"], field=f"{field}.terminal_binding"
    )
    _validate_recovery_flags(payload, field=field)
    return dict(payload)


def _validate_recovery_resolution_core(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    identity: ModalLiveCohortIdentity,
) -> dict[str, Any]:
    field = "recovery_resolution"
    if set(payload) != _RECOVERY_RESOLUTION_FIELDS:
        raise ModalActionJournalIntegrityError(
            "Modal recovery resolution has an invalid exact schema"
        )
    if (
        payload["schema_name"] != RECOVERY_RESOLUTION_SCHEMA_NAME
        or payload["schema_version"] != RECOVERY_SCHEMA_VERSION
        or payload["attempt_id"] != attempt_id
        or payload["branch"] not in RECOVERY_BRANCHES
        or payload["validated"] is not True
    ):
        raise ModalActionJournalIntegrityError(
            "Modal recovery resolution has the wrong contract"
        )
    _canonical_utc(payload["recorded_at_utc"], f"{field}.recorded_at_utc")
    if _recovery_identity(payload["identity"], field=f"{field}.identity") != identity:
        raise ModalActionJournalIntegrityError(
            "Modal recovery resolution differs from its cohort path"
        )
    for name in ("request_binding", "recovery_intent", "host_containment"):
        _recovery_binding(payload[name], field=f"{field}.{name}")
    _validate_recovery_source_evidence(
        payload["source_evidence"], field=f"{field}.source_evidence"
    )
    _recovery_binding_roster(
        payload["final_reservation_bindings"],
        field=f"{field}.final_reservation_bindings",
    )
    candidate = _attempt_id(
        payload["fresh_candidate_attempt_id"],
        f"{field}.fresh_candidate_attempt_id",
    )
    if candidate == attempt_id:
        raise ModalActionJournalIntegrityError(
            "Modal recovery resolution reuses the orphan attempt ID"
        )
    _validate_recovery_flags(payload, field=field)
    return dict(payload)


def _exact_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value <= 1:
        raise ModalActionJournalIntegrityError(
            f"{field} must be an exact positive integer"
        )
    return value


def _identity_from_payload(
    payload: Mapping[str, Any],
    *,
    image_field: str,
    allow_absent: bool,
    allow_partial: bool = False,
    field: str,
) -> ModalLiveCohortIdentity | None:
    values = (
        payload.get("source_tree_sha256"),
        payload.get(image_field),
        payload.get("cohort_id"),
    )
    if all(value is None for value in values):
        if allow_absent:
            return None
        raise ModalActionJournalIntegrityError(f"{field} identity is absent")
    if any(value is None for value in values):
        if allow_partial:
            if values[0] is not None:
                _sha256(values[0], f"{field}.source_tree_sha256")
            if values[1] is not None:
                _sha256(values[1], f"{field}.{image_field}")
            if values[2] is not None:
                try:
                    validate_run_id(values[2])
                except (TypeError, ValueError) as error:
                    raise ModalActionJournalIntegrityError(
                        f"{field}.cohort_id is invalid"
                    ) from error
            return None
        raise ModalActionJournalIntegrityError(f"{field} identity is partial")
    try:
        return ModalLiveCohortIdentity(
            source_tree_sha256=_sha256(values[0], f"{field}.source_tree_sha256"),
            image_source_sha256=_sha256(values[1], f"{field}.{image_field}"),
            cohort_id=validate_run_id(values[2]),
        )
    except (TypeError, ValueError) as error:
        if isinstance(error, ModalActionJournalIntegrityError):
            raise
        raise ModalActionJournalIntegrityError(
            f"{field} identity is invalid"
        ) from error


def _validate_containment(
    payload: Mapping[str, Any],
    *,
    field: str,
    allow_absent: bool,
) -> bool:
    names = (
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
    )
    values = tuple(payload.get(name) for name in names)
    if all(value is None for value in values):
        if allow_absent:
            return False
        raise ModalActionJournalIntegrityError(f"{field} is absent")
    if any(value is None for value in values):
        raise ModalActionJournalIntegrityError(f"{field} is partial")
    if values[0] != modal_local_host_anchor_path().as_posix():
        raise ModalActionJournalIntegrityError(
            f"{field} host-anchor path is not canonical"
        )
    _sha256(values[1], f"{field}.local_host_anchor_sha256")
    if type(values[2]) is not int or values[2] < _MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS:
        raise ModalActionJournalIntegrityError(
            f"{field} boot-start identity is invalid"
        )
    _sha256(values[3], f"{field}.local_boot_session_sha256")
    return True


def expected_modal_concrete_run_ids(
    *,
    action: str,
    run_id: str,
    verifier_run_id: str | None,
) -> tuple[str, ...]:
    """Return the canonical concrete reservation roster for one action."""

    selected = validate_run_id(run_id)
    if action not in MODAL_ACTIONS:
        raise ValueError("Modal action is unsupported")
    if action in {"download", "verify"}:
        if verifier_run_id is None:
            raise ValueError("verifier action lacks its destination run ID")
        return (validate_run_id(verifier_run_id),)
    if action == "canaries":
        return tuple(
            validate_run_id(f"{selected}-{_CANARY_SUFFIXES[harness]}")
            for harness in CANARY_ORDER
        )
    return (selected,)


def _expected_outer_cli_timeout_seconds(
    action: str, harness: str | None = None
) -> int:
    if action not in MODAL_ACTIONS:
        raise ModalActionJournalIntegrityError("Modal action is unsupported")
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
    return (
        IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS
        + runtime_seconds
        + _MODAL_CLI_ORCHESTRATION_RESERVE_SECONDS
    )


def _expected_modal_resource_profile(
    action: str,
    harness: str | None,
) -> dict[str, Any]:
    dynamic_timeout: int | None = None
    if action == EVOLUTION_ACTION:
        evolution = EvolutionRunSpec.parse(harness)
        function_names = [EVOLUTION_FUNCTION_NAME]
        dynamic_timeout = evolution.function_timeout_seconds
    elif action == "canaries":
        function_names = [f"canary_{item}" for item in CANARY_ORDER]
    elif action == "canary":
        if harness not in CANARY_ORDER:
            raise ModalActionJournalIntegrityError(
                "single-canary resource profile lacks its harness"
            )
        function_names = [f"canary_{harness}"]
    elif action in {"download", "verify"}:
        function_names = ["artifact_verify"]
    else:
        function_names = [action.replace("-", "_")]
    runtime_calls: list[dict[str, Any]] = []
    try:
        specs = [function_spec(name) for name in function_names]
    except KeyError as error:  # pragma: no cover - frozen action/spec parity
        raise ModalActionJournalIntegrityError(
            "Modal resource profile function is unsupported"
        ) from error
    for function_name, spec in zip(function_names, specs, strict=True):
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
                    "provider_egress_enabled" if spec.provider_secret else "blocked"
                ),
            }
        )
    return {
        "modal_environment": MODAL_ENVIRONMENT_NAME,
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


def _positive_decimal_text(value: object, field: str) -> Decimal:
    if not isinstance(value, str) or _CANONICAL_DECIMAL.fullmatch(value) is None:
        raise ModalActionJournalIntegrityError(f"{field} is not canonical decimal text")
    try:
        selected = Decimal(value)
    except InvalidOperation as error:  # pragma: no cover - regex excludes this
        raise ModalActionJournalIntegrityError(f"{field} is invalid") from error
    if not selected.is_finite() or selected <= 0:
        raise ModalActionJournalIntegrityError(f"{field} must be positive")
    return selected


def _nonnegative_decimal_text(value: object, field: str) -> Decimal:
    if not isinstance(value, str) or _CANONICAL_DECIMAL.fullmatch(value) is None:
        raise ModalActionJournalIntegrityError(f"{field} is not canonical decimal text")
    try:
        selected = Decimal(value)
    except InvalidOperation as error:  # pragma: no cover - regex excludes this
        raise ModalActionJournalIntegrityError(f"{field} is invalid") from error
    if not selected.is_finite() or selected < 0:
        raise ModalActionJournalIntegrityError(f"{field} must be nonnegative")
    return selected


def _expected_predecessor_gate_roster(
    action: str,
    bindings: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    action_gates = {
        "cuda-environment": (),
        "offline-smoke": ("modal_cuda_environment_validated",),
        "candidate-smoke": (
            "modal_cuda_environment_validated",
            "modal_offline_smoke_validated",
        ),
        "checkpoint-resume": ("modal_artifact_round_trip_validated",),
        "canary": ("candidate_resume_preflight_validated",),
        "canaries": ("candidate_resume_preflight_validated",),
        "exploratory_c0c3_pilot": ("candidate_resume_preflight_validated",),
        OPENEVOLVE_60_ACTION: ("candidate_resume_preflight_validated",),
        EVOLUTION_ACTION: ("candidate_resume_preflight_validated",),
    }
    if action in {"download", "verify"}:
        source_gates = (
            "source_action_intent",
            "source_action_attempt_terminal",
            "source_local_process_start",
        )
        observed_action = tuple(item["gate"] for item in bindings[3:])
        if observed_action == (*source_gates, "provider_canary_aggregate_outcomes"):
            selected = observed_action
        else:
            selected = source_gates
    else:
        selected = action_gates[action]
    return (*_LOCAL_ENGINEERING_FREEZE_GATES, *selected)


def _validate_approved_action_contract(
    payload: Mapping[str, Any],
    *,
    action: str,
    harness: str | None,
    field: str,
) -> None:
    if payload["outer_cli_timeout_seconds"] != _expected_outer_cli_timeout_seconds(
        action, harness
    ):
        raise ModalActionJournalIntegrityError(
            f"{field} timeout differs from its action"
        )
    if payload["modal_resource_profile"] != _expected_modal_resource_profile(
        action,
        harness,
    ):
        raise ModalActionJournalIntegrityError(
            f"{field} resource profile differs from its action"
        )
    modal_cap = _positive_decimal_text(
        payload["modal_cost_cap_usd"],
        f"{field}.modal_cost_cap_usd",
    )
    estimate = payload["modal_cost_estimate"]
    if not isinstance(estimate, dict) or "action_estimate_usd" not in estimate:
        raise ModalActionJournalIntegrityError(
            f"{field} Modal cost estimate lacks its action total"
        )
    action_estimate = _positive_decimal_text(
        estimate["action_estimate_usd"],
        f"{field}.modal_cost_estimate.action_estimate_usd",
    )
    if modal_cap < action_estimate or payload["modal_cost_approved"] is not True:
        raise ModalActionJournalIntegrityError(
            f"{field} lacks sufficient Modal cost approval"
        )
    provider_action = action in _PROVIDER_ACTIONS
    if payload["provider_cost_approved"] is not provider_action:
        raise ModalActionJournalIntegrityError(
            f"{field} provider approval differs from its action"
        )
    provider_fields = (
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
    )
    if provider_action:
        if any(payload[name] is None for name in provider_fields):
            raise ModalActionJournalIntegrityError(
                f"{field} provider approval core is incomplete"
            )
        _positive_decimal_text(
            payload["provider_cost_cap_usd"],
            f"{field}.provider_cost_cap_usd",
        )
    elif any(payload[name] is not None for name in provider_fields):
        raise ModalActionJournalIntegrityError(
            f"{field} non-provider action contains provider approval fields"
        )
    bindings = payload["predecessor_receipts"]
    expected_gates = _expected_predecessor_gate_roster(action, bindings)
    if tuple(record["gate"] for record in bindings) != expected_gates:
        raise ModalActionJournalIntegrityError(
            f"{field} predecessor gate roster differs from its action"
        )


def _exclusive_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def build_modal_remote_run_reservation_specs(
    *,
    concrete_remote_run_ids: Sequence[str],
    attempt_id: str,
    action: str,
    identity: ModalLiveCohortIdentity,
    created_at_utc: str,
    launch_capability_sha256: str,
    local_host_anchor_path: str,
    local_host_anchor_sha256: str,
    local_boot_started_at_unix_microseconds: int,
    local_boot_session_sha256: str,
) -> tuple[ModalRemoteRunReservationSpec, ...]:
    """Build the one canonical reservation payload for every concrete run."""

    owner = _attempt_id(attempt_id, "reservation.owner_attempt_id")
    if action not in MODAL_ACTIONS:
        raise ModalActionJournalIntegrityError("reservation action is invalid")
    _utc(created_at_utc, "reservation.created_at_utc")
    capability = _sha256(
        launch_capability_sha256,
        "reservation.launch_capability_sha256",
    )
    containment = {
        "local_host_anchor_path": local_host_anchor_path,
        "local_host_anchor_sha256": local_host_anchor_sha256,
        "local_boot_started_at_unix_microseconds": (
            local_boot_started_at_unix_microseconds
        ),
        "local_boot_session_sha256": local_boot_session_sha256,
    }
    _validate_containment(
        containment,
        field="reservation.local_containment",
        allow_absent=False,
    )
    specs: list[ModalRemoteRunReservationSpec] = []
    observed: set[str] = set()
    for raw_run_id in concrete_remote_run_ids:
        run_id = validate_run_id(raw_run_id)
        if run_id in observed:
            raise ModalActionJournalIntegrityError(
                "reservation roster contains duplicate run IDs"
            )
        observed.add(run_id)
        path = modal_remote_run_reservation_path(run_id).as_posix()
        payload = {
            "schema_name": "ModalRemoteRunReservation",
            "schema_version": "1.2",
            "remote_run_id": run_id,
            "owner_attempt_id": owner,
            "action": action,
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "modal_environment": MODAL_ENVIRONMENT_NAME,
            "created_at_utc": created_at_utc,
            "launch_capability_sha256": capability,
            **containment,
        }
        binding = {
            "run_id": run_id,
            "path": path,
            "sha256": hashlib.sha256(_exclusive_json_bytes(payload)).hexdigest(),
        }
        specs.append(ModalRemoteRunReservationSpec(binding=binding, payload=payload))
    return tuple(specs)


def _validate_binding_roster(
    value: object,
    *,
    expected_run_ids: Sequence[str],
    field: str,
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or len(value) != len(expected_run_ids):
        raise ModalActionJournalIntegrityError(f"{field} is not exact")
    records: list[dict[str, str]] = []
    for index, (raw, run_id) in enumerate(zip(value, expected_run_ids, strict=True)):
        if not isinstance(raw, dict) or set(raw) != _RESERVATION_BINDING_FIELDS:
            raise ModalActionJournalIntegrityError(
                f"{field}[{index}] has an invalid exact schema"
            )
        expected_path = modal_remote_run_reservation_path(run_id).as_posix()
        if raw["run_id"] != run_id or raw["path"] != expected_path:
            raise ModalActionJournalIntegrityError(f"{field}[{index}] is not canonical")
        digest = _sha256(raw["sha256"], f"{field}[{index}].sha256")
        records.append({"run_id": run_id, "path": expected_path, "sha256": digest})
    return tuple(records)


def _validate_predecessor_roster(value: object, field: str) -> None:
    if not isinstance(value, list):
        raise ModalActionJournalIntegrityError(f"{field} must be a list")
    observed: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or set(raw) != {"gate", "path", "sha256"}:
            raise ModalActionJournalIntegrityError(
                f"{field}[{index}] has an invalid exact schema"
            )
        gate = raw["gate"]
        if not isinstance(gate, str) or not gate or not gate.replace("_", "").isalnum():
            raise ModalActionJournalIntegrityError(f"{field}[{index}].gate is invalid")
        try:
            path = safe_relative_path(raw["path"]).as_posix()
        except (TypeError, ValueError) as error:
            raise ModalActionJournalIntegrityError(
                f"{field}[{index}].path is invalid"
            ) from error
        _sha256(raw["sha256"], f"{field}[{index}].sha256")
        key = (gate, path)
        if key in observed:
            raise ModalActionJournalIntegrityError(f"{field} contains a duplicate")
        observed.add(key)


def _validated_action_core(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str, str, str | None, str | None, str | None]:
    try:
        return validate_modal_action_identity(
            action=payload["action"],
            run_id=payload["run_id"],
            source_run_id=payload["source_run_id"],
            verifier_run_id=payload["verifier_run_id"],
            harness=payload["harness"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            f"{field} action identity is invalid"
        ) from error


def _validated_partial_action_core(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Validate the sanitized identity retained by a pre-ownership rejection."""

    action = payload["action"]
    if action is not None and action not in MODAL_ACTIONS:
        raise ModalActionJournalIntegrityError(f"{field} action is invalid")
    selected: list[str | None] = []
    for name in ("run_id", "source_run_id", "verifier_run_id"):
        value = payload[name]
        if value is not None:
            try:
                value = validate_run_id(value)
            except (TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    f"{field}.{name} is invalid"
                ) from error
        selected.append(value)
    harness = payload["harness"]
    if harness is not None and harness not in CANARY_ORDER:
        raise ModalActionJournalIntegrityError(f"{field}.harness is invalid")
    return action, selected[0], selected[1], selected[2], harness


def _validate_action_intent_core(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    identity: ModalLiveCohortIdentity,
) -> dict[str, Any]:
    if set(payload) != _ACTION_INTENT_FIELDS:
        raise ModalActionJournalIntegrityError(
            "Modal action intent has an invalid exact schema"
        )
    if (
        payload["schema_name"] != "ModalActionIntent"
        or payload["schema_version"] != "1.6"
        or payload["attempt_id"] != attempt_id
    ):
        raise ModalActionJournalIntegrityError(
            "Modal action intent has the wrong contract"
        )
    _attempt_id(attempt_id, "intent.attempt_id")
    _utc(payload["created_at_utc"], "intent.created_at_utc")
    selected_identity = _identity_from_payload(
        payload,
        image_field="approved_image_source_sha256",
        allow_absent=False,
        field="intent",
    )
    if selected_identity != identity:
        raise ModalActionJournalIntegrityError(
            "Modal action intent differs from its cohort path"
        )
    action, run_id, _source, verifier, harness = _validated_action_core(
        payload,
        field="intent",
    )
    if action == "cuda-environment" and run_id != identity.cohort_id:
        raise ModalActionJournalIntegrityError(
            "first CUDA action run ID differs from its cohort"
        )
    expected_runs = expected_modal_concrete_run_ids(
        action=action,
        run_id=run_id,
        verifier_run_id=verifier,
    )
    concrete = payload["concrete_remote_run_ids"]
    if not isinstance(concrete, list) or tuple(concrete) != expected_runs:
        raise ModalActionJournalIntegrityError(
            "Modal action intent concrete run roster is invalid"
        )
    _validate_containment(
        payload,
        field="intent.local_containment",
        allow_absent=False,
    )
    _validate_binding_roster(
        payload["remote_run_reservations"],
        expected_run_ids=expected_runs,
        field="intent.remote_run_reservations",
    )
    for name in (
        "modal_command_sha256",
        "launch_capability_sha256",
        "modal_price_basis_sha256",
    ):
        _sha256(payload[name], f"intent.{name}")
    if payload["modal_profile"] != "scalingintelligence":
        raise ModalActionJournalIntegrityError("intent Modal profile is invalid")
    if payload["modal_environment"] != MODAL_ENVIRONMENT_NAME:
        raise ModalActionJournalIntegrityError("intent Modal environment is invalid")
    if (
        type(payload["outer_cli_timeout_seconds"]) is not int
        or payload["outer_cli_timeout_seconds"] <= 0
        or not isinstance(payload["modal_resource_profile"], dict)
        or not isinstance(payload["modal_cost_estimate"], dict)
        or not isinstance(payload["modal_cost_cap_usd"], str)
        or payload["modal_cost_approved"] is not True
    ):
        raise ModalActionJournalIntegrityError("intent Modal approval core is invalid")
    try:
        safe_relative_path(payload["modal_price_basis_path"])
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "intent Modal price-basis path is invalid"
        ) from error
    provider_action = action in _PROVIDER_ACTIONS
    provider_fields = (
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
    )
    if type(payload["provider_cost_approved"]) is not bool or (
        payload["provider_cost_approved"] is not provider_action
    ):
        raise ModalActionJournalIntegrityError(
            "intent provider approval differs from its action"
        )
    if provider_action:
        if any(payload[name] is None for name in provider_fields):
            raise ModalActionJournalIntegrityError(
                "provider intent approval core is incomplete"
            )
        for name in ("approval_plan_sha256", "provider_price_basis_sha256"):
            _sha256(payload[name], f"intent.{name}")
        for name in ("provider_approval_plan_path", "provider_price_basis_path"):
            try:
                safe_relative_path(payload[name])
            except (TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    f"intent.{name} is invalid"
                ) from error
    elif any(payload[name] is not None for name in provider_fields):
        raise ModalActionJournalIntegrityError(
            "non-provider intent contains provider approval fields"
        )
    if type(payload["source_evidence_recovery"]) is not bool or (
        payload["source_evidence_recovery"] and action not in {"download", "verify"}
    ):
        raise ModalActionJournalIntegrityError(
            "intent source-evidence recovery flag is invalid"
        )
    _validate_predecessor_roster(
        payload["predecessor_receipts"],
        "intent.predecessor_receipts",
    )
    _validate_approved_action_contract(
        payload,
        action=action,
        harness=harness,
        field="intent",
    )
    return dict(payload)


def _validate_terminal_status(payload: Mapping[str, Any]) -> None:
    status = payload["status"]
    failure = payload["failure_kind"]
    started = payload["modal_cli_process_started"]
    returncode = payload["returncode"]
    closed = payload["process_group_closed"]
    if status == "succeeded":
        valid = (
            failure is None and started is True and returncode == 0 and closed is True
        )
    elif status == "failed":
        valid = (
            failure == "modal_cli_exit"
            and started is True
            and type(returncode) is int
            and returncode != 0
            and closed is True
        )
    elif status == "timed_out":
        valid = (
            failure == "outer_cli_timeout"
            and started is True
            and returncode is None
            and closed is True
        )
    elif status == "preflight_failed":
        valid = (
            failure
            in {
                "preflight",
                "action_intent_persistence",
                "action_intent_post_persistence",
                "action_intent_persistence_uncertain",
            }
            and started is False
            and returncode is None
            and closed is None
        )
    elif status == "preflight_rejected":
        valid = (
            failure == "preflight"
            and started is False
            and returncode is None
            and closed is None
        )
    elif status == "lock_contended":
        valid = (
            failure == "local_launcher_lock"
            and started is False
            and returncode is None
            and closed is None
        )
    elif status == "interrupted":
        valid = (
            failure == "interrupt"
            and returncode is None
            and (
                (started is True and closed is True)
                or (started is False and closed is None)
            )
        )
    elif status == "cli_failed":
        valid = returncode is None and (
            (started is False and closed is None and failure == "process_launch")
            or (
                started is True
                and closed is True
                and failure
                in {
                    "modal_cli",
                    "process_launch",
                    "process_start_receipt_persistence",
                }
            )
        )
    elif status == "cleanup_failed":
        open_group_failures = {
            "process_group_cleanup",
            "process_group_and_python_execution_cleanup",
            "process_start_receipt_and_process_group_cleanup",
            "process_start_receipt_process_group_and_python_execution_cleanup",
        }
        closed_group_failures = {"process_start_receipt_and_python_execution_cleanup"}
        valid = returncode is None and (
            (started is True and closed is False and failure in open_group_failures)
            or (started is True and closed is True and failure in closed_group_failures)
            or (failure == "python_execution_cleanup" and closed in {None, True})
        )
    else:
        valid = False
    if not valid:
        raise ModalActionJournalIntegrityError(
            "Modal action terminal status fields do not reconcile"
        )


def _validate_action_terminal_core(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    location: Literal["cohort", "global_rejection"],
    path_identity: ModalLiveCohortIdentity | None,
) -> dict[str, Any]:
    if set(payload) != _ACTION_ATTEMPT_FIELDS:
        raise ModalActionJournalIntegrityError(
            "Modal action terminal has an invalid exact schema"
        )
    if (
        payload["schema_name"] != "ModalActionAttemptReceipt"
        or payload["schema_version"] != "3.6"
        or payload["attempt_id"] != attempt_id
    ):
        raise ModalActionJournalIntegrityError(
            "Modal action terminal has the wrong contract"
        )
    _attempt_id(attempt_id, "terminal.attempt_id")
    started_at = _utc(payload["started_at_utc"], "terminal.started_at_utc")
    finished_at = _utc(payload["finished_at_utc"], "terminal.finished_at_utc")
    if finished_at < started_at:
        raise ModalActionJournalIntegrityError(
            "Modal action terminal finished before it started"
        )
    if not isinstance(payload["status"], str) or not isinstance(
        payload["failure_kind"], (str, type(None))
    ):
        raise ModalActionJournalIntegrityError(
            "Modal action terminal status identity is invalid"
        )
    reservations = payload["remote_run_reservations"]
    if not isinstance(reservations, list):
        raise ModalActionJournalIntegrityError(
            "terminal reservation roster must be a list"
        )
    preownership_rejection = location == "global_rejection" and not reservations
    identity = _identity_from_payload(
        payload,
        image_field="approved_image_source_sha256",
        allow_absent=True,
        allow_partial=preownership_rejection,
        field="terminal",
    )
    if location == "cohort":
        if identity is None or identity != path_identity:
            raise ModalActionJournalIntegrityError(
                "cohort terminal differs from its cohort path"
            )
        if payload["failure_kind"] in {
            "action_intent_persistence",
            "action_intent_persistence_uncertain",
        }:
            raise ModalActionJournalIntegrityError(
                "intent-persistence failure is global-rejection-only"
            )
    elif payload["failure_kind"] == "action_intent_post_persistence":
        raise ModalActionJournalIntegrityError(
            "post-intent-persistence failure is cohort-only"
        )

    if preownership_rejection:
        action, run_id, _source, verifier, _harness = _validated_partial_action_core(
            payload, field="terminal"
        )
        if action in {"download", "verify"} and verifier is not None:
            expected_runs = (verifier,)
        elif action == "canaries" and run_id is not None:
            expected_runs = expected_modal_concrete_run_ids(
                action=action,
                run_id=run_id,
                verifier_run_id=None,
            )
        elif action is not None and run_id is not None:
            expected_runs = (run_id,)
        else:
            expected_runs = ()
    else:
        action = payload["action"]
        run_id = payload["run_id"]
        if (action is None) is not (run_id is None):
            raise ModalActionJournalIntegrityError(
                "unbound terminal contains a partial action identity"
            )
        if action is None:
            if any(
                payload[name] is not None
                for name in ("source_run_id", "verifier_run_id", "harness")
            ):
                raise ModalActionJournalIntegrityError(
                    "unbound terminal contains partial action identity"
                )
            expected_runs = ()
        else:
            selected_action, selected_run, _source, verifier, _harness = (
                _validated_action_core(payload, field="terminal")
            )
            expected_runs = expected_modal_concrete_run_ids(
                action=selected_action,
                run_id=selected_run,
                verifier_run_id=verifier,
            )
            if (
                selected_action == "cuda-environment"
                and identity is not None
                and selected_run != identity.cohort_id
            ):
                raise ModalActionJournalIntegrityError(
                    "first CUDA terminal run ID differs from its cohort"
                )
    concrete = payload["concrete_remote_run_ids"]
    if not isinstance(concrete, list) or tuple(concrete) != expected_runs:
        raise ModalActionJournalIntegrityError(
            "Modal action terminal concrete run roster is invalid"
        )

    containment_present = _validate_containment(
        payload,
        field="terminal.local_containment",
        allow_absent=location == "global_rejection",
    )
    if identity is None and containment_present and not preownership_rejection:
        raise ModalActionJournalIntegrityError(
            "unbound terminal contains local containment fields"
        )
    if reservations:
        if (
            identity is None
            or action is None
            or not containment_present
            or payload["launch_capability_sha256"] is None
        ):
            raise ModalActionJournalIntegrityError(
                "terminal reservation roster lacks its ownership core"
            )
        _validate_binding_roster(
            reservations,
            expected_run_ids=expected_runs,
            field="terminal.remote_run_reservations",
        )

    for name in (
        "approved_image_source_sha256",
        "modal_command_sha256",
        "launch_capability_sha256",
        "modal_price_basis_sha256",
        "approval_plan_sha256",
        "provider_price_basis_sha256",
        "local_process_start_receipt_sha256",
    ):
        if payload[name] is not None:
            _sha256(payload[name], f"terminal.{name}")
    if payload["modal_profile"] != "scalingintelligence":
        raise ModalActionJournalIntegrityError("terminal Modal profile is invalid")
    if payload["modal_environment"] != MODAL_ENVIRONMENT_NAME:
        raise ModalActionJournalIntegrityError("terminal Modal environment is invalid")
    for name in ("modal_cost_approved", "provider_cost_approved"):
        if type(payload[name]) is not bool:
            raise ModalActionJournalIntegrityError(f"terminal.{name} must be boolean")
    if type(payload["source_evidence_recovery"]) is not bool or (
        payload["source_evidence_recovery"] and action not in {"download", "verify"}
    ):
        raise ModalActionJournalIntegrityError(
            "terminal source-evidence recovery flag is invalid"
        )
    _validate_predecessor_roster(
        payload["predecessor_receipts"],
        "terminal.predecessor_receipts",
    )
    for name in (
        "modal_price_basis_path",
        "provider_approval_plan_path",
        "provider_price_basis_path",
    ):
        if payload[name] is not None:
            try:
                safe_relative_path(payload[name])
            except (TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    f"terminal.{name} is invalid"
                ) from error
    for name in (
        "outer_cli_timeout_seconds",
        "local_process_id",
        "local_process_group_id",
        "local_session_id",
    ):
        value = payload[name]
        if value is not None and (type(value) is not int or value <= 0):
            raise ModalActionJournalIntegrityError(f"terminal.{name} is invalid")
    if not preownership_rejection:
        if action is None:
            raise ModalActionJournalIntegrityError(
                "owned terminal lacks its action identity"
            )
        _validate_approved_action_contract(
            payload,
            action=action,
            harness=payload["harness"],
            field="terminal",
        )
    if payload["returncode"] is not None and type(payload["returncode"]) is not int:
        raise ModalActionJournalIntegrityError("terminal.returncode is invalid")
    if (
        payload["process_group_closed"] is not None
        and type(payload["process_group_closed"]) is not bool
    ):
        raise ModalActionJournalIntegrityError(
            "terminal.process_group_closed is invalid"
        )
    if type(payload["modal_cli_process_started"]) is not bool:
        raise ModalActionJournalIntegrityError(
            "terminal.modal_cli_process_started is invalid"
        )
    expected_remote_state = (
        "may_have_started"
        if payload["modal_cli_process_started"]
        else "definitely_not_started"
    )
    if payload["remote_execution_state"] != expected_remote_state:
        raise ModalActionJournalIntegrityError(
            "terminal process and remote-execution states differ"
        )
    marker_fields = (
        "local_process_start_receipt_path",
        "local_process_start_receipt_sha256",
        "local_process_id",
        "local_process_group_id",
        "local_session_id",
    )
    if payload["modal_cli_process_started"]:
        expected_marker = modal_local_process_start_receipt_path(attempt_id).as_posix()
        if payload["local_process_start_receipt_path"] != expected_marker:
            raise ModalActionJournalIntegrityError(
                "started terminal marker path is not canonical"
            )
        process_id = payload["local_process_id"]
        if process_id is not None and (
            payload["local_process_group_id"] != process_id
            or payload["local_session_id"] != process_id
        ):
            raise ModalActionJournalIntegrityError(
                "started terminal process identity is inconsistent"
            )
    elif any(payload[name] is not None for name in marker_fields):
        raise ModalActionJournalIntegrityError(
            "unstarted terminal contains process-start evidence"
        )
    _validate_terminal_status(payload)
    if location == "global_rejection" and (
        payload["modal_cli_process_started"]
        or payload["remote_execution_state"] != "definitely_not_started"
        or payload["returncode"] is not None
        or payload["process_group_closed"] is not None
    ):
        raise ModalActionJournalIntegrityError(
            "global rejection claims a started process"
        )
    return dict(payload)


def _validate_reservation_core(
    payload: Mapping[str, Any],
    *,
    run_id: str,
) -> dict[str, Any]:
    if set(payload) != _RESERVATION_FIELDS:
        raise ModalActionJournalIntegrityError(
            "global reservation has an invalid exact schema"
        )
    if (
        payload["schema_name"] != "ModalRemoteRunReservation"
        or payload["schema_version"] != "1.2"
        or payload["remote_run_id"] != run_id
        or payload["modal_environment"] != MODAL_ENVIRONMENT_NAME
    ):
        raise ModalActionJournalIntegrityError(
            "global reservation has the wrong contract"
        )
    _attempt_id(payload["owner_attempt_id"], "reservation.owner_attempt_id")
    if payload["action"] not in MODAL_ACTIONS:
        raise ModalActionJournalIntegrityError("reservation action is invalid")
    _identity_from_payload(
        payload,
        image_field="image_source_sha256",
        allow_absent=False,
        field="reservation",
    )
    _utc(payload["created_at_utc"], "reservation.created_at_utc")
    _sha256(
        payload["launch_capability_sha256"],
        "reservation.launch_capability_sha256",
    )
    _validate_containment(
        payload,
        field="reservation.local_containment",
        allow_absent=False,
    )
    return dict(payload)


def _validate_process_marker_core(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
) -> dict[str, Any]:
    if set(payload) != _LOCAL_PROCESS_START_FIELDS:
        raise ModalActionJournalIntegrityError(
            "process-start marker has an invalid exact schema"
        )
    if (
        payload["schema_name"] != "ModalLocalProcessStart"
        or payload["schema_version"] != "1.1"
        or payload["attempt_id"] != attempt_id
    ):
        raise ModalActionJournalIntegrityError(
            "process-start marker has the wrong contract"
        )
    _attempt_id(attempt_id, "process_marker.attempt_id")
    _utc(payload["created_at_utc"], "process_marker.created_at_utc")
    if payload["action"] not in MODAL_ACTIONS:
        raise ModalActionJournalIntegrityError("process marker action is invalid")
    try:
        validate_run_id(payload["run_id"])
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "process marker run ID is invalid"
        ) from error
    identity = _identity_from_payload(
        payload,
        image_field="image_source_sha256",
        allow_absent=False,
        field="process_marker",
    )
    if identity is None:  # pragma: no cover - guarded above
        raise AssertionError("validated marker lacks a cohort identity")
    if (
        payload["intent_path"]
        != modal_action_intent_receipt_path(
            identity,
            attempt_id,
        ).as_posix()
    ):
        raise ModalActionJournalIntegrityError(
            "process marker intent path is not canonical"
        )
    for name in (
        "intent_sha256",
        "modal_command_sha256",
        "launch_capability_sha256",
        "process_birth_identity_sha256",
    ):
        _sha256(payload[name], f"process_marker.{name}")
    _validate_containment(
        payload,
        field="process_marker.local_containment",
        allow_absent=False,
    )
    process_id = _exact_positive_int(payload["process_id"], "process_marker.pid")
    if (
        _exact_positive_int(
            payload["expected_process_group_id"],
            "process_marker.process_group_id",
        )
        != process_id
        or _exact_positive_int(
            payload["expected_session_id"],
            "process_marker.session_id",
        )
        != process_id
    ):
        raise ModalActionJournalIntegrityError(
            "process marker is not an isolated session leader"
        )
    if not isinstance(payload["modal_cost_cap_usd"], str):
        raise ModalActionJournalIntegrityError(
            "process marker Modal cost cap is invalid"
        )
    provider_cap = payload["provider_cost_cap_usd"]
    if (payload["action"] in _PROVIDER_ACTIONS) is not (
        isinstance(provider_cap, str)
    ):
        raise ModalActionJournalIntegrityError(
            "process marker provider cap differs from its action"
        )
    return dict(payload)


def _lineage_identity(
    value: object,
    *,
    field: str,
) -> ModalLiveCohortIdentity:
    if not isinstance(value, dict) or set(value) != {
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
    }:
        raise ModalActionJournalIntegrityError(f"{field} has an invalid exact schema")
    try:
        return ModalLiveCohortIdentity(
            source_tree_sha256=_sha256(
                value["source_tree_sha256"],
                f"{field}.source_tree_sha256",
            ),
            image_source_sha256=_sha256(
                value["image_source_sha256"],
                f"{field}.image_source_sha256",
            ),
            cohort_id=validate_run_id(value["cohort_id"]),
        )
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(f"{field} is invalid") from error


def _lineage_file_binding(
    value: object,
    *,
    field: str,
    expected_path: str | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _LINEAGE_FILE_BINDING_FIELDS:
        raise ModalActionJournalIntegrityError(f"{field} has an invalid exact schema")
    try:
        logical = safe_relative_path(value["path"]).as_posix()
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(f"{field}.path is invalid") from error
    if expected_path is not None and logical != expected_path:
        raise ModalActionJournalIntegrityError(f"{field}.path is not canonical")
    digest = _sha256(value["sha256"], f"{field}.sha256")
    size = value["size_bytes"]
    if type(size) is not int or size < 0 or (size == 0 and not allow_empty):
        raise ModalActionJournalIntegrityError(f"{field}.size_bytes is invalid")
    return {"path": logical, "sha256": digest, "size_bytes": size}


def _lineage_action_journal(
    value: object,
    *,
    identity: ModalLiveCohortIdentity,
    field: str,
) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict) or set(value) != _LINEAGE_JOURNAL_FIELDS:
        raise ModalActionJournalIntegrityError(f"{field} has an invalid exact schema")
    patterns = {
        "intent_receipts": _INTENT_NAME,
        "terminal_receipts": _TERMINAL_NAME,
        "aggregate_receipts": _AGGREGATE_NAME,
    }
    expected_parent = modal_action_attempt_directory(identity).as_posix()
    result: dict[str, list[dict[str, Any]]] = {}
    all_paths: set[str] = set()
    for kind, pattern in patterns.items():
        raw_bindings = value[kind]
        if not isinstance(raw_bindings, list):
            raise ModalActionJournalIntegrityError(f"{field}.{kind} must be a list")
        bindings = [
            _lineage_file_binding(
                raw,
                field=f"{field}.{kind}[{index}]",
            )
            for index, raw in enumerate(raw_bindings)
        ]
        paths = [record["path"] for record in bindings]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ModalActionJournalIntegrityError(
                f"{field}.{kind} must be sorted and unique"
            )
        for logical in paths:
            selected = safe_relative_path(logical)
            if (
                selected.parent.as_posix() != expected_parent
                or pattern.fullmatch(selected.name) is None
                or logical in all_paths
            ):
                raise ModalActionJournalIntegrityError(
                    f"{field}.{kind} contains a noncanonical binding"
                )
            all_paths.add(logical)
        result[kind] = bindings
    return result


def _lineage_reservation_bindings(
    value: object,
    *,
    field: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ModalActionJournalIntegrityError(f"{field} must be a list")
    bindings = [
        _lineage_file_binding(raw, field=f"{field}[{index}]")
        for index, raw in enumerate(value)
    ]
    paths = [record["path"] for record in bindings]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ModalActionJournalIntegrityError(f"{field} must be sorted and unique")
    expected_parent = MODAL_REMOTE_RUN_RESERVATION_ROOT.as_posix()
    for logical in paths:
        selected = safe_relative_path(logical)
        try:
            run_id = validate_run_id(selected.stem)
        except (TypeError, ValueError) as error:
            raise ModalActionJournalIntegrityError(
                f"{field} contains an invalid run ID"
            ) from error
        if (
            selected.parent.as_posix() != expected_parent
            or selected.suffix != ".json"
            or logical != modal_remote_run_reservation_path(run_id).as_posix()
        ):
            raise ModalActionJournalIntegrityError(
                f"{field} contains a noncanonical binding"
            )
    return bindings


def _sorted_unique_text_list(value: object, *, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or value != sorted(set(value))
    ):
        raise ModalActionJournalIntegrityError(f"{field} must be sorted unique text")
    return list(value)


def _lineage_execution_context(
    value: object,
    *,
    field: str,
) -> ExecutionContextV1:
    try:
        return ExecutionContextV1.from_dict(value)
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(f"{field} is invalid") from error


def _lineage_remote_executions(
    value: object,
    *,
    identity: ModalLiveCohortIdentity,
    field: str,
) -> dict[tuple[str, str], tuple[dict[str, Any], ExecutionContextV1]]:
    if not isinstance(value, list):
        raise ModalActionJournalIntegrityError(f"{field} must be a list")
    result: dict[tuple[str, str], tuple[dict[str, Any], ExecutionContextV1]] = {}
    keys: list[tuple[str, str]] = []
    app_owners: dict[str, str] = {}
    call_ids: set[str] = set()
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(raw, dict) or set(raw) != _LINEAGE_REMOTE_EXECUTION_FIELDS:
            raise ModalActionJournalIntegrityError(
                f"{item_field} has an invalid exact schema"
            )
        attempt_id = _attempt_id(raw["attempt_id"], f"{item_field}.attempt_id")
        try:
            run_id = validate_run_id(raw["run_id"])
        except (TypeError, ValueError) as error:
            raise ModalActionJournalIntegrityError(
                f"{item_field}.run_id is invalid"
            ) from error
        action = raw["action"]
        evidence_kind = raw["evidence_kind"]
        if (
            not isinstance(action, str)
            or action not in MODAL_ACTIONS
            or not isinstance(evidence_kind, str)
            or evidence_kind not in _LINEAGE_REMOTE_EXECUTION_EVIDENCE_KINDS
        ):
            raise ModalActionJournalIntegrityError(
                f"{item_field} action or evidence kind is invalid"
            )
        verifier = action in {"download", "verify"}
        if verifier is not evidence_kind.startswith(("remote_", "volume_")):
            raise ModalActionJournalIntegrityError(
                f"{item_field} evidence kind differs from its action"
            )
        expected_evidence_path = None
        if not verifier:
            expected_evidence_path = (
                PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT)
                / run_id
                / "execution_context.json"
            ).as_posix()
        _lineage_file_binding(
            raw["evidence"],
            field=f"{item_field}.evidence",
            expected_path=expected_evidence_path,
        )
        context = _lineage_execution_context(
            raw["execution_context"],
            field=f"{item_field}.execution_context",
        )
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
            raise ModalActionJournalIntegrityError(
                f"{item_field}.execution_context identity is incomplete"
            )
        expected_function = (
            "artifact_verify" if verifier else _ORDINARY_ACTION_FUNCTIONS.get(action)
        )
        if expected_function is not None:
            if context.function_name != expected_function:
                raise ModalActionJournalIntegrityError(
                    f"{item_field}.execution_context function differs from its action"
                )
        elif action in {"canary", "canaries"}:
            if context.function_name not in {
                f"canary_{harness}" for harness in CANARY_ORDER
            }:
                raise ModalActionJournalIntegrityError(
                    f"{item_field}.execution_context canary function is invalid"
                )
        else:  # pragma: no cover - frozen MODAL_ACTIONS parity
            raise ModalActionJournalIntegrityError(
                f"{item_field}.execution_context action is unsupported"
            )
        key = (attempt_id, run_id)
        keys.append(key)
        result[key] = (dict(raw), context)
        owner = app_owners.setdefault(context.modal_app_id, attempt_id)
        if owner != attempt_id:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal reuses one Modal App across attempts"
            )
        if context.modal_call_id in call_ids:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal reuses one Modal call ID"
            )
        call_ids.add(context.modal_call_id)
    if keys != sorted(set(keys)):
        raise ModalActionJournalIntegrityError(
            f"{field} must be sorted and unique by attempt/run"
        )
    return result


def _lineage_artifact_manifests(
    value: object,
    *,
    remote_executions: Mapping[
        tuple[str, str], tuple[Mapping[str, Any], ExecutionContextV1]
    ],
    field: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(value, list):
        raise ModalActionJournalIntegrityError(f"{field} must be a list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    keys: list[tuple[str, str]] = []
    paths: list[str] = []
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(raw, dict) or set(raw) != _LINEAGE_ARTIFACT_MANIFEST_FIELDS:
            raise ModalActionJournalIntegrityError(
                f"{item_field} has an invalid exact schema"
            )
        attempt_id = _attempt_id(raw["attempt_id"], f"{item_field}.attempt_id")
        try:
            run_id = validate_run_id(raw["run_id"])
            logical = safe_relative_path(raw["path"]).as_posix()
        except (TypeError, ValueError) as error:
            raise ModalActionJournalIntegrityError(
                f"{item_field} identity or path is invalid"
            ) from error
        _sha256(raw["sha256"], f"{item_field}.sha256")
        _sha256(
            raw["canonical_manifest_sha256"],
            f"{item_field}.canonical_manifest_sha256",
        )
        if type(raw["size_bytes"]) is not int or raw["size_bytes"] <= 0:
            raise ModalActionJournalIntegrityError(
                f"{item_field}.size_bytes is invalid"
            )
        key = (attempt_id, run_id)
        execution = remote_executions.get(key)
        if execution is None:
            raise ModalActionJournalIntegrityError(f"{item_field} invents an execution")
        evidence_kind = execution[0]["evidence_kind"]
        if evidence_kind == "downloaded_execution_context":
            selected = safe_relative_path(logical)
            expected_parent = PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT) / run_id
            if (
                selected.parent != expected_parent
                or selected.name not in ARTIFACT_MANIFEST_FILENAMES
            ):
                raise ModalActionJournalIntegrityError(
                    f"{item_field}.path is not canonical"
                )
        elif evidence_kind not in {"volume_success_capture", "volume_failure_capture"}:
            raise ModalActionJournalIntegrityError(
                f"{item_field} is forbidden for its evidence kind"
            )
        keys.append(key)
        paths.append(logical)
        result[key] = dict(raw)
    if keys != sorted(set(keys)) or len(paths) != len(set(paths)):
        raise ModalActionJournalIntegrityError(
            f"{field} must be sorted and unique by attempt/run and path"
        )
    expected_keys = {
        key
        for key, (record, _context) in remote_executions.items()
        if record["evidence_kind"]
        in {
            "downloaded_execution_context",
            "volume_success_capture",
            "volume_failure_capture",
        }
    }
    if set(result) != expected_keys:
        raise ModalActionJournalIntegrityError(
            f"{field} omits or invents a remote execution manifest"
        )
    return result


def _lineage_optional_file_binding(
    value: object,
    *,
    field: str,
    expected_path: str,
    allow_empty: bool = False,
) -> dict[str, Any] | None:
    if value is None:
        return None
    return _lineage_file_binding(
        value,
        field=field,
        expected_path=expected_path,
        allow_empty=allow_empty,
    )


def _lineage_provider_attempt_evidence(
    value: object,
    *,
    remote_executions: Mapping[
        tuple[str, str], tuple[Mapping[str, Any], ExecutionContextV1]
    ],
    field: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(value, list):
        raise ModalActionJournalIntegrityError(f"{field} must be a list")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    keys: list[tuple[str, str]] = []
    request_ids: set[str] = set()
    response_ids: set[str] = set()
    for index, raw in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(raw, dict) or set(raw) != _LINEAGE_PROVIDER_EVIDENCE_FIELDS:
            raise ModalActionJournalIntegrityError(
                f"{item_field} has an invalid exact schema"
            )
        attempt_id = _attempt_id(raw["attempt_id"], f"{item_field}.attempt_id")
        try:
            run_id = validate_run_id(raw["run_id"])
        except (TypeError, ValueError) as error:
            raise ModalActionJournalIntegrityError(
                f"{item_field}.run_id is invalid"
            ) from error
        harness = raw["harness"]
        if not isinstance(harness, str) or harness not in CANARY_ORDER:
            raise ModalActionJournalIntegrityError(f"{item_field}.harness is invalid")
        key = (attempt_id, run_id)
        execution = remote_executions.get(key)
        binding_state = raw["binding_state"]
        if not isinstance(binding_state, str) or binding_state not in {
            "execution_context_bound",
            "unbound_observed",
        }:
            raise ModalActionJournalIntegrityError(
                f"{item_field}.binding_state is invalid"
            )
        if (binding_state == "execution_context_bound") is not (execution is not None):
            raise ModalActionJournalIntegrityError(
                f"{item_field}.binding_state differs from remote execution evidence"
            )
        if execution is not None and (
            execution[0]["action"] not in {"canary", "canaries"}
            or execution[1].function_name != f"canary_{harness}"
        ):
            raise ModalActionJournalIntegrityError(
                f"{item_field} harness differs from its execution"
            )
        controller = PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT) / run_id / "controller"
        ledger = _lineage_optional_file_binding(
            raw["ledger"],
            field=f"{item_field}.ledger",
            expected_path=(controller / "provider_attempts.jsonl").as_posix(),
            allow_empty=True,
        )
        uncertainty = _lineage_optional_file_binding(
            raw["uncertainty"],
            field=f"{item_field}.uncertainty",
            expected_path=(
                controller / "provider_request_start_uncertain.json"
            ).as_posix(),
        )
        if ledger is None and uncertainty is None:
            raise ModalActionJournalIntegrityError(
                f"{item_field} contains no provider evidence binding"
            )
        dispositions = raw["parse_dispositions"]
        if (
            not isinstance(dispositions, list)
            or not dispositions
            or any(
                not isinstance(item, str)
                or item
                not in {
                    "valid_terminal_records",
                    "exact_empty",
                    "valid_start_uncertain",
                    "partial_unparseable",
                }
                for item in dispositions
            )
        ):
            raise ModalActionJournalIntegrityError(
                f"{item_field}.parse_dispositions is invalid"
            )
        if (
            type(raw["provider_attempt_count"]) is not int
            or not 0 <= raw["provider_attempt_count"] <= 1
        ):
            raise ModalActionJournalIntegrityError(
                f"{item_field}.provider_attempt_count is invalid"
            )
        selected_requests = _sorted_unique_text_list(
            raw["request_ids"], field=f"{item_field}.request_ids"
        )
        selected_responses = _sorted_unique_text_list(
            raw["response_ids"], field=f"{item_field}.response_ids"
        )
        if request_ids.intersection(selected_requests) or response_ids.intersection(
            selected_responses
        ):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal reuses provider request or response IDs"
            )
        request_ids.update(selected_requests)
        response_ids.update(selected_responses)
        keys.append(key)
        result[key] = dict(raw)
    if keys != sorted(set(keys)):
        raise ModalActionJournalIntegrityError(
            f"{field} must be sorted and unique by attempt/run"
        )
    return result


def _lineage_provider_spend_estimate(value: object, *, field: str) -> None:
    if not isinstance(value, dict) or set(value) != _LINEAGE_PROVIDER_SPEND_FIELDS:
        raise ModalActionJournalIntegrityError(f"{field} has an invalid exact schema")
    if not isinstance(value["accounting_label"], str) or not value["accounting_label"]:
        raise ModalActionJournalIntegrityError(f"{field}.accounting_label is invalid")
    count_fields = (
        "provider_launcher_attempt_count",
        "provider_terminal_attempt_record_count",
        "provider_attempt_count_lower_bound",
        "provider_attempt_count_upper_bound",
        "successful_provider_attempt_count",
        "failed_provider_attempt_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
    )
    if any(type(value[name]) is not int or value[name] < 0 for name in count_fields):
        raise ModalActionJournalIntegrityError(f"{field} contains an invalid count")
    if (
        value["provider_attempt_count_lower_bound"]
        > value["provider_attempt_count_upper_bound"]
        or value["total_tokens"] != value["input_tokens"] + value["output_tokens"]
    ):
        raise ModalActionJournalIntegrityError(f"{field} counts do not reconcile")
    decimals = {
        name: _nonnegative_decimal_text(value[name], f"{field}.{name}")
        for name in (
            "known_success_usage_estimate_usd",
            "failed_attempt_reserve_usd",
            "uncertain_request_start_reserve_usd",
            "conservative_provider_spend_bound_usd",
            "approved_provider_cap_total_usd",
        )
    }
    if (
        decimals["conservative_provider_spend_bound_usd"]
        != (
            decimals["known_success_usage_estimate_usd"]
            + decimals["failed_attempt_reserve_usd"]
            + decimals["uncertain_request_start_reserve_usd"]
        )
        or decimals["conservative_provider_spend_bound_usd"]
        > decimals["approved_provider_cap_total_usd"]
    ):
        raise ModalActionJournalIntegrityError(f"{field} totals do not reconcile")
    _sorted_unique_text_list(
        value["provider_request_ids"],
        field=f"{field}.provider_request_ids",
    )
    _sorted_unique_text_list(
        value["provider_response_ids"],
        field=f"{field}.provider_response_ids",
    )
    for name in ("run_cost_dispositions", "launcher_approval_bounds"):
        records = value[name]
        if not isinstance(records, list) or any(
            not isinstance(record, dict) or not record for record in records
        ):
            raise ModalActionJournalIntegrityError(f"{field}.{name} is invalid")


def _lineage_modal_compute_exposure(value: object, *, field: str) -> None:
    if not isinstance(value, dict) or set(value) != _LINEAGE_MODAL_EXPOSURE_FIELDS:
        raise ModalActionJournalIntegrityError(f"{field} has an invalid exact schema")
    if not isinstance(value["accounting_label"], str) or not value["accounting_label"]:
        raise ModalActionJournalIntegrityError(f"{field}.accounting_label is invalid")
    measured = _nonnegative_decimal_text(
        value["measured_app_billing_usd"],
        f"{field}.measured_app_billing_usd",
    )
    reserve = _nonnegative_decimal_text(
        value["unresolved_compute_reserve_usd"],
        f"{field}.unresolved_compute_reserve_usd",
    )
    conservative = _nonnegative_decimal_text(
        value["conservative_compute_exposure_usd"],
        f"{field}.conservative_compute_exposure_usd",
    )
    _nonnegative_decimal_text(
        value["measured_over_local_authorization_cap_usd"],
        f"{field}.measured_over_local_authorization_cap_usd",
    )
    if conservative != measured + reserve:
        raise ModalActionJournalIntegrityError(f"{field} totals do not reconcile")
    _sorted_unique_text_list(
        value["local_authorization_cap_breach_attempt_ids"],
        field=f"{field}.local_authorization_cap_breach_attempt_ids",
    )
    if value["local_authorization_is_platform_hard_bound"] is not False:
        raise ModalActionJournalIntegrityError(
            f"{field} misstates the authorization-cap semantics"
        )
    if not isinstance(value["attempts"], list) or any(
        not isinstance(record, dict) or not record for record in value["attempts"]
    ):
        raise ModalActionJournalIntegrityError(f"{field}.attempts is invalid")


def _validate_migration_terminal_seal_core(
    payload: Mapping[str, Any],
    *,
    identity: ModalLiveCohortIdentity,
) -> dict[str, Any]:
    if set(payload) != _MIGRATION_LINEAGE_FIELDS:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal has an invalid exact schema"
        )
    if (
        payload["schema_name"] != "ModalMigrationLineage"
        or payload["schema_version"] != "1.1"
        or payload["validated"] is not True
        or payload["global_uniqueness_validated"] is not True
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal has the wrong contract"
        )
    _utc(payload["recorded_at_utc"], "migration_terminal_seal.recorded_at_utc")
    selected = payload["selected_final"]
    if not isinstance(selected, dict) or set(selected) != (
        _LINEAGE_SELECTED_FINAL_FIELDS
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal final selection has an invalid exact schema"
        )
    selected_identity = _lineage_identity(
        selected["identity"],
        field="migration_terminal_seal.selected_final.identity",
    )
    if selected_identity != identity:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal differs from its cohort path"
        )

    accepted_runs = selected["accepted_primary_runs"]
    accepted_attempts = selected["accepted_attempt_ids"]
    if not isinstance(accepted_runs, dict) or set(accepted_runs) != set(
        _LINEAGE_PRIMARY_ACTIONS
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal accepted run roster is not exact"
        )
    if not isinstance(accepted_attempts, dict) or set(accepted_attempts) != set(
        _LINEAGE_PRIMARY_ACTIONS
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal accepted attempt roster is not exact"
        )
    try:
        validated_runs = {
            label: validate_run_id(accepted_runs[label])
            for label in _LINEAGE_PRIMARY_ACTIONS
        }
        validated_attempts = {
            label: _attempt_id(
                accepted_attempts[label],
                f"migration_terminal_seal.accepted_attempt_ids.{label}",
            )
            for label in _LINEAGE_PRIMARY_ACTIONS
        }
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal accepted identity is invalid"
        ) from error
    if len(set(validated_runs.values())) != len(_LINEAGE_PRIMARY_ACTIONS) or len(
        set(validated_attempts.values())
    ) != len(_LINEAGE_PRIMARY_ACTIONS):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal accepted identities are not unique"
        )

    journal = _lineage_action_journal(
        selected["action_journal"],
        identity=identity,
        field="migration_terminal_seal.selected_final.action_journal",
    )
    _lineage_reservation_bindings(
        selected["remote_run_reservations"],
        field="migration_terminal_seal.selected_final.remote_run_reservations",
    )
    run_dispositions = selected["run_dispositions"]
    if not isinstance(run_dispositions, list):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal run dispositions must be a list"
        )
    disposition_keys: list[tuple[str, str]] = []
    for index, record in enumerate(run_dispositions):
        if not isinstance(record, dict) or set(record) != (
            _LINEAGE_RUN_DISPOSITION_FIELDS
        ):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal run disposition schema drifted"
            )
        attempt_id = _attempt_id(
            record["attempt_id"],
            f"migration_terminal_seal.run_dispositions[{index}].attempt_id",
        )
        try:
            run_id = validate_run_id(record["run_id"])
        except (TypeError, ValueError) as error:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal run disposition run ID is invalid"
            ) from error
        if (
            not isinstance(record["action"], str)
            or record["action"] not in MODAL_ACTIONS
            or not isinstance(record["status"], str)
                or record["status"]
                not in {
                    "succeeded",
                    "failed",
                    "timed_out",
                    "preflight_failed",
                    "preflight_rejected",
                    "lock_contended",
                    "interrupted",
                    "cli_failed",
                    "cleanup_failed",
                }
            or not isinstance(record["failure_kind"], (str, type(None)))
            or type(record["modal_cli_process_started"]) is not bool
            or record["remote_execution_state"]
            not in {"may_have_started", "definitely_not_started"}
            or not isinstance(record["execution_disposition"], str)
            or record["execution_disposition"]
            not in {
                "definitely_not_started",
                "remote_execution_bound",
                "may_have_started_unresolved_quarantined",
            }
            or not isinstance(record["provider_disposition"], str)
            or record["provider_disposition"]
            not in {
                "not_applicable",
                "definitely_not_started",
                "evidence_bound",
                "start_unresolved_conservative",
            }
        ):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal run disposition is invalid"
            )
        provider_action = record["action"] in {"canary", "canaries"}
        process_started = record["modal_cli_process_started"]
        if (
            (
                (not process_started)
                and (
                    record["execution_disposition"] != "definitely_not_started"
                    or record["provider_disposition"]
                    != (
                        "definitely_not_started"
                        if provider_action
                        else "not_applicable"
                    )
                )
            )
            or (
                process_started
                and record["execution_disposition"] == "definitely_not_started"
            )
            or (
                provider_action
                and process_started
                and record["provider_disposition"]
                not in {"evidence_bound", "start_unresolved_conservative"}
            )
            or (
                not provider_action
                and record["provider_disposition"] != "not_applicable"
            )
        ):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal run disposition semantics are invalid"
            )
        disposition_keys.append((attempt_id, run_id))
    if disposition_keys != sorted(set(disposition_keys)):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal run dispositions are not sorted and unique"
        )

    if selected["aggregate_receipts"] != journal["aggregate_receipts"]:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal aggregate rosters differ"
        )
    remote_executions = _lineage_remote_executions(
        selected["remote_executions"],
        identity=identity,
        field="migration_terminal_seal.selected_final.remote_executions",
    )
    _lineage_artifact_manifests(
        selected["artifact_manifests"],
        remote_executions=remote_executions,
        field="migration_terminal_seal.selected_final.artifact_manifests",
    )
    provider_evidence = _lineage_provider_attempt_evidence(
        selected["provider_attempt_evidence"],
        remote_executions=remote_executions,
        field="migration_terminal_seal.selected_final.provider_attempt_evidence",
    )
    remote_ids = selected["remote_object_ids"]
    if not isinstance(remote_ids, dict) or set(remote_ids) != (
        _LINEAGE_REMOTE_OBJECT_FIELDS
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal remote object roster is not exact"
        )
    validated_remote_ids = {
        name: _sorted_unique_text_list(
            remote_ids[name],
            field=f"migration_terminal_seal.remote_object_ids.{name}",
        )
        for name in _LINEAGE_REMOTE_OBJECT_FIELDS
    }
    contexts = [context for _record, context in remote_executions.values()]
    expected_remote_ids = {
        "app_ids": sorted({context.modal_app_id for context in contexts}),
        "function_ids": sorted({context.modal_function_id for context in contexts}),
        "call_ids": sorted({context.modal_call_id for context in contexts}),
        "image_ids": sorted({context.modal_image_id for context in contexts}),
    }
    if validated_remote_ids != expected_remote_ids:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal remote object roster differs from executions"
        )
    _lineage_provider_spend_estimate(
        selected["provider_spend_estimate"],
        field="migration_terminal_seal.selected_final.provider_spend_estimate",
    )
    provider_requests = sorted(
        request_id
        for record in provider_evidence.values()
        for request_id in record["request_ids"]
    )
    provider_responses = sorted(
        response_id
        for record in provider_evidence.values()
        for response_id in record["response_ids"]
    )
    if (
        selected["provider_spend_estimate"]["provider_request_ids"] != provider_requests
        or selected["provider_spend_estimate"]["provider_response_ids"]
        != provider_responses
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal provider ID rosters do not reconcile"
        )

    prior = payload["prior_quarantined_cohorts"]
    if not isinstance(prior, list):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal prior cohort roster must be a list"
        )
    prior_identities: list[ModalLiveCohortIdentity] = []
    for index, record in enumerate(prior):
        field = f"migration_terminal_seal.prior_quarantined_cohorts[{index}]"
        if not isinstance(record, dict) or set(record) != _LINEAGE_PRIOR_FIELDS:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal prior cohort schema drifted"
            )
        prior_identity = _lineage_identity(
            record["identity"],
            field=f"{field}.identity",
        )
        if record["disposition"] != "quarantined":
            raise ModalActionJournalIntegrityError(
                "migration terminal seal prior cohort is not quarantined"
            )
        expected_accounting = (
            MODAL_LIVE_COHORT_ROOT
            / prior_identity.source_tree_sha256
            / prior_identity.image_source_sha256
            / prior_identity.cohort_id
            / "quarantine_accounting.v1.1.json"
        ).as_posix()
        _lineage_file_binding(
            record["accounting_receipt"],
            field=f"{field}.accounting_receipt",
            expected_path=expected_accounting,
        )
        _lineage_action_journal(
            record["action_journal"],
            identity=prior_identity,
            field=f"{field}.action_journal",
        )
        _lineage_reservation_bindings(
            record["remote_run_reservations"],
            field=f"{field}.remote_run_reservations",
        )
        _lineage_provider_spend_estimate(
            record["provider_spend_estimate"],
            field=f"{field}.provider_spend_estimate",
        )
        _lineage_modal_compute_exposure(
            record["modal_compute_exposure"],
            field=f"{field}.modal_compute_exposure",
        )
        prior_identities.append(prior_identity)
    prior_keys = [
        (item.source_tree_sha256, item.image_source_sha256, item.cohort_id)
        for item in prior_identities
    ]
    if identity in prior_identities or prior_keys != sorted(set(prior_keys)):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal prior identities are not sorted and unique"
        )

    _lineage_reservation_bindings(
        payload["global_remote_run_reservations"],
        field="migration_terminal_seal.global_remote_run_reservations",
    )
    legacy = payload["legacy_superseded_usage"]
    if not isinstance(legacy, dict) or set(legacy) != {
        "run_id",
        "amount_usd",
        "accounting_basis",
    }:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal legacy usage schema drifted"
        )
    if (
        legacy["run_id"] != "modal-cuda-env-20260809-02"
        or legacy["amount_usd"] != "0.00643852"
        or legacy["accounting_basis"]
        != "preserved_legacy_measurement_excluded_from_all_cohort_snapshots"
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal legacy usage changed"
        )
    totals = {
        name: _nonnegative_decimal_text(
            payload[name],
            f"migration_terminal_seal.{name}",
        )
        for name in (
            "prior_app_compute_total_usd",
            "final_provider_spend_bound_usd",
            "prior_provider_spend_bound_usd",
            "migration_provider_spend_bound_usd",
            "prior_modal_measured_app_billing_usd",
            "prior_modal_unresolved_compute_reserve_usd",
            "prior_modal_conservative_exposure_usd",
        )
    }
    if totals["migration_provider_spend_bound_usd"] != (
        totals["final_provider_spend_bound_usd"]
        + totals["prior_provider_spend_bound_usd"]
    ) or totals["prior_modal_conservative_exposure_usd"] != (
        totals["prior_modal_measured_app_billing_usd"]
        + totals["prior_modal_unresolved_compute_reserve_usd"]
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal totals do not reconcile"
        )
    retained = payload["retained_storage_estimate"]
    if not isinstance(retained, dict) or set(retained) != (
        _LINEAGE_RETAINED_STORAGE_FIELDS
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal retained-storage schema drifted"
        )
    if (
        not isinstance(retained["prior_cohort_estimates"], list)
        or len(retained["prior_cohort_estimates"]) != len(prior)
        or any(
            not isinstance(record, dict) or not record
            for record in retained["prior_cohort_estimates"]
        )
        or retained["final_cohort_included"] is not False
        or retained["basis"]
        != (
            "prior_quarantine_receipts_only; final retained storage is "
            "reported by the cleanup receipt"
        )
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal retained-storage contract changed"
        )
    return dict(payload)


def _validate_global_launch_rejection_seal_core(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if set(payload) != _GLOBAL_LAUNCH_REJECTION_SEAL_FIELDS:
        raise ModalActionJournalIntegrityError(
            "global launch-rejection seal has an invalid exact schema"
        )
    if (
        payload["schema_name"] != "ModalGlobalLaunchRejectionSeal"
        or payload["schema_version"] != "1.0"
        or payload["validated"] is not True
    ):
        raise ModalActionJournalIntegrityError(
            "global launch-rejection seal has the wrong contract"
        )
    _canonical_utc(
        payload["recorded_at_utc"],
        "global_launch_rejection_seal.recorded_at_utc",
    )
    roster = payload["rejection_receipts"]
    if not isinstance(roster, list):
        raise ModalActionJournalIntegrityError(
            "global launch-rejection seal roster must be a list"
        )
    bindings = [
        _lineage_file_binding(
            raw,
            field=f"global_launch_rejection_seal.rejection_receipts[{index}]",
        )
        for index, raw in enumerate(roster)
    ]
    paths = [binding["path"] for binding in bindings]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ModalActionJournalIntegrityError(
            "global launch-rejection seal roster must be sorted and unique"
        )
    for logical in paths:
        selected = safe_relative_path(logical)
        match = _TERMINAL_NAME.fullmatch(selected.name)
        if (
            selected.parent != MODAL_LAUNCH_REJECTION_ROOT
            or match is None
            or logical != modal_launch_rejection_receipt_path(match.group(1)).as_posix()
        ):
            raise ModalActionJournalIntegrityError(
                "global launch-rejection seal contains a noncanonical binding"
            )
    return dict(payload)


def _directory_flags() -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory is None:
        raise ModalActionJournalIntegrityError(
            "platform cannot enforce no-follow journal discovery"
        )
    return os.O_RDONLY | os.O_CLOEXEC | no_follow | directory


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_uid,
        metadata.st_nlink,
        stat.S_IMODE(metadata.st_mode),
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_uid,
        metadata.st_nlink,
        stat.S_IMODE(metadata.st_mode),
    )


def _validate_directory_metadata(
    metadata: os.stat_result,
    *,
    field: str,
    required_mode: int | None = None,
) -> None:
    mode = stat.S_IMODE(metadata.st_mode)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or mode & 0o022
        or (required_mode is not None and mode != required_mode)
    ):
        raise ModalActionJournalIntegrityError(
            f"{field} must be an owned, non-writable directory"
        )


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModalActionJournalIntegrityError(
                f"journal JSON contains duplicate key: {key}"
            )
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ModalActionJournalIntegrityError(
        f"journal JSON contains non-finite value: {value}"
    )


class _NamespaceReader:
    """Hold stable witnesses for every directory touched by one global scan."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        try:
            before = os.lstat(project_root)
        except OSError as error:
            raise ModalActionJournalIntegrityError(
                "journal project root changed before discovery"
            ) from error
        if stat.S_ISLNK(before.st_mode):
            raise ModalActionJournalIntegrityError(
                "journal project root may not be a symlink"
            )
        _validate_directory_metadata(before, field="journal project root")
        try:
            self.root_descriptor = os.open(project_root, _directory_flags())
        except OSError as error:
            raise ModalActionJournalIntegrityError(
                "journal project root changed while opening"
            ) from error
        opened = os.fstat(self.root_descriptor)
        if _directory_identity(before) != _directory_identity(opened):
            os.close(self.root_descriptor)
            raise ModalActionJournalIntegrityError(
                "journal project root changed while opening"
            )
        self._witnesses: dict[tuple[str, ...], _DirectoryWitness] = {}
        self._file_witnesses: dict[tuple[tuple[str, ...], str], _FileWitness] = {}
        self._absent_directories: set[tuple[str, ...]] = set()
        self._register_witness((), self.root_descriptor)

    def _register_witness(self, parts: tuple[str, ...], descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        _validate_directory_metadata(
            metadata,
            field="journal directory " + ("/".join(parts) or "."),
        )
        names = tuple(sorted(os.listdir(descriptor)))
        previous = self._witnesses.get(parts)
        if previous is not None:
            if previous.identity != _directory_identity(metadata):
                raise ModalActionJournalIntegrityError(
                    "journal directory has ambiguous identities"
                )
            return
        self._witnesses[parts] = _DirectoryWitness(
            parts=parts,
            descriptor=os.dup(descriptor),
            identity=_directory_identity(metadata),
            names=names,
        )

    def open_directory(
        self,
        parts: Sequence[str],
        *,
        field: str,
        optional: bool,
        required_mode: int | None = None,
    ) -> int | None:
        selected = tuple(parts)
        descriptor = os.dup(self.root_descriptor)
        traversed: list[str] = []
        try:
            for component in selected:
                if component in {"", ".", ".."} or "/" in component:
                    raise ModalActionJournalIntegrityError(
                        f"{field} contains an unsafe path component"
                    )
                try:
                    before = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    if optional:
                        self._absent_directories.add(selected)
                        os.close(descriptor)
                        return None
                    raise ModalActionJournalIntegrityError(
                        f"{field} is missing"
                    ) from None
                if stat.S_ISLNK(before.st_mode):
                    raise ModalActionJournalIntegrityError(
                        f"{field} may not traverse symlinks"
                    )
                _validate_directory_metadata(before, field=field)
                try:
                    child = os.open(
                        component,
                        _directory_flags(),
                        dir_fd=descriptor,
                    )
                except OSError as error:
                    raise ModalActionJournalIntegrityError(
                        f"{field} changed while opening"
                    ) from error
                opened = os.fstat(child)
                if _directory_identity(before) != _directory_identity(opened):
                    os.close(child)
                    raise ModalActionJournalIntegrityError(
                        f"{field} changed while opening"
                    )
                os.close(descriptor)
                descriptor = child
                traversed.append(component)
                self._register_witness(tuple(traversed), descriptor)
            if required_mode is not None:
                _validate_directory_metadata(
                    os.fstat(descriptor),
                    field=field,
                    required_mode=required_mode,
                )
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def read_json_leaf(
        self,
        parent_descriptor: int,
        filename: str,
        *,
        logical_path: str,
        field: str,
    ) -> ModalJournalRecord:
        try:
            before = os.stat(
                filename,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ModalActionJournalIntegrityError(f"{field} disappeared") from error
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > _MAX_JSON_BYTES
        ):
            raise ModalActionJournalIntegrityError(
                f"{field} must be an owned single-link 0600 regular file"
            )
        flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(filename, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise ModalActionJournalIntegrityError(
                f"{field} changed while opening"
            ) from error
        try:
            opened = os.fstat(descriptor)
            if _file_identity(opened) != _file_identity(before):
                raise ModalActionJournalIntegrityError(f"{field} changed while opening")
            raw = bytearray()
            while len(raw) <= _MAX_JSON_BYTES:
                try:
                    chunk = os.read(
                        descriptor,
                        min(64 * 1024, _MAX_JSON_BYTES + 1 - len(raw)),
                    )
                except InterruptedError:
                    continue
                if not chunk:
                    break
                raw.extend(chunk)
            if len(raw) > _MAX_JSON_BYTES:
                raise ModalActionJournalIntegrityError(
                    f"{field} exceeds its size limit"
                )
            after = os.fstat(descriptor)
            if (
                _file_identity(after) != _file_identity(opened)
                or len(raw) != after.st_size
            ):
                raise ModalActionJournalIntegrityError(f"{field} changed while reading")
            witness_descriptor = os.dup(descriptor)
        except OSError as error:
            raise ModalActionJournalIntegrityError(
                f"{field} changed while reading"
            ) from error
        finally:
            os.close(descriptor)
        try:
            rebound = os.stat(
                filename,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if _file_identity(rebound) != _file_identity(before):
                raise ModalActionJournalIntegrityError(f"{field} changed after reading")
            payload = json.loads(
                bytes(raw).decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except OSError as error:
            os.close(witness_descriptor)
            raise ModalActionJournalIntegrityError(
                f"{field} changed after reading"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            os.close(witness_descriptor)
            raise ModalActionJournalIntegrityError(
                f"{field} is not valid UTF-8 JSON"
            ) from error
        except ModalActionJournalIntegrityError:
            os.close(witness_descriptor)
            raise
        if not isinstance(payload, dict):
            os.close(witness_descriptor)
            raise ModalActionJournalIntegrityError(
                f"{field} must contain one JSON object"
            )
        encoded = bytes(raw)
        try:
            canonical = _exclusive_json_bytes(payload)
        except (TypeError, ValueError) as error:
            os.close(witness_descriptor)
            raise ModalActionJournalIntegrityError(
                f"{field} contains unsupported JSON values"
            ) from error
        if encoded != canonical:
            os.close(witness_descriptor)
            raise ModalActionJournalIntegrityError(
                f"{field} does not use canonical create-only JSON bytes"
            )
        components = tuple(logical_path.split("/"))
        if (
            not components
            or components[-1] != filename
            or any(component in {"", ".", ".."} for component in components)
        ):
            os.close(witness_descriptor)
            raise ModalActionJournalIntegrityError(
                f"{field} has an unsafe logical path"
            )
        witness_key = (components[:-1], filename)
        if witness_key in self._file_witnesses:
            os.close(witness_descriptor)
            raise ModalActionJournalIntegrityError(
                f"{field} was discovered more than once"
            )
        self._file_witnesses[witness_key] = _FileWitness(
            parent_parts=components[:-1],
            filename=filename,
            descriptor=witness_descriptor,
            identity=_file_identity(after),
        )
        return ModalJournalRecord(
            binding=ModalJournalFileBinding(
                path=logical_path,
                sha256=hashlib.sha256(encoded).hexdigest(),
                size_bytes=len(encoded),
            ),
            payload=payload,
        )

    def read_bound_file(
        self,
        logical_path: str,
        *,
        field: str,
        optional: bool,
        maximum_bytes: int,
        required_mode: int | None = None,
    ) -> tuple[bytes, ModalJournalFileBinding] | None:
        """Read arbitrary immutable evidence bytes through the held namespace."""

        try:
            selected = safe_relative_path(logical_path)
        except (TypeError, ValueError) as error:
            raise ModalActionJournalIntegrityError(
                f"{field} has an unsafe logical path"
            ) from error
        if selected.as_posix() != logical_path or maximum_bytes <= 0:
            raise ModalActionJournalIntegrityError(
                f"{field} has an unsafe logical path or byte limit"
            )
        parent_parts = tuple(selected.parent.parts)
        parent = self.open_directory(
            parent_parts,
            field=f"{field} parent",
            optional=optional,
        )
        if parent is None:
            return None
        filename = selected.name
        try:
            try:
                before = os.stat(
                    filename,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if optional:
                    return None
                raise ModalActionJournalIntegrityError(f"{field} is missing") from None
            if (
                stat.S_ISLNK(before.st_mode)
                or not stat.S_ISREG(before.st_mode)
                or before.st_uid != os.getuid()
                or before.st_nlink != 1
                or before.st_size > maximum_bytes
                or (
                    required_mode is not None
                    and stat.S_IMODE(before.st_mode) != required_mode
                )
            ):
                raise ModalActionJournalIntegrityError(
                    f"{field} must be an owned single-link bounded regular file"
                )
            flags = (
                os.O_RDONLY
                | os.O_CLOEXEC
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                descriptor = os.open(filename, flags, dir_fd=parent)
            except OSError as error:
                raise ModalActionJournalIntegrityError(
                    f"{field} changed while opening"
                ) from error
            try:
                opened = os.fstat(descriptor)
                if _file_identity(opened) != _file_identity(before):
                    raise ModalActionJournalIntegrityError(
                        f"{field} changed while opening"
                    )
                raw = bytearray()
                while len(raw) <= maximum_bytes:
                    try:
                        chunk = os.read(
                            descriptor,
                            min(64 * 1024, maximum_bytes + 1 - len(raw)),
                        )
                    except InterruptedError:
                        continue
                    if not chunk:
                        break
                    raw.extend(chunk)
                if len(raw) > maximum_bytes:
                    raise ModalActionJournalIntegrityError(
                        f"{field} exceeds its size limit"
                    )
                after = os.fstat(descriptor)
                if (
                    _file_identity(after) != _file_identity(opened)
                    or len(raw) != after.st_size
                ):
                    raise ModalActionJournalIntegrityError(
                        f"{field} changed while reading"
                    )
                witness_descriptor = os.dup(descriptor)
            except OSError as error:
                raise ModalActionJournalIntegrityError(
                    f"{field} changed while reading"
                ) from error
            finally:
                os.close(descriptor)
            try:
                rebound = os.stat(
                    filename,
                    dir_fd=parent,
                    follow_symlinks=False,
                )
                if _file_identity(rebound) != _file_identity(before):
                    raise ModalActionJournalIntegrityError(
                        f"{field} changed after reading"
                    )
            except OSError as error:
                os.close(witness_descriptor)
                raise ModalActionJournalIntegrityError(
                    f"{field} changed after reading"
                ) from error
            witness_key = (parent_parts, filename)
            if witness_key in self._file_witnesses:
                os.close(witness_descriptor)
                raise ModalActionJournalIntegrityError(
                    f"{field} was discovered more than once"
                )
            self._file_witnesses[witness_key] = _FileWitness(
                parent_parts=parent_parts,
                filename=filename,
                descriptor=witness_descriptor,
                identity=_file_identity(after),
            )
            encoded = bytes(raw)
            return encoded, ModalJournalFileBinding(
                path=logical_path,
                sha256=hashlib.sha256(encoded).hexdigest(),
                size_bytes=len(encoded),
            )
        finally:
            os.close(parent)

    def _reopen_directory(self, parts: tuple[str, ...]) -> int:
        descriptor = os.dup(self.root_descriptor)
        try:
            for component in parts:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(before.st_mode):
                    raise ModalActionJournalIntegrityError(
                        "journal directory became a symlink"
                    )
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
                opened = os.fstat(child)
                if _directory_identity(before) != _directory_identity(opened):
                    os.close(child)
                    raise ModalActionJournalIntegrityError(
                        "journal directory changed while reopening"
                    )
                os.close(descriptor)
                descriptor = child
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def require_stable(self) -> None:
        for parts, witness in self._witnesses.items():
            current = os.fstat(witness.descriptor)
            if (
                _directory_identity(current) != witness.identity
                or tuple(sorted(os.listdir(witness.descriptor))) != witness.names
            ):
                raise ModalActionJournalIntegrityError(
                    "journal directory changed during the global scan"
                )
            try:
                reopened = self._reopen_directory(parts)
            except OSError as error:
                raise ModalActionJournalIntegrityError(
                    "journal directory path changed during the global scan"
                ) from error
            try:
                if _directory_identity(os.fstat(reopened)) != witness.identity:
                    raise ModalActionJournalIntegrityError(
                        "journal directory path was replaced during the global scan"
                    )
            finally:
                os.close(reopened)
        for witness in self._file_witnesses.values():
            if _file_identity(os.fstat(witness.descriptor)) != witness.identity:
                raise ModalActionJournalIntegrityError(
                    "journal file changed during the global scan"
                )
            try:
                parent = self._reopen_directory(witness.parent_parts)
                try:
                    rebound = os.stat(
                        witness.filename,
                        dir_fd=parent,
                        follow_symlinks=False,
                    )
                finally:
                    os.close(parent)
            except OSError as error:
                raise ModalActionJournalIntegrityError(
                    "journal file path changed during the global scan"
                ) from error
            if _file_identity(rebound) != witness.identity:
                raise ModalActionJournalIntegrityError(
                    "journal file path changed during the global scan"
                )
        for parts in self._absent_directories:
            try:
                descriptor = self._reopen_directory(parts)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise ModalActionJournalIntegrityError(
                    "absent journal namespace changed during the global scan"
                ) from error
            else:
                os.close(descriptor)
                raise ModalActionJournalIntegrityError(
                    "journal namespace appeared during the global scan"
                )

    def close(self) -> None:
        for witness in self._file_witnesses.values():
            os.close(witness.descriptor)
        self._file_witnesses.clear()
        for witness in self._witnesses.values():
            os.close(witness.descriptor)
        self._witnesses.clear()
        os.close(self.root_descriptor)


def _logical(parts: Sequence[str], filename: str | None = None) -> str:
    selected = [*parts]
    if filename is not None:
        selected.append(filename)
    return "/".join(selected)


def _stat_names_as_directories(
    descriptor: int,
    names: Sequence[str],
    *,
    field: str,
) -> None:
    for name in names:
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except OSError as error:
            raise ModalActionJournalIntegrityError(
                f"{field} changed during discovery"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ModalActionJournalIntegrityError(
                f"{field} contains a non-directory entry"
            )


def _scan_action_attempt_directory(
    reader: _NamespaceReader,
    identity: ModalLiveCohortIdentity,
    descriptor: int,
) -> tuple[
    tuple[ModalJournalRecord, ...],
    tuple[ModalJournalRecord, ...],
    tuple[ModalJournalRecord, ...],
]:
    parts = tuple(modal_action_attempt_directory(identity).parts)
    names = tuple(sorted(os.listdir(descriptor)))
    intents: list[ModalJournalRecord] = []
    terminals: list[ModalJournalRecord] = []
    aggregates: list[ModalJournalRecord] = []
    for filename in names:
        intent_match = _INTENT_NAME.fullmatch(filename)
        terminal_match = _TERMINAL_NAME.fullmatch(filename)
        aggregate_match = _AGGREGATE_NAME.fullmatch(filename)
        matches = sum(
            match is not None
            for match in (intent_match, terminal_match, aggregate_match)
        )
        if matches != 1:
            raise ModalActionJournalIntegrityError(
                "cohort action journal contains an unsupported filename"
            )
        attempt_id = (intent_match or terminal_match or aggregate_match).group(1)
        record = reader.read_json_leaf(
            descriptor,
            filename,
            logical_path=_logical(parts, filename),
            field=f"cohort action journal {attempt_id}",
        )
        if intent_match is not None:
            expected = modal_action_intent_receipt_path(
                identity,
                attempt_id,
            ).as_posix()
            if record.binding.path != expected:
                raise ModalActionJournalIntegrityError(
                    "cohort intent path is not canonical"
                )
            _validate_action_intent_core(
                record.payload,
                attempt_id=attempt_id,
                identity=identity,
            )
            intents.append(record)
        elif terminal_match is not None:
            expected = modal_action_terminal_receipt_path(
                identity,
                attempt_id,
            ).as_posix()
            if record.binding.path != expected:
                raise ModalActionJournalIntegrityError(
                    "cohort terminal path is not canonical"
                )
            _validate_action_terminal_core(
                record.payload,
                attempt_id=attempt_id,
                location="cohort",
                path_identity=identity,
            )
            terminals.append(record)
        else:
            try:
                validate_provider_canary_aggregate_outcome_receipt(
                    record.payload,
                    expected_attempt_id=attempt_id,
                    expected_source_tree_sha256=identity.source_tree_sha256,
                    expected_image_source_sha256=identity.image_source_sha256,
                    expected_cohort_id=identity.cohort_id,
                )
            except (TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    "provider aggregate receipt is invalid"
                ) from error
            aggregates.append(record)
    return tuple(intents), tuple(terminals), tuple(aggregates)


def _recovery_name(
    filename: str,
) -> tuple[str, Literal["intent", "host_containment", "resolution"]]:
    matches: list[tuple[str, Literal["intent", "host_containment", "resolution"]]] = []
    for pattern, kind in _RECOVERY_NAMES:
        match = pattern.fullmatch(filename)
        if match is not None:
            matches.append((match.group(1), kind))
    if len(matches) != 1:
        raise ModalActionJournalIntegrityError(
            "cohort recovery journal contains an unsupported filename"
        )
    return matches[0]


def _scan_recovery_directory(
    reader: _NamespaceReader,
    identity: ModalLiveCohortIdentity,
    descriptor: int,
) -> tuple[ModalRecoveryJournalRecord, ...]:
    parts = tuple(modal_action_recovery_directory(identity).parts)
    records: list[ModalRecoveryJournalRecord] = []
    seen: set[tuple[str, str]] = set()
    for filename in sorted(os.listdir(descriptor)):
        attempt_id, kind = _recovery_name(filename)
        expected_paths = {
            "intent": modal_action_recovery_intent_path(identity, attempt_id),
            "host_containment": modal_action_host_containment_path(
                identity,
                attempt_id,
            ),
            "resolution": modal_action_recovery_resolution_path(
                identity,
                attempt_id,
            ),
        }
        logical_path = _logical(parts, filename)
        if logical_path != expected_paths[kind].as_posix():
            raise ModalActionJournalIntegrityError(
                "cohort recovery path is not canonical"
            )
        key = (attempt_id, kind)
        if key in seen:
            raise ModalActionJournalIntegrityError(
                "cohort recovery journal duplicates a receipt kind"
            )
        seen.add(key)
        record = reader.read_json_leaf(
            descriptor,
            filename,
            logical_path=logical_path,
            field=f"cohort recovery journal {attempt_id}",
        )
        validators = {
            "intent": _validate_recovery_intent_core,
            "host_containment": _validate_recovery_host_core,
            "resolution": _validate_recovery_resolution_core,
        }
        validators[kind](
            record.payload,
            attempt_id=attempt_id,
            identity=identity,
        )
        records.append(
            ModalRecoveryJournalRecord(
                attempt_id=attempt_id,
                kind=kind,
                record=record,
            )
        )
    return tuple(records)


def _scan_cohort_journals(
    reader: _NamespaceReader,
) -> tuple[ModalCohortActionJournal, ...]:
    live_parts = tuple(MODAL_LIVE_COHORT_ROOT.parts)
    live_descriptor = reader.open_directory(
        live_parts,
        field="Modal live-cohort root",
        optional=True,
    )
    if live_descriptor is None:
        return ()
    cohorts: list[ModalCohortActionJournal] = []
    try:
        source_names = tuple(sorted(os.listdir(live_descriptor)))
        if any(_SHA256.fullmatch(name) is None for name in source_names):
            raise ModalActionJournalIntegrityError(
                "Modal live-cohort source namespace is not canonical"
            )
        _stat_names_as_directories(
            live_descriptor,
            source_names,
            field="Modal live-cohort source namespace",
        )
        for source_sha256 in source_names:
            source_parts = (*live_parts, source_sha256)
            source_descriptor = reader.open_directory(
                source_parts,
                field="Modal live-cohort source directory",
                optional=False,
            )
            if source_descriptor is None:  # pragma: no cover - required above
                raise AssertionError("required source directory is absent")
            try:
                image_names = tuple(sorted(os.listdir(source_descriptor)))
                if any(_SHA256.fullmatch(name) is None for name in image_names):
                    raise ModalActionJournalIntegrityError(
                        "Modal live-cohort image namespace is not canonical"
                    )
                _stat_names_as_directories(
                    source_descriptor,
                    image_names,
                    field="Modal live-cohort image namespace",
                )
                for image_sha256 in image_names:
                    image_parts = (*source_parts, image_sha256)
                    image_descriptor = reader.open_directory(
                        image_parts,
                        field="Modal live-cohort image directory",
                        optional=False,
                    )
                    if image_descriptor is None:  # pragma: no cover
                        raise AssertionError("required image directory is absent")
                    try:
                        cohort_names = tuple(sorted(os.listdir(image_descriptor)))
                        _stat_names_as_directories(
                            image_descriptor,
                            cohort_names,
                            field="Modal live-cohort identity namespace",
                        )
                        for cohort_name in cohort_names:
                            try:
                                cohort_id = validate_run_id(cohort_name)
                            except (TypeError, ValueError) as error:
                                raise ModalActionJournalIntegrityError(
                                    "Modal live-cohort ID is not canonical"
                                ) from error
                            identity = ModalLiveCohortIdentity(
                                source_tree_sha256=source_sha256,
                                image_source_sha256=image_sha256,
                                cohort_id=cohort_id,
                            )
                            cohort_parts = (*image_parts, cohort_name)
                            cohort_descriptor = reader.open_directory(
                                cohort_parts,
                                field="Modal live-cohort directory",
                                optional=False,
                            )
                            if cohort_descriptor is None:  # pragma: no cover
                                raise AssertionError(
                                    "required cohort directory is absent"
                                )
                            try:
                                leaf_names = tuple(
                                    sorted(os.listdir(cohort_descriptor))
                                )
                                seal_name = modal_migration_lineage_path(identity).name
                                seal: ModalJournalRecord | None = None
                                if seal_name in leaf_names:
                                    seal = reader.read_json_leaf(
                                        cohort_descriptor,
                                        seal_name,
                                        logical_path=modal_migration_lineage_path(
                                            identity
                                        ).as_posix(),
                                        field="migration terminal seal",
                                    )
                                    _validate_migration_terminal_seal_core(
                                        seal.payload,
                                        identity=identity,
                                    )

                                action_parts = tuple(
                                    modal_action_attempt_directory(identity).parts
                                )
                                action_descriptor = reader.open_directory(
                                    action_parts,
                                    field="cohort action-attempt journal",
                                    optional=True,
                                )
                                if action_descriptor is None:
                                    intents: tuple[ModalJournalRecord, ...] = ()
                                    terminals: tuple[ModalJournalRecord, ...] = ()
                                    aggregates: tuple[ModalJournalRecord, ...] = ()
                                else:
                                    try:
                                        intents, terminals, aggregates = (
                                            _scan_action_attempt_directory(
                                                reader,
                                                identity,
                                                action_descriptor,
                                            )
                                        )
                                    finally:
                                        os.close(action_descriptor)

                                recovery_parts = tuple(
                                    modal_action_recovery_directory(identity).parts
                                )
                                recovery_descriptor = reader.open_directory(
                                    recovery_parts,
                                    field="cohort action-recovery journal",
                                    optional=True,
                                )
                                if recovery_descriptor is None:
                                    recoveries: tuple[
                                        ModalRecoveryJournalRecord, ...
                                    ] = ()
                                else:
                                    try:
                                        if seal is not None and os.listdir(
                                            recovery_descriptor
                                        ):
                                            raise ModalActionJournalIntegrityError(
                                        "sealed cohort contains recovery "
                                        "content"
                                            )
                                        recoveries = _scan_recovery_directory(
                                            reader,
                                            identity,
                                            recovery_descriptor,
                                        )
                                    finally:
                                        os.close(recovery_descriptor)
                                cohorts.append(
                                    ModalCohortActionJournal(
                                        identity=identity,
                                        migration_terminal_seal=seal,
                                        intents=intents,
                                        terminals=terminals,
                                        aggregates=aggregates,
                                        recoveries=recoveries,
                                    )
                                )
                            finally:
                                os.close(cohort_descriptor)
                    finally:
                        os.close(image_descriptor)
            finally:
                os.close(source_descriptor)
    finally:
        os.close(live_descriptor)
    return tuple(cohorts)


def _scan_global_reservations(
    reader: _NamespaceReader,
) -> tuple[ModalJournalRecord, ...]:
    parts = tuple(MODAL_REMOTE_RUN_RESERVATION_ROOT.parts)
    descriptor = reader.open_directory(
        parts,
        field="global remote-run reservation root",
        optional=True,
    )
    if descriptor is None:
        return ()
    records: list[ModalJournalRecord] = []
    try:
        for filename in sorted(os.listdir(descriptor)):
            if not filename.endswith(".json"):
                raise ModalActionJournalIntegrityError(
                    "global reservation namespace contains an unsupported filename"
                )
            try:
                run_id = validate_run_id(filename.removesuffix(".json"))
            except (TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    "global reservation filename is invalid"
                ) from error
            logical = modal_remote_run_reservation_path(run_id).as_posix()
            if logical != _logical(parts, filename):
                raise ModalActionJournalIntegrityError(
                    "global reservation path is not canonical"
                )
            record = reader.read_json_leaf(
                descriptor,
                filename,
                logical_path=logical,
                field=f"global reservation {run_id}",
            )
            _validate_reservation_core(record.payload, run_id=run_id)
            records.append(record)
    finally:
        os.close(descriptor)
    return tuple(records)


def _scan_global_rejections(
    reader: _NamespaceReader,
) -> tuple[tuple[ModalJournalRecord, ...], ModalJournalRecord | None]:
    parts = tuple(MODAL_LAUNCH_REJECTION_ROOT.parts)
    descriptor = reader.open_directory(
        parts,
        field="global launch-rejection root",
        optional=True,
    )
    if descriptor is None:
        return (), None
    records: list[ModalJournalRecord] = []
    seal: ModalJournalRecord | None = None
    try:
        for filename in sorted(os.listdir(descriptor)):
            if filename == modal_global_launch_rejection_seal_path().name:
                logical = modal_global_launch_rejection_seal_path().as_posix()
                seal = reader.read_json_leaf(
                    descriptor,
                    filename,
                    logical_path=logical,
                    field="global launch-rejection seal",
                )
                _validate_global_launch_rejection_seal_core(seal.payload)
                continue
            match = _TERMINAL_NAME.fullmatch(filename)
            if match is None:
                raise ModalActionJournalIntegrityError(
                    "global rejection namespace contains an unsupported filename"
                )
            attempt_id = match.group(1)
            logical = modal_launch_rejection_receipt_path(attempt_id).as_posix()
            if logical != _logical(parts, filename):
                raise ModalActionJournalIntegrityError(
                    "global rejection path is not canonical"
                )
            record = reader.read_json_leaf(
                descriptor,
                filename,
                logical_path=logical,
                field=f"global launch rejection {attempt_id}",
            )
            _validate_action_terminal_core(
                record.payload,
                attempt_id=attempt_id,
                location="global_rejection",
                path_identity=None,
            )
            records.append(record)
    finally:
        os.close(descriptor)
    return tuple(records), seal


def _scan_private_process_markers(
    reader: _NamespaceReader,
) -> tuple[ModalJournalRecord, ...]:
    containment_parts = tuple(MODAL_LOCAL_PROCESS_START_ROOT.parent.parts)
    containment = reader.open_directory(
        containment_parts,
        field="private local-containment root",
        optional=True,
        required_mode=0o700,
    )
    if containment is None:
        return ()
    os.close(containment)
    parts = tuple(MODAL_LOCAL_PROCESS_START_ROOT.parts)
    descriptor = reader.open_directory(
        parts,
        field="private process-start root",
        optional=True,
        required_mode=0o700,
    )
    if descriptor is None:
        return ()
    records: list[ModalJournalRecord] = []
    try:
        for filename in sorted(os.listdir(descriptor)):
            match = _TERMINAL_NAME.fullmatch(filename)
            if match is None:
                raise ModalActionJournalIntegrityError(
                    "process-marker namespace contains an unsupported filename"
                )
            attempt_id = match.group(1)
            logical = modal_local_process_start_receipt_path(attempt_id).as_posix()
            if logical != _logical(parts, filename):
                raise ModalActionJournalIntegrityError(
                    "process-marker path is not canonical"
                )
            record = reader.read_json_leaf(
                descriptor,
                filename,
                logical_path=logical,
                field=f"process-start marker {attempt_id}",
            )
            _validate_process_marker_core(
                record.payload,
                attempt_id=attempt_id,
            )
            records.append(record)
    finally:
        os.close(descriptor)
    return tuple(records)


@dataclass(slots=True)
class _AttemptAccumulator:
    attempt_id: str
    identity: ModalLiveCohortIdentity | None = None
    intent: ModalJournalRecord | None = None
    terminal: ModalJournalRecord | None = None
    rejection: ModalJournalRecord | None = None
    reservations: list[ModalJournalRecord] = dataclass_field(default_factory=list)
    marker: ModalJournalRecord | None = None
    aggregates: list[ModalJournalRecord] = dataclass_field(default_factory=list)
    recoveries: list[ModalRecoveryJournalRecord] = dataclass_field(default_factory=list)


def _accumulator(
    attempts: dict[str, _AttemptAccumulator],
    attempt_id: str,
) -> _AttemptAccumulator:
    selected = _attempt_id(attempt_id, "journal.attempt_id")
    return attempts.setdefault(selected, _AttemptAccumulator(selected))


def _bind_attempt_identity(
    attempt: _AttemptAccumulator,
    identity: ModalLiveCohortIdentity | None,
    *,
    field: str,
) -> None:
    if identity is None:
        return
    if attempt.identity is not None and attempt.identity != identity:
        raise ModalActionJournalIntegrityError(
            f"attempt {attempt.attempt_id} has conflicting {field} cohort identities"
        )
    attempt.identity = identity


def _single_record(
    current: ModalJournalRecord | None,
    replacement: ModalJournalRecord,
    *,
    field: str,
) -> ModalJournalRecord:
    if current is not None:
        raise ModalActionJournalIntegrityError(
            f"attempt appears in multiple {field} locations"
        )
    return replacement


def _collect_attempts(
    cohorts: Sequence[ModalCohortActionJournal],
    reservations: Sequence[ModalJournalRecord],
    rejections: Sequence[ModalJournalRecord],
    markers: Sequence[ModalJournalRecord],
) -> dict[str, _AttemptAccumulator]:
    attempts: dict[str, _AttemptAccumulator] = {}
    for cohort in cohorts:
        identity = cohort.identity
        for record in cohort.intents:
            attempt_id = record.payload["attempt_id"]
            attempt = _accumulator(attempts, attempt_id)
            _bind_attempt_identity(attempt, identity, field="intent")
            attempt.intent = _single_record(
                attempt.intent,
                record,
                field="intent",
            )
        for record in cohort.terminals:
            attempt_id = record.payload["attempt_id"]
            attempt = _accumulator(attempts, attempt_id)
            _bind_attempt_identity(attempt, identity, field="terminal")
            attempt.terminal = _single_record(
                attempt.terminal,
                record,
                field="terminal",
            )
        for record in cohort.aggregates:
            attempt_id = record.payload["attempt_id"]
            attempt = _accumulator(attempts, attempt_id)
            _bind_attempt_identity(attempt, identity, field="aggregate")
            attempt.aggregates.append(record)
        for recovery in cohort.recoveries:
            attempt = _accumulator(attempts, recovery.attempt_id)
            _bind_attempt_identity(attempt, identity, field="recovery")
            attempt.recoveries.append(recovery)
    for record in reservations:
        attempt_id = record.payload["owner_attempt_id"]
        attempt = _accumulator(attempts, attempt_id)
        identity = _identity_from_payload(
            record.payload,
            image_field="image_source_sha256",
            allow_absent=False,
            field="reservation",
        )
        _bind_attempt_identity(attempt, identity, field="reservation")
        attempt.reservations.append(record)
    for record in rejections:
        attempt_id = record.payload["attempt_id"]
        attempt = _accumulator(attempts, attempt_id)
        if attempt.intent is not None or attempt.terminal is not None:
            raise ModalActionJournalIntegrityError(
                "attempt is present in both cohort and global terminal namespaces"
            )
        identity = _identity_from_payload(
            record.payload,
            image_field="approved_image_source_sha256",
            allow_absent=True,
            allow_partial=not record.payload["remote_run_reservations"],
            field="global_rejection",
        )
        _bind_attempt_identity(attempt, identity, field="global rejection")
        attempt.rejection = _single_record(
            attempt.rejection,
            record,
            field="global rejection",
        )
    for record in markers:
        attempt_id = record.payload["attempt_id"]
        attempt = _accumulator(attempts, attempt_id)
        identity = _identity_from_payload(
            record.payload,
            image_field="image_source_sha256",
            allow_absent=False,
            field="process_marker",
        )
        _bind_attempt_identity(attempt, identity, field="process marker")
        attempt.marker = _single_record(
            attempt.marker,
            record,
            field="process marker",
        )
    return attempts


def _reservation_specs_for_claim(
    payload: Mapping[str, Any],
    *,
    identity: ModalLiveCohortIdentity,
    timestamp_field: Literal["created_at_utc", "started_at_utc"],
) -> tuple[ModalRemoteRunReservationSpec, ...]:
    return build_modal_remote_run_reservation_specs(
        concrete_remote_run_ids=payload["concrete_remote_run_ids"],
        attempt_id=payload["attempt_id"],
        action=payload["action"],
        identity=identity,
        created_at_utc=payload[timestamp_field],
        launch_capability_sha256=payload["launch_capability_sha256"],
        local_host_anchor_path=payload["local_host_anchor_path"],
        local_host_anchor_sha256=payload["local_host_anchor_sha256"],
        local_boot_started_at_unix_microseconds=payload[
            "local_boot_started_at_unix_microseconds"
        ],
        local_boot_session_sha256=payload["local_boot_session_sha256"],
    )


def _validate_claim_reservation_specs(
    payload: Mapping[str, Any],
    specs: Sequence[ModalRemoteRunReservationSpec],
    *,
    field: str,
) -> None:
    expected_bindings = [dict(spec.binding) for spec in specs]
    if payload["remote_run_reservations"] != expected_bindings:
        raise ModalActionJournalIntegrityError(
            f"{field} reservation binding roster is not canonical"
        )


def _validate_actual_reservation(
    record: ModalJournalRecord,
    spec: ModalRemoteRunReservationSpec,
) -> None:
    if (
        record.binding.path != spec.binding["path"]
        or record.binding.sha256 != spec.binding["sha256"]
        or dict(record.payload) != dict(spec.payload)
    ):
        raise ModalActionJournalIntegrityError(
            "global reservation differs from its canonical owner binding"
        )


def _canary_aggregate_prefix(remote_run_id: str) -> str:
    matches = [
        remote_run_id.removesuffix(f"-{suffix}")
        for suffix in _CANARY_SUFFIXES.values()
        if remote_run_id.endswith(f"-{suffix}")
    ]
    if len(matches) != 1:
        raise ModalActionJournalIntegrityError(
            "aggregate reservation run ID lacks one exact canary suffix"
        )
    try:
        return validate_run_id(matches[0])
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "aggregate reservation prefix is invalid"
        ) from error


def _validate_reservation_only_group(
    attempt: _AttemptAccumulator,
) -> None:
    records = sorted(
        attempt.reservations,
        key=lambda item: item.payload["remote_run_id"],
    )
    if not records:
        return
    first = records[0].payload
    common_fields = (
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
    )
    if any(
        any(record.payload[name] != first[name] for name in common_fields)
        for record in records[1:]
    ):
        raise ModalActionJournalIntegrityError(
            "reservation-only attempt has conflicting ownership fields"
        )
    action = first["action"]
    if action == "canaries":
        prefixes = {
            _canary_aggregate_prefix(record.payload["remote_run_id"])
            for record in records
        }
        if len(prefixes) != 1:
            raise ModalActionJournalIntegrityError(
                "reservation-only aggregate has conflicting run prefixes"
            )
        prefix = next(iter(prefixes))
        expected_runs = expected_modal_concrete_run_ids(
            action="canaries",
            run_id=prefix,
            verifier_run_id=None,
        )
        observed_runs = {record.payload["remote_run_id"] for record in records}
        if not observed_runs <= set(expected_runs):
            raise ModalActionJournalIntegrityError(
                "reservation-only aggregate contains a foreign sibling"
            )
        identity = _identity_from_payload(
            first,
            image_field="image_source_sha256",
            allow_absent=False,
            field="reservation_only",
        )
        if identity is None:  # pragma: no cover - guarded above
            raise AssertionError("reservation-only group lacks identity")
        specs = build_modal_remote_run_reservation_specs(
            concrete_remote_run_ids=expected_runs,
            attempt_id=attempt.attempt_id,
            action="canaries",
            identity=identity,
            created_at_utc=first["created_at_utc"],
            launch_capability_sha256=first["launch_capability_sha256"],
            local_host_anchor_path=first["local_host_anchor_path"],
            local_host_anchor_sha256=first["local_host_anchor_sha256"],
            local_boot_started_at_unix_microseconds=first[
                "local_boot_started_at_unix_microseconds"
            ],
            local_boot_session_sha256=first["local_boot_session_sha256"],
        )
        specs_by_run = {spec.binding["run_id"]: spec for spec in specs}
        for record in records:
            _validate_actual_reservation(
                record,
                specs_by_run[record.payload["remote_run_id"]],
            )
    elif len(records) != 1:
        raise ModalActionJournalIntegrityError(
            "non-aggregate reservation-only attempt owns multiple run IDs"
        )


def _validate_intent_terminal_pair(attempt: _AttemptAccumulator) -> None:
    if attempt.intent is None or attempt.terminal is None:
        return
    intent = attempt.intent.payload
    terminal = attempt.terminal.payload
    if intent["created_at_utc"] != terminal["started_at_utc"] or any(
        intent[field] != terminal[field] for field in _INTENT_TERMINAL_SHARED_FIELDS
    ):
        raise ModalActionJournalIntegrityError(
            "Modal action intent and cohort terminal differ"
        )


def _validate_marker_links(attempt: _AttemptAccumulator) -> None:
    marker_record = attempt.marker
    if marker_record is None:
        return
    if attempt.intent is None:
        raise ModalActionJournalIntegrityError(
            "process marker lacks its durable cohort intent"
        )
    marker = marker_record.payload
    intent = attempt.intent.payload
    marker_created_at = _utc(
        marker["created_at_utc"],
        "process_marker.created_at_utc",
    )
    if marker_created_at < _utc(intent["created_at_utc"], "intent.created_at_utc"):
        raise ModalActionJournalIntegrityError(
            "process marker predates its durable intent"
        )
    expected = {
        "action": intent["action"],
        "run_id": intent["run_id"],
        "intent_path": attempt.intent.binding.path,
        "intent_sha256": attempt.intent.binding.sha256,
        "source_tree_sha256": intent["source_tree_sha256"],
        "image_source_sha256": intent["approved_image_source_sha256"],
        "cohort_id": intent["cohort_id"],
        "modal_command_sha256": intent["modal_command_sha256"],
        "launch_capability_sha256": intent["launch_capability_sha256"],
        "modal_cost_cap_usd": intent["modal_cost_cap_usd"],
        "provider_cost_cap_usd": intent["provider_cost_cap_usd"],
        "local_host_anchor_path": intent["local_host_anchor_path"],
        "local_host_anchor_sha256": intent["local_host_anchor_sha256"],
        "local_boot_started_at_unix_microseconds": intent[
            "local_boot_started_at_unix_microseconds"
        ],
        "local_boot_session_sha256": intent["local_boot_session_sha256"],
    }
    if any(marker[field] != value for field, value in expected.items()):
        raise ModalActionJournalIntegrityError(
            "process marker differs from its durable intent"
        )
    if attempt.terminal is not None:
        terminal = attempt.terminal.payload
        if not (
            _utc(terminal["started_at_utc"], "terminal.started_at_utc")
            <= marker_created_at
            <= _utc(terminal["finished_at_utc"], "terminal.finished_at_utc")
        ):
            raise ModalActionJournalIntegrityError(
                "process marker timestamp is outside its terminal attempt"
            )
        marker_sha256 = terminal["local_process_start_receipt_sha256"]
        if marker_sha256 is not None and (
            terminal["local_process_start_receipt_path"] != marker_record.binding.path
            or marker_sha256 != marker_record.binding.sha256
        ):
            raise ModalActionJournalIntegrityError(
                "terminal process-marker binding changed"
            )
        if marker_sha256 is not None and (
            terminal["local_process_id"] != marker["process_id"]
            or terminal["local_process_group_id"] != marker["expected_process_group_id"]
            or terminal["local_session_id"] != marker["expected_session_id"]
        ):
            raise ModalActionJournalIntegrityError(
                "terminal process identity differs from its bound marker"
            )


def _validate_aggregates(attempt: _AttemptAccumulator) -> None:
    terminal = attempt.terminal.payload if attempt.terminal is not None else None
    eligible = bool(
        terminal is not None
        and terminal["action"] == "canaries"
        and terminal["modal_cli_process_started"] is True
        and terminal["returncode"] in {0, 2}
    )
    if not attempt.aggregates:
        if eligible:
            raise ModalActionJournalIntegrityError(
                "completed aggregate terminal lacks its outcome receipt"
            )
        return
    if len(attempt.aggregates) != 1 or attempt.terminal is None:
        raise ModalActionJournalIntegrityError(
            "provider aggregate lacks one cohort terminal owner"
        )
    terminal = attempt.terminal.payload
    if not eligible:
        raise ModalActionJournalIntegrityError(
            "provider aggregate lacks one eligible completed terminal"
        )
    try:
        aggregate = validate_provider_canary_aggregate_outcome_receipt(
            attempt.aggregates[0].payload,
            expected_attempt_id=attempt.attempt_id,
            expected_run_id_prefix=terminal["run_id"],
            expected_source_tree_sha256=terminal["source_tree_sha256"],
            expected_image_source_sha256=terminal["approved_image_source_sha256"],
            expected_cohort_id=terminal["cohort_id"],
        )
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "provider aggregate differs from its cohort terminal"
        ) from error
    expected_all_succeeded = terminal["returncode"] == 0
    if aggregate["all_succeeded"] is not expected_all_succeeded or terminal[
        "status"
    ] != ("succeeded" if expected_all_succeeded else "failed"):
        raise ModalActionJournalIntegrityError(
            "provider aggregate and terminal statuses differ"
        )


def _probe_process_marker(
    project_root: Path,
    marker: ModalJournalRecord,
    process_probe: ProcessProbe | None,
) -> str | None:
    if process_probe is None:
        return None
    result = process_probe(
        project_root,
        marker.binding.path,
        marker.binding.sha256,
    )
    if result not in _PROCESS_PROBE_RESULTS:
        raise ModalActionJournalIntegrityError(
            "process probe returned an unsupported disposition"
        )
    return result


def _add_blocker(
    blockers: list[ModalActionJournalBlocker],
    codes: list[str],
    *,
    code: str,
    attempt: _AttemptAccumulator,
    paths: Sequence[str],
) -> None:
    if code in codes:
        return
    codes.append(code)
    blockers.append(
        ModalActionJournalBlocker(
            code=code,
            attempt_id=attempt.attempt_id,
            identity=attempt.identity,
            paths=tuple(sorted(set(paths))),
        )
    )


def _expected_recovery_source_evidence(
    attempt: _AttemptAccumulator,
) -> dict[str, Any]:
    return {
        "action_intent": (
            _record_binding_dict(attempt.intent)
            if attempt.intent is not None
            else None
        ),
        "action_terminal": (
            _record_binding_dict(attempt.terminal)
            if attempt.terminal is not None
            else None
        ),
        "global_rejection": (
            _record_binding_dict(attempt.rejection)
            if attempt.rejection is not None
            else None
        ),
        "remote_run_reservations": [
            _record_binding_dict(record)
            for record in sorted(
                attempt.reservations,
                key=lambda item: item.binding.path,
            )
        ],
        "process_marker": (
            _record_binding_dict(attempt.marker)
            if attempt.marker is not None
            else None
        ),
        "aggregate_receipts": [
            _record_binding_dict(record)
            for record in sorted(
                attempt.aggregates,
                key=lambda item: item.binding.path,
            )
        ],
    }


def _recovery_payload_by_kind(
    attempt: _AttemptAccumulator,
) -> dict[str, ModalRecoveryJournalRecord]:
    records: dict[str, ModalRecoveryJournalRecord] = {}
    for item in attempt.recoveries:
        if item.kind in records:
            raise ModalActionJournalIntegrityError(
                "attempt has duplicate recovery receipt kinds"
            )
        records[item.kind] = item
    return records


def _absolute_request_snapshot_path(
    reader: _NamespaceReader,
    value: object,
    *,
    field: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ModalActionJournalIntegrityError(f"{field} must be an absolute path")
    selected = Path(os.path.abspath(value))
    if str(selected) != value:
        raise ModalActionJournalIntegrityError(f"{field} is not canonical")
    try:
        relative = selected.relative_to(reader.project_root)
        logical = safe_relative_path(relative.as_posix())
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            f"{field} must remain inside the locked project"
        ) from error
    return logical.as_posix()


def _validate_recovery_request(
    reader: _NamespaceReader,
    declared: Mapping[str, Any],
    *,
    attempt_id: str,
    expected_branch: str,
    fresh_candidate_attempt_id: str,
    snapshot_manifest: Mapping[str, Any] | None,
    cache: dict[str, tuple[bytes, ModalJournalFileBinding]],
) -> dict[str, Any]:
    binding = _recovery_binding(declared, field="recovery.request_binding")
    logical = binding["path"]
    observed = cache.get(logical)
    if observed is None:
        selected = reader.read_bound_file(
            logical,
            field="recovery operator request",
            optional=False,
            maximum_bytes=_MAX_JSON_BYTES,
            required_mode=0o600,
        )
        if selected is None:  # pragma: no cover - optional=False
            raise AssertionError("required recovery request disappeared")
        observed = selected
        cache[logical] = observed
    _declared_binding_matches(
        binding,
        observed[1],
        field="recovery operator request",
    )
    request = _bound_json_object(observed[0], field="recovery operator request")
    if set(request) != _RECOVERY_REQUEST_FIELDS or (
        request["schema_name"] != RECOVERY_REQUEST_SCHEMA_NAME
        or request["schema_version"] != RECOVERY_REQUEST_SCHEMA_VERSION
        or request["attempt_id"] != attempt_id
        or request["fresh_candidate_attempt_id"] != fresh_candidate_attempt_id
        or request["expected_branch"] != expected_branch
    ):
        raise ModalActionJournalIntegrityError(
            "recovery operator request has an invalid exact contract"
        )
    _attempt_id(request["attempt_id"], "recovery_request.attempt_id")
    _attempt_id(
        request["fresh_candidate_attempt_id"],
        "recovery_request.fresh_candidate_attempt_id",
    )
    _recovery_binding_roster(
        request["initial_reservation_bindings"],
        field="recovery_request.initial_reservation_bindings",
    )
    snapshot_logical = _absolute_request_snapshot_path(
        reader,
        request["snapshot_manifest_path"],
        field="recovery_request.snapshot_manifest_path",
    )
    expected_snapshot = snapshot_manifest["path"] if snapshot_manifest else None
    if snapshot_logical != expected_snapshot:
        raise ModalActionJournalIntegrityError(
            "recovery request snapshot selection changed"
        )
    return request


def _parse_recovery_utc_hour(value: object, *, field: str) -> datetime:
    parsed = _canonical_utc(value, field)
    if parsed.minute or parsed.second or parsed.microsecond:
        raise ModalActionJournalIntegrityError(f"{field} is not UTC-hour aligned")
    return parsed


def _snapshot_command_suffix(
    snapshot_name: str,
    *,
    billing_start: str,
    billing_end: str,
) -> tuple[str, ...]:
    commands = {
        "app_list": ("app", "list", "--env", MODAL_ENVIRONMENT_NAME, "--json"),
        "container_list": (
            "container",
            "list",
            "--env",
            MODAL_ENVIRONMENT_NAME,
            "--json",
        ),
        "endpoint_list": (
            "endpoint",
            "list",
            "--env",
            MODAL_ENVIRONMENT_NAME,
            "--json",
        ),
        "volume_list": (
            "volume",
            "list",
            "--env",
            MODAL_ENVIRONMENT_NAME,
            "--json",
        ),
        "run_directory_list": (
            "volume",
            "ls",
            "--env",
            MODAL_ENVIRONMENT_NAME,
            "--json",
            VOLUME_NAME,
            "/runs",
        ),
        "billing_report": (
            "billing",
            "report",
            "--start",
            billing_start,
            "--end",
            billing_end,
            "--resolution",
            "h",
            "--tz",
            "UTC",
            "--show-resources",
            "--json",
        ),
    }
    return commands[snapshot_name]


def _strict_bound_json_value(raw: bytes, *, field: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModalActionJournalIntegrityError(
            f"{field} is not valid UTF-8 JSON"
        ) from error


def _snapshot_text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ModalActionJournalIntegrityError(f"{field} is not valid text")
    if len(value.encode("utf-8")) > 4096:
        raise ModalActionJournalIntegrityError(f"{field} exceeds its text limit")
    return value


def _snapshot_timestamp(
    value: object,
    *,
    field: str,
    allow_naive: bool = False,
) -> datetime:
    text = _snapshot_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ModalActionJournalIntegrityError(
            f"{field} is not an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        if not allow_naive:
            raise ModalActionJournalIntegrityError(f"{field} lacks a timezone")
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validate_recovery_snapshot_rows(
    raw: bytes,
    *,
    snapshot_name: str,
    billing_start: datetime,
    billing_end: datetime,
) -> list[dict[str, Any]]:
    payload = _strict_bound_json_value(raw, field=f"recovery {snapshot_name}")
    if not isinstance(payload, list) or len(payload) > 100_000:
        raise ModalActionJournalIntegrityError(
            f"recovery {snapshot_name} must be a bounded JSON row list"
        )
    rows: list[dict[str, Any]] = []
    canonical_rows: set[str] = set()
    identifiers: set[str] = set()
    identifier_fields = {
        "app_list": "app_id",
        "container_list": "container_id",
        "endpoint_list": "endpoint_id",
        "volume_list": "name",
        "run_directory_list": "filename",
    }
    for index, raw_row in enumerate(payload):
        field = f"recovery {snapshot_name}[{index}]"
        if (
            not isinstance(raw_row, dict)
            or set(raw_row) != _RECOVERY_RAW_SNAPSHOT_FIELDS[snapshot_name]
        ):
            raise ModalActionJournalIntegrityError(
                f"{field} differs from the Modal 1.5.3 JSON schema"
            )
        row = dict(raw_row)
        for key, value in row.items():
            if snapshot_name == "app_list" and key == "stopped_at" and value is None:
                continue
            _snapshot_text(
                value,
                field=f"{field}.{key}",
                allow_empty=(
                    snapshot_name == "billing_report"
                    and key in {"description", "environment"}
                ),
            )
        identifier_name = identifier_fields.get(snapshot_name)
        if identifier_name is not None:
            identifier = row[identifier_name]
            if identifier in identifiers:
                raise ModalActionJournalIntegrityError(
                    f"recovery {snapshot_name} contains duplicate {identifier_name}"
                )
            identifiers.add(identifier)
        if snapshot_name == "app_list":
            _snapshot_timestamp(row["created_at"], field=f"{field}.created_at")
            if row["stopped_at"] is not None:
                _snapshot_timestamp(row["stopped_at"], field=f"{field}.stopped_at")
            if not row["tasks"].isdigit():
                raise ModalActionJournalIntegrityError(
                    f"{field}.tasks is not a decimal integer"
                )
        elif snapshot_name == "container_list":
            if row["start_time"] != "Pending":
                _snapshot_timestamp(
                    row["start_time"], field=f"{field}.start_time"
                )
        elif snapshot_name in {"endpoint_list", "volume_list"}:
            _snapshot_timestamp(row["created_at"], field=f"{field}.created_at")
        elif snapshot_name == "run_directory_list":
            if row["type"] not in {"dir", "fifo", "file", "link", "socket"}:
                raise ModalActionJournalIntegrityError(
                    f"{field}.type is unsupported"
                )
            if _RECOVERY_MODAL_TIMESTAMP.fullmatch(row["created_modified"]) is None:
                raise ModalActionJournalIntegrityError(
                    f"{field}.created_modified is not Modal CLI timestamp text"
                )
            try:
                datetime.strptime(
                    row["created_modified"][:16], "%Y-%m-%d %H:%M"
                )
            except ValueError as error:
                raise ModalActionJournalIntegrityError(
                    f"{field}.created_modified is invalid"
                ) from error
            if _RECOVERY_MODAL_SIZE.fullmatch(row["size"]) is None:
                raise ModalActionJournalIntegrityError(
                    f"{field}.size is not Modal CLI size text"
                )
            normalized = row["filename"].removeprefix("/").removesuffix("/")
            parts = PurePosixPath(normalized).parts
            if len(parts) != 2 or parts[0] != "runs":
                raise ModalActionJournalIntegrityError(
                    f"{field}.filename is outside /runs"
                )
            try:
                validate_run_id(parts[1])
            except (TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    f"{field}.filename has an invalid run ID"
                ) from error
        elif snapshot_name == "billing_report":
            interval = _snapshot_timestamp(
                row["interval_start"], field=f"{field}.interval_start"
            )
            if interval.minute or interval.second or interval.microsecond:
                raise ModalActionJournalIntegrityError(
                    f"{field}.interval_start is not hourly aligned"
                )
            if not billing_start <= interval < billing_end:
                raise ModalActionJournalIntegrityError(
                    f"{field}.interval_start lies outside the captured window"
                )
            _nonnegative_decimal_text(row["cost"], f"{field}.cost")
            if row["description"] == APP_NAME and (
                row["environment"] != MODAL_ENVIRONMENT_NAME
            ):
                raise ModalActionJournalIntegrityError(
                    f"{field} names the target App outside environment main"
                )
        canonical = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        if canonical in canonical_rows:
            raise ModalActionJournalIntegrityError(
                f"recovery {snapshot_name} contains a duplicate row"
            )
        canonical_rows.add(canonical)
        rows.append(row)
    return rows


def _attempt_time_window(attempt: _AttemptAccumulator) -> tuple[datetime, datetime]:
    if attempt.intent is not None:
        started = _utc(
            attempt.intent.payload["created_at_utc"],
            "intent.created_at_utc",
        )
    elif attempt.rejection is not None:
        started = _utc(
            attempt.rejection.payload["started_at_utc"],
            "global_rejection.started_at_utc",
        )
    elif attempt.reservations:
        started = min(
            _utc(record.payload["created_at_utc"], "reservation.created_at_utc")
            for record in attempt.reservations
        )
    else:  # pragma: no cover - recovery source closure guards this
        raise ModalActionJournalIntegrityError("recovery attempt has no start time")
    if attempt.terminal is not None:
        finished = _utc(
            attempt.terminal.payload["finished_at_utc"],
            "terminal.finished_at_utc",
        )
    elif attempt.rejection is not None:
        finished = _utc(
            attempt.rejection.payload["finished_at_utc"],
            "global_rejection.finished_at_utc",
        )
    else:
        finished = started
    return started.astimezone(UTC), finished.astimezone(UTC)


def _read_recovery_bound_file(
    reader: _NamespaceReader,
    declared: Mapping[str, Any],
    *,
    field: str,
    cache: dict[str, tuple[bytes, ModalJournalFileBinding]],
    maximum_bytes: int = _MAX_JSON_BYTES,
) -> tuple[bytes, ModalJournalFileBinding]:
    binding = _recovery_binding(declared, field=field)
    logical = binding["path"]
    observed = cache.get(logical)
    if observed is None:
        selected = reader.read_bound_file(
            logical,
            field=field,
            optional=False,
            maximum_bytes=maximum_bytes,
            required_mode=0o600,
        )
        if selected is None:  # pragma: no cover - optional=False
            raise AssertionError(f"required {field} disappeared")
        observed = selected
        cache[logical] = observed
    _declared_binding_matches(binding, observed[1], field=field)
    return observed


def _validate_recovery_snapshot(
    reader: _NamespaceReader,
    declared_manifest: Mapping[str, Any],
    *,
    identity: ModalLiveCohortIdentity,
    attempt: _AttemptAccumulator,
    cache: dict[str, tuple[bytes, ModalJournalFileBinding]],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    raw, manifest_binding = _read_recovery_bound_file(
        reader,
        declared_manifest,
        field="recovery snapshot manifest",
        cache=cache,
    )
    manifest = _bound_json_object(raw, field="recovery snapshot manifest")
    if set(manifest) != _RECOVERY_SNAPSHOT_MANIFEST_FIELDS or (
        manifest["schema_name"] != RECOVERY_SNAPSHOT_MANIFEST_SCHEMA_NAME
        or manifest["schema_version"] != RECOVERY_SNAPSHOT_MANIFEST_SCHEMA_VERSION
        or manifest["source_tree_sha256"] != identity.source_tree_sha256
        or manifest["image_source_sha256"] != identity.image_source_sha256
        or manifest["cohort_id"] != identity.cohort_id
        or manifest["modal_profile"] != "scalingintelligence"
        or manifest["modal_environment"] != MODAL_ENVIRONMENT_NAME
        or manifest["modal_cli_version"] != MODAL_VERSION
        or manifest["command_retry_count"] != 0
        or type(manifest["command_timeout_seconds"]) not in {int, float}
        or type(manifest["outer_timeout_seconds"]) not in {int, float}
        or manifest["command_timeout_seconds"] <= 0
        or manifest["outer_timeout_seconds"] <= 0
    ):
        raise ModalActionJournalIntegrityError(
            "recovery snapshot manifest has an invalid exact contract"
        )
    try:
        capture_id = validate_run_id(manifest["capture_id"])
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "recovery snapshot capture ID is invalid"
        ) from error
    expected_manifest_path = (
        modal_live_cohort_root(identity)
        / "resource_cleanup"
        / "snapshot_captures"
        / capture_id
        / RECOVERY_SNAPSHOT_MANIFEST_FILENAME
    ).as_posix()
    if manifest_binding.path != expected_manifest_path:
        raise ModalActionJournalIntegrityError(
            "recovery snapshot manifest path is not canonical"
        )
    billing_start = _parse_recovery_utc_hour(
        manifest["billing_window_start_utc"],
        field="recovery snapshot billing_window_start_utc",
    )
    billing_end = _parse_recovery_utc_hour(
        manifest["billing_window_end_utc"],
        field="recovery snapshot billing_window_end_utc",
    )
    if (
        billing_start >= billing_end
        or billing_end - billing_start > MAX_MODAL_BILLING_WINDOW
    ):
        raise ModalActionJournalIntegrityError(
            "recovery snapshot billing window is empty or exceeds 31 days"
        )
    manifest_started = _canonical_utc(
        manifest["started_at_utc"], "recovery snapshot started_at_utc"
    )
    manifest_finished = _canonical_utc(
        manifest["finished_at_utc"], "recovery snapshot finished_at_utc"
    )
    if (
        manifest_finished < manifest_started
        or billing_end > manifest_finished.astimezone(UTC).replace(
            minute=0,
            second=0,
            microsecond=0,
        )
    ):
        raise ModalActionJournalIntegrityError(
            "recovery snapshot timing does not prove completed billing hours"
        )
    attempt_started, attempt_finished = _attempt_time_window(attempt)
    if not (
        billing_start <= attempt_started
        and attempt_finished < billing_end
    ):
        raise ModalActionJournalIntegrityError(
            "recovery snapshot billing window does not cover the action attempt"
        )
    snapshots = manifest["snapshots"]
    if not isinstance(snapshots, dict) or set(snapshots) != set(
        RECOVERY_SNAPSHOT_NAMES
    ):
        raise ModalActionJournalIntegrityError(
            "recovery snapshot manifest does not bind the exact six-read roster"
        )
    rows_by_name: dict[str, list[dict[str, Any]]] = {}
    snapshot_bindings: dict[str, dict[str, Any]] = {}
    capture_root = PurePosixPath(expected_manifest_path).parent
    executable: str | None = None
    for name in RECOVERY_SNAPSHOT_NAMES:
        record = snapshots[name]
        if (
            not isinstance(record, dict)
            or set(record) != _RECOVERY_SNAPSHOT_RECORD_FIELDS
        ):
            raise ModalActionJournalIntegrityError(
                f"recovery snapshot {name} has an invalid exact binding schema"
            )
        binding = _recovery_binding(
            {
                "path": record["path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            },
            field=f"recovery snapshot {name}",
        )
        if binding["path"] != (capture_root / f"{name}.json").as_posix():
            raise ModalActionJournalIntegrityError(
                f"recovery snapshot {name} path is not canonical"
            )
        argv = record["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise ModalActionJournalIntegrityError(
                f"recovery snapshot {name} argv is invalid"
            )
        if executable is None:
            executable = argv[0]
        if argv[0] != executable or tuple(argv[1:]) != _snapshot_command_suffix(
            name,
            billing_start=manifest["billing_window_start_utc"],
            billing_end=manifest["billing_window_end_utc"],
        ):
            raise ModalActionJournalIntegrityError(
                f"recovery snapshot {name} command is not the frozen read-only command"
            )
        captured_at = _canonical_utc(
            record["captured_at_utc"],
            f"recovery snapshot {name}.captured_at_utc",
        )
        if not manifest_started <= captured_at <= manifest_finished:
            raise ModalActionJournalIntegrityError(
                f"recovery snapshot {name} timestamp is outside its capture"
            )
        leaf_raw, observed_binding = _read_recovery_bound_file(
            reader,
            binding,
            field=f"recovery snapshot {name}",
            cache=cache,
        )
        rows_by_name[name] = _validate_recovery_snapshot_rows(
            leaf_raw,
            snapshot_name=name,
            billing_start=billing_start,
            billing_end=billing_end,
        )
        snapshot_bindings[name] = _record_binding_dict(
            ModalJournalRecord(binding=observed_binding, payload={})
        )

    app_rows = [
        row for row in rows_by_name["app_list"] if row["description"] == APP_NAME
    ]
    target_app_ids = {row["app_id"] for row in app_rows}
    active_app_ids = sorted(
        row["app_id"]
        for row in app_rows
        if row["state"].lower() not in {"stopped", "completed"}
        or int(row["tasks"]) != 0
    )
    active_container_ids = sorted(
        row["container_id"]
        for row in rows_by_name["container_list"]
        if row["app_name"] == APP_NAME or row["app_id"] in target_app_ids
    )
    inactive_endpoint_states = {"disabled", "inactive", "stopped"}
    active_endpoint_ids = sorted(
        row["endpoint_id"]
        for row in rows_by_name["endpoint_list"]
        if (
            row["name"] == APP_NAME
            or row["name"].startswith(f"{APP_NAME}.")
            or row["created_by"] in target_app_ids
        )
        and row["status"].lower() not in inactive_endpoint_states
    )
    if active_app_ids or active_container_ids or active_endpoint_ids:
        raise ModalActionJournalIntegrityError(
            "recovery snapshot still contains active target App resources"
        )
    target_volumes = [
        row for row in rows_by_name["volume_list"] if row["name"] == VOLUME_NAME
    ]
    if len(target_volumes) != 1:
        raise ModalActionJournalIntegrityError(
            "recovery snapshot does not contain exactly one target Volume"
        )
    observed_run_ids = {
        row["filename"].removeprefix("/runs/").removesuffix("/")
        for row in rows_by_name["run_directory_list"]
    }
    concrete_run_ids = _recovery_attempt_concrete_run_ids(attempt)
    run_dispositions = [
        {
            "run_id": run_id,
            "present": run_id in observed_run_ids,
            "disposition": (
                "present_quarantined"
                if run_id in observed_run_ids
                else "absent_quarantined"
            ),
        }
        for run_id in sorted(concrete_run_ids)
    ]
    measured = sum(
        (
            _nonnegative_decimal_text(row["cost"], "recovery billing cost")
            for row in rows_by_name["billing_report"]
            if row["description"] == APP_NAME
            and row["environment"] == MODAL_ENVIRONMENT_NAME
        ),
        Decimal("0"),
    )
    owner = _recovery_attempt_owner_payload(attempt)
    reserve = _positive_decimal_text(
        owner["modal_cost_cap_usd"],
        "recovery unresolved Modal authorization reserve",
    )
    amount = format(measured, "f")
    reserve_amount = format(reserve, "f")
    conservative_amount = format(measured + reserve, "f")
    snapshot_evidence = {
        "manifest": dict(declared_manifest),
        "capture_id": capture_id,
        "snapshots": snapshot_bindings,
        "billing_window_start_utc": manifest["billing_window_start_utc"],
        "billing_window_end_utc": manifest["billing_window_end_utc"],
        "target_volume_name": VOLUME_NAME,
        "target_volume_present": True,
        "active_app_ids": active_app_ids,
        "active_container_ids": active_container_ids,
        "active_endpoint_ids": active_endpoint_ids,
    }
    modal_exposure = {
        "basis": (
            "complete_app_name_main_billing_snapshot_plus_full_local_"
            "authorization_reserve_for_unresolved_start"
        ),
        "measured_app_name_main_billing_usd": amount,
        "unresolved_compute_reserve_usd": reserve_amount,
        "conservative_app_name_main_billing_usd": conservative_amount,
        "complete_hourly_window": True,
        "local_authorization_is_platform_hard_bound": False,
        "modal_api_requests_performed": 0,
        "snapshot_requests_performed": 0,
        "billing_requests_performed": 0,
        "price_requests_performed": 0,
    }
    known_objects = {
        "app_ids": {"coverage": "partial", "ids": sorted(target_app_ids)},
        "function_ids": {"coverage": "partial", "ids": []},
        "call_ids": {"coverage": "partial", "ids": []},
        "image_ids": {"coverage": "partial", "ids": []},
    }
    return snapshot_evidence, modal_exposure, known_objects, run_dispositions


def _recovery_attempt_owner_payload(
    attempt: _AttemptAccumulator,
) -> Mapping[str, Any]:
    if attempt.intent is not None:
        return attempt.intent.payload
    if attempt.rejection is not None:
        return attempt.rejection.payload
    if attempt.reservations:
        return attempt.reservations[0].payload
    raise ModalActionJournalIntegrityError("recovery attempt lacks ownership evidence")


def _recovery_attempt_concrete_run_ids(
    attempt: _AttemptAccumulator,
) -> tuple[str, ...]:
    if attempt.intent is not None:
        return tuple(attempt.intent.payload["concrete_remote_run_ids"])
    if attempt.rejection is not None and attempt.rejection.payload[
        "concrete_remote_run_ids"
    ]:
        return tuple(attempt.rejection.payload["concrete_remote_run_ids"])
    return tuple(
        sorted(record.payload["remote_run_id"] for record in attempt.reservations)
    )


def _recovery_harness_for_run(
    owner: Mapping[str, Any],
    run_id: str,
) -> str:
    if owner["action"] == "canary":
        harness = owner["harness"]
        if harness not in CANARY_ORDER or run_id != owner["run_id"]:
            raise ModalActionJournalIntegrityError(
                "recovery provider run differs from its single-canary owner"
            )
        return harness
    if owner["action"] != "canaries":
        raise ModalActionJournalIntegrityError(
            "recovery provider ledger belongs to a non-provider action"
        )
    matches = [
        harness
        for harness in CANARY_ORDER
        if run_id == f"{owner['run_id']}-{_CANARY_SUFFIXES[harness]}"
    ]
    if len(matches) != 1:
        raise ModalActionJournalIntegrityError(
            "recovery provider aggregate child is ambiguous"
        )
    return matches[0]


def _validate_recovery_price_basis(
    reader: _NamespaceReader,
    owner: Mapping[str, Any],
    *,
    cache: dict[str, tuple[bytes, ModalJournalFileBinding]],
) -> tuple[dict[str, Any], ModalJournalFileBinding, Decimal, Decimal, Decimal]:
    logical = owner["provider_price_basis_path"]
    if not isinstance(logical, str):
        raise ModalActionJournalIntegrityError(
            "recovery exact provider usage lacks a price-basis path"
        )
    selected = reader.read_bound_file(
        logical,
        field="recovery provider price basis",
        optional=False,
        maximum_bytes=_MAX_JSON_BYTES,
        required_mode=0o600,
    )
    if selected is None:  # pragma: no cover - optional=False
        raise AssertionError("required provider price basis disappeared")
    cache[logical] = selected
    raw, binding = selected
    if binding.sha256 != owner["provider_price_basis_sha256"]:
        raise ModalActionJournalIntegrityError(
            "recovery provider price-basis bytes differ from the action intent"
        )
    payload = _bound_json_object(raw, field="recovery provider price basis")
    if set(payload) != _RECOVERY_PRICE_BASIS_FIELDS or (
        payload["schema_name"] != "ProviderPriceBasis"
        or payload["schema_version"] != "1.0"
        or payload["model"] != TARGET_MODEL
        or not isinstance(payload["official_source_url"], str)
        or re.fullmatch(
            r"https://(?:platform\.)?openai\.com/[^\s]*",
            payload["official_source_url"],
        )
        is None
    ):
        raise ModalActionJournalIntegrityError(
            "recovery provider price basis has an invalid exact contract"
        )
    _canonical_utc(
        payload["retrieved_at_utc"],
        "recovery provider price_basis.retrieved_at_utc",
    )
    input_rate = _nonnegative_decimal_text(
        payload["uncached_input_usd_per_million_tokens"],
        "recovery provider price_basis.input_rate",
    )
    output_rate = _nonnegative_decimal_text(
        payload["output_usd_per_million_tokens"],
        "recovery provider price_basis.output_rate",
    )
    request_fee = _nonnegative_decimal_text(
        payload["per_request_fee_usd"],
        "recovery provider price_basis.request_fee",
    )
    if input_rate <= 0 or output_rate <= 0:
        raise ModalActionJournalIntegrityError(
            "recovery provider token rates must be positive"
        )
    return payload, binding, input_rate, output_rate, request_fee


def _expected_recovery_provider_exposure(
    reader: _NamespaceReader,
    attempt: _AttemptAccumulator,
    *,
    branch: str,
    cache: dict[str, tuple[bytes, ModalJournalFileBinding]],
) -> dict[str, Any]:
    owner = _recovery_attempt_owner_payload(attempt)
    provider_action = owner["action"] in _PROVIDER_ACTIONS
    approved_bound = owner.get("provider_cost_cap_usd") if provider_action else None
    if approved_bound is not None:
        _positive_decimal_text(
            approved_bound,
            "recovery provider frozen approval bound",
        )
    zero = {
        "applicable": provider_action,
        "basis": "definite_pre_popen_zero_exposure",
        "ledger_bindings": [],
        "provider_price_basis_binding": None,
        "attempt_count": 0,
        "success_count": 0,
        "error_count": 0,
        "usage_known_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "exact_usage_cost_usd": "0",
        "frozen_provider_approval_bound_usd": approved_bound,
        "conservative_provider_exposure_usd": "0",
        "provider_requests_performed": 0,
        "price_requests_performed": 0,
    }
    if branch == "definitely_not_started" or not provider_action:
        if not provider_action:
            zero["basis"] = "non_provider_action_zero_exposure"
        return zero

    concrete_run_ids = _recovery_attempt_concrete_run_ids(attempt)
    ledger_bindings: list[dict[str, Any]] = []
    records_by_run: dict[str, list[ProviderAttemptRecord]] = {}
    for run_id in concrete_run_ids:
        logical = (
            PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT)
            / run_id
            / "controller"
            / "provider_attempts.jsonl"
        ).as_posix()
        selected = reader.read_bound_file(
            logical,
            field=f"recovery provider ledger {run_id}",
            optional=True,
            maximum_bytes=_MAX_PROVIDER_EVIDENCE_BYTES,
            required_mode=0o600,
        )
        if selected is None:
            continue
        cache[logical] = selected
        raw, binding = selected
        records = _parse_provider_ledger_bytes(
            raw,
            field=f"recovery provider ledger {run_id}",
        )
        evolution = (
            EvolutionRunSpec.parse(owner.get("harness"))
            if owner["action"] == EVOLUTION_ACTION
            else None
        )
        harness = (
            evolution.harness
            if evolution is not None
            else (
                "openevolve_generic"
                if owner["action"] == OPENEVOLVE_60_ACTION
                else _recovery_harness_for_run(owner, run_id)
            )
        )
        maximum_attempts = (
            evolution.iterations
            if evolution is not None
            else 60 if owner["action"] == OPENEVOLVE_60_ACTION else 1
        )
        expected_ledger_action = (
            "evolution_run"
            if evolution is not None
            else (
                OPENEVOLVE_60_ACTION.replace("-", "_")
                if owner["action"] == OPENEVOLVE_60_ACTION
                else "one_opportunity_engineering_canary"
            )
        )
        if (
            len(records) > maximum_attempts
            or [record.attempt_ordinal for record in records]
            != list(range(1, len(records) + 1))
            or any(
                record.execution_backend != "modal"
                or record.action_run_id != run_id
                or record.harness != harness
                or record.action != expected_ledger_action
                for record in records
            )
        ):
            raise ModalActionJournalIntegrityError(
                "recovery provider ledger identity or opportunity count changed"
            )
        records_by_run[run_id] = records
        ledger_bindings.append(
            {
                "path": binding.path,
                "sha256": binding.sha256,
                "size_bytes": binding.size_bytes,
            }
        )
    ledger_bindings.sort(key=lambda item: item["path"])
    records = [
        record
        for run_id in concrete_run_ids
        for record in records_by_run.get(run_id, [])
    ]
    attempt_count = len(records)
    success_count = sum(record.status == "success" for record in records)
    error_count = sum(record.status == "error" for record in records)
    usage_known_count = sum(record.usage_known for record in records)
    input_tokens = sum(record.input_tokens or 0 for record in records)
    output_tokens = sum(record.output_tokens or 0 for record in records)
    fully_bound = (
        set(records_by_run) == set(concrete_run_ids)
        and all(
            len(records_by_run[run_id])
            == (
                EvolutionRunSpec.parse(owner.get("harness")).iterations
                if owner["action"] == EVOLUTION_ACTION
                else 60 if owner["action"] == OPENEVOLVE_60_ACTION else 1
            )
            for run_id in concrete_run_ids
        )
        and attempt_count
        == len(concrete_run_ids)
        * (
            EvolutionRunSpec.parse(owner.get("harness")).iterations
            if owner["action"] == EVOLUTION_ACTION
            else 60 if owner["action"] == OPENEVOLVE_60_ACTION else 1
        )
        and success_count == attempt_count
        and usage_known_count == attempt_count
    )
    price_binding: dict[str, Any] | None = None
    exact_cost: str | None = None
    if fully_bound:
        _payload, binding, input_rate, output_rate, request_fee = (
            _validate_recovery_price_basis(reader, owner, cache=cache)
        )
        price_binding = {
            "path": binding.path,
            "sha256": binding.sha256,
            "size_bytes": binding.size_bytes,
        }
        cost = (
            Decimal(input_tokens) * input_rate / Decimal(1_000_000)
            + Decimal(output_tokens) * output_rate / Decimal(1_000_000)
            + Decimal(attempt_count) * request_fee
        )
        exact_cost = format(cost, "f")
        basis = "exact_provider_attempt_ledger_usage"
        conservative = exact_cost
    else:
        if approved_bound is None:  # pragma: no cover - provider core validates it
            raise ModalActionJournalIntegrityError(
                "recovery provider action lacks its frozen approval bound"
            )
        basis = "frozen_full_provider_approval_bound"
        conservative = approved_bound
    return {
        "applicable": True,
        "basis": basis,
        "ledger_bindings": ledger_bindings,
        "provider_price_basis_binding": price_binding,
        "attempt_count": attempt_count,
        "success_count": success_count,
        "error_count": error_count,
        "usage_known_count": usage_known_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "exact_usage_cost_usd": exact_cost,
        "frozen_provider_approval_bound_usd": approved_bound,
        "conservative_provider_exposure_usd": conservative,
        "provider_requests_performed": 0,
        "price_requests_performed": 0,
    }


def _recovery_claim_specs(
    attempt: _AttemptAccumulator,
) -> tuple[ModalRemoteRunReservationSpec, ...]:
    if attempt.identity is None:
        raise ModalActionJournalIntegrityError(
            "recovery reservation claim lacks a cohort identity"
        )
    if attempt.intent is not None:
        return _reservation_specs_for_claim(
            attempt.intent.payload,
            identity=attempt.identity,
            timestamp_field="created_at_utc",
        )
    if attempt.rejection is not None:
        return _reservation_specs_for_claim(
            attempt.rejection.payload,
            identity=attempt.identity,
            timestamp_field="started_at_utc",
        )
    if not attempt.reservations:
        raise ModalActionJournalIntegrityError(
            "recovery reservation inference lacks any reservation"
        )
    first = attempt.reservations[0].payload
    if first["action"] == "canaries":
        prefixes = {
            _canary_aggregate_prefix(record.payload["remote_run_id"])
            for record in attempt.reservations
        }
        if len(prefixes) != 1:
            raise ModalActionJournalIntegrityError(
                "recovery reservation inference has an ambiguous canary prefix"
            )
        run_id = next(iter(prefixes))
        concrete = expected_modal_concrete_run_ids(
            action="canaries",
            run_id=run_id,
            verifier_run_id=None,
        )
    else:
        if len(attempt.reservations) != 1:
            raise ModalActionJournalIntegrityError(
                "recovery cannot infer multiple non-canary reservations"
            )
        concrete = (first["remote_run_id"],)
    return build_modal_remote_run_reservation_specs(
        concrete_remote_run_ids=concrete,
        attempt_id=attempt.attempt_id,
        action=first["action"],
        identity=attempt.identity,
        created_at_utc=first["created_at_utc"],
        launch_capability_sha256=first["launch_capability_sha256"],
        local_host_anchor_path=first["local_host_anchor_path"],
        local_host_anchor_sha256=first["local_host_anchor_sha256"],
        local_boot_started_at_unix_microseconds=first[
            "local_boot_started_at_unix_microseconds"
        ],
        local_boot_session_sha256=first["local_boot_session_sha256"],
    )


def _recovery_action_and_run_id(
    attempt: _AttemptAccumulator,
) -> tuple[str, str]:
    owner = _recovery_attempt_owner_payload(attempt)
    action = owner["action"]
    if attempt.intent is not None or attempt.rejection is not None:
        run_id = owner["run_id"]
    elif action == "canaries":
        run_id = _canary_aggregate_prefix(
            attempt.reservations[0].payload["remote_run_id"]
        )
    else:
        run_id = attempt.reservations[0].payload["remote_run_id"]
    try:
        return action, validate_run_id(run_id)
    except (TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "recovery action/run identity is invalid"
        ) from error


def _recovery_original_boot_identity(
    attempt: _AttemptAccumulator,
) -> tuple[int, str]:
    owner = _recovery_attempt_owner_payload(attempt)
    started = owner["local_boot_started_at_unix_microseconds"]
    session = owner["local_boot_session_sha256"]
    if (
        type(started) is not int
        or started < _MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS
    ):
        raise ModalActionJournalIntegrityError(
            "recovery source boot-start identity is invalid"
        )
    return started, _sha256(session, "recovery source boot-session SHA-256")


def _expected_zero_modal_exposure() -> dict[str, Any]:
    return {
        "basis": "definite_pre_popen_journal_state",
        "measured_app_name_main_billing_usd": "0",
        "unresolved_compute_reserve_usd": "0",
        "conservative_app_name_main_billing_usd": "0",
        "complete_hourly_window": False,
        "local_authorization_is_platform_hard_bound": False,
        "modal_api_requests_performed": 0,
        "snapshot_requests_performed": 0,
        "billing_requests_performed": 0,
        "price_requests_performed": 0,
    }


def _expected_empty_known_objects() -> dict[str, Any]:
    return {
        name: {"coverage": "partial", "ids": []}
        for name in sorted(_RECOVERY_KNOWN_OBJECT_FIELDS)
    }


def _validate_recovery_records(
    reader: _NamespaceReader,
    attempt: _AttemptAccumulator,
    *,
    process_probe_result: str | None,
    pre_recovery_codes: Sequence[str],
) -> bool:
    """Validate present recovery stages and return whether the triplet closes."""

    records = _recovery_payload_by_kind(attempt)
    if not records:
        return False
    if "intent" not in records:
        raise ModalActionJournalIntegrityError(
            "recovery containment/resolution exists without recovery intent"
        )
    if "resolution" in records and "host_containment" not in records:
        raise ModalActionJournalIntegrityError(
            "recovery resolution exists without host containment"
        )
    if not any(
        (
            attempt.intent,
            attempt.terminal,
            attempt.rejection,
            attempt.reservations,
            attempt.marker,
        )
    ):
        raise ModalActionJournalIntegrityError(
            "recovery journal lacks discovered source evidence"
        )
    if not pre_recovery_codes:
        raise ModalActionJournalIntegrityError(
            "an already closed attempt cannot be retroactively recovered"
        )
    intent_record = records["intent"].record
    intent = intent_record.payload
    branch = intent["branch"]
    expected_source = _expected_recovery_source_evidence(attempt)
    if intent["source_evidence"] != expected_source:
        raise ModalActionJournalIntegrityError(
            "recovery intent source-evidence closure changed"
        )
    action, run_id = _recovery_action_and_run_id(attempt)
    if intent["action"] != action or intent["run_id"] != run_id:
        raise ModalActionJournalIntegrityError(
            "recovery intent action identity changed"
        )
    repair = intent["reservation_repair"]
    final_bindings = expected_source["remote_run_reservations"]
    if (
        repair["final_reservation_bindings"] != final_bindings
        or intent["source_evidence"]["remote_run_reservations"] != final_bindings
    ):
        raise ModalActionJournalIntegrityError(
            "recovery final reservation binding roster changed"
        )
    initial = repair["initial_reservation_bindings"]
    published = repair["published_reservation_bindings"]
    initial_paths = {item["path"] for item in initial}
    published_paths = {item["path"] for item in published}
    if initial_paths & published_paths or sorted(
        [*initial, *published], key=lambda item: item["path"]
    ) != final_bindings:
        raise ModalActionJournalIntegrityError(
            "recovery initial/published/final reservation bindings do not reconcile"
        )
    specs = _recovery_claim_specs(attempt)
    expected_paths = {spec.binding["path"] for spec in specs}
    if {item["path"] for item in final_bindings} != expected_paths:
        raise ModalActionJournalIntegrityError(
            "recovery final reservation namespace is incomplete"
        )
    cache: dict[str, tuple[bytes, ModalJournalFileBinding]] = {}
    snapshot_manifest = intent["snapshot_manifest"]
    request = _validate_recovery_request(
        reader,
        intent["request_binding"],
        attempt_id=attempt.attempt_id,
        expected_branch=branch,
        fresh_candidate_attempt_id=intent["fresh_candidate_attempt_id"],
        snapshot_manifest=snapshot_manifest,
        cache=cache,
    )
    if repair["initial_reservation_bindings"] != request[
        "initial_reservation_bindings"
    ]:
        raise ModalActionJournalIntegrityError(
            "recovery intent changed the request-frozen initial reservation roster"
        )
    if branch == "definitely_not_started":
        if (
            attempt.intent is not None
            or attempt.terminal is not None
            or attempt.marker is not None
            or attempt.aggregates
            or not attempt.reservations
            or snapshot_manifest is not None
            or repair["basis"]
            not in {
                "global_rejection_planned_roster",
                "canonical_reservation_inference",
            }
        ):
            raise ModalActionJournalIntegrityError(
                "definitely-not-started recovery lacks exact pre-Popen proof"
            )
        if attempt.rejection is not None and repair["basis"] != (
            "global_rejection_planned_roster"
        ):
            raise ModalActionJournalIntegrityError(
                "reservation-bearing rejection was not preferred for repair"
            )
    else:
        if (
            attempt.intent is None
            or attempt.rejection is not None
            or repair["basis"] != "not_applicable"
            or initial != final_bindings
            or published
            or snapshot_manifest is None
        ):
            raise ModalActionJournalIntegrityError(
                "may-have-started recovery repaired or omitted durable ownership"
            )
        _read_recovery_bound_file(
            reader,
            snapshot_manifest,
            field="recovery selected snapshot manifest",
            cache=cache,
        )
    if set(records) == {"intent"}:
        return False

    host_record = records["host_containment"].record
    host = host_record.payload
    if (
        host["branch"] != branch
        or host["request_binding"] != intent["request_binding"]
        or host["source_evidence"] != expected_source
        or host["recovery_intent"] != _record_binding_dict(intent_record)
        or host["recorded_at_utc"] < intent["recorded_at_utc"]
    ):
        raise ModalActionJournalIntegrityError(
            "recovery host containment does not extend its exact intent"
        )
    original_start, original_session = _recovery_original_boot_identity(attempt)
    if (
        host["original_boot_started_at_unix_microseconds"] != original_start
        or host["original_boot_session_sha256"] != original_session
    ):
        raise ModalActionJournalIntegrityError(
            "recovery host containment changed the original boot identity"
        )
    if branch == "definitely_not_started":
        if any(
            host[name] is not None
            for name in (
                "current_boot_started_at_unix_microseconds",
                "current_boot_session_sha256",
                "process_identity",
                "marker_binding",
                "terminal_binding",
            )
        ) or (
            host["containment_basis"] != "definite_pre_popen_journal_state"
            or host["process_probe_result"] != "not_applicable"
        ):
            raise ModalActionJournalIntegrityError(
                "definitely-not-started recovery contains a process/boot probe claim"
            )
    else:
        if host["containment_basis"] == "same_boot_process_group_absent":
            terminal = attempt.terminal.payload if attempt.terminal else None
            marker = attempt.marker.payload if attempt.marker else None
            if (
                terminal is None
                or marker is None
                or terminal["modal_cli_process_started"] is not True
                or terminal["local_process_start_receipt_sha256"]
                != attempt.marker.binding.sha256
                or host["marker_binding"]
                != _record_binding_dict(attempt.marker)
                or host["terminal_binding"]
                != _record_binding_dict(attempt.terminal)
                or host["current_boot_started_at_unix_microseconds"]
                != original_start
                or host["current_boot_session_sha256"] != original_session
                or host["process_probe_result"]
                != "same_boot_process_group_absent"
                or process_probe_result
                not in {
                    "same_boot_process_group_absent",
                    "different_boot_session",
                }
            ):
                raise ModalActionJournalIntegrityError(
                    "same-boot recovery lacks exact absent process-group proof"
                )
            expected_process_identity = {
                "process_id": marker["process_id"],
                "process_group_id": marker["expected_process_group_id"],
                "session_id": marker["expected_session_id"],
                "process_birth_identity_sha256": marker[
                    "process_birth_identity_sha256"
                ],
            }
            if host["process_identity"] != expected_process_identity:
                raise ModalActionJournalIntegrityError(
                    "same-boot recovery process identity changed"
                )
        elif host["containment_basis"] == "strictly_later_boot_session":
            current_start = host["current_boot_started_at_unix_microseconds"]
            current_session = host["current_boot_session_sha256"]
            if (
                type(current_start) is not int
                or current_start <= original_start
                or current_session == original_session
                or host["terminal_binding"]
                != (
                    _record_binding_dict(attempt.terminal)
                    if attempt.terminal is not None
                    else None
                )
                or host["marker_binding"]
                != (
                    _record_binding_dict(attempt.marker)
                    if attempt.marker is not None
                    else None
                )
                or host["process_identity"]
                != (
                    {
                        "process_id": attempt.marker.payload["process_id"],
                        "process_group_id": attempt.marker.payload[
                            "expected_process_group_id"
                        ],
                        "session_id": attempt.marker.payload["expected_session_id"],
                        "process_birth_identity_sha256": attempt.marker.payload[
                            "process_birth_identity_sha256"
                        ],
                    }
                    if attempt.marker is not None
                    else None
                )
                or host["process_probe_result"]
                != (
                    "different_boot_session"
                    if attempt.marker is not None
                    else "not_applicable"
                )
            ):
                raise ModalActionJournalIntegrityError(
                    "later-boot recovery lacks strict changed-session proof"
                )
        else:
            raise ModalActionJournalIntegrityError(
                "may-have-started recovery has an invalid containment basis"
            )
    if set(records) == {"intent", "host_containment"}:
        return False

    resolution_record = records["resolution"].record
    resolution = resolution_record.payload
    if (
        resolution["branch"] != branch
        or resolution["request_binding"] != intent["request_binding"]
        or resolution["source_evidence"] != expected_source
        or resolution["recovery_intent"] != _record_binding_dict(intent_record)
        or resolution["host_containment"] != _record_binding_dict(host_record)
        or resolution["final_reservation_bindings"] != final_bindings
        or resolution["fresh_candidate_attempt_id"]
        != intent["fresh_candidate_attempt_id"]
        or resolution["recorded_at_utc"] < host["recorded_at_utc"]
    ):
        raise ModalActionJournalIntegrityError(
            "recovery resolution does not close its exact predecessor stages"
        )
    _validate_recovery_request(
        reader,
        resolution["request_binding"],
        attempt_id=attempt.attempt_id,
        expected_branch=branch,
        fresh_candidate_attempt_id=resolution["fresh_candidate_attempt_id"],
        snapshot_manifest=snapshot_manifest,
        cache=cache,
    )
    if branch == "definitely_not_started":
        expected_snapshot = None
        expected_modal = _expected_zero_modal_exposure()
        expected_objects = _expected_empty_known_objects()
        expected_runs = [
            {
                "run_id": run_id,
                "present": False,
                "disposition": "not_applicable_pre_popen",
            }
            for run_id in sorted(_recovery_attempt_concrete_run_ids(attempt))
        ]
    else:
        (
            expected_snapshot,
            expected_modal,
            expected_objects,
            expected_runs,
        ) = _validate_recovery_snapshot(
            reader,
            snapshot_manifest,
            identity=attempt.identity,
            attempt=attempt,
            cache=cache,
        )
    expected_provider = _expected_recovery_provider_exposure(
        reader,
        attempt,
        branch=branch,
        cache=cache,
    )
    if (
        resolution["snapshot_evidence"] != expected_snapshot
        or resolution["modal_exposure"] != expected_modal
        or resolution["provider_exposure"] != expected_provider
        or resolution["known_remote_objects"] != expected_objects
        or resolution["run_directory_dispositions"] != expected_runs
    ):
        raise ModalActionJournalIntegrityError(
            "recovery resolution exposure or quarantine claims changed"
        )
    return True


def _classify_attempts(
    reader: _NamespaceReader,
    project_root: Path,
    cohorts: Sequence[ModalCohortActionJournal],
    attempts: Mapping[str, _AttemptAccumulator],
    *,
    process_probe: ProcessProbe | None,
) -> tuple[
    tuple[ModalAttemptJournalState, ...],
    tuple[ModalActionJournalBlocker, ...],
]:
    sealed_identities = {cohort.identity for cohort in cohorts if cohort.sealed}
    blockers: list[ModalActionJournalBlocker] = []
    states: list[ModalAttemptJournalState] = []
    for attempt_id in sorted(attempts):
        attempt = attempts[attempt_id]
        blocker_start = len(blockers)
        codes: list[str] = []
        paths = [
            record.binding.path
            for record in (
                attempt.intent,
                attempt.terminal,
                attempt.rejection,
                attempt.marker,
            )
            if record is not None
        ]
        paths.extend(record.binding.path for record in attempt.reservations)
        paths.extend(item.record.binding.path for item in attempt.recoveries)

        if attempt.terminal is not None and attempt.intent is None:
            raise ModalActionJournalIntegrityError(
                "cohort terminal lacks its durable intent"
            )
        if attempt.intent is not None and attempt.rejection is not None:
            raise ModalActionJournalIntegrityError(
                "attempt has both a cohort intent and global rejection"
            )
        _validate_intent_terminal_pair(attempt)
        _validate_marker_links(attempt)
        _validate_aggregates(attempt)

        actual_by_run = {
            record.payload["remote_run_id"]: record for record in attempt.reservations
        }
        if len(actual_by_run) != len(attempt.reservations):
            raise ModalActionJournalIntegrityError(
                "attempt owns duplicate remote-run reservation IDs"
            )
        if attempt.intent is not None:
            if attempt.identity is None:  # pragma: no cover - core invariant
                raise AssertionError("intent attempt lacks identity")
            specs = _reservation_specs_for_claim(
                attempt.intent.payload,
                identity=attempt.identity,
                timestamp_field="created_at_utc",
            )
            _validate_claim_reservation_specs(
                attempt.intent.payload,
                specs,
                field="intent",
            )
            expected_by_run = {spec.binding["run_id"]: spec for spec in specs}
            if set(actual_by_run) != set(expected_by_run):
                raise ModalActionJournalIntegrityError(
                    "durable intent reservation namespace is incomplete"
                )
            for run_id, spec in expected_by_run.items():
                _validate_actual_reservation(actual_by_run[run_id], spec)
        elif (
            attempt.rejection is not None
            and attempt.rejection.payload["remote_run_reservations"]
        ):
            if attempt.identity is None:
                raise ModalActionJournalIntegrityError(
                    "reservation-bearing rejection lacks a cohort identity"
                )
            specs = _reservation_specs_for_claim(
                attempt.rejection.payload,
                identity=attempt.identity,
                timestamp_field="started_at_utc",
            )
            _validate_claim_reservation_specs(
                attempt.rejection.payload,
                specs,
                field="global rejection",
            )
            expected_by_run = {spec.binding["run_id"]: spec for spec in specs}
            if not set(actual_by_run) <= set(expected_by_run):
                raise ModalActionJournalIntegrityError(
                    "global rejection owns an unplanned reservation"
                )
            for run_id, record in actual_by_run.items():
                _validate_actual_reservation(record, expected_by_run[run_id])
            _add_blocker(
                blockers,
                codes,
                code="global_rejection_reservations_require_recovery",
                attempt=attempt,
                paths=paths,
            )
            if set(actual_by_run) != set(expected_by_run):
                _add_blocker(
                    blockers,
                    codes,
                    code="partial_reservation_publication",
                    attempt=attempt,
                    paths=paths,
                )
        elif attempt.reservations:
            if attempt.rejection is not None:
                raise ModalActionJournalIntegrityError(
                    "global rejection omits its owned reservation bindings"
                )
            _validate_reservation_only_group(attempt)
            _add_blocker(
                blockers,
                codes,
                code="reservation_without_journal_owner",
                attempt=attempt,
                paths=paths,
            )

        if attempt.intent is not None and attempt.terminal is None:
            _add_blocker(
                blockers,
                codes,
                code="intent_without_terminal",
                attempt=attempt,
                paths=paths,
            )
        if attempt.marker is not None and attempt.terminal is not None:
            terminal = attempt.terminal.payload
            if terminal["local_process_start_receipt_sha256"] is None:
                _add_blocker(
                    blockers,
                    codes,
                    code="unbound_process_marker",
                    attempt=attempt,
                    paths=paths,
                )
        if attempt.terminal is not None:
            terminal = attempt.terminal.payload
            if terminal["modal_cli_process_started"] and (
                attempt.marker is None
                or terminal["local_process_start_receipt_sha256"] is None
            ):
                _add_blocker(
                    blockers,
                    codes,
                    code="started_process_marker_missing_or_unbound",
                    attempt=attempt,
                    paths=paths,
                )
            if terminal["modal_cli_process_started"] and (
                terminal["process_group_closed"] is not True
            ):
                _add_blocker(
                    blockers,
                    codes,
                    code="process_group_not_closed",
                    attempt=attempt,
                    paths=paths,
                )

        probe_result = (
            _probe_process_marker(project_root, attempt.marker, process_probe)
            if attempt.marker is not None
            else None
        )
        if probe_result == "same_boot_process_group_exists":
            _add_blocker(
                blockers,
                codes,
                code="owned_process_group_exists",
                attempt=attempt,
                paths=paths,
            )
        elif attempt.marker is not None and process_probe is None and codes:
            _add_blocker(
                blockers,
                codes,
                code="process_probe_required",
                attempt=attempt,
                paths=paths,
            )

        recovery_complete = False
        if attempt.recoveries:
            pre_recovery_codes = tuple(codes)
            recovery_complete = _validate_recovery_records(
                reader,
                attempt,
                process_probe_result=probe_result,
                pre_recovery_codes=pre_recovery_codes,
            )
            if recovery_complete:
                del blockers[blocker_start:]
                codes.clear()
            else:
                _add_blocker(
                    blockers,
                    codes,
                    code="incomplete_recovery_journal",
                    attempt=attempt,
                    paths=paths,
                )

        if not any(
            (
                attempt.intent,
                attempt.terminal,
                attempt.rejection,
                attempt.reservations,
                attempt.marker,
                attempt.recoveries,
            )
        ):
            raise ModalActionJournalIntegrityError(
                "attempt has no owning journal evidence"
            )
        if attempt.identity in sealed_identities and codes:
            raise ModalActionJournalIntegrityError(
                "sealed cohort contains unresolved action state"
            )
        disposition: Literal["closed", "rejected", "unresolved"]
        if recovery_complete:
            disposition = "closed"
        elif codes:
            disposition = "unresolved"
        elif attempt.rejection is not None:
            disposition = "rejected"
        elif attempt.intent is not None and attempt.terminal is not None:
            disposition = "closed"
        else:
            raise ModalActionJournalIntegrityError(
                "attempt state is neither closed nor explicitly unresolved"
            )
        states.append(
            ModalAttemptJournalState(
                attempt_id=attempt.attempt_id,
                identity=attempt.identity,
                disposition=disposition,
                intent=attempt.intent,
                terminal=attempt.terminal,
                rejection=attempt.rejection,
                reservations=tuple(
                    sorted(
                        attempt.reservations,
                        key=lambda item: item.binding.path,
                    )
                ),
                process_marker=attempt.marker,
                recoveries=tuple(
                    sorted(
                        attempt.recoveries,
                        key=lambda item: item.record.binding.path,
                    )
                ),
                process_probe_result=probe_result,
                blocker_codes=tuple(codes),
            )
        )
    return tuple(states), tuple(blockers)


def _record_binding_dict(record: ModalJournalRecord) -> dict[str, Any]:
    return {
        "path": record.binding.path,
        "sha256": record.binding.sha256,
        "size_bytes": record.binding.size_bytes,
    }


def _validate_global_launch_rejection_seal_snapshot(
    rejections: Sequence[ModalJournalRecord],
    seal: ModalJournalRecord | None,
) -> None:
    if seal is None:
        return
    expected = [
        _record_binding_dict(record)
        for record in sorted(rejections, key=lambda item: item.binding.path)
    ]
    if seal.payload["rejection_receipts"] != expected:
        raise ModalActionJournalIntegrityError(
            "global launch-rejection seal snapshot changed"
        )
    sealed_at = _utc(
        seal.payload["recorded_at_utc"],
        "global_launch_rejection_seal.recorded_at_utc",
    )
    if any(
        _utc(
            record.payload["finished_at_utc"],
            "global_launch_rejection_seal.rejection.finished_at_utc",
        )
        > sealed_at
        for record in rejections
    ):
        raise ModalActionJournalIntegrityError(
            "global launch-rejection seal predates a frozen rejection"
        )


def _cohort_journal_binding_snapshot(
    cohort: ModalCohortActionJournal,
) -> dict[str, list[dict[str, Any]]]:
    return {
        "intent_receipts": [
            _record_binding_dict(record)
            for record in sorted(cohort.intents, key=lambda item: item.binding.path)
        ],
        "terminal_receipts": [
            _record_binding_dict(record)
            for record in sorted(cohort.terminals, key=lambda item: item.binding.path)
        ],
        "aggregate_receipts": [
            _record_binding_dict(record)
            for record in sorted(cohort.aggregates, key=lambda item: item.binding.path)
        ],
    }


def _bound_json_object(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModalActionJournalIntegrityError(
            f"{field} is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise ModalActionJournalIntegrityError(f"{field} is not one JSON object")
    return value


def _declared_binding_matches(
    declared: Mapping[str, Any],
    observed: ModalJournalFileBinding,
    *,
    field: str,
) -> None:
    if any(
        declared.get(name) != expected
        for name, expected in (
            ("path", observed.path),
            ("sha256", observed.sha256),
            ("size_bytes", observed.size_bytes),
        )
    ):
        raise ModalActionJournalIntegrityError(f"{field} byte binding changed")


def _provider_harness_from_terminal(
    terminal: Mapping[str, Any],
    run_id: str,
) -> str:
    if terminal["action"] == "canary":
        harness = terminal["harness"]
        if harness not in CANARY_ORDER or run_id != terminal["run_id"]:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal single-canary identity changed"
            )
        return harness
    if terminal["action"] != "canaries":
        raise ModalActionJournalIntegrityError(
            "migration terminal seal provider evidence names a non-provider action"
        )
    matches = [
        harness
        for harness in CANARY_ORDER
        if run_id == f"{terminal['run_id']}-{_CANARY_SUFFIXES[harness]}"
    ]
    if len(matches) != 1:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal aggregate child harness is ambiguous"
        )
    return matches[0]


def _validate_provider_uncertainty_payload(
    value: Mapping[str, Any],
    *,
    harness: str,
    run_id: str,
    modal_call_id: str | None,
    ledger_present: bool,
) -> None:
    if set(value) != _PROVIDER_START_UNCERTAIN_FIELDS:
        raise ModalActionJournalIntegrityError(
            "provider uncertainty evidence has an invalid exact schema"
        )
    if (
        value["schema_name"] != "ProviderRequestStartUncertainEvidence"
        or value["schema_version"] != "1.0"
        or value["harness"] != harness
        or value["action"] != "one_opportunity_engineering_canary"
        or value["execution_backend"] != "modal"
        or value["action_run_id"] != run_id
        or not isinstance(value["modal_call_id"], str)
        or not value["modal_call_id"]
        or (modal_call_id is not None and value["modal_call_id"] != modal_call_id)
        or value["api_endpoint"] != OFFICIAL_OPENAI_API_BASE
        or value["model"] != TARGET_MODEL
        or type(value["provider_attempt_count_lower_bound"]) is not int
        or value["provider_attempt_count_lower_bound"] != 0
        or type(value["provider_attempt_count_upper_bound"]) is not int
        or value["provider_attempt_count_upper_bound"] != 1
        or value["provider_request_started"] != "unknown"
        or value["provider_attempt_ledger_state"]
        != ("present" if ledger_present else "missing")
        or value["billing_treatment"] != "reserve_one_full_approved_request"
        or value["reason"] != "controller_terminated_without_terminal_attempt_record"
    ):
        raise ModalActionJournalIntegrityError(
            "provider uncertainty evidence identity or semantics changed"
        )


def _parse_provider_ledger_bytes(
    raw: bytes,
    *,
    field: str,
) -> list[ProviderAttemptRecord]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ModalActionJournalIntegrityError(f"{field} is not valid UTF-8") from error
    if text and not text.endswith("\n"):
        raise ModalActionJournalIntegrityError(f"{field} is truncated")
    records: list[ProviderAttemptRecord] = []
    try:
        for line in text.splitlines():
            if not line:
                raise ModalActionJournalIntegrityError(
                    f"{field} contains a blank record"
                )
            payload = json.loads(
                line,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
            if not isinstance(payload, dict):
                raise ModalActionJournalIntegrityError(
                    f"{field} contains a non-object record"
                )
            records.append(ProviderAttemptRecord.from_dict(payload))
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, ModalActionJournalIntegrityError):
            raise
        raise ModalActionJournalIntegrityError(
            f"{field} contains an invalid provider record"
        ) from error
    return records


def _migration_verifier_source_run_id(
    terminal: Mapping[str, Any],
    *,
    concrete_run_id: str,
) -> str:
    """Return the source run using the canonical Modal action identity contract."""

    try:
        action, source, source_run_id, verifier_run_id, harness = (
            validate_modal_action_identity(
                action=terminal["action"],
                run_id=terminal["run_id"],
                source_run_id=terminal["source_run_id"],
                verifier_run_id=terminal["verifier_run_id"],
                harness=terminal["harness"],
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ModalActionJournalIntegrityError(
            "migration verifier terminal identity changed"
        ) from error
    if (
        action not in {"download", "verify"}
        or source_run_id is not None
        or verifier_run_id != concrete_run_id
        or harness is not None
    ):
        raise ModalActionJournalIntegrityError(
            "migration verifier terminal identity changed"
        )
    return source


def _validate_migration_terminal_evidence_snapshot(
    reader: _NamespaceReader,
    *,
    identity: ModalLiveCohortIdentity,
    selected: Mapping[str, Any],
    states_by_attempt: Mapping[str, ModalAttemptJournalState],
) -> None:
    """Rederive every nested final-evidence record from stable local bytes."""

    cache: dict[str, tuple[bytes, ModalJournalFileBinding] | None] = {}

    def bound_file(
        logical: str,
        *,
        field: str,
        maximum_bytes: int = _MAX_JSON_BYTES,
    ) -> tuple[bytes, ModalJournalFileBinding] | None:
        if logical not in cache:
            cache[logical] = reader.read_bound_file(
                logical,
                field=field,
                optional=True,
                maximum_bytes=maximum_bytes,
            )
        return cache[logical]

    disposition_map = {
        (record["attempt_id"], record["run_id"]): record
        for record in selected["run_dispositions"]
    }
    remote_map = {
        (record["attempt_id"], record["run_id"]): record
        for record in selected["remote_executions"]
    }
    artifact_map = {
        (record["attempt_id"], record["run_id"]): record
        for record in selected["artifact_manifests"]
    }
    provider_map = {
        (record["attempt_id"], record["run_id"]): record
        for record in selected["provider_attempt_evidence"]
    }
    observed_remote_keys: set[tuple[str, str]] = set()
    observed_artifact_keys: set[tuple[str, str]] = set()
    execution_contexts: dict[tuple[str, str], ExecutionContextV1] = {}

    for key, disposition in disposition_map.items():
        attempt_id, run_id = key
        state = states_by_attempt.get(attempt_id)
        terminal = state.terminal.payload if state and state.terminal else None
        if state is None or state.identity != identity or terminal is None:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal evidence names an unknown final attempt"
            )
        action = terminal["action"]
        process_started = terminal["modal_cli_process_started"]
        remote_record = remote_map.get(key)
        artifact_record = artifact_map.get(key)
        observed_evidence: tuple[str, str, bytes, ModalJournalFileBinding] | None = None
        observed_manifest: tuple[bytes, ModalJournalFileBinding] | None = None
        source_run_id: str | None = None

        if action not in {"download", "verify"}:
            evidence_path = (
                PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT)
                / run_id
                / "execution_context.json"
            ).as_posix()
            evidence_file = bound_file(
                evidence_path,
                field=f"migration execution evidence {attempt_id}/{run_id}",
            )
            manifest_files = [
                (
                    (
                        PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT) / run_id / filename
                    ).as_posix(),
                    bound_file(
                        (
                            PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT)
                            / run_id
                            / filename
                        ).as_posix(),
                        field=f"migration artifact manifest {attempt_id}/{run_id}",
                        maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
                    ),
                )
                for filename in ARTIFACT_MANIFEST_FILENAMES
            ]
            present_manifests = [
                (logical, snapshot)
                for logical, snapshot in manifest_files
                if snapshot is not None
            ]
            if len(present_manifests) > 1:
                raise ModalActionJournalIntegrityError(
                    "migration execution has multiple artifact manifests"
                )
            if evidence_file is None and present_manifests:
                raise ModalActionJournalIntegrityError(
                    "migration artifact manifest lacks its execution context"
                )
            if evidence_file is not None:
                evidence_kind = (
                    "downloaded_execution_context"
                    if present_manifests
                    else "downloaded_execution_context_without_artifact_manifest"
                )
                observed_evidence = (
                    evidence_kind,
                    evidence_path,
                    evidence_file[0],
                    evidence_file[1],
                )
            if present_manifests:
                _manifest_path, observed_manifest = present_manifests[0]
        else:
            source_run_id = _migration_verifier_source_run_id(
                terminal,
                concrete_run_id=run_id,
            )
            capture = modal_artifact_verifier_capture_directory_path(
                identity,
                source_run_id,
                run_id,
                attempt_id,
            )
            evidence_candidates = (
                (
                    "remote_verification_receipt",
                    modal_remote_verification_receipt_path(
                        identity,
                        source_run_id,
                        run_id,
                        attempt_id,
                    ).as_posix(),
                ),
                (
                    "volume_success_capture",
                    (capture / "artifact_verification_result.json").as_posix(),
                ),
                (
                    "volume_failure_capture",
                    (capture / "artifact_verification_failure.json").as_posix(),
                ),
            )
            present_evidence = []
            for evidence_kind, logical in evidence_candidates:
                snapshot = bound_file(
                    logical,
                    field=f"migration verifier evidence {attempt_id}/{run_id}",
                )
                if snapshot is not None:
                    present_evidence.append(
                        (evidence_kind, logical, snapshot[0], snapshot[1])
                    )
            if len(present_evidence) > 1:
                raise ModalActionJournalIntegrityError(
                    "migration verifier has multiple execution captures"
                )
            if present_evidence:
                observed_evidence = present_evidence[0]
                if observed_evidence[0] in {
                    "volume_success_capture",
                    "volume_failure_capture",
                }:
                    manifest_path = (capture / "artifact_manifest.json").as_posix()
                    observed_manifest = bound_file(
                        manifest_path,
                        field=f"migration verifier manifest {attempt_id}/{run_id}",
                        maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
                    )
                    if observed_manifest is None:
                        raise ModalActionJournalIntegrityError(
                            "migration verifier capture lacks its artifact manifest"
                        )

        if not process_started:
            if observed_evidence is not None or observed_manifest is not None:
                raise ModalActionJournalIntegrityError(
                    "definitely-not-started migration run has execution evidence"
                )
            expected_execution_disposition = "definitely_not_started"
        elif observed_evidence is not None:
            expected_execution_disposition = "remote_execution_bound"
        else:
            if terminal["status"] == "succeeded":
                raise ModalActionJournalIntegrityError(
                    "successful migration run lacks execution evidence"
                )
            expected_execution_disposition = "may_have_started_unresolved_quarantined"
        if disposition["execution_disposition"] != expected_execution_disposition:
            raise ModalActionJournalIntegrityError(
                "migration execution disposition differs from local evidence"
            )

        if observed_evidence is None:
            if remote_record is not None or artifact_record is not None:
                raise ModalActionJournalIntegrityError(
                    "migration seal invents remote execution evidence"
                )
            continue
        if remote_record is None:
            raise ModalActionJournalIntegrityError(
                "migration seal omits remote execution evidence"
            )
        evidence_kind, evidence_path, evidence_raw, evidence_binding = observed_evidence
        if (
            remote_record["action"] != action
            or remote_record["evidence_kind"] != evidence_kind
        ):
            raise ModalActionJournalIntegrityError(
                "migration remote execution action or evidence kind changed"
            )
        _declared_binding_matches(
            remote_record["evidence"],
            evidence_binding,
            field="migration remote execution evidence",
        )
        if remote_record["evidence"]["path"] != evidence_path:
            raise ModalActionJournalIntegrityError(
                "migration remote execution evidence path changed"
            )
        embedded_context = _lineage_execution_context(
            remote_record["execution_context"],
            field="migration embedded execution context",
        )
        evidence_payload = _bound_json_object(
            evidence_raw,
            field="migration remote execution evidence",
        )
        if evidence_kind.startswith("downloaded_"):
            evidence_context = _lineage_execution_context(
                evidence_payload,
                field="migration downloaded execution context",
            )
        elif evidence_kind in {
            "remote_verification_receipt",
            "volume_success_capture",
        }:
            try:
                verification = ArtifactVerificationV1.from_dict(evidence_payload)
            except (TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    "migration verifier success evidence is invalid"
                ) from error
            if (
                verification.source_run_id != source_run_id
                or verification.verifier_run_id != run_id
            ):
                raise ModalActionJournalIntegrityError(
                    "migration verifier success evidence identity changed"
                )
            evidence_context = verification.verifier_execution_context
        else:
            if set(evidence_payload) != _FAILED_VERIFIER_RECEIPT_FIELDS or (
                evidence_payload["schema_name"] != "ModalArtifactVerificationFailure"
                or evidence_payload["schema_version"] != "1.0"
                or evidence_payload["source_run_id"] != source_run_id
                or evidence_payload["verifier_run_id"] != run_id
                or evidence_payload["message"]
                != "artifact verification failed; details suppressed"
                or not isinstance(evidence_payload["error_type"], str)
                or re.fullmatch(
                    r"[A-Za-z][A-Za-z0-9_]{0,127}",
                    evidence_payload["error_type"],
                )
                is None
            ):
                raise ModalActionJournalIntegrityError(
                    "migration verifier failure evidence is invalid"
                )
            evidence_context = _lineage_execution_context(
                evidence_payload["verifier_execution_context"],
                field="migration verifier failure execution context",
            )
        expected_harness = (
            _provider_harness_from_terminal(terminal, run_id)
            if action in {"canary", "canaries"}
            else None
        )
        expected_function = (
            "artifact_verify"
            if action in {"download", "verify"}
            else f"canary_{expected_harness}"
            if expected_harness is not None
            else _ORDINARY_ACTION_FUNCTIONS[action]
        )
        expected_artifact_uri = volume_artifact_uri(
            source_run_id if source_run_id is not None else run_id
        )
        if (
            evidence_context != embedded_context
            or embedded_context.run_id != run_id
            or embedded_context.function_name != expected_function
            or embedded_context.artifact_uri != expected_artifact_uri
            or embedded_context.image_source_sha256 != identity.image_source_sha256
        ):
            raise ModalActionJournalIntegrityError(
                "migration embedded execution context differs from bound evidence"
            )
        observed_remote_keys.add(key)
        execution_contexts[key] = embedded_context

        if observed_manifest is None:
            if artifact_record is not None:
                raise ModalActionJournalIntegrityError(
                    "migration seal invents an artifact manifest"
                )
        else:
            if artifact_record is None:
                raise ModalActionJournalIntegrityError(
                    "migration seal omits an artifact manifest"
                )
            manifest_raw, manifest_binding = observed_manifest
            _declared_binding_matches(
                artifact_record,
                manifest_binding,
                field="migration artifact manifest",
            )
            try:
                manifest = parse_artifact_manifest_bytes(manifest_raw)
            except (ArtifactIntegrityError, TypeError, ValueError) as error:
                raise ModalActionJournalIntegrityError(
                    "migration artifact manifest is invalid"
                ) from error
            evidence_relative = PurePosixPath(evidence_path).name
            evidence_item = next(
                (
                    item
                    for item in manifest.files
                    if item.relative_path == evidence_relative
                ),
                None,
            )
            if (
                manifest.run_id != run_id
                or manifest.image_source_sha256 != identity.image_source_sha256
                or manifest.manifest_sha256
                != artifact_record["canonical_manifest_sha256"]
                or evidence_item is None
                or evidence_item.sha256 != evidence_binding.sha256
                or evidence_item.size_bytes != evidence_binding.size_bytes
            ):
                raise ModalActionJournalIntegrityError(
                    "migration artifact manifest identity or evidence binding changed"
                )
            observed_artifact_keys.add(key)

    if set(remote_map) != observed_remote_keys or set(artifact_map) != (
        observed_artifact_keys
    ):
        raise ModalActionJournalIntegrityError(
            "migration nested execution or artifact roster is incomplete"
        )

    observed_provider_keys: set[tuple[str, str]] = set()
    actual_request_ids: set[str] = set()
    actual_response_ids: set[str] = set()
    provider_launcher_attempt_ids: set[str] = set()
    provider_terminal_record_count = 0
    provider_attempt_lower_bound = 0
    provider_attempt_upper_bound = 0
    successful_provider_attempt_count = 0
    failed_provider_attempt_count = 0
    provider_input_tokens = 0
    provider_output_tokens = 0
    for key, disposition in disposition_map.items():
        attempt_id, run_id = key
        state = states_by_attempt[attempt_id]
        terminal = state.terminal.payload if state.terminal else None
        if terminal is None or terminal["action"] not in {"canary", "canaries"}:
            continue
        provider_launcher_attempt_ids.add(attempt_id)
        harness = _provider_harness_from_terminal(terminal, run_id)
        controller = PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT) / run_id / "controller"
        ledger_path = (controller / "provider_attempts.jsonl").as_posix()
        uncertainty_path = (
            controller / "provider_request_start_uncertain.json"
        ).as_posix()
        ledger = bound_file(
            ledger_path,
            field=f"migration provider ledger {attempt_id}/{run_id}",
            maximum_bytes=_MAX_PROVIDER_EVIDENCE_BYTES,
        )
        uncertainty = bound_file(
            uncertainty_path,
            field=f"migration provider uncertainty {attempt_id}/{run_id}",
            maximum_bytes=_MAX_PROVIDER_EVIDENCE_BYTES,
        )
        provider_record = provider_map.get(key)
        context = execution_contexts.get(key)
        process_started = terminal["modal_cli_process_started"]
        if not process_started and (ledger is not None or uncertainty is not None):
            raise ModalActionJournalIntegrityError(
                "definitely-not-started migration provider run has evidence"
            )
        if context is not None:
            expected_provider_disposition = "evidence_bound"
            expected_binding_state = "execution_context_bound"
            if ledger is None and uncertainty is None:
                raise ModalActionJournalIntegrityError(
                    "bound migration provider run lacks provider evidence"
                )
        elif process_started:
            expected_provider_disposition = "start_unresolved_conservative"
            expected_binding_state = "unbound_observed"
        else:
            expected_provider_disposition = "definitely_not_started"
            expected_binding_state = None
        if disposition["provider_disposition"] != expected_provider_disposition:
            raise ModalActionJournalIntegrityError(
                "migration provider disposition differs from local evidence"
            )
        if ledger is None and uncertainty is None:
            if provider_record is not None:
                raise ModalActionJournalIntegrityError(
                    "migration seal invents provider evidence"
                )
            if process_started:
                provider_attempt_upper_bound += 1
            continue
        if provider_record is None:
            raise ModalActionJournalIntegrityError(
                "migration seal omits provider evidence"
            )
        if (
            provider_record["harness"] != harness
            or provider_record["binding_state"] != expected_binding_state
        ):
            raise ModalActionJournalIntegrityError(
                "migration provider evidence run or harness binding changed"
            )
        if ledger is None:
            if provider_record["ledger"] is not None:
                raise ModalActionJournalIntegrityError(
                    "migration seal invents a provider ledger"
                )
        else:
            if provider_record["ledger"] is None:
                raise ModalActionJournalIntegrityError(
                    "migration seal omits a provider ledger"
                )
            _declared_binding_matches(
                provider_record["ledger"],
                ledger[1],
                field="migration provider ledger",
            )
        if uncertainty is None:
            if provider_record["uncertainty"] is not None:
                raise ModalActionJournalIntegrityError(
                    "migration seal invents provider uncertainty evidence"
                )
        else:
            if provider_record["uncertainty"] is None:
                raise ModalActionJournalIntegrityError(
                    "migration seal omits provider uncertainty evidence"
                )
            _declared_binding_matches(
                provider_record["uncertainty"],
                uncertainty[1],
                field="migration provider uncertainty",
            )

        parse_dispositions: list[str] = []
        records: list[ProviderAttemptRecord] = []
        if ledger is not None:
            try:
                records = _parse_provider_ledger_bytes(
                    ledger[0], field="migration provider ledger"
                )
            except ModalActionJournalIntegrityError:
                if expected_binding_state == "execution_context_bound":
                    raise
                parse_dispositions.append("partial_unparseable")
                records = []
            else:
                if (
                    len(records) > 1
                    or [record.attempt_ordinal for record in records]
                    != list(range(1, len(records) + 1))
                    or any(
                        record.execution_backend != "modal"
                        or record.action_run_id != run_id
                        or record.harness != harness
                        or record.action != "one_opportunity_engineering_canary"
                        or (
                            context is not None
                            and record.modal_call_id != context.modal_call_id
                        )
                        for record in records
                    )
                ):
                    raise ModalActionJournalIntegrityError(
                        "migration provider ledger identity changed"
                    )
                launcher_started = _utc(
                    terminal["started_at_utc"],
                    "migration provider launcher started_at_utc",
                )
                launcher_finished = _utc(
                    terminal["finished_at_utc"],
                    "migration provider launcher finished_at_utc",
                )
                if any(
                    not (
                        launcher_started
                        <= _utc(
                            record.started_at_utc,
                            "migration provider record started_at_utc",
                        )
                        <= _utc(
                            record.ended_at_utc,
                            "migration provider record ended_at_utc",
                        )
                        <= launcher_finished
                    )
                    for record in records
                ):
                    raise ModalActionJournalIntegrityError(
                        "migration provider record escapes its launcher interval"
                    )
                parse_dispositions.append(
                    "valid_terminal_records" if records else "exact_empty"
                )
        if uncertainty is not None:
            try:
                uncertainty_payload = _bound_json_object(
                    uncertainty[0], field="migration provider uncertainty"
                )
                _validate_provider_uncertainty_payload(
                    uncertainty_payload,
                    harness=harness,
                    run_id=run_id,
                    modal_call_id=(context.modal_call_id if context else None),
                    ledger_present=ledger is not None,
                )
            except ModalActionJournalIntegrityError:
                if expected_binding_state == "execution_context_bound":
                    raise
                parse_dispositions.append("partial_unparseable")
            else:
                parse_dispositions.append("valid_start_uncertain")
        if (
            expected_binding_state == "execution_context_bound"
            and not records
            and uncertainty is None
        ):
            raise ModalActionJournalIntegrityError(
                "bound migration provider ledger proves no terminal attempt"
            )
        expected_parse_dispositions = (
            ["valid_terminal_records"]
            if expected_binding_state == "execution_context_bound" and records
            else ["valid_start_uncertain"]
            if expected_binding_state == "execution_context_bound"
            else parse_dispositions
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
        if any(
            record.status == "success"
            and (
                record.provider_request_id is None
                or record.provider_response_id is None
                or record.usage_known is not True
            )
            for record in records
        ):
            raise ModalActionJournalIntegrityError(
                "successful migration provider record lacks exact usage identity"
            )
        provider_terminal_record_count += len(records)
        if records:
            provider_attempt_lower_bound += 1
            provider_attempt_upper_bound += 1
            record = records[0]
            if record.status == "success":
                successful_provider_attempt_count += 1
                if record.input_tokens is None or record.output_tokens is None:
                    raise ModalActionJournalIntegrityError(
                        "successful migration provider usage is incomplete"
                    )
                provider_input_tokens += record.input_tokens
                provider_output_tokens += record.output_tokens
            else:
                failed_provider_attempt_count += 1
        else:
            provider_attempt_upper_bound += 1
        if (
            provider_record["parse_dispositions"] != expected_parse_dispositions
            or provider_record["provider_attempt_count"] != len(records)
            or provider_record["request_ids"] != request_ids
            or provider_record["response_ids"] != response_ids
        ):
            raise ModalActionJournalIntegrityError(
                "migration provider evidence accounting changed"
            )
        if actual_request_ids.intersection(
            request_ids
        ) or actual_response_ids.intersection(response_ids):
            raise ModalActionJournalIntegrityError(
                "migration provider evidence reuses provider IDs"
            )
        actual_request_ids.update(request_ids)
        actual_response_ids.update(response_ids)
        observed_provider_keys.add(key)

    if set(provider_map) != observed_provider_keys:
        raise ModalActionJournalIntegrityError(
            "migration provider evidence roster is incomplete"
        )
    spend = selected["provider_spend_estimate"]
    expected_provider_counts = {
        "provider_launcher_attempt_count": len(provider_launcher_attempt_ids),
        "provider_terminal_attempt_record_count": provider_terminal_record_count,
        "provider_attempt_count_lower_bound": provider_attempt_lower_bound,
        "provider_attempt_count_upper_bound": provider_attempt_upper_bound,
        "successful_provider_attempt_count": successful_provider_attempt_count,
        "failed_provider_attempt_count": failed_provider_attempt_count,
        "input_tokens": provider_input_tokens,
        "output_tokens": provider_output_tokens,
        "total_tokens": provider_input_tokens + provider_output_tokens,
    }
    if (
        any(
            spend[name] != expected
            for name, expected in expected_provider_counts.items()
        )
        or spend["provider_request_ids"] != sorted(actual_request_ids)
        or spend["provider_response_ids"] != sorted(actual_response_ids)
    ):
        raise ModalActionJournalIntegrityError(
            "migration provider spend counts or IDs differ from bound evidence"
        )


def _validate_migration_terminal_seal_snapshot(
    reader: _NamespaceReader,
    cohorts: Sequence[ModalCohortActionJournal],
    reservations: Sequence[ModalJournalRecord],
    global_rejection_seal: ModalJournalRecord | None,
    process_markers: Sequence[ModalJournalRecord],
    attempts: Sequence[ModalAttemptJournalState],
) -> None:
    sealed = [cohort for cohort in cohorts if cohort.sealed]
    if not sealed:
        return
    if len(sealed) != 1:
        raise ModalActionJournalIntegrityError(
            "global journal contains multiple migration terminal seals"
        )
    final_cohort = sealed[0]
    seal_record = final_cohort.migration_terminal_seal
    if seal_record is None:  # pragma: no cover - narrowed by ``sealed``
        raise AssertionError("sealed cohort lacks its seal record")
    payload = seal_record.payload
    sealed_at = _utc(
        payload["recorded_at_utc"],
        "migration_terminal_seal.recorded_at_utc",
    )
    if global_rejection_seal is None:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal lacks its global launch-rejection seal"
        )
    rejection_sealed_at = _utc(
        global_rejection_seal.payload["recorded_at_utc"],
        "migration_terminal_seal.global_rejection_seal.recorded_at_utc",
    )
    if rejection_sealed_at > sealed_at:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal predates its global launch-rejection seal"
        )
    if any(
        _utc(
            record.payload["created_at_utc"],
            "migration_terminal_seal.process_marker.created_at_utc",
        )
        > sealed_at
        for record in process_markers
    ):
        raise ModalActionJournalIntegrityError(
            "global action journal contains content created after its migration seal"
        )
    selected = payload["selected_final"]
    prior_records = payload["prior_quarantined_cohorts"]
    prior_by_identity = {
        _lineage_identity(
            record["identity"],
            field="migration_terminal_seal.prior.identity",
        ): record
        for record in prior_records
    }
    expected_identities = {final_cohort.identity, *prior_by_identity}
    actual_by_identity = {cohort.identity: cohort for cohort in cohorts}
    if set(actual_by_identity) != expected_identities:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal omits or invents a live cohort"
        )
    frozen_times = [
        _utc(
            record.payload["created_at_utc"],
            "migration_terminal_seal.intent.created_at_utc",
        )
        for cohort in cohorts
        for record in cohort.intents
    ]
    frozen_times.extend(
        _utc(
            record.payload["finished_at_utc"],
            "migration_terminal_seal.terminal.finished_at_utc",
        )
        for cohort in cohorts
        for record in cohort.terminals
    )
    frozen_times.extend(
        _utc(
            record.payload["created_at_utc"],
            "migration_terminal_seal.reservation.created_at_utc",
        )
        for record in reservations
    )
    if any(timestamp > sealed_at for timestamp in frozen_times):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal predates its frozen journal evidence"
        )

    if selected["action_journal"] != _cohort_journal_binding_snapshot(final_cohort):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal final action-journal snapshot changed"
        )
    for prior_identity, prior in prior_by_identity.items():
        if prior["action_journal"] != _cohort_journal_binding_snapshot(
            actual_by_identity[prior_identity]
        ):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal prior action-journal snapshot changed"
            )

    global_reservations = [
        _record_binding_dict(record)
        for record in sorted(reservations, key=lambda item: item.binding.path)
    ]
    if payload["global_remote_run_reservations"] != global_reservations:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal global reservation snapshot changed"
        )
    reservations_by_identity: dict[ModalLiveCohortIdentity, list[dict[str, Any]]] = {
        identity: [] for identity in expected_identities
    }
    for record in reservations:
        owner = _identity_from_payload(
            record.payload,
            image_field="image_source_sha256",
            allow_absent=False,
            field="migration_terminal_seal.reservation",
        )
        if owner not in reservations_by_identity:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal reservation names an unknown cohort"
            )
        reservations_by_identity[owner].append(_record_binding_dict(record))
    for bindings in reservations_by_identity.values():
        bindings.sort(key=lambda item: item["path"])
    if (
        selected["remote_run_reservations"]
        != reservations_by_identity[final_cohort.identity]
    ):
        raise ModalActionJournalIntegrityError(
            "migration terminal seal final reservation snapshot changed"
        )
    for prior_identity, prior in prior_by_identity.items():
        if prior["remote_run_reservations"] != reservations_by_identity[prior_identity]:
            raise ModalActionJournalIntegrityError(
                "migration terminal seal prior reservation snapshot changed"
            )

    states_by_attempt = {state.attempt_id: state for state in attempts}
    accepted_runs = selected["accepted_primary_runs"]
    accepted_attempt_ids = selected["accepted_attempt_ids"]
    for label, (expected_action, expected_harness) in _LINEAGE_PRIMARY_ACTIONS.items():
        attempt_id = accepted_attempt_ids[label]
        state = states_by_attempt.get(attempt_id)
        terminal = state.terminal.payload if state and state.terminal else None
        if (
            state is None
            or state.identity != final_cohort.identity
            or state.disposition != "closed"
            or terminal is None
            or terminal["status"] != "succeeded"
            or terminal["action"] != expected_action
            or terminal["harness"] != expected_harness
            or terminal["run_id"] != accepted_runs[label]
            or terminal["concrete_remote_run_ids"] != [accepted_runs[label]]
        ):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal accepted primary roster changed"
            )

    actual_dispositions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for state in attempts:
        if state.identity not in expected_identities or state.terminal is None:
            continue
        terminal = state.terminal.payload
        for run_id in terminal["concrete_remote_run_ids"]:
            actual_dispositions[(state.attempt_id, run_id)] = terminal
    sealed_dispositions = {
        (record["attempt_id"], record["run_id"]): record
        for record in selected["run_dispositions"]
    }
    final_actual_keys = {
        key
        for key, terminal in actual_dispositions.items()
        if _identity_from_payload(
            terminal,
            image_field="approved_image_source_sha256",
            allow_absent=False,
            field="migration_terminal_seal.terminal",
        )
        == final_cohort.identity
    }
    if set(sealed_dispositions) != final_actual_keys:
        raise ModalActionJournalIntegrityError(
            "migration terminal seal run-disposition roster changed"
        )
    for key, record in sealed_dispositions.items():
        terminal = actual_dispositions[key]
        if any(
            record[name] != terminal[name]
            for name in (
                "action",
                "status",
                "failure_kind",
                "modal_cli_process_started",
                "remote_execution_state",
            )
        ):
            raise ModalActionJournalIntegrityError(
                "migration terminal seal run disposition changed"
            )
    _validate_migration_terminal_evidence_snapshot(
        reader,
        identity=final_cohort.identity,
        selected=selected,
        states_by_attempt=states_by_attempt,
    )


def scan_modal_global_action_journal(
    *,
    lock_descriptor: int,
    process_probe: ProcessProbe | None = None,
) -> ModalGlobalJournalScan:
    """Return one stable, complete, held-lock snapshot of local action state.

    A process probe, when supplied, is always passed the project root derived
    from ``lock_descriptor``.  The callback therefore cannot accidentally be
    bound by the scanner to an independently selected checkout.
    """

    project_root = held_modal_action_lock_project_root(lock_descriptor)
    reader = _NamespaceReader(project_root)
    try:
        cohorts = _scan_cohort_journals(reader)
        reservations = _scan_global_reservations(reader)
        rejections, global_rejection_seal = _scan_global_rejections(reader)
        _validate_global_launch_rejection_seal_snapshot(
            rejections,
            global_rejection_seal,
        )
        markers = _scan_private_process_markers(reader)
        collected = _collect_attempts(
            cohorts,
            reservations,
            rejections,
            markers,
        )
        attempts, blockers = _classify_attempts(
            reader,
            project_root,
            cohorts,
            collected,
            process_probe=process_probe,
        )
        _validate_migration_terminal_seal_snapshot(
            reader,
            cohorts,
            reservations,
            global_rejection_seal,
            markers,
            attempts,
        )
        reader.require_stable()
        assert_modal_action_lock_identity(lock_descriptor)
        if held_modal_action_lock_project_root(lock_descriptor) != project_root:
            raise ModalActionJournalIntegrityError(
                "held lock project root changed during journal discovery"
            )
        return ModalGlobalJournalScan(
            project_root=project_root,
            cohorts=tuple(
                sorted(
                    cohorts,
                    key=lambda item: (
                        item.identity.source_tree_sha256,
                        item.identity.image_source_sha256,
                        item.identity.cohort_id,
                    ),
                )
            ),
            attempts=attempts,
            reservations=tuple(
                sorted(reservations, key=lambda item: item.binding.path)
            ),
            rejections=tuple(sorted(rejections, key=lambda item: item.binding.path)),
            global_rejection_seal=global_rejection_seal,
            process_markers=tuple(sorted(markers, key=lambda item: item.binding.path)),
            blockers=blockers,
        )
    finally:
        reader.close()


def require_modal_global_action_journal_resolved(
    scan: ModalGlobalJournalScan,
) -> None:
    """Require no unresolved attempts while deliberately permitting seals."""

    if scan.blockers:
        codes = ", ".join(sorted({blocker.code for blocker in scan.blockers}))
        raise ModalActionJournalBlockedError(
            f"global Modal action journal is unresolved: {codes}"
        )


def build_modal_global_launch_rejection_seal_payload(
    scan: ModalGlobalJournalScan,
    *,
    recorded_at_utc: str,
) -> dict[str, Any]:
    """Build the exact create-only rejection roster from one held-lock scan."""

    require_modal_global_action_journal_resolved(scan)
    if scan.global_rejection_seal is not None:
        raise ModalActionJournalIntegrityError(
            "global launch-rejection journal is already sealed"
        )
    if any(cohort.sealed for cohort in scan.cohorts):
        raise ModalActionJournalIntegrityError("migration terminal seal already exists")
    recorded_at = _canonical_utc(
        recorded_at_utc,
        "global_launch_rejection_seal.recorded_at_utc",
    )
    if any(
        _utc(
            record.payload["finished_at_utc"],
            "global_launch_rejection.finished_at_utc",
        )
        > recorded_at
        for record in scan.rejections
    ):
        raise ModalActionJournalIntegrityError(
            "global launch-rejection seal predates a frozen rejection"
        )
    payload: dict[str, Any] = {
        "schema_name": "ModalGlobalLaunchRejectionSeal",
        "schema_version": "1.0",
        "recorded_at_utc": recorded_at_utc,
        "rejection_receipts": [
            _record_binding_dict(record)
            for record in sorted(
                scan.rejections,
                key=lambda item: item.binding.path,
            )
        ],
        "validated": True,
    }
    return _validate_global_launch_rejection_seal_core(payload)


def require_modal_global_action_gate_clear(
    scan: ModalGlobalJournalScan,
    *,
    candidate_attempt_id: str,
) -> None:
    """Reject a new attempt ID, a seal, or unresolved global action state."""

    selected = _attempt_id(candidate_attempt_id, "candidate_attempt_id")
    if any(attempt.attempt_id == selected for attempt in scan.attempts):
        raise ModalActionJournalIntegrityError(
            "candidate attempt ID is already present in the global journal"
        )
    if any(cohort.sealed for cohort in scan.cohorts):
        raise ModalActionJournalBlockedError(
            "global Modal action journal is terminally sealed: "
            "migration_terminal_seal_present"
        )
    if scan.global_rejection_seal is not None:
        raise ModalActionJournalBlockedError(
            "global Modal launch-rejection journal is sealed: "
            "global_launch_rejection_seal_present"
        )
    require_modal_global_action_journal_resolved(scan)
