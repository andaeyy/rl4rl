from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import modal_action_journal as journal
import pytest
from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from common.modal_action_lock import (
    acquire_modal_action_lock,
    release_modal_action_lock,
)
from modal_boundary import (
    APP_NAME,
    CANARY_ORDER,
    MODAL_DOWNLOAD_OUTPUT_ROOT,
    MODAL_LAUNCH_REJECTION_ROOT,
    MODAL_LIVE_COHORT_ROOT,
    MODAL_REMOTE_RUN_RESERVATION_ROOT,
    ArtifactManifestV1,
    ModalLiveCohortIdentity,
    canary_run_suffix,
    modal_action_host_containment_path,
    modal_action_intent_receipt_path,
    modal_action_recovery_intent_path,
    modal_action_terminal_receipt_path,
    modal_global_launch_rejection_seal_path,
    modal_launch_rejection_receipt_path,
    modal_local_host_anchor_path,
    modal_local_process_start_receipt_path,
    modal_migration_lineage_path,
    provider_canary_aggregate_outcome_receipt_path,
    volume_artifact_uri,
)
from scripts import recover_modal_action_journal as recovery

ATTEMPT_ID = "1" * 32
SOURCE_SHA256 = "2" * 64
IMAGE_SHA256 = "3" * 64
HOST_ANCHOR_SHA256 = "4" * 64
BOOT_SESSION_SHA256 = "5" * 64
MODAL_COMMAND_SHA256 = "6" * 64
LAUNCH_CAPABILITY_SHA256 = "7" * 64
PRICE_BASIS_SHA256 = "8" * 64
PROCESS_BIRTH_SHA256 = "9" * 64
CREATED_AT_UTC = "2026-08-10T00:00:00Z"
FINISHED_AT_UTC = "2026-08-10T00:00:01Z"
BOOT_STARTED_AT_UNIX_MICROSECONDS = 1_754_780_000_000_000


def _identity(
    *,
    source_tree_sha256: str = SOURCE_SHA256,
    image_source_sha256: str = IMAGE_SHA256,
    cohort_id: str = "modal-test-cohort",
) -> ModalLiveCohortIdentity:
    return ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256,
        image_source_sha256=image_source_sha256,
        cohort_id=cohort_id,
    )


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _write_json(path: Path, payload: Any) -> str:
    encoded = _canonical_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _record_binding(record: journal.ModalJournalRecord) -> dict[str, Any]:
    return {
        "path": record.binding.path,
        "sha256": record.binding.sha256,
        "size_bytes": record.binding.size_bytes,
    }


def _write_recovery_request(
    root: Path,
    scan: journal.ModalGlobalJournalScan,
    *,
    branch: str,
    snapshot_manifest_path: Path | None = None,
    fresh_candidate_attempt_id: str = "a" * 32,
) -> Path:
    state = next(item for item in scan.attempts if item.attempt_id == ATTEMPT_ID)
    request_path = root / "operator" / f"recovery-{branch}.json"
    _write_json(
        request_path,
        {
            "schema_name": journal.RECOVERY_REQUEST_SCHEMA_NAME,
            "schema_version": journal.RECOVERY_REQUEST_SCHEMA_VERSION,
            "attempt_id": ATTEMPT_ID,
            "fresh_candidate_attempt_id": fresh_candidate_attempt_id,
            "expected_branch": branch,
            "snapshot_manifest_path": (
                str(snapshot_manifest_path)
                if snapshot_manifest_path is not None
                else None
            ),
            "initial_reservation_bindings": [
                _record_binding(record) for record in state.reservations
            ],
        },
    )
    return request_path


def _write_recovery_snapshot(
    root: Path,
    identity: ModalLiveCohortIdentity,
    *,
    billing_rows: list[dict[str, Any]] | None = None,
    app_state: str = "stopped",
    app_tasks: str = "0",
    include_target_volume: bool = True,
) -> Path:
    capture_id = "recovery-capture-001"
    capture_root = (
        root
        / journal.modal_live_cohort_root(identity)
        / "resource_cleanup"
        / "snapshot_captures"
        / capture_id
    )
    rows: dict[str, list[dict[str, Any]]] = {
        "app_list": [
            {
                "app_id": "ap-recovery-test",
                "description": APP_NAME,
                "state": app_state,
                "tasks": app_tasks,
                "created_at": "2026-08-10T00:00:00+00:00",
                "stopped_at": (
                    "2026-08-10T00:30:00+00:00"
                    if app_state == "stopped"
                    else None
                ),
            }
        ],
        "container_list": [],
        "endpoint_list": [],
        "volume_list": (
            [
                {
                    "name": journal.VOLUME_NAME,
                    "created_at": "2026-08-10T00:00:00+00:00",
                    "created_by": "test-user",
                }
            ]
            if include_target_volume
            else []
        ),
        "run_directory_list": [
            {
                "filename": f"/runs/{identity.cohort_id}",
                "type": "dir",
                "created_modified": "2026-08-10 00:30 UTC",
                "size": "0 B",
            }
        ],
        "billing_report": (
            billing_rows
            if billing_rows is not None
            else [
                {
                    "object_id": "ap-recovery-test",
                    "description": APP_NAME,
                    "environment": "main",
                    "interval_start": "2026-08-10T00:00:00+00:00",
                    "resource": "T4 GPU",
                    "cost": "0.01",
                }
            ]
        ),
    }
    snapshots: dict[str, dict[str, Any]] = {}
    executable = "/dev/fd/99"
    for name in journal.RECOVERY_SNAPSHOT_NAMES:
        leaf = capture_root / f"{name}.json"
        digest = _write_json(leaf, rows[name])
        snapshots[name] = {
            "path": leaf.relative_to(root).as_posix(),
            "sha256": digest,
            "size_bytes": leaf.stat().st_size,
            "argv": [
                executable,
                *journal._snapshot_command_suffix(
                    name,
                    billing_start="2026-08-10T00:00:00Z",
                    billing_end="2026-08-10T01:00:00Z",
                ),
            ],
            "captured_at_utc": "2026-08-10T02:00:00Z",
        }
    manifest_path = capture_root / journal.RECOVERY_SNAPSHOT_MANIFEST_FILENAME
    _write_json(
        manifest_path,
        {
            "schema_name": journal.RECOVERY_SNAPSHOT_MANIFEST_SCHEMA_NAME,
            "schema_version": journal.RECOVERY_SNAPSHOT_MANIFEST_SCHEMA_VERSION,
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "capture_id": capture_id,
            "modal_profile": "scalingintelligence",
            "modal_environment": "main",
            "modal_cli_version": journal.MODAL_VERSION,
            "billing_window_start_utc": "2026-08-10T00:00:00Z",
            "billing_window_end_utc": "2026-08-10T01:00:00Z",
            "started_at_utc": "2026-08-10T02:00:00Z",
            "finished_at_utc": "2026-08-10T02:00:00Z",
            "command_timeout_seconds": 60,
            "outer_timeout_seconds": 420,
            "command_retry_count": 0,
            "snapshots": snapshots,
        },
    )
    return manifest_path


def _containment() -> dict[str, Any]:
    return {
        "local_host_anchor_path": modal_local_host_anchor_path().as_posix(),
        "local_host_anchor_sha256": HOST_ANCHOR_SHA256,
        "local_boot_started_at_unix_microseconds": (BOOT_STARTED_AT_UNIX_MICROSECONDS),
        "local_boot_session_sha256": BOOT_SESSION_SHA256,
    }


def _intent_and_reservations(
    *,
    identity: ModalLiveCohortIdentity | None = None,
    attempt_id: str = ATTEMPT_ID,
) -> tuple[dict[str, Any], tuple[journal.ModalRemoteRunReservationSpec, ...]]:
    selected = identity or _identity()
    concrete_run_ids = (selected.cohort_id,)
    specs = journal.build_modal_remote_run_reservation_specs(
        concrete_remote_run_ids=concrete_run_ids,
        attempt_id=attempt_id,
        action="cuda-environment",
        identity=selected,
        created_at_utc=CREATED_AT_UTC,
        launch_capability_sha256=LAUNCH_CAPABILITY_SHA256,
        **_containment(),
    )
    intent: dict[str, Any] = {
        "schema_name": "ModalActionIntent",
        "schema_version": "1.6",
        "attempt_id": attempt_id,
        "created_at_utc": CREATED_AT_UTC,
        "action": "cuda-environment",
        "run_id": selected.cohort_id,
        "concrete_remote_run_ids": list(concrete_run_ids),
        "remote_run_reservations": [dict(spec.binding) for spec in specs],
        **_containment(),
        "source_run_id": None,
        "verifier_run_id": None,
        "harness": None,
        "source_tree_sha256": selected.source_tree_sha256,
        "cohort_id": selected.cohort_id,
        "approved_image_source_sha256": selected.image_source_sha256,
        "modal_command_sha256": MODAL_COMMAND_SHA256,
        "launch_capability_sha256": LAUNCH_CAPABILITY_SHA256,
        "modal_profile": "scalingintelligence",
        "modal_environment": "main",
        "outer_cli_timeout_seconds": journal._expected_outer_cli_timeout_seconds(
            "cuda-environment"
        ),
        "modal_cost_cap_usd": "0.25",
        "modal_resource_profile": journal._expected_modal_resource_profile(
            "cuda-environment",
            None,
        ),
        "modal_price_basis_path": "outputs/readiness/modal-price.json",
        "modal_price_basis_sha256": PRICE_BASIS_SHA256,
        "modal_cost_estimate": {"action_estimate_usd": "0.01"},
        "modal_cost_approved": True,
        "provider_cost_approved": False,
        "provider_cost_cap_usd": None,
        "provider_approval_plan_path": None,
        "approval_plan_sha256": None,
        "provider_price_basis_path": None,
        "provider_price_basis_sha256": None,
        "predecessor_receipts": [
            {
                "gate": gate,
                "path": f"outputs/readiness/local-freeze-{index}.json",
                "sha256": "c" * 64,
            }
            for index, gate in enumerate(journal._LOCAL_ENGINEERING_FREEZE_GATES)
        ],
        "source_evidence_recovery": False,
    }
    return intent, specs


def _terminal(
    intent: Mapping[str, Any],
    *,
    started: bool = False,
    marker_sha256: str | None = None,
    process_id: int = 424_242,
) -> dict[str, Any]:
    terminal = {
        key: value
        for key, value in intent.items()
        if key
        not in {
            "schema_name",
            "schema_version",
            "attempt_id",
            "created_at_utc",
        }
    }
    if started:
        status = "succeeded" if marker_sha256 is not None else "cli_failed"
        failure_kind = (
            None if marker_sha256 is not None else "process_start_receipt_persistence"
        )
        terminal.update(
            {
                "local_process_start_receipt_path": (
                    modal_local_process_start_receipt_path(
                        intent["attempt_id"]
                    ).as_posix()
                ),
                "local_process_start_receipt_sha256": marker_sha256,
                "local_process_id": (process_id if marker_sha256 is not None else None),
                "local_process_group_id": (
                    process_id if marker_sha256 is not None else None
                ),
                "local_session_id": (process_id if marker_sha256 is not None else None),
                "modal_cli_process_started": True,
                "remote_execution_state": "may_have_started",
                "returncode": 0 if marker_sha256 is not None else None,
                "process_group_closed": True,
            }
        )
    else:
        status = "preflight_rejected"
        failure_kind = "preflight"
        terminal.update(
            {
                "local_process_start_receipt_path": None,
                "local_process_start_receipt_sha256": None,
                "local_process_id": None,
                "local_process_group_id": None,
                "local_session_id": None,
                "modal_cli_process_started": False,
                "remote_execution_state": "definitely_not_started",
                "returncode": None,
                "process_group_closed": None,
            }
        )
    terminal.update(
        {
            "schema_name": "ModalActionAttemptReceipt",
            "schema_version": "3.6",
            "attempt_id": intent["attempt_id"],
            "started_at_utc": intent["created_at_utc"],
            "finished_at_utc": FINISHED_AT_UTC,
            "status": status,
            "failure_kind": failure_kind,
        }
    )
    return terminal


def _global_rejection(
    intent: Mapping[str, Any],
    *,
    retain_reservation_roster: bool,
) -> dict[str, Any]:
    payload = _terminal(intent)
    payload.update(
        {
            "status": "preflight_failed",
            "failure_kind": "action_intent_persistence_uncertain",
            "remote_run_reservations": (
                payload["remote_run_reservations"] if retain_reservation_roster else []
            ),
        }
    )
    return payload


def _partial_preownership_rejection(
    intent: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _global_rejection(intent, retain_reservation_roster=False)
    payload.update(
        {
            "status": "lock_contended",
            "failure_kind": "local_launcher_lock",
            "source_tree_sha256": None,
            "modal_command_sha256": None,
            "launch_capability_sha256": None,
            "modal_cost_estimate": None,
            **{name: None for name in _containment()},
        }
    )
    return payload


def _canaries_intent_and_reservations(
    *,
    identity: ModalLiveCohortIdentity | None = None,
    attempt_id: str = ATTEMPT_ID,
) -> tuple[dict[str, Any], tuple[journal.ModalRemoteRunReservationSpec, ...]]:
    selected = identity or _identity()
    intent, _old_specs = _intent_and_reservations(
        identity=selected,
        attempt_id=attempt_id,
    )
    run_id = "provider-canaries"
    concrete_run_ids = tuple(
        f"{run_id}-{canary_run_suffix(harness)}" for harness in CANARY_ORDER
    )
    specs = journal.build_modal_remote_run_reservation_specs(
        concrete_remote_run_ids=concrete_run_ids,
        attempt_id=attempt_id,
        action="canaries",
        identity=selected,
        created_at_utc=CREATED_AT_UTC,
        launch_capability_sha256=LAUNCH_CAPABILITY_SHA256,
        **_containment(),
    )
    intent.update(
        {
            "action": "canaries",
            "run_id": run_id,
            "concrete_remote_run_ids": list(concrete_run_ids),
            "remote_run_reservations": [dict(spec.binding) for spec in specs],
            "provider_cost_approved": True,
            "provider_cost_cap_usd": "1",
            "provider_approval_plan_path": "outputs/readiness/provider-plan.json",
            "approval_plan_sha256": "a" * 64,
            "provider_price_basis_path": "outputs/readiness/provider-price.json",
            "provider_price_basis_sha256": "b" * 64,
            "outer_cli_timeout_seconds": journal._expected_outer_cli_timeout_seconds(
                "canaries"
            ),
            "modal_resource_profile": journal._expected_modal_resource_profile(
                "canaries",
                None,
            ),
            "predecessor_receipts": [
                *intent["predecessor_receipts"],
                {
                    "gate": "candidate_resume_preflight_validated",
                    "path": "outputs/readiness/candidate-preflight.json",
                    "sha256": "d" * 64,
                },
            ],
        }
    )
    return intent, specs


def _provider_aggregate(
    identity: ModalLiveCohortIdentity,
    *,
    all_succeeded: bool,
) -> dict[str, Any]:
    return {
        "schema_name": "ProviderCanaryAggregateOutcomeReceipt",
        "schema_version": "1.1",
        "attempt_id": ATTEMPT_ID,
        "run_id_prefix": "provider-canaries",
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
        "harness_order": list(CANARY_ORDER),
        "outcomes": [
            {
                "harness": harness,
                "run_id": f"provider-canaries-{canary_run_suffix(harness)}",
                "status": ("success" if all_succeeded or index else "failed"),
                "error_type": (None if all_succeeded or index else "RuntimeError"),
            }
            for index, harness in enumerate(CANARY_ORDER)
        ],
        "all_succeeded": all_succeeded,
    }


def _process_marker(
    intent: Mapping[str, Any],
    *,
    intent_sha256: str,
    process_id: int = 424_242,
) -> dict[str, Any]:
    return {
        "schema_name": "ModalLocalProcessStart",
        "schema_version": "1.1",
        "attempt_id": intent["attempt_id"],
        "created_at_utc": "2026-08-10T00:00:00.500000Z",
        "action": intent["action"],
        "run_id": intent["run_id"],
        "intent_path": modal_action_intent_receipt_path(
            _identity(
                source_tree_sha256=intent["source_tree_sha256"],
                image_source_sha256=intent["approved_image_source_sha256"],
                cohort_id=intent["cohort_id"],
            ),
            intent["attempt_id"],
        ).as_posix(),
        "intent_sha256": intent_sha256,
        "source_tree_sha256": intent["source_tree_sha256"],
        "image_source_sha256": intent["approved_image_source_sha256"],
        "cohort_id": intent["cohort_id"],
        "modal_command_sha256": intent["modal_command_sha256"],
        "launch_capability_sha256": intent["launch_capability_sha256"],
        "modal_cost_cap_usd": intent["modal_cost_cap_usd"],
        "provider_cost_cap_usd": intent["provider_cost_cap_usd"],
        **_containment(),
        "process_id": process_id,
        "expected_process_group_id": process_id,
        "expected_session_id": process_id,
        "process_birth_identity_sha256": PROCESS_BIRTH_SHA256,
    }


def _migration_terminal_seal(
    identity: ModalLiveCohortIdentity,
    *,
    project_root: Path,
    accepted_primary_runs: Mapping[str, str] | None = None,
    accepted_attempt_ids: Mapping[str, str] | None = None,
    recorded_at_utc: str = FINISHED_AT_UTC,
) -> dict[str, Any]:
    def binding(path: Path, logical: str) -> dict[str, Any]:
        raw = path.read_bytes()
        return {
            "path": logical,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }

    action_root = modal_action_intent_receipt_path(identity, ATTEMPT_ID).parent
    action_directory = project_root / action_root
    journal_bindings: dict[str, list[dict[str, Any]]] = {
        "intent_receipts": [],
        "terminal_receipts": [],
        "aggregate_receipts": [],
    }
    terminal_payloads: list[dict[str, Any]] = []
    if action_directory.is_dir():
        for path in sorted(action_directory.iterdir()):
            logical = (action_root / path.name).as_posix()
            if path.name.endswith(".intent.json"):
                field = "intent_receipts"
            elif path.name.endswith(".aggregate.json"):
                field = "aggregate_receipts"
            else:
                field = "terminal_receipts"
                terminal_payloads.append(json.loads(path.read_text()))
            journal_bindings[field].append(binding(path, logical))

    reservation_bindings = []
    reservation_directory = project_root / MODAL_REMOTE_RUN_RESERVATION_ROOT
    if reservation_directory.is_dir():
        for path in sorted(reservation_directory.iterdir()):
            reservation_bindings.append(
                binding(
                    path,
                    (MODAL_REMOTE_RUN_RESERVATION_ROOT / path.name).as_posix(),
                )
            )
    labels = tuple(journal._LINEAGE_PRIMARY_ACTIONS)
    accepted_runs = (
        dict(accepted_primary_runs)
        if accepted_primary_runs
        else {
            label: (identity.cohort_id if index == 0 else f"accepted-run-{index}")
            for index, label in enumerate(labels)
        }
    )
    accepted_attempts = (
        dict(accepted_attempt_ids)
        if accepted_attempt_ids
        else {label: f"{index + 1:x}" * 32 for index, label in enumerate(labels)}
    )
    run_dispositions = [
        {
            "attempt_id": terminal["attempt_id"],
            "action": terminal["action"],
            "status": terminal["status"],
            "failure_kind": terminal["failure_kind"],
            "run_id": run_id,
            "modal_cli_process_started": terminal["modal_cli_process_started"],
            "remote_execution_state": terminal["remote_execution_state"],
            "execution_disposition": (
                "remote_execution_bound"
                if terminal["modal_cli_process_started"]
                else "definitely_not_started"
            ),
            "provider_disposition": (
                "evidence_bound"
                if terminal["action"] in {"canary", "canaries"}
                and terminal["modal_cli_process_started"]
                else "definitely_not_started"
                if terminal["action"] in {"canary", "canaries"}
                else "not_applicable"
            ),
        }
        for terminal in terminal_payloads
        for run_id in terminal["concrete_remote_run_ids"]
    ]
    remote_executions: list[dict[str, Any]] = []
    artifact_manifests: list[dict[str, Any]] = []
    provider_attempt_evidence: list[dict[str, Any]] = []
    remote_object_ids: dict[str, set[str]] = {
        "app_ids": set(),
        "function_ids": set(),
        "call_ids": set(),
        "image_ids": set(),
    }
    for terminal in terminal_payloads:
        for run_id in terminal["concrete_remote_run_ids"]:
            logical_root = PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT) / run_id
            context_path = project_root / logical_root / "execution_context.json"
            if not context_path.is_file():
                continue
            context = json.loads(context_path.read_text())
            remote_executions.append(
                {
                    "attempt_id": terminal["attempt_id"],
                    "run_id": run_id,
                    "action": terminal["action"],
                    "evidence_kind": "downloaded_execution_context",
                    "evidence": binding(
                        context_path,
                        (logical_root / "execution_context.json").as_posix(),
                    ),
                    "execution_context": context,
                }
            )
            for field, context_field in (
                ("app_ids", "modal_app_id"),
                ("function_ids", "modal_function_id"),
                ("call_ids", "modal_call_id"),
                ("image_ids", "modal_image_id"),
            ):
                remote_object_ids[field].add(context[context_field])
            manifest_path = project_root / logical_root / "artifact_manifest.json"
            manifest = ArtifactManifestV1.from_dict(
                json.loads(manifest_path.read_text())
            )
            artifact_manifests.append(
                {
                    "attempt_id": terminal["attempt_id"],
                    "run_id": run_id,
                    **binding(
                        manifest_path,
                        (logical_root / "artifact_manifest.json").as_posix(),
                    ),
                    "canonical_manifest_sha256": manifest.manifest_sha256,
                }
            )
            if terminal["action"] in {"canary", "canaries"}:
                harness = (
                    terminal["harness"]
                    if terminal["action"] == "canary"
                    else next(
                        item
                        for item in CANARY_ORDER
                        if run_id == f"{terminal['run_id']}-{canary_run_suffix(item)}"
                    )
                )
                uncertainty_path = (
                    project_root
                    / logical_root
                    / "controller"
                    / "provider_request_start_uncertain.json"
                )
                provider_attempt_evidence.append(
                    {
                        "attempt_id": terminal["attempt_id"],
                        "run_id": run_id,
                        "harness": harness,
                        "binding_state": "execution_context_bound",
                        "ledger": None,
                        "uncertainty": binding(
                            uncertainty_path,
                            (
                                logical_root
                                / "controller"
                                / "provider_request_start_uncertain.json"
                            ).as_posix(),
                        ),
                        "parse_dispositions": ["valid_start_uncertain"],
                        "provider_attempt_count": 0,
                        "request_ids": [],
                        "response_ids": [],
                    }
                )
    remote_executions.sort(key=lambda item: (item["attempt_id"], item["run_id"]))
    artifact_manifests.sort(key=lambda item: (item["attempt_id"], item["run_id"]))
    provider_attempt_evidence.sort(
        key=lambda item: (item["attempt_id"], item["run_id"])
    )
    provider_launcher_attempt_count = len(
        {
            terminal["attempt_id"]
            for terminal in terminal_payloads
            if terminal["action"] in {"canary", "canaries"}
        }
    )
    provider_attempt_upper_bound = sum(
        len(terminal["concrete_remote_run_ids"])
        for terminal in terminal_payloads
        if terminal["action"] in {"canary", "canaries"}
        and terminal["modal_cli_process_started"]
    )
    provider_spend_estimate = {
        "accounting_label": "test_fixture_zero_provider_spend",
        "provider_launcher_attempt_count": provider_launcher_attempt_count,
        "provider_terminal_attempt_record_count": 0,
        "provider_attempt_count_lower_bound": 0,
        "provider_attempt_count_upper_bound": provider_attempt_upper_bound,
        "successful_provider_attempt_count": 0,
        "failed_provider_attempt_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "known_success_usage_estimate_usd": "0",
        "failed_attempt_reserve_usd": "0",
        "uncertain_request_start_reserve_usd": "0",
        "conservative_provider_spend_bound_usd": "0",
        "approved_provider_cap_total_usd": "0",
        "provider_request_ids": [],
        "provider_response_ids": [],
        "run_cost_dispositions": [],
        "launcher_approval_bounds": [],
    }
    return {
        "schema_name": "ModalMigrationLineage",
        "schema_version": "1.1",
        "recorded_at_utc": recorded_at_utc,
        "selected_final": {
            "identity": {
                "source_tree_sha256": identity.source_tree_sha256,
                "image_source_sha256": identity.image_source_sha256,
                "cohort_id": identity.cohort_id,
            },
            "accepted_primary_runs": accepted_runs,
            "accepted_attempt_ids": accepted_attempts,
            "action_journal": journal_bindings,
            "remote_run_reservations": reservation_bindings,
            "run_dispositions": run_dispositions,
            "aggregate_receipts": journal_bindings["aggregate_receipts"],
            "remote_executions": remote_executions,
            "remote_object_ids": {
                field: sorted(values) for field, values in remote_object_ids.items()
            },
            "provider_attempt_evidence": provider_attempt_evidence,
            "provider_spend_estimate": provider_spend_estimate,
            "artifact_manifests": artifact_manifests,
        },
        "prior_quarantined_cohorts": [],
        "global_remote_run_reservations": reservation_bindings,
        "legacy_superseded_usage": {
            "run_id": "modal-cuda-env-20260809-02",
            "amount_usd": "0.00643852",
            "accounting_basis": (
                "preserved_legacy_measurement_excluded_from_all_cohort_snapshots"
            ),
        },
        "prior_app_compute_total_usd": "0",
        "final_provider_spend_bound_usd": "0",
        "prior_provider_spend_bound_usd": "0",
        "migration_provider_spend_bound_usd": "0",
        "prior_modal_measured_app_billing_usd": "0",
        "prior_modal_unresolved_compute_reserve_usd": "0",
        "prior_modal_conservative_exposure_usd": "0",
        "retained_storage_estimate": {
            "prior_cohort_estimates": [],
            "final_cohort_included": False,
            "basis": (
                "prior_quarantine_receipts_only; final retained storage is "
                "reported by the cleanup receipt"
            ),
        },
        "global_uniqueness_validated": True,
        "validated": True,
    }


def _publish_global_rejection_seal(
    project_root: Path,
    *,
    recorded_at_utc: str = FINISHED_AT_UTC,
) -> dict[str, Any]:
    bindings = []
    directory = project_root / MODAL_LAUNCH_REJECTION_ROOT
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if path.name == modal_global_launch_rejection_seal_path().name:
                continue
            raw = path.read_bytes()
            bindings.append(
                {
                    "path": (MODAL_LAUNCH_REJECTION_ROOT / path.name).as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "size_bytes": len(raw),
                }
            )
    payload = {
        "schema_name": "ModalGlobalLaunchRejectionSeal",
        "schema_version": "1.0",
        "recorded_at_utc": recorded_at_utc,
        "rejection_receipts": bindings,
        "validated": True,
    }
    _write_json(project_root / modal_global_launch_rejection_seal_path(), payload)
    return payload


def _publish_intent_and_reservations(
    project_root: Path,
    *,
    identity: ModalLiveCohortIdentity | None = None,
    attempt_id: str = ATTEMPT_ID,
    publish_reservations: bool = True,
) -> tuple[dict[str, Any], str, tuple[journal.ModalRemoteRunReservationSpec, ...]]:
    selected = identity or _identity()
    intent, specs = _intent_and_reservations(
        identity=selected,
        attempt_id=attempt_id,
    )
    intent_sha256 = _write_json(
        project_root / modal_action_intent_receipt_path(selected, attempt_id),
        intent,
    )
    if publish_reservations:
        for spec in specs:
            _write_json(project_root / spec.binding["path"], spec.payload)
    return intent, intent_sha256, specs


def _publish_canaries_intent_and_reservations(
    project_root: Path,
) -> tuple[dict[str, Any], str]:
    identity = _identity()
    intent, specs = _canaries_intent_and_reservations(identity=identity)
    intent_sha256 = _write_json(
        project_root / modal_action_intent_receipt_path(identity, ATTEMPT_ID),
        intent,
    )
    for spec in specs:
        _write_json(project_root / spec.binding["path"], spec.payload)
    return intent, intent_sha256


def _publish_marker(
    project_root: Path,
    intent: Mapping[str, Any],
    *,
    intent_sha256: str,
) -> str:
    path = project_root / modal_local_process_start_receipt_path(intent["attempt_id"])
    marker_sha256 = _write_json(
        path,
        _process_marker(intent, intent_sha256=intent_sha256),
    )
    path.parent.chmod(0o700)
    path.parent.parent.chmod(0o700)
    return marker_sha256


def _publish_successful_action(
    project_root: Path,
    *,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    action: str,
    run_id: str,
    source_run_id: str | None = None,
    harness: str | None = None,
) -> dict[str, Any]:
    intent, _old_specs = _intent_and_reservations(
        identity=identity,
        attempt_id=attempt_id,
    )
    concrete = journal.expected_modal_concrete_run_ids(
        action=action,
        run_id=run_id,
        verifier_run_id=None,
    )
    specs = journal.build_modal_remote_run_reservation_specs(
        concrete_remote_run_ids=concrete,
        attempt_id=attempt_id,
        action=action,
        identity=identity,
        created_at_utc=CREATED_AT_UTC,
        launch_capability_sha256=LAUNCH_CAPABILITY_SHA256,
        **_containment(),
    )
    provider_action = action in {"canary", "canaries"}
    gates = journal._expected_predecessor_gate_roster(action, ())
    intent.update(
        {
            "action": action,
            "run_id": run_id,
            "concrete_remote_run_ids": list(concrete),
            "remote_run_reservations": [dict(spec.binding) for spec in specs],
            "source_run_id": source_run_id,
            "verifier_run_id": None,
            "harness": harness,
            "outer_cli_timeout_seconds": (
                journal._expected_outer_cli_timeout_seconds(action)
            ),
            "modal_resource_profile": journal._expected_modal_resource_profile(
                action,
                harness,
            ),
            "provider_cost_approved": provider_action,
            "provider_cost_cap_usd": "1" if provider_action else None,
            "provider_approval_plan_path": (
                "outputs/readiness/provider-plan.json" if provider_action else None
            ),
            "approval_plan_sha256": "a" * 64 if provider_action else None,
            "provider_price_basis_path": (
                "outputs/readiness/provider-price.json" if provider_action else None
            ),
            "provider_price_basis_sha256": ("b" * 64 if provider_action else None),
            "predecessor_receipts": [
                {
                    "gate": gate,
                    "path": f"outputs/readiness/{attempt_id}-{index}.json",
                    "sha256": "c" * 64,
                }
                for index, gate in enumerate(gates)
            ],
        }
    )
    intent_sha256 = _write_json(
        project_root / modal_action_intent_receipt_path(identity, attempt_id),
        intent,
    )
    for spec in specs:
        _write_json(project_root / spec.binding["path"], spec.payload)
    marker_sha256 = _publish_marker(
        project_root,
        intent,
        intent_sha256=intent_sha256,
    )
    _write_json(
        project_root / modal_action_terminal_receipt_path(identity, attempt_id),
        _terminal(intent, started=True, marker_sha256=marker_sha256),
    )
    return intent


def _publish_primary_execution_evidence(
    project_root: Path,
    *,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    action: str,
    run_id: str,
    harness: str | None,
) -> None:
    function_name = (
        f"canary_{harness}"
        if harness is not None
        else journal._ORDINARY_ACTION_FUNCTIONS[action]
    )
    context = {
        "schema_name": "ExecutionContext",
        "schema_version": "1.0",
        "execution_backend": "modal",
        "run_id": run_id,
        "app_name": APP_NAME,
        "function_name": function_name,
        "modal_app_id": f"ap-{attempt_id[:8]}",
        "modal_function_id": f"fu-{attempt_id[:8]}",
        "modal_call_id": f"fc-{attempt_id[:8]}",
        "modal_image_id": f"im-{identity.image_source_sha256[:8]}",
        "image_source_sha256": identity.image_source_sha256,
        "artifact_uri": volume_artifact_uri(run_id),
    }
    run_root = project_root / MODAL_DOWNLOAD_OUTPUT_ROOT / run_id
    context_path = run_root / "execution_context.json"
    context_sha256 = _write_json(context_path, context)
    context_size = context_path.stat().st_size
    manifest_payload = {
        "schema_name": "ModalRunArtifactManifest",
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at_utc": FINISHED_AT_UTC,
        "image_source_sha256": identity.image_source_sha256,
        "files": [
            {
                "relative_path": "execution_context.json",
                "sha256": context_sha256,
                "size_bytes": context_size,
            }
        ],
    }
    ArtifactManifestV1.from_dict(manifest_payload)
    _write_json(run_root / "artifact_manifest.json", manifest_payload)
    if action == "canary":
        _write_json(
            run_root / "controller" / "provider_request_start_uncertain.json",
            {
                "schema_name": "ProviderRequestStartUncertainEvidence",
                "schema_version": "1.0",
                "harness": harness,
                "action": "one_opportunity_engineering_canary",
                "execution_backend": "modal",
                "action_run_id": run_id,
                "modal_call_id": context["modal_call_id"],
                "api_endpoint": OFFICIAL_OPENAI_API_BASE,
                "model": TARGET_MODEL,
                "provider_attempt_count_lower_bound": 0,
                "provider_attempt_count_upper_bound": 1,
                "provider_request_started": "unknown",
                "provider_attempt_ledger_state": "missing",
                "billing_treatment": "reserve_one_full_approved_request",
                "reason": "controller_terminated_without_terminal_attempt_record",
            },
        )


def _publish_complete_primary_roster(
    project_root: Path,
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, str], dict[str, str]]:
    accepted_runs: dict[str, str] = {}
    accepted_attempts: dict[str, str] = {}
    for index, (label, (action, harness)) in enumerate(
        journal._LINEAGE_PRIMARY_ACTIONS.items(),
        start=1,
    ):
        attempt_id = f"{index:x}" * 32
        if label == "cuda_environment":
            run_id = identity.cohort_id
        elif harness is not None:
            run_id = f"provider-{canary_run_suffix(harness)}"
        else:
            run_id = f"accepted-{label.replace('_', '-')}"
        _publish_successful_action(
            project_root,
            identity=identity,
            attempt_id=attempt_id,
            action=action,
            run_id=run_id,
            source_run_id=(
                accepted_runs["candidate_smoke"]
                if action == "checkpoint-resume"
                else None
            ),
            harness=harness,
        )
        _publish_primary_execution_evidence(
            project_root,
            identity=identity,
            attempt_id=attempt_id,
            action=action,
            run_id=run_id,
            harness=harness,
        )
        accepted_runs[label] = run_id
        accepted_attempts[label] = attempt_id
    return accepted_runs, accepted_attempts


def _scan(
    project_root: Path,
    *,
    process_probe: journal.ProcessProbe | None = None,
) -> journal.ModalGlobalJournalScan:
    descriptor = acquire_modal_action_lock(project_root)
    try:
        return journal.scan_modal_global_action_journal(
            lock_descriptor=descriptor,
            process_probe=process_probe,
        )
    finally:
        release_modal_action_lock(descriptor)


def _complete_migration_payload(
    project_root: Path,
) -> tuple[ModalLiveCohortIdentity, dict[str, Any]]:
    identity = _identity()
    accepted_runs, accepted_attempts = _publish_complete_primary_roster(
        project_root,
        identity,
    )
    _publish_global_rejection_seal(project_root)
    return identity, _migration_terminal_seal(
        identity,
        project_root=project_root,
        accepted_primary_runs=accepted_runs,
        accepted_attempt_ids=accepted_attempts,
    )


def _refresh_remote_object_ids(payload: dict[str, Any]) -> None:
    executions = payload["selected_final"]["remote_executions"]
    payload["selected_final"]["remote_object_ids"] = {
        field: sorted(
            {record["execution_context"][context_field] for record in executions}
        )
        for field, context_field in (
            ("app_ids", "modal_app_id"),
            ("function_ids", "modal_function_id"),
            ("call_ids", "modal_call_id"),
            ("image_ids", "modal_image_id"),
        )
    }


def _replace_first_provider_uncertainty_with_ledger(
    project_root: Path,
    payload: dict[str, Any],
    *,
    extra_field: bool = False,
) -> None:
    provider = payload["selected_final"]["provider_attempt_evidence"][0]
    execution = next(
        record
        for record in payload["selected_final"]["remote_executions"]
        if (record["attempt_id"], record["run_id"])
        == (provider["attempt_id"], provider["run_id"])
    )
    uncertainty_path = project_root / provider["uncertainty"]["path"]
    uncertainty_path.unlink()
    ledger_payload: dict[str, Any] = {
        "schema_name": "ProviderAttemptRecord",
        "schema_version": "1.0",
        "harness": provider["harness"],
        "action": "one_opportunity_engineering_canary",
        "controller_run_id": provider["run_id"],
        "execution_backend": "modal",
        "action_run_id": provider["run_id"],
        "modal_call_id": execution["execution_context"]["modal_call_id"],
        "attempt_ordinal": 1,
        "started_at_utc": "2026-08-10T00:00:00.600000Z",
        "ended_at_utc": "2026-08-10T00:00:00.700000Z",
        "status": "success",
        "api_endpoint": OFFICIAL_OPENAI_API_BASE,
        "model": TARGET_MODEL,
        "generation_settings_sha256": "a" * 64,
        "provider_response_id": "resp-test-1",
        "provider_request_id": "req-test-1",
        "usage_known": True,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "error_class": None,
    }
    if extra_field:
        ledger_payload["invented"] = True
    ledger_raw = (
        json.dumps(
            ledger_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    ledger_logical = (
        PurePosixPath(MODAL_DOWNLOAD_OUTPUT_ROOT)
        / provider["run_id"]
        / "controller"
        / "provider_attempts.jsonl"
    ).as_posix()
    ledger_path = project_root / ledger_logical
    ledger_path.write_bytes(ledger_raw)
    ledger_path.chmod(0o600)
    provider.update(
        {
            "ledger": {
                "path": ledger_logical,
                "sha256": hashlib.sha256(ledger_raw).hexdigest(),
                "size_bytes": len(ledger_raw),
            },
            "uncertainty": None,
            "parse_dispositions": ["valid_terminal_records"],
            "provider_attempt_count": 1,
            "request_ids": ["req-test-1"],
            "response_ids": ["resp-test-1"],
        }
    )
    spend = payload["selected_final"]["provider_spend_estimate"]
    spend.update(
        {
            "provider_terminal_attempt_record_count": 1,
            "provider_attempt_count_lower_bound": 1,
            "successful_provider_attempt_count": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
        }
    )
    spend["provider_request_ids"] = ["req-test-1"]
    spend["provider_response_ids"] = ["resp-test-1"]


def test_empty_global_journal_is_launch_clear(tmp_path: Path) -> None:
    scan = _scan(tmp_path)

    assert scan.project_root == tmp_path
    assert scan.launch_clear
    assert scan.cohorts == ()
    assert scan.attempts == ()


@pytest.mark.parametrize("action", ["download", "verify"])
def test_migration_verifier_source_uses_canonical_action_run_id(
    action: str,
) -> None:
    terminal = {
        "action": action,
        "run_id": "source-artifact-run",
        "source_run_id": None,
        "verifier_run_id": "verifier-destination-run",
        "harness": None,
    }

    assert journal._migration_verifier_source_run_id(
        terminal,
        concrete_run_id="verifier-destination-run",
    ) == "source-artifact-run"

    terminal["source_run_id"] = "legacy-wrong-source-field"
    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="migration verifier terminal identity changed",
    ):
        journal._migration_verifier_source_run_id(
            terminal,
            concrete_run_id="verifier-destination-run",
        )


def test_closed_unstarted_attempt_is_launch_clear(tmp_path: Path) -> None:
    identity = _identity()
    intent, _digest, _specs = _publish_intent_and_reservations(tmp_path)
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        _terminal(intent),
    )

    scan = _scan(tmp_path)

    assert scan.launch_clear
    assert scan.attempts[0].disposition == "closed"
    assert scan.attempts[0].blocker_codes == ()


def test_intent_without_terminal_is_a_blocker(tmp_path: Path) -> None:
    _publish_intent_and_reservations(tmp_path)

    scan = _scan(tmp_path)

    assert not scan.launch_clear
    assert scan.attempts[0].disposition == "unresolved"
    assert "intent_without_terminal" in scan.attempts[0].blocker_codes
    with pytest.raises(journal.ModalActionJournalBlockedError):
        journal.require_modal_global_action_gate_clear(
            scan,
            candidate_attempt_id="a" * 32,
        )


def test_reservation_without_owner_is_a_blocker(tmp_path: Path) -> None:
    _intent, specs = _intent_and_reservations()
    _write_json(tmp_path / specs[0].binding["path"], specs[0].payload)

    scan = _scan(tmp_path)

    assert scan.attempts[0].blocker_codes == ("reservation_without_journal_owner",)


def test_global_rejection_without_reservations_is_closed_rejection(
    tmp_path: Path,
) -> None:
    intent, _specs = _intent_and_reservations()
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        _global_rejection(intent, retain_reservation_roster=False),
    )

    scan = _scan(tmp_path)

    assert scan.launch_clear
    assert scan.attempts[0].disposition == "rejected"
    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="already present",
    ):
        journal.require_modal_global_action_gate_clear(
            scan,
            candidate_attempt_id=ATTEMPT_ID,
        )


def test_partial_preownership_global_rejection_is_closed_rejection(
    tmp_path: Path,
) -> None:
    intent, _specs = _intent_and_reservations()
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        _partial_preownership_rejection(intent),
    )

    scan = _scan(tmp_path)

    assert scan.launch_clear
    assert scan.attempts[0].identity is None
    assert scan.attempts[0].disposition == "rejected"


def test_partial_preownership_rejection_cannot_hide_owned_reservation(
    tmp_path: Path,
) -> None:
    intent, specs = _intent_and_reservations()
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        _partial_preownership_rejection(intent),
    )
    _write_json(tmp_path / specs[0].binding["path"], specs[0].payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="omits its owned reservation bindings",
    ):
        _scan(tmp_path)


def test_reservation_bearing_rejection_requires_complete_identity(
    tmp_path: Path,
) -> None:
    intent, _specs = _intent_and_reservations()
    rejection = _global_rejection(intent, retain_reservation_roster=True)
    rejection["source_tree_sha256"] = None
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        rejection,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="identity is partial",
    ):
        _scan(tmp_path)


def test_partial_global_rejection_publication_is_a_blocker(
    tmp_path: Path,
) -> None:
    intent, _specs = _intent_and_reservations()
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        _global_rejection(intent, retain_reservation_roster=True),
    )

    scan = _scan(tmp_path)

    assert set(scan.attempts[0].blocker_codes) == {
        "global_rejection_reservations_require_recovery",
        "partial_reservation_publication",
    }


@pytest.mark.parametrize("publish_marker", (False, True))
def test_started_terminal_without_exact_marker_binding_stays_blocked(
    tmp_path: Path,
    publish_marker: bool,
) -> None:
    identity = _identity()
    intent, intent_sha256, _specs = _publish_intent_and_reservations(tmp_path)
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        _terminal(intent, started=True),
    )
    if publish_marker:
        _publish_marker(tmp_path, intent, intent_sha256=intent_sha256)

    scan = _scan(
        tmp_path,
        process_probe=(
            lambda _root, _path, _sha256: (
                "different_boot_session" if publish_marker else None
            )
        ),
    )

    assert scan.attempts[0].terminal is not None
    assert scan.attempts[0].terminal.payload["process_group_closed"] is True
    assert "started_process_marker_missing_or_unbound" in scan.attempts[0].blocker_codes
    if publish_marker:
        assert "unbound_process_marker" in scan.attempts[0].blocker_codes


def test_exact_marker_binding_allows_closed_started_attempt(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, intent_sha256, _specs = _publish_intent_and_reservations(tmp_path)
    marker_sha256 = _publish_marker(
        tmp_path,
        intent,
        intent_sha256=intent_sha256,
    )
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        _terminal(intent, started=True, marker_sha256=marker_sha256),
    )

    scan = _scan(tmp_path)

    assert scan.launch_clear
    assert scan.attempts[0].disposition == "closed"


def test_completed_canaries_terminal_requires_aggregate_receipt(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, intent_sha256 = _publish_canaries_intent_and_reservations(tmp_path)
    marker_sha256 = _publish_marker(
        tmp_path,
        intent,
        intent_sha256=intent_sha256,
    )
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        _terminal(intent, started=True, marker_sha256=marker_sha256),
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="lacks its outcome receipt",
    ):
        _scan(tmp_path)


def test_provider_aggregate_requires_eligible_completed_terminal(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, _digest = _publish_canaries_intent_and_reservations(tmp_path)
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        _terminal(intent),
    )
    _write_json(
        tmp_path
        / provider_canary_aggregate_outcome_receipt_path(
            identity,
            ATTEMPT_ID,
        ),
        _provider_aggregate(identity, all_succeeded=True),
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="eligible completed terminal",
    ):
        _scan(tmp_path)


@pytest.mark.parametrize("all_succeeded", (False, True))
def test_provider_aggregate_status_must_match_completed_terminal(
    tmp_path: Path,
    all_succeeded: bool,
) -> None:
    identity = _identity()
    intent, intent_sha256 = _publish_canaries_intent_and_reservations(tmp_path)
    marker_sha256 = _publish_marker(
        tmp_path,
        intent,
        intent_sha256=intent_sha256,
    )
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        _terminal(intent, started=True, marker_sha256=marker_sha256),
    )
    _write_json(
        tmp_path
        / provider_canary_aggregate_outcome_receipt_path(
            identity,
            ATTEMPT_ID,
        ),
        _provider_aggregate(identity, all_succeeded=all_succeeded),
    )

    if not all_succeeded:
        with pytest.raises(
            journal.ModalActionJournalIntegrityError,
            match="statuses differ",
        ):
            _scan(tmp_path)
    else:
        scan = _scan(tmp_path)
        assert scan.launch_clear
        assert scan.attempts[0].disposition == "closed"


def test_failed_completed_canaries_terminal_accepts_matching_aggregate(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, intent_sha256 = _publish_canaries_intent_and_reservations(tmp_path)
    marker_sha256 = _publish_marker(
        tmp_path,
        intent,
        intent_sha256=intent_sha256,
    )
    terminal = _terminal(intent, started=True, marker_sha256=marker_sha256)
    terminal.update(
        {
            "status": "failed",
            "failure_kind": "modal_cli_exit",
            "returncode": 2,
        }
    )
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        terminal,
    )
    _write_json(
        tmp_path / provider_canary_aggregate_outcome_receipt_path(identity, ATTEMPT_ID),
        _provider_aggregate(identity, all_succeeded=False),
    )

    scan = _scan(tmp_path)

    assert scan.launch_clear
    assert scan.attempts[0].disposition == "closed"


def test_bound_marker_process_identity_mismatch_is_integrity_invalid(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, intent_sha256, _specs = _publish_intent_and_reservations(tmp_path)
    marker_sha256 = _publish_marker(
        tmp_path,
        intent,
        intent_sha256=intent_sha256,
    )
    terminal = _terminal(intent, started=True, marker_sha256=marker_sha256)
    terminal.update(
        {
            "local_process_id": 424_243,
            "local_process_group_id": 424_243,
            "local_session_id": 424_243,
        }
    )
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        terminal,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="process identity differs",
    ):
        _scan(tmp_path)


def test_process_probe_is_bound_to_held_lock_project_root(
    tmp_path: Path,
) -> None:
    intent, intent_sha256, _specs = _publish_intent_and_reservations(tmp_path)
    marker_sha256 = _publish_marker(
        tmp_path,
        intent,
        intent_sha256=intent_sha256,
    )
    observed: list[tuple[Path, str, str]] = []

    def probe(project_root: Path, path: str, digest: str) -> str:
        observed.append((project_root, path, digest))
        return "same_boot_process_group_absent"

    scan = _scan(tmp_path, process_probe=probe)

    assert observed == [
        (
            tmp_path,
            modal_local_process_start_receipt_path(ATTEMPT_ID).as_posix(),
            marker_sha256,
        )
    ]
    assert scan.attempts[0].process_probe_result == ("same_boot_process_group_absent")


def test_namespace_change_during_process_probe_is_integrity_error(
    tmp_path: Path,
) -> None:
    intent, intent_sha256, specs = _publish_intent_and_reservations(tmp_path)
    _publish_marker(tmp_path, intent, intent_sha256=intent_sha256)

    def mutate_namespace(_root: Path, _path: str, _digest: str) -> str:
        late = dict(specs[0].payload)
        late["remote_run_id"] = "late-run"
        _write_json(
            tmp_path / MODAL_REMOTE_RUN_RESERVATION_ROOT / "late-run.json",
            late,
        )
        return "different_boot_session"

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="changed during the global scan",
    ):
        _scan(tmp_path, process_probe=mutate_namespace)


def test_leaf_change_after_read_during_process_probe_is_integrity_error(
    tmp_path: Path,
) -> None:
    intent, intent_sha256, specs = _publish_intent_and_reservations(tmp_path)
    _publish_marker(tmp_path, intent, intent_sha256=intent_sha256)

    def mutate_leaf(_root: Path, _path: str, _digest: str) -> str:
        _write_json(
            tmp_path / specs[0].binding["path"],
            {"tampered_after_secure_read": True},
        )
        return "different_boot_session"

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="journal file changed during the global scan",
    ):
        _scan(tmp_path, process_probe=mutate_leaf)


def test_malformed_recovery_intent_is_integrity_invalid(
    tmp_path: Path,
) -> None:
    _intent, specs = _intent_and_reservations()
    identity = _identity()
    _write_json(tmp_path / specs[0].binding["path"], specs[0].payload)
    recovery_path = modal_action_recovery_intent_path(identity, ATTEMPT_ID)
    _write_json(tmp_path / recovery_path, {"opaque_until_contract_freeze": True})

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="recovery intent has an invalid exact schema",
    ):
        _scan(tmp_path)


def test_definitely_not_started_recovery_closes_quarantined_attempt(
    tmp_path: Path,
) -> None:
    _intent, specs = _intent_and_reservations()
    _write_json(tmp_path / specs[0].binding["path"], specs[0].payload)
    # Initialize the shared lock before the genuinely read-only inspect call.
    initial_scan = _scan(tmp_path)
    assert initial_scan.attempts[0].disposition == "unresolved"
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="definitely_not_started",
    )
    before = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
    )
    inspection = recovery.inspect_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: pytest.fail("pre-Popen branch probed a process"),
        boot_state_provider=lambda _sha: pytest.fail(
            "pre-Popen branch queried boot state"
        ),
    )
    after = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
    )
    assert before == after
    assert inspection.ready_to_publish
    assert inspection.branch == "definitely_not_started"

    result = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: pytest.fail("pre-Popen branch probed a process"),
        boot_state_provider=lambda _sha: pytest.fail(
            "pre-Popen branch queried boot state"
        ),
        now_factory=lambda: datetime(2026, 8, 10, 1, tzinfo=UTC),
    )

    assert result.branch == "definitely_not_started"
    assert result.reservation_paths_created == ()
    scan = _scan(tmp_path)
    state = scan.attempts[0]
    assert state.disposition == "closed"
    assert state.blocker_codes == ()
    assert {item.kind for item in state.recoveries} == {
        "intent",
        "host_containment",
        "resolution",
    }
    for item in state.recoveries:
        assert item.record.payload["quarantined"] is True
        assert item.record.payload["eligible_for_final_acceptance"] is False
        assert item.record.payload["fresh_attempt_required"] is True


def _publish_same_boot_may_have_started_orphan(
    root: Path,
) -> tuple[ModalLiveCohortIdentity, journal.ModalGlobalJournalScan]:
    identity = _identity()
    intent, intent_sha256, _specs = _publish_intent_and_reservations(root)
    marker_sha256 = _publish_marker(
        root,
        intent,
        intent_sha256=intent_sha256,
    )
    terminal = _terminal(intent, started=True, marker_sha256=marker_sha256)
    terminal.update(
        {
            "status": "cleanup_failed",
            "failure_kind": "process_group_cleanup",
            "returncode": None,
            "process_group_closed": False,
        }
    )
    _write_json(
        root / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        terminal,
    )
    scan = _scan(
        root,
        process_probe=lambda *_args: "same_boot_process_group_absent",
    )
    assert scan.attempts[0].disposition == "unresolved"
    assert "process_group_not_closed" in scan.attempts[0].blocker_codes
    return identity, scan


def _publish_same_boot_provider_orphan(
    root: Path,
) -> tuple[ModalLiveCohortIdentity, journal.ModalGlobalJournalScan]:
    identity = _identity()
    intent, intent_sha256 = _publish_canaries_intent_and_reservations(root)
    marker_sha256 = _publish_marker(
        root,
        intent,
        intent_sha256=intent_sha256,
    )
    terminal = _terminal(intent, started=True, marker_sha256=marker_sha256)
    terminal.update(
        {
            "status": "cleanup_failed",
            "failure_kind": "process_group_cleanup",
            "returncode": None,
            "process_group_closed": False,
        }
    )
    _write_json(
        root / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        terminal,
    )
    scan = _scan(
        root,
        process_probe=lambda *_args: "same_boot_process_group_absent",
    )
    assert scan.attempts[0].disposition == "unresolved"
    assert "process_group_not_closed" in scan.attempts[0].blocker_codes
    return identity, scan


def test_may_have_started_same_boot_absent_recovery_is_conservative_and_resumable(
    tmp_path: Path,
) -> None:
    identity, initial_scan = _publish_same_boot_may_have_started_orphan(tmp_path)
    snapshot_path = _write_recovery_snapshot(tmp_path, identity)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )
    probes: list[str] = []

    def absent_probe(_root: Path, _path: str, _digest: str) -> str:
        probes.append("probe")
        return "same_boot_process_group_absent"

    inspection = recovery.inspect_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=absent_probe,
        boot_state_provider=lambda _sha: (
            BOOT_STARTED_AT_UNIX_MICROSECONDS,
            BOOT_SESSION_SHA256,
        ),
    )
    assert inspection.ready_to_publish is True
    assert inspection.branch == "may_have_started_contained"

    result = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=absent_probe,
        boot_state_provider=lambda _sha: (
            BOOT_STARTED_AT_UNIX_MICROSECONDS,
            BOOT_SESSION_SHA256,
        ),
        now_factory=lambda: datetime(2026, 8, 10, 3, tzinfo=UTC),
    )
    assert result.resumed is False
    resolution_path = tmp_path / result.resolution_path
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    assert resolution["modal_exposure"] == {
        "basis": (
            "complete_app_name_main_billing_snapshot_plus_full_local_"
            "authorization_reserve_for_unresolved_start"
        ),
        "measured_app_name_main_billing_usd": "0.01",
        "unresolved_compute_reserve_usd": "0.25",
        "conservative_app_name_main_billing_usd": "0.26",
        "complete_hourly_window": True,
        "local_authorization_is_platform_hard_bound": False,
        "modal_api_requests_performed": 0,
        "snapshot_requests_performed": 0,
        "billing_requests_performed": 0,
        "price_requests_performed": 0,
    }
    assert resolution["known_remote_objects"]["app_ids"] == {
        "coverage": "partial",
        "ids": ["ap-recovery-test"],
    }
    assert resolution["quarantined"] is True
    assert resolution["eligible_for_final_acceptance"] is False
    assert resolution["fresh_attempt_required"] is True
    assert probes

    repeated = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=absent_probe,
        boot_state_provider=lambda _sha: (
            BOOT_STARTED_AT_UNIX_MICROSECONDS,
            BOOT_SESSION_SHA256,
        ),
    )
    assert repeated.resumed is True
    assert resolution_path.read_bytes() == _canonical_bytes(resolution)


def test_provider_recovery_without_ledgers_reserves_full_frozen_approval(
    tmp_path: Path,
) -> None:
    identity, initial_scan = _publish_same_boot_provider_orphan(tmp_path)
    snapshot_path = _write_recovery_snapshot(tmp_path, identity)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )

    result = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: "same_boot_process_group_absent",
        boot_state_provider=lambda _sha: (
            BOOT_STARTED_AT_UNIX_MICROSECONDS,
            BOOT_SESSION_SHA256,
        ),
    )
    resolution = json.loads((tmp_path / result.resolution_path).read_text())

    assert resolution["provider_exposure"] == {
        "applicable": True,
        "basis": "frozen_full_provider_approval_bound",
        "ledger_bindings": [],
        "provider_price_basis_binding": None,
        "attempt_count": 0,
        "success_count": 0,
        "error_count": 0,
        "usage_known_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "exact_usage_cost_usd": None,
        "frozen_provider_approval_bound_usd": "1",
        "conservative_provider_exposure_usd": "1",
        "provider_requests_performed": 0,
        "price_requests_performed": 0,
    }
    assert not list(
        tmp_path.glob(
            "outputs/development/modal_downloads/*/controller/"
            "provider_attempts.jsonl"
        )
    )


@pytest.mark.parametrize(
    ("probe_result", "expected_blocker"),
    (
        ("same_boot_process_group_exists", "same_boot_process_group_still_exists"),
        (
            "same_boot_process_identity_changed",
            "same_boot_process_identity_changed_or_reused",
        ),
    ),
)
def test_may_have_started_same_boot_live_or_reused_process_is_blocked(
    tmp_path: Path,
    probe_result: str,
    expected_blocker: str,
) -> None:
    identity, initial_scan = _publish_same_boot_may_have_started_orphan(tmp_path)
    snapshot_path = _write_recovery_snapshot(tmp_path, identity)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )

    inspection = recovery.inspect_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: probe_result,
        boot_state_provider=lambda _sha: (
            BOOT_STARTED_AT_UNIX_MICROSECONDS,
            BOOT_SESSION_SHA256,
        ),
    )

    assert inspection.ready_to_publish is False
    assert expected_blocker in inspection.blockers
    with pytest.raises(recovery.ModalActionRecoveryBlockedError):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: probe_result,
            boot_state_provider=lambda _sha: (
                BOOT_STARTED_AT_UNIX_MICROSECONDS,
                BOOT_SESSION_SHA256,
            ),
        )
    assert not (
        tmp_path / modal_action_recovery_intent_path(identity, ATTEMPT_ID)
    ).exists()


@pytest.mark.parametrize(
    ("snapshot_overrides", "message"),
    (
        ({"app_state": "running", "app_tasks": "1"}, "active target App"),
        ({"include_target_volume": False}, "exactly one target Volume"),
    ),
)
def test_may_have_started_recovery_rejects_incomplete_cleanup_snapshot(
    tmp_path: Path,
    snapshot_overrides: dict[str, Any],
    message: str,
) -> None:
    identity, initial_scan = _publish_same_boot_may_have_started_orphan(tmp_path)
    snapshot_path = _write_recovery_snapshot(
        tmp_path,
        identity,
        **snapshot_overrides,
    )
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match=message,
    ):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: "same_boot_process_group_absent",
            boot_state_provider=lambda _sha: (
                BOOT_STARTED_AT_UNIX_MICROSECONDS,
                BOOT_SESSION_SHA256,
            ),
        )


def test_zero_billing_rows_still_reserve_full_modal_authorization(
    tmp_path: Path,
) -> None:
    identity, initial_scan = _publish_same_boot_may_have_started_orphan(tmp_path)
    snapshot_path = _write_recovery_snapshot(
        tmp_path,
        identity,
        billing_rows=[],
    )
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )

    result = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: "same_boot_process_group_absent",
        boot_state_provider=lambda _sha: (
            BOOT_STARTED_AT_UNIX_MICROSECONDS,
            BOOT_SESSION_SHA256,
        ),
    )
    resolution = json.loads((tmp_path / result.resolution_path).read_text())
    exposure = resolution["modal_exposure"]
    assert exposure["measured_app_name_main_billing_usd"] == "0"
    assert exposure["unresolved_compute_reserve_usd"] == "0.25"
    assert exposure["conservative_app_name_main_billing_usd"] == "0.25"
    assert exposure["local_authorization_is_platform_hard_bound"] is False


def test_later_boot_recovers_missing_terminal_and_marker_but_same_boot_blocks(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _publish_intent_and_reservations(tmp_path)
    initial_scan = _scan(tmp_path)
    assert "intent_without_terminal" in initial_scan.attempts[0].blocker_codes
    snapshot_path = _write_recovery_snapshot(tmp_path, identity)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )

    same_boot = recovery.inspect_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: pytest.fail(
            "missing marker unexpectedly triggered a process probe"
        ),
        boot_state_provider=lambda _sha: (
            BOOT_STARTED_AT_UNIX_MICROSECONDS,
            BOOT_SESSION_SHA256,
        ),
    )
    assert same_boot.ready_to_publish is False
    assert "same_boot_marker_or_terminal_is_missing_or_unbound" in (
        same_boot.blockers
    )

    later_boot = (
        BOOT_STARTED_AT_UNIX_MICROSECONDS + 1_000_000,
        "f" * 64,
    )
    result = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: pytest.fail(
            "missing marker unexpectedly triggered a process probe"
        ),
        boot_state_provider=lambda _sha: later_boot,
    )
    resolution = json.loads((tmp_path / result.resolution_path).read_text())
    host = json.loads((tmp_path / result.host_containment_path).read_text())
    assert host["containment_basis"] == "strictly_later_boot_session"
    assert host["current_boot_started_at_unix_microseconds"] == later_boot[0]
    assert host["current_boot_session_sha256"] == later_boot[1]
    assert resolution["fresh_attempt_required"] is True
    assert _scan(tmp_path).attempts[0].disposition == "closed"


def test_pre_popen_reservation_repair_crash_preserves_request_frozen_history(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, specs = _canaries_intent_and_reservations(identity=identity)
    rejection = _global_rejection(intent, retain_reservation_roster=True)
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        rejection,
    )
    _write_json(tmp_path / specs[0].binding["path"], specs[0].payload)
    initial_scan = _scan(tmp_path)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="definitely_not_started",
    )
    crashed_after: list[str] = []

    def crash_after_first_repair(stage: str, path: Path) -> None:
        if stage == "reservation":
            crashed_after.append(path.relative_to(tmp_path).as_posix())
            raise RuntimeError("simulated crash during reservation repair")

    with pytest.raises(RuntimeError, match="simulated crash"):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: pytest.fail(
                "pre-Popen recovery probed a process"
            ),
            boot_state_provider=lambda _sha: pytest.fail(
                "pre-Popen recovery queried boot state"
            ),
            publish_hook=crash_after_first_repair,
        )
    assert len(crashed_after) == 1
    assert not (
        tmp_path / modal_action_recovery_intent_path(identity, ATTEMPT_ID)
    ).exists()

    result = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: pytest.fail(
            "pre-Popen recovery probed a process"
        ),
        boot_state_provider=lambda _sha: pytest.fail(
            "pre-Popen recovery queried boot state"
        ),
    )
    assert result.resumed is True
    assert len(result.reservation_paths_created) == len(specs) - 2
    recovery_intent = json.loads(
        (tmp_path / result.recovery_intent_path).read_text()
    )
    repair = recovery_intent["reservation_repair"]
    assert len(repair["initial_reservation_bindings"]) == 1
    assert len(repair["published_reservation_bindings"]) == len(specs) - 1
    assert len(repair["final_reservation_bindings"]) == len(specs)
    assert {
        item["path"] for item in repair["initial_reservation_bindings"]
    }.isdisjoint(
        item["path"] for item in repair["published_reservation_bindings"]
    )
    assert _scan(tmp_path).attempts[0].disposition == "closed"


@pytest.mark.parametrize(
    ("crash_stage", "expected_stage_names"),
    (
        ("intent", ("intent",)),
        ("host_containment", ("intent", "host_containment")),
    ),
)
def test_recovery_stage_crash_resumes_without_rewriting_prior_bytes(
    tmp_path: Path,
    crash_stage: str,
    expected_stage_names: tuple[str, ...],
) -> None:
    identity = _identity()
    _intent, specs = _intent_and_reservations()
    _write_json(tmp_path / specs[0].binding["path"], specs[0].payload)
    initial_scan = _scan(tmp_path)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="definitely_not_started",
    )

    def crash_after_stage(stage: str, _path: Path) -> None:
        if stage == crash_stage:
            raise RuntimeError(f"simulated crash after {stage}")

    with pytest.raises(RuntimeError, match="simulated crash after"):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: pytest.fail(
                "pre-Popen recovery probed a process"
            ),
            boot_state_provider=lambda _sha: pytest.fail(
                "pre-Popen recovery queried boot state"
            ),
            now_factory=lambda: datetime(2026, 8, 10, 4, tzinfo=UTC),
            publish_hook=crash_after_stage,
        )

    paths = {
        "intent": modal_action_recovery_intent_path(identity, ATTEMPT_ID),
        "host_containment": modal_action_host_containment_path(
            identity,
            ATTEMPT_ID,
        ),
    }
    preserved = {
        name: (tmp_path / paths[name]).read_bytes()
        for name in expected_stage_names
    }
    partial = _scan(tmp_path)
    state = partial.attempts[0]
    assert state.disposition == "unresolved"
    assert "incomplete_recovery_journal" in state.blocker_codes

    result = recovery.resolve_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: pytest.fail(
            "pre-Popen recovery probed a process"
        ),
        boot_state_provider=lambda _sha: pytest.fail(
            "pre-Popen recovery queried boot state"
        ),
        now_factory=lambda: datetime(2026, 8, 10, 5, tzinfo=UTC),
    )

    assert result.resumed is True
    for name, raw in preserved.items():
        assert (tmp_path / paths[name]).read_bytes() == raw
    assert _scan(tmp_path).attempts[0].disposition == "closed"


def test_recovery_request_tamper_after_intent_is_integrity_invalid(
    tmp_path: Path,
) -> None:
    _intent, specs = _intent_and_reservations()
    _write_json(tmp_path / specs[0].binding["path"], specs[0].payload)
    request_path = _write_recovery_request(
        tmp_path,
        _scan(tmp_path),
        branch="definitely_not_started",
    )

    def crash_after_intent(stage: str, _path: Path) -> None:
        if stage == "intent":
            raise RuntimeError("simulated crash after intent")

    with pytest.raises(RuntimeError, match="simulated crash after intent"):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: pytest.fail(
                "pre-Popen recovery probed a process"
            ),
            boot_state_provider=lambda _sha: pytest.fail(
                "pre-Popen recovery queried boot state"
            ),
            publish_hook=crash_after_intent,
        )

    tampered = json.loads(request_path.read_text(encoding="utf-8"))
    tampered["fresh_candidate_attempt_id"] = "b" * 32
    _write_json(request_path, tampered)
    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="recovery operator request byte binding changed",
    ):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: pytest.fail(
                "pre-Popen recovery probed a process"
            ),
            boot_state_provider=lambda _sha: pytest.fail(
                "pre-Popen recovery queried boot state"
            ),
        )


def test_recovery_request_frozen_reservation_binding_change_is_blocked(
    tmp_path: Path,
) -> None:
    _intent, specs = _intent_and_reservations()
    reservation_path = tmp_path / specs[0].binding["path"]
    _write_json(reservation_path, specs[0].payload)
    request_path = _write_recovery_request(
        tmp_path,
        _scan(tmp_path),
        branch="definitely_not_started",
    )
    changed = dict(specs[0].payload)
    changed["launch_capability_sha256"] = "f" * 64
    _write_json(reservation_path, changed)

    inspection = recovery.inspect_modal_action_recovery(
        project_root=tmp_path,
        request_path=request_path,
        process_probe=lambda *_args: pytest.fail(
            "pre-Popen recovery probed a process"
        ),
        boot_state_provider=lambda _sha: pytest.fail(
            "pre-Popen recovery queried boot state"
        ),
    )
    assert inspection.ready_to_publish is False
    assert "request_initial_reservation_binding_changed" in inspection.blockers
    with pytest.raises(
        recovery.ModalActionRecoveryBlockedError,
        match="request_initial_reservation_binding_changed",
    ):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: pytest.fail(
                "pre-Popen recovery probed a process"
            ),
            boot_state_provider=lambda _sha: pytest.fail(
                "pre-Popen recovery queried boot state"
            ),
        )
    assert not list(
        tmp_path.glob("outputs/readiness/modal_only_final/*/action_recoveries/*.json")
    )


def test_recovery_snapshot_leaf_tamper_is_integrity_invalid(
    tmp_path: Path,
) -> None:
    identity, initial_scan = _publish_same_boot_may_have_started_orphan(tmp_path)
    snapshot_path = _write_recovery_snapshot(tmp_path, identity)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )
    app_list_path = snapshot_path.parent / "app_list.json"
    _write_json(app_list_path, [])

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="recovery snapshot app_list byte binding changed",
    ):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: "same_boot_process_group_absent",
            boot_state_provider=lambda _sha: (
                BOOT_STARTED_AT_UNIX_MICROSECONDS,
                BOOT_SESSION_SHA256,
            ),
        )


def test_recovery_snapshot_rejects_incomplete_billing_horizon(
    tmp_path: Path,
) -> None:
    identity, initial_scan = _publish_same_boot_may_have_started_orphan(tmp_path)
    snapshot_path = _write_recovery_snapshot(tmp_path, identity)
    manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    manifest["billing_window_end_utc"] = "2026-08-10T03:00:00Z"
    _write_json(snapshot_path, manifest)
    request_path = _write_recovery_request(
        tmp_path,
        initial_scan,
        branch="may_have_started_contained",
        snapshot_manifest_path=snapshot_path,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="timing does not prove completed billing hours",
    ):
        recovery.resolve_modal_action_recovery(
            project_root=tmp_path,
            request_path=request_path,
            process_probe=lambda *_args: "same_boot_process_group_absent",
            boot_state_provider=lambda _sha: (
                BOOT_STARTED_AT_UNIX_MICROSECONDS,
                BOOT_SESSION_SHA256,
            ),
        )


def test_recovery_cli_has_no_external_service_or_snapshot_imports() -> None:
    source = Path(recovery.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint(
        {"httpx", "modal", "openai", "requests", "socket", "subprocess", "urllib"}
    )
    assert "capture_modal_cleanup_snapshots" not in source


def test_sealed_cohort_with_recovery_content_is_integrity_invalid(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _publish_global_rejection_seal(tmp_path)
    _write_json(
        tmp_path / modal_migration_lineage_path(identity),
        _migration_terminal_seal(identity, project_root=tmp_path),
    )
    _write_json(
        tmp_path / modal_action_recovery_intent_path(identity, ATTEMPT_ID),
        {"opaque_until_contract_freeze": True},
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="sealed cohort contains recovery content",
    ):
        _scan(tmp_path)


def test_sealed_cohort_with_unresolved_action_is_integrity_invalid(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _publish_intent_and_reservations(tmp_path)
    _publish_global_rejection_seal(tmp_path)
    _write_json(
        tmp_path / modal_migration_lineage_path(identity),
        _migration_terminal_seal(identity, project_root=tmp_path),
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="sealed cohort contains unresolved action state",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_selected_final_placeholder(
    tmp_path: Path,
) -> None:
    identity = _identity()
    seal = _migration_terminal_seal(identity, project_root=tmp_path)
    _publish_global_rejection_seal(tmp_path)
    seal["selected_final"] = {"identity": seal["selected_final"]["identity"]}
    _write_json(tmp_path / modal_migration_lineage_path(identity), seal)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="final selection has an invalid exact schema",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_post_seal_closed_attempt(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _publish_global_rejection_seal(tmp_path)
    _write_json(
        tmp_path / modal_migration_lineage_path(identity),
        _migration_terminal_seal(identity, project_root=tmp_path),
    )
    intent, _intent_sha256, _specs = _publish_intent_and_reservations(tmp_path)
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        _terminal(intent),
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="action-journal snapshot changed",
    ):
        _scan(tmp_path)


def test_global_journal_rejects_multiple_migration_terminal_seals(
    tmp_path: Path,
) -> None:
    first = _identity()
    second = _identity(
        source_tree_sha256="a" * 64,
        image_source_sha256="b" * 64,
        cohort_id="second-modal-cohort",
    )
    _publish_global_rejection_seal(tmp_path)
    _write_json(
        tmp_path / modal_migration_lineage_path(first),
        _migration_terminal_seal(first, project_root=tmp_path),
    )
    _write_json(
        tmp_path / modal_migration_lineage_path(second),
        _migration_terminal_seal(second, project_root=tmp_path),
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="multiple migration terminal seals",
    ):
        _scan(tmp_path)


def test_global_rejection_seal_rejects_post_seal_rejection(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _publish_global_rejection_seal(tmp_path)
    _write_json(
        tmp_path / modal_migration_lineage_path(identity),
        _migration_terminal_seal(identity, project_root=tmp_path),
    )
    intent, _specs = _intent_and_reservations(identity=identity)
    rejection = _partial_preownership_rejection(intent)
    rejection["finished_at_utc"] = "2026-08-10T00:00:02Z"
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        rejection,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="launch-rejection seal snapshot changed",
    ):
        _scan(tmp_path)


def test_valid_terminal_seal_blocks_every_fresh_candidate_without_mutation(
    tmp_path: Path,
) -> None:
    identity = _identity()
    accepted_runs, accepted_attempts = _publish_complete_primary_roster(
        tmp_path,
        identity,
    )
    _publish_global_rejection_seal(tmp_path)
    _write_json(
        tmp_path / modal_migration_lineage_path(identity),
        _migration_terminal_seal(
            identity,
            project_root=tmp_path,
            accepted_primary_runs=accepted_runs,
            accepted_attempt_ids=accepted_attempts,
        ),
    )
    scan = _scan(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert scan.launch_clear
    journal.require_modal_global_action_journal_resolved(scan)
    with pytest.raises(
        journal.ModalActionJournalBlockedError,
        match="migration_terminal_seal_present",
    ):
        journal.require_modal_global_action_gate_clear(
            scan,
            candidate_attempt_id="f" * 32,
        )

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize(
    "mutation",
    ("extra_remote_field", "missing_provider_field", "mistyped_artifact_field"),
)
def test_migration_terminal_seal_rejects_nested_exact_schema_spoofs(
    tmp_path: Path,
    mutation: str,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    selected = payload["selected_final"]
    if mutation == "extra_remote_field":
        selected["remote_executions"][0]["invented"] = True
    elif mutation == "missing_provider_field":
        selected["provider_attempt_evidence"][0].pop("harness")
    else:
        selected["artifact_manifests"][0]["size_bytes"] = "1"
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="invalid exact schema|size_bytes is invalid",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_duplicate_modal_call_id(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    executions = payload["selected_final"]["remote_executions"]
    executions[1]["execution_context"]["modal_call_id"] = executions[0][
        "execution_context"
    ]["modal_call_id"]
    _refresh_remote_object_ids(payload)
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="reuses one Modal call ID",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_validates_bound_provider_ledger_records(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    _replace_first_provider_uncertainty_with_ledger(tmp_path, payload)
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    assert _scan(tmp_path).launch_clear


def test_migration_terminal_seal_rejects_provider_ledger_schema_spoof(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    _replace_first_provider_uncertainty_with_ledger(
        tmp_path,
        payload,
        extra_field=True,
    )
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="invalid provider record",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_swapped_provider_harness(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    provider = payload["selected_final"]["provider_attempt_evidence"]
    provider[0]["harness"], provider[1]["harness"] = (
        provider[1]["harness"],
        provider[0]["harness"],
    )
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="harness differs from its execution",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_swapped_execution_run_identity(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    executions = payload["selected_final"]["remote_executions"]
    executions[0]["execution_context"], executions[1]["execution_context"] = (
        executions[1]["execution_context"],
        executions[0]["execution_context"],
    )
    _refresh_remote_object_ids(payload)
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="execution_context identity is incomplete",
    ):
        _scan(tmp_path)


@pytest.mark.parametrize(
    ("roster", "binding", "field", "value"),
    (
        ("remote_executions", "evidence", "sha256", "f" * 64),
        ("artifact_manifests", None, "sha256", "e" * 64),
        ("provider_attempt_evidence", "uncertainty", "sha256", "d" * 64),
        (
            "remote_executions",
            "evidence",
            "path",
            "outputs/development/modal_downloads/invented/execution_context.json",
        ),
    ),
)
def test_migration_terminal_seal_rejects_wrong_nested_file_binding(
    tmp_path: Path,
    roster: str,
    binding: str | None,
    field: str,
    value: str,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    record = payload["selected_final"][roster][0]
    target = record if binding is None else record[binding]
    target[field] = value
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(journal.ModalActionJournalIntegrityError):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_omitted_remote_execution_record(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    selected = payload["selected_final"]
    omitted = selected["remote_executions"].pop(0)
    selected["artifact_manifests"] = [
        record
        for record in selected["artifact_manifests"]
        if (record["attempt_id"], record["run_id"])
        != (omitted["attempt_id"], omitted["run_id"])
    ]
    _refresh_remote_object_ids(payload)
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="omits remote execution evidence",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_omitted_artifact_manifest_record(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    payload["selected_final"]["artifact_manifests"].pop(0)
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="omits or invents a remote execution manifest",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_omitted_provider_evidence_record(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    payload["selected_final"]["provider_attempt_evidence"].pop(0)
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="omits provider evidence",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_rejects_invented_provider_evidence_record(
    tmp_path: Path,
) -> None:
    identity, payload = _complete_migration_payload(tmp_path)
    invented = dict(payload["selected_final"]["provider_attempt_evidence"][-1])
    invented.update(
        {
            "attempt_id": "f" * 32,
            "run_id": "invented-provider-run",
            "binding_state": "unbound_observed",
            "ledger": {
                "path": (
                    "outputs/development/modal_downloads/invented-provider-run/"
                    "controller/provider_attempts.jsonl"
                ),
                "sha256": "f" * 64,
                "size_bytes": 1,
            },
            "uncertainty": None,
            "parse_dispositions": ["partial_unparseable"],
        }
    )
    payload["selected_final"]["provider_attempt_evidence"].append(invented)
    _write_json(tmp_path / modal_migration_lineage_path(identity), payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="provider evidence roster is incomplete",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_requires_global_rejection_roster_seal(
    tmp_path: Path,
) -> None:
    identity = _identity()
    _write_json(
        tmp_path / modal_migration_lineage_path(identity),
        _migration_terminal_seal(identity, project_root=tmp_path),
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="lacks its global launch-rejection seal",
    ):
        _scan(tmp_path)


def test_migration_terminal_seal_cannot_predate_frozen_journal(
    tmp_path: Path,
) -> None:
    identity = _identity()
    accepted_runs, accepted_attempts = _publish_complete_primary_roster(
        tmp_path,
        identity,
    )
    early = "2026-08-10T00:00:00.750000Z"
    _publish_global_rejection_seal(tmp_path, recorded_at_utc=early)
    _write_json(
        tmp_path / modal_migration_lineage_path(identity),
        _migration_terminal_seal(
            identity,
            project_root=tmp_path,
            accepted_primary_runs=accepted_runs,
            accepted_attempt_ids=accepted_attempts,
            recorded_at_utc=early,
        ),
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="predates its frozen journal evidence",
    ):
        _scan(tmp_path)


def test_global_rejection_seal_is_a_prelaunch_gate_without_scan_blockers(
    tmp_path: Path,
) -> None:
    intent, _specs = _intent_and_reservations()
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        _partial_preownership_rejection(intent),
    )
    _publish_global_rejection_seal(tmp_path)

    scan = _scan(tmp_path)

    assert scan.launch_clear
    assert scan.global_rejection_seal is not None
    journal.require_modal_global_action_journal_resolved(scan)
    with pytest.raises(
        journal.ModalActionJournalBlockedError,
        match="global_launch_rejection_seal_present",
    ):
        journal.require_modal_global_action_gate_clear(
            scan,
            candidate_attempt_id="f" * 32,
        )


def test_rejection_seal_builder_is_pure_exact_and_held_scan_derived(
    tmp_path: Path,
) -> None:
    intent, _specs = _intent_and_reservations()
    rejection_path = tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID)
    rejection_sha256 = _write_json(
        rejection_path,
        _partial_preownership_rejection(intent),
    )
    descriptor = acquire_modal_action_lock(tmp_path)
    try:
        scan = journal.scan_modal_global_action_journal(
            lock_descriptor=descriptor,
        )
        before = {
            path.relative_to(tmp_path).as_posix(): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }

        payload = journal.build_modal_global_launch_rejection_seal_payload(
            scan,
            recorded_at_utc=FINISHED_AT_UTC,
        )

        assert payload == {
            "schema_name": "ModalGlobalLaunchRejectionSeal",
            "schema_version": "1.0",
            "recorded_at_utc": FINISHED_AT_UTC,
            "rejection_receipts": [
                {
                    "path": modal_launch_rejection_receipt_path(ATTEMPT_ID).as_posix(),
                    "sha256": rejection_sha256,
                    "size_bytes": rejection_path.stat().st_size,
                }
            ],
            "validated": True,
        }
        after = {
            path.relative_to(tmp_path).as_posix(): path.read_bytes()
            for path in tmp_path.rglob("*")
            if path.is_file()
        }
        assert after == before
    finally:
        release_modal_action_lock(descriptor)


@pytest.mark.parametrize(
    "recorded_at_utc",
    (
        "2026-08-10T00:00:00.500000Z",
        "2026-08-10T08:00:01+08:00",
    ),
)
def test_rejection_seal_builder_rejects_early_or_noncanonical_time(
    tmp_path: Path,
    recorded_at_utc: str,
) -> None:
    intent, _specs = _intent_and_reservations()
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        _partial_preownership_rejection(intent),
    )
    scan = _scan(tmp_path)

    with pytest.raises(journal.ModalActionJournalIntegrityError):
        journal.build_modal_global_launch_rejection_seal_payload(
            scan,
            recorded_at_utc=recorded_at_utc,
        )


@pytest.mark.parametrize("mutation", ("delete", "tamper", "add"))
def test_global_rejection_seal_detects_namespace_change(
    tmp_path: Path,
    mutation: str,
) -> None:
    intent, _specs = _intent_and_reservations()
    rejection_path = tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID)
    rejection = _partial_preownership_rejection(intent)
    _write_json(rejection_path, rejection)
    _publish_global_rejection_seal(tmp_path)

    if mutation == "delete":
        rejection_path.unlink()
    elif mutation == "tamper":
        rejection["finished_at_utc"] = "2026-08-10T00:00:00.750000Z"
        _write_json(rejection_path, rejection)
    else:
        second_intent, _second_specs = _intent_and_reservations(attempt_id="2" * 32)
        _write_json(
            tmp_path / modal_launch_rejection_receipt_path("2" * 32),
            _partial_preownership_rejection(second_intent),
        )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="launch-rejection seal snapshot changed",
    ):
        _scan(tmp_path)


def test_global_scanner_rejects_symlinked_namespace(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    live_root = tmp_path / MODAL_LIVE_COHORT_ROOT
    live_root.parent.mkdir(parents=True)
    live_root.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="symlinks",
    ):
        _scan(tmp_path)


@pytest.mark.parametrize(
    ("filename", "encoded", "mode", "message"),
    (
        ("unsupported.txt", b"{}\n", 0o600, "unsupported filename"),
        ("bad-run.json", b'{"x": 1, "x": 2}\n', 0o600, "duplicate key"),
        ("nonfinite.json", b'{"x": NaN}\n', 0o600, "non-finite value"),
        ("bad-mode.json", b"{}\n", 0o644, "single-link 0600"),
    ),
)
def test_global_scanner_rejects_malformed_or_unsafe_reservation_leaf(
    tmp_path: Path,
    filename: str,
    encoded: bytes,
    mode: int,
    message: str,
) -> None:
    path = tmp_path / MODAL_REMOTE_RUN_RESERVATION_ROOT / filename
    path.parent.mkdir(parents=True)
    path.write_bytes(encoded)
    path.chmod(mode)

    with pytest.raises(journal.ModalActionJournalIntegrityError, match=message):
        _scan(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "outer_cli_timeout_seconds",
            journal._expected_outer_cli_timeout_seconds("cuda-environment") + 1,
            "timeout differs from its action",
        ),
        ("modal_resource_profile", {}, "resource profile differs from its action"),
        ("modal_cost_cap_usd", "0.001", "sufficient Modal cost approval"),
        ("modal_cost_approved", False, "Modal approval core is invalid"),
        ("provider_cost_approved", True, "provider approval differs"),
    ),
)
def test_intent_intrinsic_action_contract_cannot_be_weakened(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    identity = _identity()
    intent, specs = _intent_and_reservations(identity=identity)
    intent[field] = value
    _write_json(
        tmp_path / modal_action_intent_receipt_path(identity, ATTEMPT_ID),
        intent,
    )
    for spec in specs:
        _write_json(tmp_path / spec.binding["path"], spec.payload)

    with pytest.raises(journal.ModalActionJournalIntegrityError, match=message):
        _scan(tmp_path)


def test_intent_requires_the_action_exact_predecessor_gate_roster(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, specs = _intent_and_reservations(identity=identity)
    intent["predecessor_receipts"][0]["gate"] = "invented_gate"
    _write_json(
        tmp_path / modal_action_intent_receipt_path(identity, ATTEMPT_ID),
        intent,
    )
    for spec in specs:
        _write_json(tmp_path / spec.binding["path"], spec.payload)

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="predecessor gate roster differs from its action",
    ):
        _scan(tmp_path)


def test_reservation_bearing_rejection_enforces_terminal_action_contract(
    tmp_path: Path,
) -> None:
    intent, specs = _intent_and_reservations()
    rejection = _global_rejection(intent, retain_reservation_roster=True)
    rejection["modal_resource_profile"] = {}
    for spec in specs:
        _write_json(tmp_path / spec.binding["path"], spec.payload)
    _write_json(
        tmp_path / modal_launch_rejection_receipt_path(ATTEMPT_ID),
        rejection,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="resource profile differs from its action",
    ):
        _scan(tmp_path)


def test_invalid_terminal_status_does_not_clear_the_global_gate(
    tmp_path: Path,
) -> None:
    identity = _identity()
    intent, _intent_sha256, _specs = _publish_intent_and_reservations(tmp_path)
    terminal = _terminal(intent)
    terminal.update(
        {
            "status": "succeeded",
            "failure_kind": None,
        }
    )
    _write_json(
        tmp_path / modal_action_terminal_receipt_path(identity, ATTEMPT_ID),
        terminal,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="terminal status fields do not reconcile",
    ):
        _scan(tmp_path)


def test_scanner_schema_fields_match_launcher_dataclasses_without_importing_it() -> (
    None
):
    source = (Path(__file__).parents[1] / "scripts" / "launch_modal.py").read_text()
    tree = ast.parse(source)
    class_fields = {
        node.name: {
            statement.target.id
            for statement in node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    assert class_fields["ModalActionIntent"] == journal._ACTION_INTENT_FIELDS
    assert class_fields["ModalActionAttemptReceipt"] == (journal._ACTION_ATTEMPT_FIELDS)
    assert class_fields["ModalLocalProcessStartReceipt"] == (
        journal._LOCAL_PROCESS_START_FIELDS
    )


def test_same_attempt_id_cannot_bind_two_cohorts(tmp_path: Path) -> None:
    first = _identity()
    second = _identity(
        source_tree_sha256="a" * 64,
        image_source_sha256="b" * 64,
        cohort_id="other-cohort",
    )
    _publish_intent_and_reservations(
        tmp_path,
        identity=first,
        publish_reservations=False,
    )
    _publish_intent_and_reservations(
        tmp_path,
        identity=second,
        publish_reservations=False,
    )

    with pytest.raises(
        journal.ModalActionJournalIntegrityError,
        match="conflicting .* cohort identities",
    ):
        _scan(tmp_path)
