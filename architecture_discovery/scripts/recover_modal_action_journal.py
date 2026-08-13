#!/usr/bin/env python3
"""Inspect or close one orphaned local Modal action journal, without API calls.

``inspect`` acquires the already-existing global action lock and performs no
persistent writes.  ``resolve`` and its alias ``publish`` acquire the same lock,
repair only a proven pre-Popen reservation roster, then publish the three
canonical v1.0 recovery stages in order.  Existing stages are accepted only
when their complete raw bytes are identical to the expected canonical bytes.

The command never imports Modal, invokes a provider, captures a snapshot,
queries billing/prices, signals a process, edits launcher receipts, or deletes
evidence.  A may-have-started resolution consumes an operator-selected,
already-published six-read cleanup snapshot manifest.  Both the request and
snapshot manifest must be explicit absolute paths to owned 0600 regular files
inside the locked project.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import modal_action_journal as journal  # noqa: E402
from common.modal_action_lock import (  # noqa: E402
    acquire_existing_modal_action_lock,
    acquire_modal_action_lock,
    assert_modal_action_lock_identity,
    held_modal_action_lock_project_root,
    release_modal_action_lock,
)
from modal_boundary import (  # noqa: E402
    ModalLiveCohortIdentity,
    modal_action_host_containment_path,
    modal_action_recovery_intent_path,
    modal_action_recovery_resolution_path,
    safe_relative_path,
)
from study.serialization import create_json_exclusive  # noqa: E402


class ModalActionRecoveryError(RuntimeError):
    """Base error for safe local orphan-recovery operations."""


class ModalActionRecoveryBlockedError(ModalActionRecoveryError):
    """Raised when the journal does not prove one allowed recovery branch."""


ProcessProbe = Callable[[Path, str, str], str]
BootStateProvider = Callable[[str], tuple[int, str]]
PublishHook = Callable[[str, Path], None]


@dataclass(frozen=True, slots=True)
class ModalActionRecoveryInspection:
    """Secret-free, exact result of one held-lock read-only inspection."""

    attempt_id: str
    branch: str | None
    fresh_candidate_attempt_id: str
    blockers: tuple[str, ...]
    ready_to_publish: bool
    recovery_stage: str
    recovery_paths: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "branch": self.branch,
            "fresh_candidate_attempt_id": self.fresh_candidate_attempt_id,
            "blockers": list(self.blockers),
            "ready_to_publish": self.ready_to_publish,
            "recovery_stage": self.recovery_stage,
            "recovery_paths": dict(self.recovery_paths),
        }


@dataclass(frozen=True, slots=True)
class ModalActionRecoveryResult:
    """Secret-free selector for one complete immutable recovery triplet."""

    attempt_id: str
    branch: str
    fresh_candidate_attempt_id: str
    recovery_intent_path: str
    host_containment_path: str
    resolution_path: str
    resumed: bool
    reservation_paths_created: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "branch": self.branch,
            "fresh_candidate_attempt_id": self.fresh_candidate_attempt_id,
            "recovery_intent_path": self.recovery_intent_path,
            "host_containment_path": self.host_containment_path,
            "resolution_path": self.resolution_path,
            "resumed": self.resumed,
            "reservation_paths_created": list(self.reservation_paths_created),
        }


@dataclass(slots=True)
class _RecoveryPlan:
    request: dict[str, Any]
    request_binding: dict[str, Any]
    scan: journal.ModalGlobalJournalScan
    state: journal.ModalAttemptJournalState | None
    accumulator: Any | None
    branch: str | None
    blockers: list[str]
    specs: tuple[journal.ModalRemoteRunReservationSpec, ...]
    snapshot_manifest: dict[str, Any] | None
    current_boot_started_at_unix_microseconds: int | None
    current_boot_session_sha256: str | None
    containment_basis: str | None
    process_probe_result: str | None


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
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


def _canonical_utc_now(now_factory: Callable[[], datetime]) -> str:
    value = now_factory()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("recovery clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _logical_from_absolute_input(
    project_root: Path,
    path: str | Path,
    *,
    field: str,
) -> str:
    raw = os.fspath(path)
    selected = Path(raw)
    if not selected.is_absolute() or str(Path(os.path.abspath(raw))) != raw:
        raise ValueError(f"{field} must be an explicit canonical absolute path")
    try:
        relative = selected.relative_to(project_root)
        logical = safe_relative_path(relative.as_posix())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must remain inside the locked project") from error
    return logical.as_posix()


def _load_request_locked(
    lock_descriptor: int,
    request_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    project_root = held_modal_action_lock_project_root(lock_descriptor)
    logical = _logical_from_absolute_input(
        project_root,
        request_path,
        field="recovery request path",
    )
    reader = journal._NamespaceReader(project_root)
    try:
        selected = reader.read_bound_file(
            logical,
            field="recovery operator request",
            optional=False,
            maximum_bytes=journal._MAX_JSON_BYTES,
            required_mode=0o600,
        )
        if selected is None:  # pragma: no cover - optional=False
            raise AssertionError("recovery request disappeared")
        raw, binding = selected
        request = journal._bound_json_object(raw, field="recovery operator request")
        if set(request) != journal._RECOVERY_REQUEST_FIELDS or (
            request["schema_name"] != journal.RECOVERY_REQUEST_SCHEMA_NAME
            or request["schema_version"] != journal.RECOVERY_REQUEST_SCHEMA_VERSION
            or request["expected_branch"] not in journal.RECOVERY_BRANCHES
        ):
            raise ValueError("recovery request has an invalid exact contract")
        journal._attempt_id(request["attempt_id"], "recovery_request.attempt_id")
        candidate = journal._attempt_id(
            request["fresh_candidate_attempt_id"],
            "recovery_request.fresh_candidate_attempt_id",
        )
        if candidate == request["attempt_id"]:
            raise ValueError("fresh candidate attempt ID reuses the orphan attempt")
        request["initial_reservation_bindings"] = (
            journal._recovery_binding_roster(
                request["initial_reservation_bindings"],
                field="recovery_request.initial_reservation_bindings",
            )
        )
        snapshot = request["snapshot_manifest_path"]
        if snapshot is not None:
            _logical_from_absolute_input(
                project_root,
                snapshot,
                field="recovery snapshot manifest path",
            )
        reader.require_stable()
        assert_modal_action_lock_identity(lock_descriptor)
        return request, {
            "path": binding.path,
            "sha256": binding.sha256,
            "size_bytes": binding.size_bytes,
        }
    finally:
        reader.close()


def _default_process_probe(project_root: Path, path: str, digest: str) -> str:
    from scripts.launch_modal import probe_same_boot_modal_process_group

    return probe_same_boot_modal_process_group(
        project_root,
        process_start_receipt_path=path,
        process_start_receipt_sha256=digest,
    )


def _default_boot_state_provider(host_anchor_sha256: str) -> tuple[int, str]:
    from scripts import launch_modal

    started = launch_modal._default_boot_session_provider()
    session = launch_modal._local_boot_session_sha256(
        host_anchor_sha256,
        launch_modal._default_boot_identity_provider(),
    )
    return started, session


def _identity_dict(identity: ModalLiveCohortIdentity) -> dict[str, str]:
    return {
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
    }


def _binding(record: journal.ModalJournalRecord) -> dict[str, Any]:
    return {
        "path": record.binding.path,
        "sha256": record.binding.sha256,
        "size_bytes": record.binding.size_bytes,
    }


def _find_state(
    scan: journal.ModalGlobalJournalScan,
    attempt_id: str,
) -> journal.ModalAttemptJournalState | None:
    return next(
        (state for state in scan.attempts if state.attempt_id == attempt_id),
        None,
    )


def _accumulator_from_scan(
    scan: journal.ModalGlobalJournalScan,
    state: journal.ModalAttemptJournalState,
) -> Any:
    accumulator = journal._AttemptAccumulator(attempt_id=state.attempt_id)
    accumulator.identity = state.identity
    accumulator.intent = state.intent
    accumulator.terminal = state.terminal
    accumulator.rejection = state.rejection
    accumulator.reservations = list(state.reservations)
    accumulator.marker = state.process_marker
    accumulator.recoveries = list(state.recoveries)
    if state.identity is not None:
        cohort = next(
            (item for item in scan.cohorts if item.identity == state.identity),
            None,
        )
        if cohort is not None:
            accumulator.aggregates = [
                record
                for record in cohort.aggregates
                if record.payload["attempt_id"] == state.attempt_id
            ]
    return accumulator


def _recovery_paths(
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> dict[str, str]:
    return {
        "intent": modal_action_recovery_intent_path(identity, attempt_id).as_posix(),
        "host_containment": modal_action_host_containment_path(
            identity, attempt_id
        ).as_posix(),
        "resolution": modal_action_recovery_resolution_path(
            identity, attempt_id
        ).as_posix(),
    }


def _snapshot_binding_from_request(
    lock_descriptor: int,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    selected = request["snapshot_manifest_path"]
    if selected is None:
        return None
    project_root = held_modal_action_lock_project_root(lock_descriptor)
    logical = _logical_from_absolute_input(
        project_root,
        selected,
        field="recovery snapshot manifest path",
    )
    reader = journal._NamespaceReader(project_root)
    try:
        observed = reader.read_bound_file(
            logical,
            field="recovery selected snapshot manifest",
            optional=False,
            maximum_bytes=journal._MAX_JSON_BYTES,
            required_mode=0o600,
        )
        if observed is None:  # pragma: no cover - optional=False
            raise AssertionError("recovery snapshot disappeared")
        _raw, binding = observed
        reader.require_stable()
        return {
            "path": binding.path,
            "sha256": binding.sha256,
            "size_bytes": binding.size_bytes,
        }
    finally:
        reader.close()


def _validate_boot_state(value: object) -> tuple[int, str]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("trusted boot-state provider returned an invalid pair")
    started, session = value
    if (
        type(started) is not int
        or started < journal._MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS
    ):
        raise ValueError("trusted boot-state provider returned an invalid start")
    journal._sha256(session, "trusted current boot-session SHA-256")
    return started, session


def _stage_name(state: journal.ModalAttemptJournalState | None) -> str:
    if state is None or not state.recoveries:
        return "not_started"
    kinds = {item.kind for item in state.recoveries}
    if kinds == {"intent"}:
        return "intent_published"
    if kinds == {"intent", "host_containment"}:
        return "host_containment_published"
    if kinds == {"intent", "host_containment", "resolution"}:
        return "resolved"
    return "invalid_partial"


def _plan_locked(
    lock_descriptor: int,
    request_path: str | Path,
    *,
    process_probe: ProcessProbe,
    boot_state_provider: BootStateProvider,
) -> _RecoveryPlan:
    request, request_binding = _load_request_locked(lock_descriptor, request_path)
    scan = journal.scan_modal_global_action_journal(
        lock_descriptor=lock_descriptor,
        process_probe=process_probe,
    )
    state = _find_state(scan, request["attempt_id"])
    blockers: list[str] = []
    if state is None:
        blockers.append("attempt_not_found")
        return _RecoveryPlan(
            request,
            request_binding,
            scan,
            None,
            None,
            None,
            blockers,
            (),
            None,
            None,
            None,
            None,
            None,
        )
    if state.identity is None:
        blockers.append("attempt_lacks_complete_cohort_identity")
    accumulator = _accumulator_from_scan(scan, state)
    existing = {item.kind: item for item in state.recoveries}
    if "intent" in existing:
        branch = existing["intent"].record.payload["branch"]
    elif state.intent is None and state.reservations and (
        state.process_marker is None and state.terminal is None
    ):
        branch = "definitely_not_started"
    elif state.intent is not None and state.disposition == "unresolved":
        branch = "may_have_started_contained"
    else:
        branch = None
        blockers.append("attempt_is_not_an_allowed_orphan_state")
    if branch is not None and branch != request["expected_branch"]:
        blockers.append("request_branch_does_not_match_journal")
    candidate = request["fresh_candidate_attempt_id"]
    candidate_state = _find_state(scan, candidate)
    if candidate_state is not None:
        blockers.append("fresh_candidate_attempt_id_is_already_used")
    specs: tuple[journal.ModalRemoteRunReservationSpec, ...] = ()
    if branch is not None:
        try:
            specs = journal._recovery_claim_specs(accumulator)
        except (TypeError, ValueError) as error:
            blockers.append(f"reservation_claim_invalid:{type(error).__name__}")
        else:
            expected_paths = {spec.binding["path"] for spec in specs}
            current_by_path = {
                record.binding.path: _binding(record)
                for record in state.reservations
            }
            requested_by_path = {
                item["path"]: item
                for item in request["initial_reservation_bindings"]
            }
            if not set(requested_by_path) <= set(current_by_path):
                blockers.append("request_initial_reservation_is_missing")
            elif any(
                current_by_path[path] != binding
                for path, binding in requested_by_path.items()
            ):
                blockers.append("request_initial_reservation_binding_changed")
            if not set(current_by_path) <= expected_paths:
                blockers.append("reservation_namespace_contains_foreign_sibling")
            if branch == "may_have_started_contained" and (
                request["initial_reservation_bindings"]
                != sorted(current_by_path.values(), key=lambda item: item["path"])
            ):
                blockers.append("may_have_started_request_initial_roster_changed")
    snapshot_binding: dict[str, Any] | None = None
    current_start: int | None = None
    current_session: str | None = None
    containment_basis: str | None = None
    probe_result = state.process_probe_result
    if branch == "definitely_not_started":
        if state.intent is not None or state.process_marker is not None:
            blockers.append("pre_popen_proof_is_not_exact")
        if request["snapshot_manifest_path"] is not None:
            blockers.append("pre_popen_recovery_forbids_snapshot_input")
        containment_basis = "definite_pre_popen_journal_state"
    elif branch == "may_have_started_contained" and state.identity is not None:
        if request["snapshot_manifest_path"] is None:
            blockers.append("may_have_started_requires_snapshot_manifest")
        else:
            try:
                snapshot_binding = _snapshot_binding_from_request(
                    lock_descriptor, request
                )
            except (OSError, TypeError, ValueError) as error:
                blockers.append(f"snapshot_manifest_invalid:{type(error).__name__}")
        original_start, original_session = journal._recovery_original_boot_identity(
            accumulator
        )
        try:
            current_start, current_session = _validate_boot_state(
                boot_state_provider(
                    journal._recovery_attempt_owner_payload(accumulator)[
                        "local_host_anchor_sha256"
                    ]
                )
            )
        except (OSError, TypeError, ValueError) as error:
            blockers.append(f"trusted_boot_state_invalid:{type(error).__name__}")
        else:
            if current_session == original_session:
                if current_start != original_start:
                    blockers.append("same_boot_hash_has_changed_start_time")
                elif (
                    state.terminal is None
                    or state.process_marker is None
                    or state.terminal.payload["modal_cli_process_started"] is not True
                    or state.terminal.payload["local_process_start_receipt_sha256"]
                    != state.process_marker.binding.sha256
                ):
                    blockers.append("same_boot_marker_or_terminal_is_missing_or_unbound")
                elif probe_result == "same_boot_process_group_absent":
                    containment_basis = "same_boot_process_group_absent"
                elif probe_result == "same_boot_process_group_exists":
                    blockers.append("same_boot_process_group_still_exists")
                elif probe_result == "same_boot_process_identity_changed":
                    blockers.append("same_boot_process_identity_changed_or_reused")
                else:
                    blockers.append("same_boot_process_probe_is_ambiguous")
            elif current_start <= original_start:
                blockers.append("changed_boot_hash_is_not_strictly_later")
            elif state.process_marker is not None and probe_result != (
                "different_boot_session"
            ):
                blockers.append("later_boot_marker_probe_did_not_confirm_reboot")
            else:
                containment_basis = "strictly_later_boot_session"
    return _RecoveryPlan(
        request=request,
        request_binding=request_binding,
        scan=scan,
        state=state,
        accumulator=accumulator,
        branch=branch,
        blockers=sorted(set(blockers)),
        specs=specs,
        snapshot_manifest=snapshot_binding,
        current_boot_started_at_unix_microseconds=current_start,
        current_boot_session_sha256=current_session,
        containment_basis=containment_basis,
        process_probe_result=probe_result,
    )


def inspect_modal_action_recovery(
    *,
    project_root: str | Path,
    request_path: str | Path,
    process_probe: ProcessProbe | None = None,
    boot_state_provider: BootStateProvider | None = None,
) -> ModalActionRecoveryInspection:
    """Return the exact recovery branch/candidate/blockers without writes."""

    lock_descriptor = acquire_existing_modal_action_lock(project_root)
    try:
        plan = _plan_locked(
            lock_descriptor,
            request_path,
            process_probe=process_probe or _default_process_probe,
            boot_state_provider=boot_state_provider or _default_boot_state_provider,
        )
        paths = (
            _recovery_paths(plan.state.identity, plan.request["attempt_id"])
            if plan.state is not None and plan.state.identity is not None
            else {}
        )
        return ModalActionRecoveryInspection(
            attempt_id=plan.request["attempt_id"],
            branch=plan.branch,
            fresh_candidate_attempt_id=plan.request[
                "fresh_candidate_attempt_id"
            ],
            blockers=tuple(plan.blockers),
            ready_to_publish=not plan.blockers and plan.branch is not None,
            recovery_stage=_stage_name(plan.state),
            recovery_paths=paths,
        )
    finally:
        release_modal_action_lock(lock_descriptor)


def _read_existing_canonical_file(path: Path, *, expected: bytes) -> bool:
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return False
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_uid != os.getuid()
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != 0o600
        or before.st_size > journal._MAX_JSON_BYTES
    ):
        raise ModalActionRecoveryError(
            "existing recovery stage is not an owned single-link 0600 file"
        )
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        raw = bytearray()
        while len(raw) <= journal._MAX_JSON_BYTES:
            chunk = os.read(
                descriptor,
                min(64 * 1024, journal._MAX_JSON_BYTES + 1 - len(raw)),
            )
            if not chunk:
                break
            raw.extend(chunk)
        after = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened != after
            or len(raw) != after.st_size
        ):
            raise ModalActionRecoveryError(
                "existing recovery stage changed while it was read"
            )
    finally:
        os.close(descriptor)
    rebound = os.lstat(path)
    if (rebound.st_dev, rebound.st_ino) != (before.st_dev, before.st_ino):
        raise ModalActionRecoveryError(
            "existing recovery stage path changed while it was read"
        )
    if bytes(raw) != expected:
        raise ModalActionRecoveryError(
            "existing recovery stage is not byte-identical to the expected stage"
        )
    return True


def _publish_or_resume(
    project_root: Path,
    logical_path: str,
    payload: Mapping[str, Any],
    *,
    lock_descriptor: int,
) -> tuple[dict[str, Any], bool]:
    assert_modal_action_lock_identity(lock_descriptor)
    logical = safe_relative_path(logical_path)
    destination = project_root.joinpath(*logical.parts)
    encoded = _canonical_json_bytes(payload)
    resumed = _read_existing_canonical_file(destination, expected=encoded)
    if not resumed:
        try:
            create_json_exclusive(destination, payload)
        except FileExistsError as error:
            resumed = _read_existing_canonical_file(destination, expected=encoded)
            if not resumed:  # pragma: no cover - FileExistsError proves existence
                raise AssertionError(
                    "competing recovery file disappeared"
                ) from error
    assert_modal_action_lock_identity(lock_descriptor)
    return {
        "path": logical.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
    }, resumed


def _rescan(
    lock_descriptor: int,
    *,
    process_probe: ProcessProbe,
) -> journal.ModalGlobalJournalScan:
    return journal.scan_modal_global_action_journal(
        lock_descriptor=lock_descriptor,
        process_probe=process_probe,
    )


def _publish_reservation_repairs(
    plan: _RecoveryPlan,
    *,
    lock_descriptor: int,
    process_probe: ProcessProbe,
    publish_hook: PublishHook | None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if plan.state is None or plan.state.identity is None:
        raise ModalActionRecoveryBlockedError("recovery attempt has no identity")
    if plan.state.recoveries:
        return [], ()
    project_root = held_modal_action_lock_project_root(lock_descriptor)
    initial_paths = {record.binding.path for record in plan.state.reservations}
    published: list[dict[str, Any]] = []
    paths_created: list[str] = []
    for spec in plan.specs:
        logical = spec.binding["path"]
        if logical in initial_paths:
            continue
        recovery_intent = project_root.joinpath(
            *modal_action_recovery_intent_path(
                plan.state.identity,
                plan.state.attempt_id,
            ).parts
        )
        if recovery_intent.exists() or recovery_intent.is_symlink():
            raise ModalActionRecoveryError(
                "reservation repair is forbidden after recovery intent publication"
            )
        binding, resumed = _publish_or_resume(
            project_root,
            logical,
            spec.payload,
            lock_descriptor=lock_descriptor,
        )
        if resumed:
            raise ModalActionRecoveryError(
                "reservation sibling appeared after the held-lock inspection"
            )
        published.append(binding)
        paths_created.append(logical)
        if publish_hook is not None:
            publish_hook("reservation", project_root.joinpath(*Path(logical).parts))
        _rescan(lock_descriptor, process_probe=process_probe)
    published.sort(key=lambda item: item["path"])
    return published, tuple(paths_created)


def _build_host_payload(
    plan: _RecoveryPlan,
    *,
    intent_payload: Mapping[str, Any],
    intent_binding: Mapping[str, Any],
    recorded_at_utc: str,
) -> dict[str, Any]:
    if plan.accumulator is None or plan.branch is None:
        raise ModalActionRecoveryBlockedError("recovery plan is incomplete")
    original_start, original_session = journal._recovery_original_boot_identity(
        plan.accumulator
    )
    marker = plan.state.process_marker if plan.state is not None else None
    terminal = plan.state.terminal if plan.state is not None else None
    process_identity = None
    if marker is not None:
        process_identity = {
            "process_id": marker.payload["process_id"],
            "process_group_id": marker.payload["expected_process_group_id"],
            "session_id": marker.payload["expected_session_id"],
            "process_birth_identity_sha256": marker.payload[
                "process_birth_identity_sha256"
            ],
        }
    if plan.branch == "definitely_not_started":
        current_start = None
        current_session = None
        probe_result = "not_applicable"
        process_identity = None
        marker_binding = None
        terminal_binding = None
    else:
        current_start = plan.current_boot_started_at_unix_microseconds
        current_session = plan.current_boot_session_sha256
        probe_result = (
            plan.process_probe_result if marker is not None else "not_applicable"
        )
        marker_binding = _binding(marker) if marker is not None else None
        terminal_binding = _binding(terminal) if terminal is not None else None
    return {
        "schema_name": journal.RECOVERY_HOST_CONTAINMENT_SCHEMA_NAME,
        "schema_version": journal.RECOVERY_SCHEMA_VERSION,
        "attempt_id": intent_payload["attempt_id"],
        "recorded_at_utc": recorded_at_utc,
        "identity": intent_payload["identity"],
        "branch": plan.branch,
        "request_binding": intent_payload["request_binding"],
        "source_evidence": intent_payload["source_evidence"],
        "recovery_intent": dict(intent_binding),
        "containment_basis": plan.containment_basis,
        "original_boot_started_at_unix_microseconds": original_start,
        "original_boot_session_sha256": original_session,
        "current_boot_started_at_unix_microseconds": current_start,
        "current_boot_session_sha256": current_session,
        "process_probe_result": probe_result,
        "process_identity": process_identity,
        "marker_binding": marker_binding,
        "terminal_binding": terminal_binding,
        "quarantined": True,
        "eligible_for_final_acceptance": False,
        "fresh_attempt_required": True,
    }


def _derive_resolution_claims(
    lock_descriptor: int,
    plan: _RecoveryPlan,
    *,
    intent_payload: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if plan.accumulator is None or plan.state is None or plan.state.identity is None:
        raise ModalActionRecoveryBlockedError("recovery plan is incomplete")
    if plan.branch == "definitely_not_started":
        snapshot = None
        modal_exposure = journal._expected_zero_modal_exposure()
        known_objects = journal._expected_empty_known_objects()
        run_dispositions = [
            {
                "run_id": run_id,
                "present": False,
                "disposition": "not_applicable_pre_popen",
            }
            for run_id in sorted(
                journal._recovery_attempt_concrete_run_ids(plan.accumulator)
            )
        ]
        reader = journal._NamespaceReader(
            held_modal_action_lock_project_root(lock_descriptor)
        )
        try:
            provider = journal._expected_recovery_provider_exposure(
                reader,
                plan.accumulator,
                branch=plan.branch,
                cache={},
            )
            reader.require_stable()
        finally:
            reader.close()
        return (
            snapshot,
            modal_exposure,
            {"known": known_objects, "provider": provider},
            run_dispositions,
        )
    reader = journal._NamespaceReader(
        held_modal_action_lock_project_root(lock_descriptor)
    )
    try:
        cache: dict[str, tuple[bytes, journal.ModalJournalFileBinding]] = {}
        snapshot, modal_exposure, known_objects, run_dispositions = (
            journal._validate_recovery_snapshot(
                reader,
                intent_payload["snapshot_manifest"],
                identity=plan.state.identity,
                attempt=plan.accumulator,
                cache=cache,
            )
        )
        provider = journal._expected_recovery_provider_exposure(
            reader,
            plan.accumulator,
            branch=plan.branch,
            cache=cache,
        )
        reader.require_stable()
        return (
            snapshot,
            modal_exposure,
            {"known": known_objects, "provider": provider},
            run_dispositions,
        )
    finally:
        reader.close()


def resolve_modal_action_recovery(
    *,
    project_root: str | Path,
    request_path: str | Path,
    process_probe: ProcessProbe | None = None,
    boot_state_provider: BootStateProvider | None = None,
    now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
    publish_hook: PublishHook | None = None,
) -> ModalActionRecoveryResult:
    """Repair an allowed roster and publish/resume the exact recovery triplet."""

    selected_probe = process_probe or _default_process_probe
    selected_boot = boot_state_provider or _default_boot_state_provider
    lock_descriptor = acquire_modal_action_lock(project_root)
    created_paths: tuple[str, ...] = ()
    any_resumed = False
    try:
        plan = _plan_locked(
            lock_descriptor,
            request_path,
            process_probe=selected_probe,
            boot_state_provider=selected_boot,
        )
        if plan.blockers or plan.branch is None or plan.state is None:
            raise ModalActionRecoveryBlockedError(
                "recovery is blocked: " + ", ".join(plan.blockers)
            )
        if plan.state.identity is None or plan.accumulator is None:
            raise ModalActionRecoveryBlockedError(
                "recovery attempt lacks a complete cohort identity"
            )
        paths = _recovery_paths(plan.state.identity, plan.state.attempt_id)
        if _stage_name(plan.state) == "resolved":
            return ModalActionRecoveryResult(
                attempt_id=plan.state.attempt_id,
                branch=plan.branch,
                fresh_candidate_attempt_id=plan.request[
                    "fresh_candidate_attempt_id"
                ],
                recovery_intent_path=paths["intent"],
                host_containment_path=paths["host_containment"],
                resolution_path=paths["resolution"],
                resumed=True,
                reservation_paths_created=(),
            )

        current_bindings_before_repair = sorted(
            (_binding(record) for record in plan.state.reservations),
            key=lambda item: item["path"],
        )
        initial_bindings = list(plan.request["initial_reservation_bindings"])
        initial_paths = {item["path"] for item in initial_bindings}
        if any(
            item["path"] not in initial_paths
            for item in current_bindings_before_repair
        ):
            any_resumed = True
        if plan.branch == "definitely_not_started" and not plan.state.recoveries:
            _new_bindings, created_paths = _publish_reservation_repairs(
                plan,
                lock_descriptor=lock_descriptor,
                process_probe=selected_probe,
                publish_hook=publish_hook,
            )
            if _new_bindings:
                refreshed = _rescan(lock_descriptor, process_probe=selected_probe)
                state = _find_state(refreshed, plan.request["attempt_id"])
                if state is None:
                    raise ModalActionRecoveryError(
                        "repaired reservation attempt disappeared"
                    )
                plan.scan = refreshed
                plan.state = state
                plan.accumulator = _accumulator_from_scan(refreshed, state)

        project = held_modal_action_lock_project_root(lock_descriptor)
        existing = {item.kind: item for item in plan.state.recoveries}
        if "intent" in existing:
            intent_record = existing["intent"].record
            intent_payload = dict(intent_record.payload)
            intent_binding = _binding(intent_record)
            any_resumed = True
        else:
            source = journal._expected_recovery_source_evidence(plan.accumulator)
            final_bindings = source["remote_run_reservations"]
            if plan.branch == "definitely_not_started":
                published_bindings = [
                    item
                    for item in final_bindings
                    if item["path"] not in initial_paths
                ]
                repair_basis = (
                    "global_rejection_planned_roster"
                    if plan.state.rejection is not None
                    else "canonical_reservation_inference"
                )
            else:
                repair_basis = "not_applicable"
                initial_bindings = list(final_bindings)
                published_bindings = []
            action, run_id = journal._recovery_action_and_run_id(plan.accumulator)
            intent_payload = {
                "schema_name": journal.RECOVERY_INTENT_SCHEMA_NAME,
                "schema_version": journal.RECOVERY_SCHEMA_VERSION,
                "attempt_id": plan.state.attempt_id,
                "recorded_at_utc": _canonical_utc_now(now_factory),
                "identity": _identity_dict(plan.state.identity),
                "branch": plan.branch,
                "action": action,
                "run_id": run_id,
                "request_binding": plan.request_binding,
                "source_evidence": source,
                "reservation_repair": {
                    "basis": repair_basis,
                    "initial_reservation_bindings": initial_bindings,
                    "published_reservation_bindings": published_bindings,
                    "final_reservation_bindings": final_bindings,
                },
                "snapshot_manifest": plan.snapshot_manifest,
                "fresh_candidate_attempt_id": plan.request[
                    "fresh_candidate_attempt_id"
                ],
                "quarantined": True,
                "eligible_for_final_acceptance": False,
                "fresh_attempt_required": True,
            }
            intent_binding, resumed = _publish_or_resume(
                project,
                paths["intent"],
                intent_payload,
                lock_descriptor=lock_descriptor,
            )
            any_resumed |= resumed
            if publish_hook is not None:
                publish_hook("intent", project.joinpath(*Path(paths["intent"]).parts))
            refreshed = _rescan(lock_descriptor, process_probe=selected_probe)
            state = _find_state(refreshed, plan.state.attempt_id)
            if state is None:
                raise ModalActionRecoveryError("published recovery intent disappeared")
            plan.scan = refreshed
            plan.state = state
            plan.accumulator = _accumulator_from_scan(refreshed, state)
            existing = {item.kind: item for item in state.recoveries}

        if "host_containment" in existing:
            host_record = existing["host_containment"].record
            host_payload = dict(host_record.payload)
            host_binding = _binding(host_record)
            any_resumed = True
        else:
            host_payload = _build_host_payload(
                plan,
                intent_payload=intent_payload,
                intent_binding=intent_binding,
                recorded_at_utc=_canonical_utc_now(now_factory),
            )
            host_binding, resumed = _publish_or_resume(
                project,
                paths["host_containment"],
                host_payload,
                lock_descriptor=lock_descriptor,
            )
            any_resumed |= resumed
            if publish_hook is not None:
                publish_hook(
                    "host_containment",
                    project.joinpath(*Path(paths["host_containment"]).parts),
                )
            refreshed = _rescan(lock_descriptor, process_probe=selected_probe)
            state = _find_state(refreshed, plan.state.attempt_id)
            if state is None:
                raise ModalActionRecoveryError(
                    "published recovery host containment disappeared"
                )
            plan.scan = refreshed
            plan.state = state
            plan.accumulator = _accumulator_from_scan(refreshed, state)
            existing = {item.kind: item for item in state.recoveries}

        if "resolution" in existing:
            any_resumed = True
        else:
            snapshot, modal_exposure, exposure, run_dispositions = (
                _derive_resolution_claims(
                    lock_descriptor,
                    plan,
                    intent_payload=intent_payload,
                )
            )
            resolution_payload = {
                "schema_name": journal.RECOVERY_RESOLUTION_SCHEMA_NAME,
                "schema_version": journal.RECOVERY_SCHEMA_VERSION,
                "attempt_id": plan.state.attempt_id,
                "recorded_at_utc": _canonical_utc_now(now_factory),
                "identity": intent_payload["identity"],
                "branch": plan.branch,
                "request_binding": intent_payload["request_binding"],
                "source_evidence": intent_payload["source_evidence"],
                "recovery_intent": dict(intent_binding),
                "host_containment": dict(host_binding),
                "final_reservation_bindings": intent_payload[
                    "reservation_repair"
                ]["final_reservation_bindings"],
                "snapshot_evidence": snapshot,
                "modal_exposure": modal_exposure,
                "provider_exposure": exposure["provider"],
                "known_remote_objects": exposure["known"],
                "run_directory_dispositions": run_dispositions,
                "fresh_candidate_attempt_id": intent_payload[
                    "fresh_candidate_attempt_id"
                ],
                "quarantined": True,
                "eligible_for_final_acceptance": False,
                "fresh_attempt_required": True,
                "validated": True,
            }
            _resolution_binding, resumed = _publish_or_resume(
                project,
                paths["resolution"],
                resolution_payload,
                lock_descriptor=lock_descriptor,
            )
            any_resumed |= resumed
            if publish_hook is not None:
                publish_hook(
                    "resolution",
                    project.joinpath(*Path(paths["resolution"]).parts),
                )
        final_scan = _rescan(lock_descriptor, process_probe=selected_probe)
        final_state = _find_state(final_scan, plan.state.attempt_id)
        if (
            final_state is None
            or final_state.disposition != "closed"
            or final_state.blocker_codes
            or {item.kind for item in final_state.recoveries}
            != {"intent", "host_containment", "resolution"}
        ):
            raise ModalActionRecoveryError(
                "published recovery triplet did not close the quarantined attempt"
            )
        return ModalActionRecoveryResult(
            attempt_id=final_state.attempt_id,
            branch=plan.branch,
            fresh_candidate_attempt_id=plan.request[
                "fresh_candidate_attempt_id"
            ],
            recovery_intent_path=paths["intent"],
            host_containment_path=paths["host_containment"],
            resolution_path=paths["resolution"],
            resumed=any_resumed,
            reservation_paths_created=created_paths,
        )
    finally:
        release_modal_action_lock(lock_descriptor)


publish_modal_action_recovery = resolve_modal_action_recovery


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    for name in ("inspect", "resolve", "publish"):
        command = subcommands.add_parser(name)
        command.add_argument("--project-root", type=Path, default=ROOT)
        command.add_argument(
            "--request",
            type=Path,
            required=True,
            help="absolute path to an owned 0600 ModalActionRecoveryRequest/1.0",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.request.is_absolute():
        raise ValueError("--request must be an explicit absolute path")
    if arguments.operation == "inspect":
        result: ModalActionRecoveryInspection | ModalActionRecoveryResult = (
            inspect_modal_action_recovery(
                project_root=arguments.project_root,
                request_path=arguments.request,
            )
        )
    else:
        result = resolve_modal_action_recovery(
            project_root=arguments.project_root,
            request_path=arguments.request,
        )
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
