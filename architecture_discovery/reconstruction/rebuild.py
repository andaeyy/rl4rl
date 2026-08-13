"""Deterministic reconstruction of state, budgets, ancestry, and outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Iterable

from analysis.outcomes import RunOutcome, RunTerminalStatus
from artifacts.failures import FailureClass, FailureDomain, FailureRecord
from artifacts.records import (
    EventKind,
    EventRecord,
    require_identifier,
    require_sha256,
)
from artifacts.store import RunArtifactStore
from reconstruction.models import ReconstructedRun


_RUN_STATUSES = {
    "initialized",
    "running",
    "completed",
    "scientific_failure",
    "candidate_failure",
    "infrastructure_failure",
}


class ReconstructionError(ValueError):
    pass


def _identifier(payload: Mapping[str, Any], field_name: str) -> str:
    value = str(payload.get(field_name, ""))
    try:
        require_identifier(value, field_name)
    except ValueError as error:
        raise ReconstructionError(str(error)) from error
    return value


def _identifiers(payload: Mapping[str, Any], field_name: str) -> tuple[str, ...]:
    raw = payload.get(field_name, ())
    if not isinstance(raw, (list, tuple)):
        raise ReconstructionError(f"{field_name} must be a sequence")
    values = tuple(str(item) for item in raw)
    for value in values:
        try:
            require_identifier(value, field_name)
        except ValueError as error:
            raise ReconstructionError(str(error)) from error
    return values


def _numeric_mapping(value: Any, field_name: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        raise ReconstructionError(f"{field_name} must be an object")
    result: dict[str, int | float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        try:
            require_identifier(key, f"{field_name} key")
        except ValueError as error:
            raise ReconstructionError(str(error)) from error
        if (
            isinstance(raw_value, bool)
            or not isinstance(raw_value, (int, float))
            or not math.isfinite(float(raw_value))
            or raw_value < 0
        ):
            raise ReconstructionError(
                f"{field_name}.{key} must be a finite non-negative number"
            )
        result[key] = raw_value
    return result


def _update_budget(
    totals: dict[str, int | float], event: EventRecord
) -> dict[str, int | float]:
    has_totals = "totals" in event.payload
    has_delta = "delta" in event.payload
    if has_totals == has_delta:
        raise ReconstructionError(
            "budget event must contain exactly one of totals or delta"
        )
    if has_delta:
        delta = _numeric_mapping(event.payload["delta"], "budget delta")
        return {
            **totals,
            **{key: totals.get(key, 0) + value for key, value in delta.items()},
        }
    snapshot = _numeric_mapping(event.payload["totals"], "budget totals")
    for key, old_value in totals.items():
        if key not in snapshot:
            raise ReconstructionError(f"budget snapshot dropped prior field {key!r}")
        if snapshot[key] < old_value:
            raise ReconstructionError(f"budget total {key!r} decreased")
    return snapshot


def _assert_acyclic(ancestry: Mapping[str, tuple[str, ...]]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(candidate_id: str) -> None:
        if candidate_id in visiting:
            raise ReconstructionError("candidate ancestry contains a cycle")
        if candidate_id in visited:
            return
        visiting.add(candidate_id)
        for parent_id in ancestry.get(candidate_id, ()):
            if parent_id in ancestry:
                visit(parent_id)
        visiting.remove(candidate_id)
        visited.add(candidate_id)

    for candidate_id in ancestry:
        visit(candidate_id)


def reconstruct_run(store: RunArtifactStore) -> ReconstructedRun:
    report = store.scan(tolerate_trailing_incomplete=True)
    context = store.context
    status = "initialized"
    budget_totals: dict[str, int | float] = {}
    accelerator_kind: str | None = None
    budget_schema_version = "1.0"
    budget_events_seen = False
    ancestry: dict[str, tuple[str, ...]] = {}
    cluster_keys: set[str] = set()
    parent_history: list[tuple[str, ...]] = []
    promotions: list[str] = []
    reviews: list[str] = []
    rerun_attempts: list[str] = []
    terminal_failure_domain = ""
    terminal_failure_class = ""
    failures: dict[str, FailureRecord] = {}
    current_attempt_id = ""
    rerun_authorized = False

    for event in report.events:
        payload = event.payload
        if event.event_kind is EventKind.CANDIDATE:
            candidate_id = _identifier(payload, "candidate_id")
            parent_ids = _identifiers(payload, "parent_candidate_ids")
            previous = ancestry.get(candidate_id)
            if previous is not None and previous != parent_ids:
                raise ReconstructionError(
                    f"candidate {candidate_id!r} has conflicting ancestry"
                )
            ancestry[candidate_id] = parent_ids
        elif event.event_kind is EventKind.BUDGET:
            budget_totals = _update_budget(budget_totals, event)
            raw_accelerator = payload.get("accelerator_kind")
            if raw_accelerator is None:
                if "accelerator_seconds" in budget_totals:
                    raise ReconstructionError(
                        "version-2 budget event lacks accelerator_kind"
                    )
                observed_accelerator = (
                    "mps" if "mps_seconds" in budget_totals else None
                )
                observed_version = "1.0"
            else:
                observed_accelerator = str(raw_accelerator)
                if not observed_accelerator or observed_accelerator != raw_accelerator:
                    raise ReconstructionError(
                        "budget accelerator_kind must be non-empty text"
                    )
                if "accelerator_seconds" not in budget_totals:
                    raise ReconstructionError(
                        "version-2 budget event lacks accelerator_seconds"
                    )
                if "mps_seconds" in budget_totals:
                    raise ReconstructionError(
                        "version-2 budget event contains legacy mps_seconds"
                    )
                observed_version = "2.0"
            if accelerator_kind is not None and observed_accelerator != accelerator_kind:
                raise ReconstructionError("budget accelerator_kind changed during the run")
            if budget_totals and observed_accelerator is not None:
                accelerator_kind = observed_accelerator
            if budget_events_seen and budget_schema_version != observed_version:
                raise ReconstructionError("budget schema version changed during the run")
            budget_schema_version = observed_version
            budget_events_seen = True
        elif event.event_kind is EventKind.PARENT_SELECTION:
            parent_history.append(_identifiers(payload, "selected_candidate_ids"))
        elif event.event_kind is EventKind.MECHANISM_CLUSTER:
            _identifier(payload, "cluster_id")
            cluster_key = str(payload.get("mechanism_cluster_key", ""))
            try:
                require_sha256(cluster_key, "mechanism_cluster_key")
            except ValueError as error:
                raise ReconstructionError(str(error)) from error
            run_ids = _identifiers(payload, "run_ids")
            qualifies = payload.get("qualifies_for_primary")
            if not isinstance(qualifies, bool):
                raise ReconstructionError(
                    "mechanism cluster requires boolean qualifies_for_primary"
                )
            if qualifies and context.run_id in run_ids:
                cluster_keys.add(cluster_key)
        elif event.event_kind is EventKind.PROMOTION:
            promotions.append(_identifier(payload, "promotion_id"))
        elif event.event_kind is EventKind.REVIEW:
            reviews.append(_identifier(payload, "review_id"))
        elif event.event_kind is EventKind.RERUN_ATTEMPT:
            rerun_attempt_id = _identifier(payload, "rerun_attempt_id")
            rerun_attempts.append(rerun_attempt_id)
            assigned_run_id = _identifier(payload, "assigned_run_id")
            if assigned_run_id != context.run_id:
                raise ReconstructionError("rerun attempt is linked to another assigned run")
            if status != "infrastructure_failure":
                raise ReconstructionError(
                    "rerun attempt must immediately follow an infrastructure failure"
                )
            triggering_failure_id = _identifier(payload, "triggering_failure_id")
            failure = failures.get(triggering_failure_id)
            if failure is None:
                raise ReconstructionError("rerun refers to an unknown failure record")
            if failure.failure_domain is not FailureDomain.INFRASTRUCTURE:
                raise ReconstructionError("rerun refers to a non-infrastructure failure")
            previous_attempt_id = _identifier(payload, "previous_attempt_id")
            if previous_attempt_id != failure.attempt_id:
                raise ReconstructionError("rerun does not link the failed attempt")
            if current_attempt_id and previous_attempt_id != current_attempt_id:
                raise ReconstructionError("rerun does not extend the active attempt chain")
            if str(payload.get("triggering_failure_class", "")) != failure.failure_class.value:
                raise ReconstructionError("rerun failure class does not match its trigger")
            policy_sha256 = str(payload.get("policy_sha256", ""))
            try:
                require_sha256(policy_sha256, "policy_sha256")
            except ValueError as error:
                raise ReconstructionError(str(error)) from error
            attempt_number = payload.get("attempt_number")
            if (
                not isinstance(attempt_number, int)
                or isinstance(attempt_number, bool)
                or attempt_number != len(rerun_attempts)
            ):
                raise ReconstructionError("rerun attempt numbers must be contiguous")
            current_attempt_id = rerun_attempt_id
            rerun_authorized = True
        elif event.event_kind is EventKind.FAILURE:
            try:
                failure = FailureRecord(
                    failure_id=_identifier(payload, "failure_id"),
                    attempt_id=_identifier(payload, "attempt_id"),
                    failure_class=FailureClass(str(payload.get("failure_class", ""))),
                    failure_domain=FailureDomain(str(payload.get("failure_domain", ""))),
                    stage=_identifier(payload, "stage"),
                    terminal=payload.get("terminal"),
                )
            except (TypeError, ValueError) as error:
                raise ReconstructionError(f"invalid failure record: {error}") from error
            if failure.failure_id in failures:
                raise ReconstructionError("duplicate failure record ID")
            if current_attempt_id and failure.attempt_id != current_attempt_id:
                raise ReconstructionError("failure does not belong to the active attempt")
            if not current_attempt_id:
                current_attempt_id = failure.attempt_id
            failures[failure.failure_id] = failure
            terminal = payload.get("terminal")
            if not isinstance(terminal, bool):
                raise ReconstructionError("failure terminal flag must be boolean")
            if terminal:
                if status == "completed":
                    raise ReconstructionError("completed run cannot acquire a terminal failure")
                terminal_failure_domain = failure.failure_domain.value
                terminal_failure_class = failure.failure_class.value
                status = {
                    FailureDomain.INFRASTRUCTURE: "infrastructure_failure",
                    FailureDomain.CANDIDATE: "candidate_failure",
                    FailureDomain.SCIENTIFIC: "scientific_failure",
                }[failure.failure_domain]
                rerun_authorized = False
        elif event.event_kind is EventKind.RUN_STATUS:
            new_status = str(payload.get("status", ""))
            if new_status not in _RUN_STATUSES:
                raise ReconstructionError(f"unknown run status {new_status!r}")
            if new_status == "running":
                if status in {"scientific_failure", "candidate_failure", "completed"}:
                    raise ReconstructionError(
                        f"terminal {status} run cannot return to running"
                    )
                if status == "infrastructure_failure" and not rerun_authorized:
                    raise ReconstructionError(
                        "infrastructure failure needs an authorized linked rerun"
                    )
                if status == "infrastructure_failure":
                    terminal_failure_domain = ""
                    terminal_failure_class = ""
                rerun_authorized = False
            elif new_status == "completed" and status != "running":
                if status != "completed":
                    raise ReconstructionError("only a running attempt can complete")
            elif new_status in {
                "scientific_failure",
                "candidate_failure",
                "infrastructure_failure",
            } and status != new_status:
                raise ReconstructionError(
                    "terminal failure status requires its explicit FailureRecord first"
                )
            elif new_status == "initialized" and status != "initialized":
                raise ReconstructionError("run cannot return to initialized")
            status = new_status
            if status == "completed":
                terminal_failure_domain = ""
                terminal_failure_class = ""

    _assert_acyclic(ancestry)
    outcome: RunOutcome | None = None
    if status in {
        "completed",
        "scientific_failure",
        "candidate_failure",
        "infrastructure_failure",
    }:
        if status == "completed":
            terminal_status = RunTerminalStatus.COMPLETED
            failure_class = ""
            qualifying_count: int | None = len(cluster_keys)
        elif status == "infrastructure_failure":
            terminal_status = RunTerminalStatus.INFRASTRUCTURE_FAILURE
            failure_class = terminal_failure_class or "unclassified_infrastructure"
            qualifying_count = None
        else:
            terminal_status = RunTerminalStatus.SCIENTIFIC_FAILURE
            failure_class = terminal_failure_class or status
            qualifying_count = None
        outcome = RunOutcome(
            study_id=context.study_id,
            block_id=context.block_id,
            run_id=context.run_id,
            condition_id=context.condition_id,
            run_seed=context.run_seed,
            terminal_status=terminal_status,
            qualifying_cluster_count=qualifying_count,
            proposal_exposure=int(budget_totals.get("proposal_opportunities", 0)),
            token_exposure=int(budget_totals.get("prompt_tokens", 0))
            + int(budget_totals.get("completion_tokens", 0)),
            failure_class=failure_class,
            assignment_hash=context.assignment_sha256,
            run_artifact_hash=report.last_event_sha256,
        )
    return ReconstructedRun(
        context=context,
        status=status,
        last_sequence=len(report.events),
        last_event_sha256=report.last_event_sha256,
        event_record_ids=tuple(event.record_id for event in report.events),
        budget_totals=dict(sorted(budget_totals.items())),
        accelerator_kind=accelerator_kind,
        ancestry={key: ancestry[key] for key in sorted(ancestry)},
        qualifying_mechanism_cluster_keys=tuple(sorted(cluster_keys)),
        parent_selection_history=tuple(parent_history),
        promotion_ids=tuple(promotions),
        review_ids=tuple(reviews),
        rerun_attempt_ids=tuple(rerun_attempts),
        failure_domain=terminal_failure_domain,
        failure_class=terminal_failure_class,
        integrity_findings=report.findings,
        outcome=outcome,
        schema_version=budget_schema_version,
    )


def reconstruct_runs(
    stores: Iterable[RunArtifactStore],
) -> tuple[ReconstructedRun, ...]:
    runs = tuple(reconstruct_run(store) for store in stores)
    identities = [(run.context.study_id, run.context.run_id) for run in runs]
    if len(identities) != len(set(identities)):
        raise ReconstructionError("duplicate assigned runs in reconstruction input")
    return tuple(sorted(runs, key=lambda item: (item.context.block_id, item.context.run_id)))
