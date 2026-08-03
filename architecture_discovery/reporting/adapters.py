"""Adapters from immutable per-run artifacts to reproducibility-report records.

The adapter deliberately requires the caller to provide the frozen assignment
record.  It never infers an assignment from whatever runs happened to finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from artifacts import EventKind, RunArtifactStore
from reconstruction import ReconstructedRun, reconstruct_run
from reporting.records import (
    ReportArtifact,
    ReportArtifactKind,
    RunReportRecord,
    RunReportStatus,
)
from study.serialization import content_hash


_EVENT_KIND_MAP: dict[EventKind, ReportArtifactKind] = {
    EventKind.PROPOSAL: ReportArtifactKind.PROPOSAL,
    EventKind.CANDIDATE: ReportArtifactKind.CANDIDATE,
    EventKind.FAILURE: ReportArtifactKind.FAILURE,
    EventKind.REPAIR: ReportArtifactKind.REPAIR,
    EventKind.TRAINING: ReportArtifactKind.TRAINING,
    EventKind.SEARCH_EVALUATION: ReportArtifactKind.EVALUATION,
    EventKind.QUALIFICATION_EVALUATION: ReportArtifactKind.EVALUATION,
    EventKind.CONFIRMATION_EVALUATION: ReportArtifactKind.EVALUATION,
    EventKind.PARENT_SELECTION: ReportArtifactKind.PARENT_SELECTION,
    EventKind.BUDGET: ReportArtifactKind.BUDGET,
    EventKind.PROMOTION: ReportArtifactKind.PROMOTION,
    EventKind.MECHANISM_CLUSTER: ReportArtifactKind.CLUSTER,
    EventKind.RUN_STATUS: ReportArtifactKind.RUN_RECORD,
    EventKind.RERUN_ATTEMPT: ReportArtifactKind.RERUN,
}


@dataclass(frozen=True)
class AdaptedRunReport:
    reconstructed: ReconstructedRun
    run_record: RunReportRecord
    artifacts: tuple[ReportArtifact, ...]
    artifact_index_sha256: str


def _review_kind(payload: Mapping[str, Any]) -> ReportArtifactKind:
    stage = str(payload.get("review_stage", payload.get("stage", "raw"))).lower()
    if "adjudicat" in stage or stage in {"final", "consensus"}:
        return ReportArtifactKind.ADJUDICATED_REVIEW
    return ReportArtifactKind.RAW_REVIEW


def _event_kind(event_kind: EventKind, payload: Mapping[str, Any]) -> ReportArtifactKind:
    if event_kind is EventKind.REVIEW:
        return _review_kind(payload)
    try:
        return _EVENT_KIND_MAP[event_kind]
    except KeyError as error:  # pragma: no cover - defensive against enum expansion
        raise ValueError(f"unsupported artifact event kind {event_kind.value!r}") from error


def _terminal_status(reconstructed: ReconstructedRun) -> RunReportStatus:
    if reconstructed.status == "completed":
        return RunReportStatus.COMPLETED
    if reconstructed.status in {"scientific_failure", "candidate_failure"}:
        return RunReportStatus.SCIENTIFIC_FAILURE
    if reconstructed.status == "infrastructure_failure":
        return RunReportStatus.INFRASTRUCTURE_FAILURE
    raise ValueError(
        f"run {reconstructed.context.run_id!r} is nonterminal "
        f"({reconstructed.status!r}); reproducibility reports require every assigned "
        "run to have a terminal outcome"
    )


def _validate_assignment(
    assignment_payload: Mapping[str, Any], store: RunArtifactStore
) -> dict[str, Any]:
    material = dict(assignment_payload)
    context = store.context
    expected: dict[str, Any] = {
        "study_id": context.study_id,
        "block_id": context.block_id,
        "run_id": context.run_id,
        "run_seed": context.run_seed,
    }
    for field_name, expected_value in expected.items():
        if material.get(field_name) != expected_value:
            raise ValueError(
                f"frozen assignment field {field_name!r} does not match the run context"
            )
    raw_condition = material.get("condition_id")
    if raw_condition is None and isinstance(material.get("condition"), Mapping):
        raw_condition = material["condition"].get("condition_id")
    if raw_condition != context.condition_id:
        raise ValueError("frozen assignment condition does not match the run context")
    if content_hash(material) != context.assignment_sha256:
        raise ValueError("frozen assignment payload does not match assignment_sha256")
    return material


def adapt_run_store(
    store: RunArtifactStore, *, assignment_payload: Mapping[str, Any]
) -> AdaptedRunReport:
    """Convert one terminal run without weakening its assignment or event provenance."""

    assignment = _validate_assignment(assignment_payload, store)
    reconstructed = reconstruct_run(store)
    terminal_status = _terminal_status(reconstructed)
    integrity = store.scan(tolerate_trailing_incomplete=True)
    artifact_index, _ = store.build_index()

    assignment_artifact = ReportArtifact.create(
        artifact_id=f"assignment-{store.context.run_id}",
        kind=ReportArtifactKind.ASSIGNMENT,
        payload=assignment,
        record_schema_name=str(assignment.get("schema_name", "FrozenAssignment")),
    )
    artifacts: list[ReportArtifact] = [assignment_artifact]
    run_artifact_ids: list[str] = []
    failure_artifact_ids: list[str] = []
    rerun_artifact_ids: list[str] = []
    budget_artifact_ids: list[str] = []

    for event in integrity.events:
        kind = _event_kind(event.event_kind, event.payload)
        artifact = ReportArtifact.create(
            artifact_id=event.record_id,
            kind=kind,
            payload=event.to_dict(),
            record_schema_name=event.schema_name,
        )
        artifacts.append(artifact)
        if kind is ReportArtifactKind.FAILURE:
            failure_artifact_ids.append(artifact.artifact_id)
        elif kind is ReportArtifactKind.RERUN:
            rerun_artifact_ids.append(artifact.artifact_id)
        elif kind is ReportArtifactKind.BUDGET:
            budget_artifact_ids.append(artifact.artifact_id)
        elif kind in {
            ReportArtifactKind.RUN_RECORD,
            ReportArtifactKind.PROPOSAL,
            ReportArtifactKind.CANDIDATE,
            ReportArtifactKind.TRAINING,
            ReportArtifactKind.EVALUATION,
            ReportArtifactKind.PARENT_SELECTION,
            ReportArtifactKind.REPAIR,
            ReportArtifactKind.PROMOTION,
        }:
            run_artifact_ids.append(artifact.artifact_id)

    reconstruction_artifact = ReportArtifact.create(
        artifact_id=f"reconstruction-{store.context.run_id}",
        kind=ReportArtifactKind.RUN_RECORD,
        payload=reconstructed.to_dict(),
        record_schema_name=reconstructed.schema_name,
    )
    index_artifact = ReportArtifact.create(
        artifact_id=f"artifact-index-{store.context.run_id}",
        kind=ReportArtifactKind.RUN_RECORD,
        payload=artifact_index.to_dict(),
        record_schema_name=artifact_index.schema_name,
    )
    artifacts.extend((reconstruction_artifact, index_artifact))
    run_artifact_ids.extend(
        (reconstruction_artifact.artifact_id, index_artifact.artifact_id)
    )

    if not budget_artifact_ids:
        raise ValueError(
            f"run {store.context.run_id!r} has no retained budget event; "
            "resource and exposure reporting would be incomplete"
        )

    run_record = RunReportRecord(
        study_id=store.context.study_id,
        block_id=store.context.block_id,
        run_id=store.context.run_id,
        condition_id=store.context.condition_id,
        run_seed=store.context.run_seed,
        assignment_sha256=store.context.assignment_sha256,
        assignment_artifact_id=assignment_artifact.artifact_id,
        terminal_status=terminal_status,
        run_artifact_ids=tuple(run_artifact_ids),
        failure_artifact_ids=tuple(failure_artifact_ids),
        rerun_artifact_ids=tuple(rerun_artifact_ids),
        budget_artifact_ids=tuple(budget_artifact_ids),
    )
    return AdaptedRunReport(
        reconstructed=reconstructed,
        run_record=run_record,
        artifacts=tuple(artifacts),
        artifact_index_sha256=artifact_index.index_sha256,
    )
