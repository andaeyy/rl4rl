from __future__ import annotations

from dataclasses import replace
import json

import pytest

from artifacts import RunArtifactStore
from reporting import ReportArtifact, ReportArtifactKind, SectionName
from reporting.adapters import adapt_run_store
from reporting.synthetic import build_synthetic_reconstruction


def test_report_retains_every_assignment_failure_rerun_and_required_section(
    tmp_path,
) -> None:
    result = build_synthetic_reconstruction(tmp_path / "synthetic-report")
    report = result.report

    assert tuple(run.run_id for run in report.runs) == (
        "synthetic-run-c0",
        "synthetic-run-c1",
        "synthetic-run-c2",
        "synthetic-run-c3",
    )
    statuses = {run.run_id: run.terminal_status.value for run in report.runs}
    assert statuses["synthetic-run-c2"] == "scientific_failure"
    assert statuses["synthetic-run-c3"] == "completed"
    c3 = next(run for run in report.runs if run.run_id == "synthetic-run-c3")
    assert len(c3.failure_artifact_ids) == 1
    assert len(c3.rerun_artifact_ids) == 1
    assert {section.name for section in report.sections} == set(SectionName)
    assert all(section.status.value == "complete" for section in report.sections)

    serialized = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert serialized["report_sha256"] == report.report_sha256


def test_report_cannot_drop_an_assigned_run_or_an_event(tmp_path) -> None:
    report = build_synthetic_reconstruction(tmp_path / "synthetic-report").report

    with pytest.raises(ValueError, match="frozen assignment roster"):
        replace(report, runs=report.runs[:-1])

    first = report.runs[0]
    shortened = replace(first, run_artifact_ids=first.run_artifact_ids[:-1])
    with pytest.raises(ValueError, match="unreferenced or omitted"):
        replace(
            report,
            runs=tuple(shortened if run.run_id == first.run_id else run for run in report.runs),
        )


def test_adapter_requires_the_exact_external_frozen_assignment(tmp_path) -> None:
    result = build_synthetic_reconstruction(tmp_path / "synthetic-report")
    manifest = json.loads(result.assignment_path.read_text(encoding="utf-8"))
    assignment = dict(manifest["assignments"][0])
    store = RunArtifactStore.open(
        result.report_path.parent / "runs" / assignment["run_id"]
    )
    assignment["run_seed"] += 1

    with pytest.raises(ValueError, match="run_seed"):
        adapt_run_store(store, assignment_payload=assignment)


def test_report_artifacts_reject_tampering_and_sensitive_fields() -> None:
    original = ReportArtifact.create(
        artifact_id="safe-record",
        kind=ReportArtifactKind.ANALYSIS,
        payload={"schema_name": "SafeRecord", "value": 1},
    )
    with pytest.raises(ValueError, match="content hash mismatch"):
        ReportArtifact(
            artifact_id=original.artifact_id,
            kind=original.kind,
            record_schema_name=original.record_schema_name,
            payload={"schema_name": "SafeRecord", "value": 2},
            content_sha256=original.content_sha256,
        )
    with pytest.raises(ValueError, match="sensitive field"):
        ReportArtifact.create(
            artifact_id="unsafe-record",
            kind=ReportArtifactKind.RUN_RECORD,
            payload={"schema_name": "UnsafeRecord", "credentials": {"token": "x"}},
        )
