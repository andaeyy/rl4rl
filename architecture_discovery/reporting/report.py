"""Complete, hash-addressed reproducibility report assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from reporting.records import (
    RUN_DETAIL_KINDS,
    SECTION_KINDS,
    ArithmeticClaim,
    DerivedArtifact,
    ExternalValidityRecord,
    ReportArtifact,
    ReportArtifactKind,
    ReportSection,
    ResourceDisclosure,
    RunReportRecord,
    SectionName,
    SectionStatus,
    StudyProvenance,
)
from research_ledger.records import require_identifier, require_sha256, require_text
from study.serialization import content_hash, create_json_exclusive


@dataclass(frozen=True)
class ReproducibilityReport:
    report_id: str
    provenance: StudyProvenance
    frozen_assignment_run_ids: tuple[str, ...]
    assignment_roster_sha256: str
    runs: tuple[RunReportRecord, ...]
    artifacts: tuple[ReportArtifact, ...]
    sections: tuple[ReportSection, ...]
    derived_artifacts: tuple[DerivedArtifact, ...]
    resources: ResourceDisclosure
    external_validity: ExternalValidityRecord
    claims: tuple[ArithmeticClaim, ...]
    limitations: tuple[str, ...]
    generated_at_utc: str
    schema_name: str = field(default="ArchitectureDiscoveryReproducibilityReport", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_identifier(self.report_id, "report_id")
        require_sha256(self.assignment_roster_sha256, "assignment_roster_sha256")
        roster = tuple(self.frozen_assignment_run_ids)
        if not roster or len(set(roster)) != len(roster):
            raise ValueError("frozen assignment roster must contain unique run IDs")
        for run_id in roster:
            require_identifier(run_id, "frozen assignment run_id")
        object.__setattr__(self, "frozen_assignment_run_ids", roster)
        object.__setattr__(self, "runs", tuple(sorted(self.runs, key=lambda item: item.run_id)))
        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(self.artifacts, key=lambda item: item.artifact_id)),
        )
        object.__setattr__(
            self,
            "sections",
            tuple(sorted(self.sections, key=lambda item: item.name.value)),
        )
        object.__setattr__(
            self,
            "derived_artifacts",
            tuple(sorted(self.derived_artifacts, key=lambda item: item.artifact_id)),
        )
        object.__setattr__(self, "claims", tuple(sorted(self.claims, key=lambda item: item.claim_id)))
        limitations = tuple(self.limitations)
        if not limitations:
            raise ValueError("reproducibility report requires explicit limitations")
        for limitation in limitations:
            require_text(limitation, "report limitation")
        object.__setattr__(self, "limitations", limitations)
        if not self.generated_at_utc.endswith("Z"):
            raise ValueError("report generation time must be explicit UTC")
        datetime.fromisoformat(self.generated_at_utc.replace("Z", "+00:00"))
        self.validate()

    @property
    def report_sha256(self) -> str:
        return content_hash(self.to_dict(include_hash=False))

    def validate(self) -> None:
        if not self.runs:
            raise ValueError("report must contain every assigned run")
        if any(run.study_id != self.provenance.study_id for run in self.runs):
            raise ValueError("run report records mix studies")
        run_ids = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("report contains duplicate assigned runs")
        if set(run_ids) != set(self.frozen_assignment_run_ids):
            missing = sorted(set(self.frozen_assignment_run_ids) - set(run_ids))
            unexpected = sorted(set(run_ids) - set(self.frozen_assignment_run_ids))
            raise ValueError(
                "report runs differ from the external frozen assignment roster: "
                f"missing={missing}, unexpected={unexpected}"
            )
        expected_roster_hash = content_hash(list(self.frozen_assignment_run_ids))
        if self.assignment_roster_sha256 != expected_roster_hash:
            raise ValueError("assignment roster hash does not match report runs")

        artifact_map = {item.artifact_id: item for item in self.artifacts}
        if len(artifact_map) != len(self.artifacts):
            raise ValueError("report artifact IDs must be globally unique")
        referenced_by_runs: dict[ReportArtifactKind, set[str]] = {
            kind: set() for kind in ReportArtifactKind
        }
        for run in self.runs:
            references = {
                ReportArtifactKind.ASSIGNMENT: (run.assignment_artifact_id,),
                ReportArtifactKind.FAILURE: run.failure_artifact_ids,
                ReportArtifactKind.RERUN: run.rerun_artifact_ids,
                ReportArtifactKind.BUDGET: run.budget_artifact_ids,
            }
            for kind, artifact_ids in references.items():
                for artifact_id in artifact_ids:
                    artifact = artifact_map.get(artifact_id)
                    if artifact is None:
                        raise ValueError(f"run references missing artifact {artifact_id!r}")
                    if artifact.kind is not kind:
                        raise ValueError(
                            f"run reference {artifact_id!r} has kind {artifact.kind}, expected {kind}"
                        )
                    referenced_by_runs[kind].add(artifact_id)
            assignment = artifact_map[run.assignment_artifact_id]
            if assignment.content_sha256 != run.assignment_sha256:
                raise ValueError("run assignment hash differs from its retained artifact")
            for artifact_id in run.run_artifact_ids:
                artifact = artifact_map.get(artifact_id)
                if artifact is None or artifact.kind not in RUN_DETAIL_KINDS:
                    raise ValueError("run detail reference has a missing or invalid artifact kind")
                referenced_by_runs[artifact.kind].add(artifact_id)

        for kind in {
            ReportArtifactKind.ASSIGNMENT,
            ReportArtifactKind.FAILURE,
            ReportArtifactKind.RERUN,
            ReportArtifactKind.BUDGET,
            *RUN_DETAIL_KINDS,
        }:
            retained = {
                artifact.artifact_id for artifact in self.artifacts if artifact.kind is kind
            }
            if retained != referenced_by_runs[kind]:
                raise ValueError(
                    f"unreferenced or omitted {kind.value} records: "
                    f"retained={sorted(retained)}, referenced={sorted(referenced_by_runs[kind])}"
                )

        section_map = {section.name: section for section in self.sections}
        if len(section_map) != len(self.sections) or set(section_map) != set(SectionName):
            raise ValueError("report must contain each required section exactly once")
        for section_name, allowed_kinds in SECTION_KINDS.items():
            section = section_map[section_name]
            expected = {
                artifact.artifact_id
                for artifact in self.artifacts
                if artifact.kind in allowed_kinds
            }
            if set(section.artifact_ids) != expected:
                raise ValueError(
                    f"section {section_name.value} omits or invents retained artifacts"
                )
            if expected and section.status in {
                SectionStatus.NOT_RUN,
                SectionStatus.NOT_APPLICABLE,
            }:
                raise ValueError(
                    f"section {section_name.value} has records but claims it was not run"
                )

        for derived in self.derived_artifacts:
            for source in derived.sources:
                artifact = artifact_map.get(source.artifact_id)
                if artifact is None:
                    raise ValueError("derived artifact source is absent from the report")
                if artifact.content_sha256 != source.content_sha256:
                    raise ValueError("derived artifact source hash does not match report content")

        for claim in self.claims:
            if not set(claim.evidence_artifact_ids).issubset(artifact_map):
                raise ValueError("claim refers to evidence absent from the report")
        external_evidence = set(self.external_validity.second_task_evidence_ids).union(
            self.external_validity.scaling_evidence_ids
        )
        if not external_evidence.issubset(artifact_map):
            raise ValueError("external-validity evidence is absent from the report")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "study_id": self.provenance.study_id,
            "generated_at_utc": self.generated_at_utc,
            "provenance": self.provenance.to_dict(),
            "frozen_assignment_run_ids": list(self.frozen_assignment_run_ids),
            "assignment_roster_sha256": self.assignment_roster_sha256,
            "runs": [item.to_dict() for item in self.runs],
            "artifacts": [item.to_dict() for item in self.artifacts],
            "sections": [item.to_dict() for item in self.sections],
            "derived_artifacts": [item.to_dict() for item in self.derived_artifacts],
            "resources": self.resources.to_dict(),
            "external_validity": self.external_validity.to_dict(),
            "claims": [item.to_dict() for item in self.claims],
            "limitations": list(self.limitations),
        }
        if include_hash:
            payload["report_sha256"] = content_hash(payload)
        return payload


def build_reproducibility_report(
    *,
    report_id: str,
    provenance: StudyProvenance,
    frozen_assignment_run_ids: Iterable[str],
    runs: Iterable[RunReportRecord],
    artifacts: Iterable[ReportArtifact],
    sections: Iterable[ReportSection],
    derived_artifacts: Iterable[DerivedArtifact],
    resources: ResourceDisclosure,
    external_validity: ExternalValidityRecord,
    claims: Iterable[ArithmeticClaim],
    limitations: Iterable[str],
    generated_at_utc: str,
) -> ReproducibilityReport:
    run_records = tuple(runs)
    frozen_roster = tuple(frozen_assignment_run_ids)
    return ReproducibilityReport(
        report_id=report_id,
        provenance=provenance,
        frozen_assignment_run_ids=frozen_roster,
        assignment_roster_sha256=content_hash(list(frozen_roster)),
        runs=run_records,
        artifacts=tuple(artifacts),
        sections=tuple(sections),
        derived_artifacts=tuple(derived_artifacts),
        resources=resources,
        external_validity=external_validity,
        claims=tuple(claims),
        limitations=tuple(limitations),
        generated_at_utc=generated_at_utc,
    )


def write_report_exclusive(
    report: ReproducibilityReport, path: str | Path
) -> Path:
    destination = Path(path)
    create_json_exclusive(destination, report.to_dict())
    return destination
