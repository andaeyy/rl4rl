"""Typed records used by the reproducibility-report builder."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from research_ledger.records import require_identifier, require_sha256, require_text
from study.serialization import content_hash


class ReportArtifactKind(StrEnum):
    ASSIGNMENT = "assignment"
    RUN_RECORD = "run_record"
    PROPOSAL = "proposal"
    CANDIDATE = "candidate"
    TRAINING = "training"
    EVALUATION = "evaluation"
    PARENT_SELECTION = "parent_selection"
    REPAIR = "repair"
    PROMOTION = "promotion"
    FAILURE = "failure"
    RERUN = "rerun"
    BUDGET = "budget"
    CLUSTER = "cluster"
    RAW_REVIEW = "raw_review"
    ADJUDICATED_REVIEW = "adjudicated_review"
    MECHANISM_DOSSIER = "mechanism_dossier"
    ANALYSIS = "analysis"
    MODEL_CARD = "model_card"
    DATA_CARD = "data_card"
    RESEARCH_LEDGER = "research_ledger"


RUN_DETAIL_KINDS = frozenset(
    {
        ReportArtifactKind.RUN_RECORD,
        ReportArtifactKind.PROPOSAL,
        ReportArtifactKind.CANDIDATE,
        ReportArtifactKind.TRAINING,
        ReportArtifactKind.EVALUATION,
        ReportArtifactKind.PARENT_SELECTION,
        ReportArtifactKind.REPAIR,
        ReportArtifactKind.PROMOTION,
    }
)


class RunReportStatus(StrEnum):
    COMPLETED = "completed"
    SCIENTIFIC_FAILURE = "scientific_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class SectionName(StrEnum):
    ASSIGNMENTS = "assignments"
    RUN_RECORDS = "run_records"
    FAILURES = "failures"
    RERUNS = "reruns"
    BUDGETS = "budgets"
    CLUSTERS = "clusters"
    RAW_REVIEWS = "raw_reviews"
    ADJUDICATED_REVIEWS = "adjudicated_reviews"
    MECHANISM_DOSSIERS = "mechanism_dossiers"
    ANALYSES = "analyses"
    MODEL_CARDS = "model_cards"
    DATA_CARDS = "data_cards"
    RESEARCH_LEDGER = "research_ledger"


SECTION_KINDS: dict[SectionName, frozenset[ReportArtifactKind]] = {
    SectionName.ASSIGNMENTS: frozenset({ReportArtifactKind.ASSIGNMENT}),
    SectionName.RUN_RECORDS: RUN_DETAIL_KINDS,
    SectionName.FAILURES: frozenset({ReportArtifactKind.FAILURE}),
    SectionName.RERUNS: frozenset({ReportArtifactKind.RERUN}),
    SectionName.BUDGETS: frozenset({ReportArtifactKind.BUDGET}),
    SectionName.CLUSTERS: frozenset({ReportArtifactKind.CLUSTER}),
    SectionName.RAW_REVIEWS: frozenset({ReportArtifactKind.RAW_REVIEW}),
    SectionName.ADJUDICATED_REVIEWS: frozenset(
        {ReportArtifactKind.ADJUDICATED_REVIEW}
    ),
    SectionName.MECHANISM_DOSSIERS: frozenset(
        {ReportArtifactKind.MECHANISM_DOSSIER}
    ),
    SectionName.ANALYSES: frozenset({ReportArtifactKind.ANALYSIS}),
    SectionName.MODEL_CARDS: frozenset({ReportArtifactKind.MODEL_CARD}),
    SectionName.DATA_CARDS: frozenset({ReportArtifactKind.DATA_CARD}),
    SectionName.RESEARCH_LEDGER: frozenset({ReportArtifactKind.RESEARCH_LEDGER}),
}


class SectionStatus(StrEnum):
    COMPLETE = "complete"
    BLOCKED = "blocked"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"


class MeasurementStatus(StrEnum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class DerivedArtifactKind(StrEnum):
    TABLE = "table"
    FIGURE = "figure"


class ExternalValidityStatus(StrEnum):
    ARITHMETIC_ONLY = "arithmetic_only"
    ARITHMETIC_SCALING_TESTED = "arithmetic_scaling_tested"
    SECOND_TASK_TESTED = "second_task_tested"
    MULTIDOMAIN_REPLICATED = "multidomain_replicated"


class ArithmeticClaimKind(StrEnum):
    SEARCH_YIELD = "search_yield"
    MECHANISM = "mechanism"
    REPLICATION = "replication"


_SENSITIVE_KEYS = {
    "api_key",
    "authorization_header",
    "credential",
    "credentials",
    "discovery_api_key",
    "secret",
}
_OVERCLAIM_PATTERNS = (
    re.compile(r"\ball tasks\b", re.IGNORECASE),
    re.compile(r"\bgeneral intelligence\b", re.IGNORECASE),
    re.compile(r"\blanguage models generally\b", re.IGNORECASE),
    re.compile(r"\bstate[- ]of[- ]the[- ]art\b", re.IGNORECASE),
    re.compile(r"\buniversally\b", re.IGNORECASE),
)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items()):
            normalized = str(key)
            if normalized.lower() in _SENSITIVE_KEYS:
                raise ValueError(f"report artifact contains sensitive field {normalized!r}")
            result[normalized] = _freeze(item)
        return MappingProxyType(result)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("report artifacts cannot contain NaN or infinity")
        return value
    raise TypeError(f"unsupported report value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ReportArtifact:
    artifact_id: str
    kind: ReportArtifactKind
    record_schema_name: str
    payload: Mapping[str, Any]
    content_sha256: str
    schema_name: str = field(default="ReproducibilityArtifact", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_identifier(self.artifact_id, "artifact_id")
        require_identifier(self.record_schema_name, "record_schema_name")
        frozen = _freeze(self.payload)
        object.__setattr__(self, "payload", frozen)
        require_sha256(self.content_sha256, "content_sha256")
        if content_hash(_thaw(frozen)) != self.content_sha256:
            raise ValueError("report artifact content hash mismatch")

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        kind: ReportArtifactKind,
        payload: Mapping[str, Any],
        record_schema_name: str | None = None,
    ) -> "ReportArtifact":
        material = _thaw(_freeze(payload))
        inferred = record_schema_name or str(material.get("schema_name", kind.value))
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            record_schema_name=inferred,
            payload=material,
            content_sha256=content_hash(material),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "record_schema_name": self.record_schema_name,
            "content_sha256": self.content_sha256,
            "payload": _thaw(self.payload),
        }


@dataclass(frozen=True)
class RunReportRecord:
    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    run_seed: int
    assignment_sha256: str
    assignment_artifact_id: str
    terminal_status: RunReportStatus
    run_artifact_ids: tuple[str, ...]
    failure_artifact_ids: tuple[str, ...]
    rerun_artifact_ids: tuple[str, ...]
    budget_artifact_ids: tuple[str, ...]
    schema_name: str = field(default="RunReproducibilityRecord", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "study_id",
            "block_id",
            "run_id",
            "condition_id",
            "assignment_artifact_id",
        ):
            require_identifier(getattr(self, field_name), field_name)
        if not isinstance(self.run_seed, int) or isinstance(self.run_seed, bool):
            raise ValueError("run_seed must be an integer")
        require_sha256(self.assignment_sha256, "assignment_sha256")
        for field_name in (
            "run_artifact_ids",
            "failure_artifact_ids",
            "rerun_artifact_ids",
            "budget_artifact_ids",
        ):
            values = tuple(sorted(set(getattr(self, field_name))))
            for value in values:
                require_identifier(value, field_name)
            object.__setattr__(self, field_name, values)
        if not self.run_artifact_ids or not self.budget_artifact_ids:
            raise ValueError("every run needs run records and budget records")
        if self.terminal_status is not RunReportStatus.COMPLETED and not self.failure_artifact_ids:
            raise ValueError("failed terminal runs require retained failure records")
        if self.rerun_artifact_ids and not self.failure_artifact_ids:
            raise ValueError("reruns require a linked retained failure")
        if (
            self.terminal_status is RunReportStatus.COMPLETED
            and self.failure_artifact_ids
            and not self.rerun_artifact_ids
        ):
            raise ValueError("a completed run with past failures needs rerun provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "block_id": self.block_id,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "run_seed": self.run_seed,
            "assignment_sha256": self.assignment_sha256,
            "assignment_artifact_id": self.assignment_artifact_id,
            "terminal_status": self.terminal_status.value,
            "run_artifact_ids": list(self.run_artifact_ids),
            "failure_artifact_ids": list(self.failure_artifact_ids),
            "rerun_artifact_ids": list(self.rerun_artifact_ids),
            "budget_artifact_ids": list(self.budget_artifact_ids),
        }


@dataclass(frozen=True)
class ReportSection:
    name: SectionName
    status: SectionStatus
    artifact_ids: tuple[str, ...]
    limitation: str | None = None

    def __post_init__(self) -> None:
        values = tuple(sorted(set(self.artifact_ids)))
        for value in values:
            require_identifier(value, "section artifact_id")
        object.__setattr__(self, "artifact_ids", values)
        if self.status is SectionStatus.COMPLETE and not values:
            raise ValueError("complete report sections cannot be empty")
        if self.status is not SectionStatus.COMPLETE:
            if self.limitation is None:
                raise ValueError("non-complete sections require an explicit limitation")
            require_text(self.limitation, "section limitation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "status": self.status.value,
            "artifact_ids": list(self.artifact_ids),
            "limitation": self.limitation,
        }


@dataclass(frozen=True)
class QuantityDisclosure:
    name: str
    unit: str
    status: MeasurementStatus
    value: float | None
    method: str

    def __post_init__(self) -> None:
        require_identifier(self.name, "quantity name")
        require_identifier(self.unit, "quantity unit")
        require_text(self.method, "quantity method")
        if self.status in {MeasurementStatus.MEASURED, MeasurementStatus.ESTIMATED}:
            if self.value is None or not math.isfinite(self.value) or self.value < 0:
                raise ValueError("measured or estimated quantities need a nonnegative value")
        elif self.value is not None:
            raise ValueError("unknown or inapplicable quantities cannot claim a value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "status": self.status.value,
            "value": self.value,
            "method": self.method,
        }


@dataclass(frozen=True)
class ResourceDisclosure:
    quantities: tuple[QuantityDisclosure, ...]
    prompt_tokens: int
    completion_tokens: int
    provider_usage_complete: bool
    notes: tuple[str, ...]
    parameter_count_role: str = field(default="descriptive_metadata_only", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantities", tuple(self.quantities))
        names = [item.name for item in self.quantities]
        if len(set(names)) != len(names):
            raise ValueError("resource quantity names must be unique")
        required = {
            "accelerator_compute",
            "cpu_compute",
            "monetary_cost",
            "energy",
        }
        if set(names) != required:
            raise ValueError(f"resource disclosure must contain {sorted(required)}")
        for value in (self.prompt_tokens, self.completion_tokens):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError("token counts must be nonnegative integers")
        notes = tuple(self.notes)
        if not notes:
            raise ValueError("resource disclosure requires methodological notes")
        for note in notes:
            require_text(note, "resource note")
        object.__setattr__(self, "notes", notes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantities": [item.to_dict() for item in self.quantities],
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "provider_usage_complete": self.provider_usage_complete,
            "notes": list(self.notes),
            "parameter_count_role": self.parameter_count_role,
        }


@dataclass(frozen=True)
class StudyProvenance:
    study_id: str
    study_spec_sha256: str
    config_sha256: str
    code_sha256: str
    environment_sha256: str
    randomization_sha256: str
    research_protocol_sha256: str
    analysis_plan_sha256: str
    artifact_index_sha256: str
    reference_corpus_sha256: str
    generator_configuration_sha256: str

    def __post_init__(self) -> None:
        require_identifier(self.study_id, "study_id")
        for field_name in (
            "study_spec_sha256",
            "config_sha256",
            "code_sha256",
            "environment_sha256",
            "randomization_sha256",
            "research_protocol_sha256",
            "analysis_plan_sha256",
            "artifact_index_sha256",
            "reference_corpus_sha256",
            "generator_configuration_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, str]:
        return {
            "study_id": self.study_id,
            "study_spec_sha256": self.study_spec_sha256,
            "config_sha256": self.config_sha256,
            "code_sha256": self.code_sha256,
            "environment_sha256": self.environment_sha256,
            "randomization_sha256": self.randomization_sha256,
            "research_protocol_sha256": self.research_protocol_sha256,
            "analysis_plan_sha256": self.analysis_plan_sha256,
            "artifact_index_sha256": self.artifact_index_sha256,
            "reference_corpus_sha256": self.reference_corpus_sha256,
            "generator_configuration_sha256": self.generator_configuration_sha256,
        }


@dataclass(frozen=True)
class SourceArtifactReference:
    artifact_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        require_identifier(self.artifact_id, "source artifact_id")
        require_sha256(self.content_sha256, "source content_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class DerivedArtifact:
    artifact_id: str
    kind: DerivedArtifactKind
    title: str
    relative_path: str
    sources: tuple[SourceArtifactReference, ...]
    transformation_id: str
    code_sha256: str
    config_sha256: str
    content_sha256: str
    generated_at_utc: str
    provenance_sha256: str
    schema_name: str = field(default="DerivedReportArtifact", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_identifier(self.artifact_id, "derived artifact_id")
        require_identifier(self.transformation_id, "transformation_id")
        require_text(self.title, "derived artifact title")
        if (
            not self.relative_path
            or self.relative_path.startswith("/")
            or ".." in self.relative_path.split("/")
        ):
            raise ValueError("derived artifact path must be safe and relative")
        sources = tuple(sorted(self.sources, key=lambda item: item.artifact_id))
        if not sources or len({item.artifact_id for item in sources}) != len(sources):
            raise ValueError("derived artifacts need unique source records")
        object.__setattr__(self, "sources", sources)
        for field_name in (
            "code_sha256",
            "config_sha256",
            "content_sha256",
            "provenance_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if content_hash(self.provenance_payload()) != self.provenance_sha256:
            raise ValueError("derived artifact provenance hash mismatch")

    def provenance_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "kind": self.kind.value,
            "title": self.title,
            "relative_path": self.relative_path,
            "sources": [item.to_dict() for item in self.sources],
            "transformation_id": self.transformation_id,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "content_sha256": self.content_sha256,
            "generated_at_utc": self.generated_at_utc,
        }

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        kind: DerivedArtifactKind,
        title: str,
        relative_path: str,
        sources: tuple[SourceArtifactReference, ...],
        transformation_id: str,
        code_sha256: str,
        config_sha256: str,
        content_sha256: str,
        generated_at_utc: str,
    ) -> "DerivedArtifact":
        payload = {
            "artifact_id": artifact_id,
            "kind": kind.value,
            "title": title,
            "relative_path": relative_path,
            "sources": [
                item.to_dict() for item in sorted(sources, key=lambda value: value.artifact_id)
            ],
            "transformation_id": transformation_id,
            "code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "content_sha256": content_sha256,
            "generated_at_utc": generated_at_utc,
        }
        return cls(
            artifact_id=artifact_id,
            kind=kind,
            title=title,
            relative_path=relative_path,
            sources=sources,
            transformation_id=transformation_id,
            code_sha256=code_sha256,
            config_sha256=config_sha256,
            content_sha256=content_sha256,
            generated_at_utc=generated_at_utc,
            provenance_sha256=content_hash(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            **self.provenance_payload(),
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True)
class ExternalValidityRecord:
    status: ExternalValidityStatus
    primary_task_id: str
    tested_task_ids: tuple[str, ...]
    second_task_evidence_ids: tuple[str, ...]
    scaling_evidence_ids: tuple[str, ...]
    limitation: str

    def __post_init__(self) -> None:
        require_identifier(self.primary_task_id, "primary_task_id")
        tasks = tuple(sorted(set(self.tested_task_ids)))
        if self.primary_task_id not in tasks:
            raise ValueError("external-validity record must include the primary task")
        for task_id in tasks:
            require_identifier(task_id, "tested_task_id")
        object.__setattr__(self, "tested_task_ids", tasks)
        for field_name in ("second_task_evidence_ids", "scaling_evidence_ids"):
            values = tuple(sorted(set(getattr(self, field_name))))
            for value in values:
                require_identifier(value, field_name)
            object.__setattr__(self, field_name, values)
        require_text(self.limitation, "external-validity limitation")
        if self.status is ExternalValidityStatus.ARITHMETIC_ONLY:
            if self.second_task_evidence_ids or self.scaling_evidence_ids:
                raise ValueError("arithmetic-only status cannot claim broader evidence")
        elif self.status is ExternalValidityStatus.ARITHMETIC_SCALING_TESTED:
            if not self.scaling_evidence_ids or self.second_task_evidence_ids:
                raise ValueError("arithmetic scaling status needs scaling evidence only")
        else:
            if not self.second_task_evidence_ids:
                raise ValueError("broader external validity requires second-task evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "primary_task_id": self.primary_task_id,
            "tested_task_ids": list(self.tested_task_ids),
            "second_task_evidence_ids": list(self.second_task_evidence_ids),
            "scaling_evidence_ids": list(self.scaling_evidence_ids),
            "limitation": self.limitation,
        }


@dataclass(frozen=True)
class ArithmeticClaim:
    claim_id: str
    kind: ArithmeticClaimKind
    result_summary: str
    evidence_artifact_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    claim_scope: str = field(default="autoregressive_arithmetic_only", init=False)

    def __post_init__(self) -> None:
        require_identifier(self.claim_id, "claim_id")
        require_text(self.result_summary, "result_summary")
        for pattern in _OVERCLAIM_PATTERNS:
            if pattern.search(self.result_summary):
                raise ValueError("claim summary exceeds the arithmetic-only evidence scope")
        evidence = tuple(sorted(set(self.evidence_artifact_ids)))
        if not evidence:
            raise ValueError("claims require linked report evidence")
        for artifact_id in evidence:
            require_identifier(artifact_id, "claim evidence_artifact_id")
        object.__setattr__(self, "evidence_artifact_ids", evidence)
        limitations = tuple(self.limitations)
        if not limitations:
            raise ValueError("arithmetic claims require explicit limitations")
        for limitation in limitations:
            require_text(limitation, "claim limitation")
        object.__setattr__(self, "limitations", limitations)

    @property
    def rendered_text(self) -> str:
        return (
            "Within the preregistered autoregressive integer-addition study, "
            f"{self.result_summary} This claim is limited to the tested arithmetic "
            "task, data generators, training budgets, and evaluation distributions; "
            "it does not establish a general language-model improvement."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind.value,
            "claim_scope": self.claim_scope,
            "result_summary": self.result_summary,
            "rendered_text": self.rendered_text,
            "evidence_artifact_ids": list(self.evidence_artifact_ids),
            "limitations": list(self.limitations),
        }
