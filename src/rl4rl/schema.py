"""Canonical, dependency-free schema for discovery trajectory events."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Paradigm(StrEnum):
    OPENEVOLVE = "openevolve"
    AUTORESEARCH = "autoresearch"
    TTT_DISCOVER = "ttt-discover"
    HUMAN_FRONTIER = "human-frontier"
    OTHER = "other"


class EventStatus(StrEnum):
    PROPOSED = "proposed"
    EVALUATED = "evaluated"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    INVALID = "invalid"
    ERROR = "error"
    STOPPED = "stopped"


class BoundaryLabel(StrEnum):
    PRESERVING = "preserving"
    CHANGING = "changing"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class AnnotationSource(StrEnum):
    HUMAN = "human"
    HEURISTIC = "heuristic"
    LLM = "llm"
    IMPORTED = "imported"


@dataclass(slots=True)
class EditAnnotation:
    component: str
    operation: str
    before: str | None = None
    after: str | None = None
    ontology_family_before: str | None = None
    ontology_family_after: str | None = None
    boundary_label: BoundaryLabel = BoundaryLabel.AMBIGUOUS
    rationale: str | None = None
    confidence: float | None = None
    annotation_source: AnnotationSource = AnnotationSource.HUMAN
    annotator_id: str | None = None
    needs_review: bool = True

    def __post_init__(self) -> None:
        if not self.component.strip():
            raise ValueError("edit component must not be empty")
        if not self.operation.strip():
            raise ValueError("edit operation must not be empty")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("edit confidence must lie in [0, 1]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EditAnnotation:
        return cls(
            component=str(value["component"]),
            operation=str(value["operation"]),
            before=_optional_str(value.get("before")),
            after=_optional_str(value.get("after")),
            ontology_family_before=_optional_str(value.get("ontology_family_before")),
            ontology_family_after=_optional_str(value.get("ontology_family_after")),
            boundary_label=BoundaryLabel(
                value.get("boundary_label", BoundaryLabel.AMBIGUOUS)
            ),
            rationale=_optional_str(value.get("rationale")),
            confidence=_optional_float(value.get("confidence")),
            annotation_source=AnnotationSource(
                value.get("annotation_source", AnnotationSource.HUMAN)
            ),
            annotator_id=_optional_str(value.get("annotator_id")),
            needs_review=bool(value.get("needs_review", True)),
        )


@dataclass(slots=True)
class ArchitectureSnapshot:
    parameters: int | None = None
    accuracy: float | None = None
    qualifies: bool | None = None
    family: str | None = None
    fingerprint: str | None = None
    features: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parameters is not None and self.parameters < 0:
            raise ValueError("parameter count must be non-negative")
        if self.accuracy is not None and not 0 <= self.accuracy <= 1:
            raise ValueError("accuracy must lie in [0, 1]")

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> ArchitectureSnapshot:
        if value is None:
            return cls()
        parameters = value.get("parameters")
        return cls(
            parameters=None if parameters is None else int(parameters),
            accuracy=_optional_float(value.get("accuracy")),
            qualifies=_optional_bool(value.get("qualifies")),
            family=_optional_str(value.get("family")),
            fingerprint=_optional_str(value.get("fingerprint")),
            features=dict(value.get("features", {})),
        )

    def design_key(self) -> str | None:
        """Return a stable key for exact revisits of an architectural region."""
        if self.fingerprint:
            return self.fingerprint
        if not self.family and not self.features:
            return None
        feature_items = ",".join(
            f"{key}={self.features[key]!r}" for key in sorted(self.features)
        )
        return f"{self.family or 'unknown'}|{feature_items}"


@dataclass(slots=True)
class RewardHackEvidence:
    suspected: bool = False
    verified: bool = False
    category: str | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> RewardHackEvidence:
        if value is None:
            return cls()
        return cls(
            suspected=bool(value.get("suspected", False)),
            verified=bool(value.get("verified", False)),
            category=_optional_str(value.get("category")),
            notes=_optional_str(value.get("notes")),
        )


@dataclass(slots=True)
class TrajectoryEvent:
    event_id: str
    run_id: str
    paradigm: Paradigm
    step: int
    status: EventStatus
    parent_ids: list[str] = field(default_factory=list)
    timestamp: str | None = None
    accepted: bool | None = None
    valid: bool | None = None
    proposal: str | None = None
    notes: str | None = None
    architecture: ArchitectureSnapshot = field(default_factory=ArchitectureSnapshot)
    edits: list[EditAnnotation] = field(default_factory=list)
    reward_hack: RewardHackEvidence = field(default_factory=RewardHackEvidence)
    artifact_refs: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id must not be empty")
        if not self.run_id.strip():
            raise ValueError("run_id must not be empty")
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if self.event_id in self.parent_ids:
            raise ValueError("an event cannot be its own parent")
        if len(self.parent_ids) != len(set(self.parent_ids)):
            raise ValueError("parent_ids must be unique")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrajectoryEvent:
        required = {"event_id", "run_id", "paradigm", "step", "status"}
        missing = sorted(required - value.keys())
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        return cls(
            event_id=str(value["event_id"]),
            run_id=str(value["run_id"]),
            paradigm=Paradigm(value["paradigm"]),
            step=int(value["step"]),
            status=EventStatus(value["status"]),
            parent_ids=[str(parent) for parent in value.get("parent_ids", [])],
            timestamp=_optional_str(value.get("timestamp")),
            accepted=_optional_bool(value.get("accepted")),
            valid=_optional_bool(value.get("valid")),
            proposal=_optional_str(value.get("proposal")),
            notes=_optional_str(value.get("notes")),
            architecture=ArchitectureSnapshot.from_dict(value.get("architecture")),
            edits=[EditAnnotation.from_dict(edit) for edit in value.get("edits", [])],
            reward_hack=RewardHackEvidence.from_dict(value.get("reward_hack")),
            artifact_refs=[str(ref) for ref in value.get("artifact_refs", [])],
            provenance=dict(value.get("provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def crosses_boundary(self) -> bool:
        return any(edit.boundary_label == BoundaryLabel.CHANGING for edit in self.edits)


def _optional_str(value: Any) -> str | None:
    return None if value is None or value == "" else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None or value == "" else float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)
