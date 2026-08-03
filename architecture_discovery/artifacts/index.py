"""Reconstructable artifact index derived only from immutable raw events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Mapping

from artifacts.records import (
    ArtifactContext,
    EventKind,
    EventRecord,
    content_sha256,
    require_identifier,
    require_sha256,
)


INDEX_CATEGORIES: tuple[str, ...] = (
    "proposals",
    "candidates",
    "failures",
    "repairs",
    "training",
    "evaluations",
    "parent_selections",
    "budgets",
    "promotions",
    "reviews",
    "mechanism_clusters",
    "run_status",
    "rerun_attempts",
)


_CATEGORY_BY_KIND = {
    EventKind.PROPOSAL: "proposals",
    EventKind.CANDIDATE: "candidates",
    EventKind.FAILURE: "failures",
    EventKind.REPAIR: "repairs",
    EventKind.TRAINING: "training",
    EventKind.SEARCH_EVALUATION: "evaluations",
    EventKind.QUALIFICATION_EVALUATION: "evaluations",
    EventKind.CONFIRMATION_EVALUATION: "evaluations",
    EventKind.PARENT_SELECTION: "parent_selections",
    EventKind.BUDGET: "budgets",
    EventKind.PROMOTION: "promotions",
    EventKind.REVIEW: "reviews",
    EventKind.MECHANISM_CLUSTER: "mechanism_clusters",
    EventKind.RUN_STATUS: "run_status",
    EventKind.RERUN_ATTEMPT: "rerun_attempts",
}


@dataclass(frozen=True)
class ArtifactIndexEntry:
    record_id: str
    sequence: int
    event_kind: str
    event_sha256: str
    object_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.record_id, "record_id")
        require_identifier(self.event_kind, "event_kind")
        require_sha256(self.event_sha256, "event_sha256")
        if self.sequence < 1:
            raise ValueError("index sequence must be positive")
        if tuple(sorted(set(self.object_sha256s))) != self.object_sha256s:
            raise ValueError("object hashes must be sorted and unique")
        for digest in self.object_sha256s:
            require_sha256(digest, "object_sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "sequence": self.sequence,
            "event_kind": self.event_kind,
            "event_sha256": self.event_sha256,
            "object_sha256s": list(self.object_sha256s),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactIndexEntry":
        return cls(
            record_id=str(payload["record_id"]),
            sequence=int(payload["sequence"]),
            event_kind=str(payload["event_kind"]),
            event_sha256=str(payload["event_sha256"]),
            object_sha256s=tuple(str(item) for item in payload["object_sha256s"]),
        )


@dataclass(frozen=True)
class ArtifactIndex:
    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    last_event_sha256: str
    event_count: int
    categories: Mapping[str, tuple[ArtifactIndexEntry, ...]]
    schema_name: str = "ArtifactIndex"
    schema_version: str = "1.0"

    REQUIRED_CATEGORIES: ClassVar[tuple[str, ...]] = INDEX_CATEGORIES

    def __post_init__(self) -> None:
        if self.schema_name != "ArtifactIndex" or self.schema_version != "1.0":
            raise ValueError("unsupported ArtifactIndex schema")
        for name in ("study_id", "block_id", "run_id", "condition_id"):
            require_identifier(getattr(self, name), name)
        if set(self.categories) != set(INDEX_CATEGORIES):
            raise ValueError("artifact index must contain every fixed category")
        require_sha256(self.last_event_sha256, "last_event_sha256")
        flattened = [
            entry for category in INDEX_CATEGORIES for entry in self.categories[category]
        ]
        if len(flattened) != self.event_count:
            raise ValueError("artifact index event count does not reconstruct")
        record_ids = [entry.record_id for entry in flattened]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("artifact index contains duplicate events")

    @property
    def index_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "block_id": self.block_id,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "last_event_sha256": self.last_event_sha256,
            "event_count": self.event_count,
            "categories": {
                category: [entry.to_dict() for entry in self.categories[category]]
                for category in INDEX_CATEGORIES
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactIndex":
        if payload.get("schema_name") != "ArtifactIndex":
            raise ValueError("expected an ArtifactIndex record")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported ArtifactIndex schema version")
        raw_categories = payload.get("categories")
        if not isinstance(raw_categories, Mapping):
            raise ValueError("artifact index categories must be an object")
        if set(raw_categories) != set(INDEX_CATEGORIES):
            raise ValueError("artifact index has missing or unexpected categories")
        return cls(
            study_id=str(payload["study_id"]),
            block_id=str(payload["block_id"]),
            run_id=str(payload["run_id"]),
            condition_id=str(payload["condition_id"]),
            last_event_sha256=str(payload["last_event_sha256"]),
            event_count=int(payload["event_count"]),
            categories={
                category: tuple(
                    ArtifactIndexEntry.from_dict(item)
                    for item in raw_categories[category]
                )
                for category in INDEX_CATEGORIES
            },
        )

    @classmethod
    def from_events(
        cls, context: ArtifactContext, events: Iterable[EventRecord]
    ) -> "ArtifactIndex":
        ordered = tuple(events)
        categories: dict[str, list[ArtifactIndexEntry]] = {
            category: [] for category in INDEX_CATEGORIES
        }
        for event in ordered:
            raw_objects = event.payload.get("object_sha256s", ())
            if not isinstance(raw_objects, (list, tuple)):
                raise ValueError("object_sha256s must be a sequence")
            object_hashes = tuple(sorted(set(str(item) for item in raw_objects)))
            categories[_CATEGORY_BY_KIND[event.event_kind]].append(
                ArtifactIndexEntry(
                    record_id=event.record_id,
                    sequence=event.sequence,
                    event_kind=event.event_kind.value,
                    event_sha256=event.event_sha256,
                    object_sha256s=object_hashes,
                )
            )
        last_hash = ordered[-1].event_sha256 if ordered else "0" * 64
        return cls(
            study_id=context.study_id,
            block_id=context.block_id,
            run_id=context.run_id,
            condition_id=context.condition_id,
            last_event_sha256=last_hash,
            event_count=len(ordered),
            categories={key: tuple(value) for key, value in categories.items()},
        )
