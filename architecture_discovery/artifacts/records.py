"""Canonical, hash-linked records for the immutable per-run event ledger."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping


SCHEMA_NAME = "RunArtifactEvent"
SCHEMA_VERSION = "1.0"
GENESIS_EVENT_SHA256 = "0" * 64


class EventKind(StrEnum):
    PROPOSAL = "proposal"
    CANDIDATE = "candidate"
    FAILURE = "failure"
    REPAIR = "repair"
    TRAINING = "training"
    SEARCH_EVALUATION = "search_evaluation"
    QUALIFICATION_EVALUATION = "qualification_evaluation"
    CONFIRMATION_EVALUATION = "confirmation_evaluation"
    PARENT_SELECTION = "parent_selection"
    BUDGET = "budget"
    PROMOTION = "promotion"
    REVIEW = "review"
    MECHANISM_CLUSTER = "mechanism_cluster"
    RUN_STATUS = "run_status"
    RERUN_ATTEMPT = "rerun_attempt"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical artifact JSON does not permit non-finite numbers")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def require_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must not contain path separators")


def require_sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ArtifactContext:
    """Frozen identity and provenance shared by every event in one assigned run."""

    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    writer_component: str
    code_sha256: str
    config_sha256: str
    environment_sha256: str
    run_seed: int = 0
    assignment_sha256: str = GENESIS_EVENT_SHA256
    schema_name: str = "ArtifactContext"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        for name in (
            "study_id",
            "block_id",
            "run_id",
            "condition_id",
            "writer_component",
        ):
            require_identifier(getattr(self, name), name)
        for name in (
            "code_sha256",
            "config_sha256",
            "environment_sha256",
            "assignment_sha256",
        ):
            require_sha256(getattr(self, name), name)
        if not isinstance(self.run_seed, int) or isinstance(self.run_seed, bool):
            raise ValueError("run_seed must be an integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactContext":
        if payload.get("schema_name") != "ArtifactContext":
            raise ValueError("expected an ArtifactContext record")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported ArtifactContext schema version")
        values = dict(payload)
        values.pop("schema_name")
        values.pop("schema_version")
        return cls(**values)


@dataclass(frozen=True)
class EventRecord:
    """One immutable event, including its payload and predecessor commitment."""

    record_id: str
    sequence: int
    event_kind: EventKind
    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    created_at_utc: str
    writer_component: str
    code_sha256: str
    config_sha256: str
    environment_sha256: str
    previous_event_sha256: str
    payload_sha256: str
    payload: Mapping[str, Any]
    event_sha256: str
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        context: ArtifactContext,
        sequence: int,
        event_kind: EventKind | str,
        payload: Mapping[str, Any],
        previous_event_sha256: str,
        created_at_utc: str | None = None,
    ) -> "EventRecord":
        normalized_payload = _jsonable(payload)
        if not isinstance(normalized_payload, dict):
            raise TypeError("event payload must be a JSON object")
        payload_sha256 = content_sha256(normalized_payload)
        kind = EventKind(event_kind)
        identity = {
            "study_id": context.study_id,
            "block_id": context.block_id,
            "run_id": context.run_id,
            "condition_id": context.condition_id,
            "sequence": sequence,
            "event_kind": kind.value,
            "payload_sha256": payload_sha256,
            "previous_event_sha256": previous_event_sha256,
        }
        record_id = f"event-{content_sha256(identity)[:24]}"
        unsigned = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "record_id": record_id,
            "sequence": sequence,
            "event_kind": kind.value,
            "study_id": context.study_id,
            "block_id": context.block_id,
            "run_id": context.run_id,
            "condition_id": context.condition_id,
            "created_at_utc": created_at_utc or utc_now(),
            "writer_component": context.writer_component,
            "code_sha256": context.code_sha256,
            "config_sha256": context.config_sha256,
            "environment_sha256": context.environment_sha256,
            "previous_event_sha256": previous_event_sha256,
            "payload_sha256": payload_sha256,
            "payload": normalized_payload,
        }
        record = cls(
            record_id=record_id,
            sequence=sequence,
            event_kind=kind,
            study_id=context.study_id,
            block_id=context.block_id,
            run_id=context.run_id,
            condition_id=context.condition_id,
            created_at_utc=str(unsigned["created_at_utc"]),
            writer_component=context.writer_component,
            code_sha256=context.code_sha256,
            config_sha256=context.config_sha256,
            environment_sha256=context.environment_sha256,
            previous_event_sha256=previous_event_sha256,
            payload_sha256=payload_sha256,
            payload=MappingProxyType(normalized_payload),
            event_sha256=content_sha256(unsigned),
        )
        record.validate(context=context)
        return record

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "sequence": self.sequence,
            "event_kind": self.event_kind.value,
            "study_id": self.study_id,
            "block_id": self.block_id,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "created_at_utc": self.created_at_utc,
            "writer_component": self.writer_component,
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "environment_sha256": self.environment_sha256,
            "previous_event_sha256": self.previous_event_sha256,
            "payload_sha256": self.payload_sha256,
            "payload": _jsonable(self.payload),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "event_sha256": self.event_sha256}

    def validate(self, *, context: ArtifactContext | None = None) -> None:
        if self.schema_name != SCHEMA_NAME or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported run-event schema")
        require_identifier(self.record_id, "record_id")
        if self.sequence < 1:
            raise ValueError("event sequence must start at one")
        for name in (
            "study_id",
            "block_id",
            "run_id",
            "condition_id",
            "writer_component",
        ):
            require_identifier(getattr(self, name), name)
        for name in (
            "code_sha256",
            "config_sha256",
            "environment_sha256",
            "previous_event_sha256",
            "payload_sha256",
            "event_sha256",
        ):
            require_sha256(getattr(self, name), name)
        try:
            parsed_time = datetime.fromisoformat(
                self.created_at_utc.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError("created_at_utc is not a valid timestamp") from error
        if parsed_time.tzinfo is None or parsed_time.utcoffset() != UTC.utcoffset(parsed_time):
            raise ValueError("created_at_utc must be UTC")
        if content_sha256(self.payload) != self.payload_sha256:
            raise ValueError("event payload hash mismatch")
        if content_sha256(self.unsigned_dict()) != self.event_sha256:
            raise ValueError("event envelope hash mismatch")
        expected_id = f"event-{content_sha256({
            'study_id': self.study_id,
            'block_id': self.block_id,
            'run_id': self.run_id,
            'condition_id': self.condition_id,
            'sequence': self.sequence,
            'event_kind': self.event_kind.value,
            'payload_sha256': self.payload_sha256,
            'previous_event_sha256': self.previous_event_sha256,
        })[:24]}"
        if self.record_id != expected_id:
            raise ValueError("event record ID does not match its stable identity")
        if context is not None:
            expected = (
                context.study_id,
                context.block_id,
                context.run_id,
                context.condition_id,
                context.writer_component,
                context.code_sha256,
                context.config_sha256,
                context.environment_sha256,
            )
            actual = (
                self.study_id,
                self.block_id,
                self.run_id,
                self.condition_id,
                self.writer_component,
                self.code_sha256,
                self.config_sha256,
                self.environment_sha256,
            )
            if actual != expected:
                raise ValueError("event provenance differs from its frozen run context")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EventRecord":
        expected_fields = {
            "schema_name",
            "schema_version",
            "record_id",
            "sequence",
            "event_kind",
            "study_id",
            "block_id",
            "run_id",
            "condition_id",
            "created_at_utc",
            "writer_component",
            "code_sha256",
            "config_sha256",
            "environment_sha256",
            "previous_event_sha256",
            "payload_sha256",
            "payload",
            "event_sha256",
        }
        if set(payload) != expected_fields:
            raise ValueError("run event has unexpected or missing fields")
        event_payload = payload["payload"]
        if not isinstance(event_payload, Mapping):
            raise ValueError("run event payload must be an object")
        record = cls(
            schema_name=str(payload["schema_name"]),
            schema_version=str(payload["schema_version"]),
            record_id=str(payload["record_id"]),
            sequence=int(payload["sequence"]),
            event_kind=EventKind(str(payload["event_kind"])),
            study_id=str(payload["study_id"]),
            block_id=str(payload["block_id"]),
            run_id=str(payload["run_id"]),
            condition_id=str(payload["condition_id"]),
            created_at_utc=str(payload["created_at_utc"]),
            writer_component=str(payload["writer_component"]),
            code_sha256=str(payload["code_sha256"]),
            config_sha256=str(payload["config_sha256"]),
            environment_sha256=str(payload["environment_sha256"]),
            previous_event_sha256=str(payload["previous_event_sha256"]),
            payload_sha256=str(payload["payload_sha256"]),
            payload=MappingProxyType(dict(event_payload)),
            event_sha256=str(payload["event_sha256"]),
        )
        record.validate()
        return record
