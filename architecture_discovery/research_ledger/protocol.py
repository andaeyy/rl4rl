"""Exclusive-write research protocols with hash verification on every use."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from research_ledger.records import (
    HypothesisSpec,
    require_identifier,
    require_sha256,
    require_text,
)
from study.serialization import (
    content_hash,
    create_json_exclusive,
    read_json,
    require_bool,
)


@dataclass(frozen=True)
class ResearchProtocol:
    protocol_id: str
    study_id: str
    research_scope: str
    hypotheses: tuple[HypothesisSpec, ...]
    code_sha256: str
    config_sha256: str
    environment_sha256: str
    pi_decision_sha256: str
    scientific: bool
    adaptive_evidence_layer: str = field(default="layer_a", init=False)
    protocol_mutation_allowed: bool = field(default=False, init=False)
    schema_name: str = field(default="ResearchProtocol", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_bool(self.scientific, "scientific")
        require_identifier(self.protocol_id, "protocol_id")
        require_identifier(self.study_id, "study_id")
        require_text(self.research_scope, "research_scope")
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        if not self.hypotheses:
            raise ValueError("research protocol requires at least one hypothesis")
        identifiers = [item.hypothesis_id for item in self.hypotheses]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("hypothesis IDs must be unique in the protocol")
        for field_name in (
            "code_sha256",
            "config_sha256",
            "environment_sha256",
            "pi_decision_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if self.adaptive_evidence_layer != "layer_a":
            raise ValueError("only Layer A may drive adaptive research updates")
        if self.protocol_mutation_allowed:
            raise ValueError("a research protocol must be immutable")

    @property
    def protocol_hash(self) -> str:
        return content_hash(self.to_dict())

    def hypothesis(self, hypothesis_id: str) -> HypothesisSpec:
        for item in self.hypotheses:
            if item.hypothesis_id == hypothesis_id:
                return item
        raise ValueError(f"unknown frozen hypothesis {hypothesis_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "study_id": self.study_id,
            "research_scope": self.research_scope,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "code_sha256": self.code_sha256,
            "config_sha256": self.config_sha256,
            "environment_sha256": self.environment_sha256,
            "pi_decision_sha256": self.pi_decision_sha256,
            "scientific": self.scientific,
            "adaptive_evidence_layer": self.adaptive_evidence_layer,
            "protocol_mutation_allowed": self.protocol_mutation_allowed,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ResearchProtocol":
        if payload.get("schema_name") != "ResearchProtocol":
            raise ValueError("expected ResearchProtocol schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported ResearchProtocol schema version")
        if payload.get("adaptive_evidence_layer") != "layer_a":
            raise ValueError("protocol adaptive layer was mutated")
        if payload.get("protocol_mutation_allowed") is not False:
            raise ValueError("protocol mutation policy was altered")
        return cls(
            protocol_id=require_identifier(payload["protocol_id"], "protocol_id"),
            study_id=require_identifier(payload["study_id"], "study_id"),
            research_scope=require_text(payload["research_scope"], "research_scope"),
            hypotheses=tuple(
                HypothesisSpec.from_dict(item) for item in payload["hypotheses"]
            ),
            code_sha256=require_sha256(payload["code_sha256"], "code_sha256"),
            config_sha256=require_sha256(payload["config_sha256"], "config_sha256"),
            environment_sha256=require_sha256(
                payload["environment_sha256"], "environment_sha256"
            ),
            pi_decision_sha256=require_sha256(
                payload["pi_decision_sha256"], "pi_decision_sha256"
            ),
            scientific=require_bool(payload["scientific"], "scientific"),
        )


@dataclass(frozen=True)
class FrozenResearchProtocol:
    path: Path
    protocol: ResearchProtocol
    protocol_sha256: str
    frozen_at_utc: str
    schema_name: str = field(default="FrozenResearchProtocol", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", self.path.resolve())
        require_sha256(self.protocol_sha256, "protocol_sha256")
        if self.protocol.protocol_hash != self.protocol_sha256:
            raise ValueError("frozen protocol receipt hash mismatch")

    def verify(self) -> None:
        loaded = load_frozen_protocol(self.path)
        if loaded.protocol_sha256 != self.protocol_sha256:
            raise ValueError("frozen protocol changed after this receipt was issued")
        if loaded.protocol.to_dict() != self.protocol.to_dict():
            raise ValueError("frozen protocol payload changed after freeze")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "frozen_at_utc": self.frozen_at_utc,
            "protocol_sha256": self.protocol_sha256,
            "protocol": self.protocol.to_dict(),
        }


def freeze_protocol(
    protocol: ResearchProtocol, path: str | Path
) -> FrozenResearchProtocol:
    frozen = FrozenResearchProtocol(
        path=Path(path),
        protocol=protocol,
        protocol_sha256=protocol.protocol_hash,
        frozen_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    )
    create_json_exclusive(frozen.path, frozen.to_dict())
    return frozen


def load_frozen_protocol(path: str | Path) -> FrozenResearchProtocol:
    payload = read_json(path)
    if payload.get("schema_name") != "FrozenResearchProtocol":
        raise ValueError("expected FrozenResearchProtocol schema")
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported frozen-protocol schema")
    protocol = ResearchProtocol.from_dict(payload["protocol"])
    stored_hash = require_sha256(payload["protocol_sha256"], "protocol_sha256")
    if stored_hash != protocol.protocol_hash:
        raise ValueError("frozen research protocol hash mismatch; file may be mutated")
    frozen_at = require_text(payload["frozen_at_utc"], "frozen_at_utc")
    if not frozen_at:
        raise ValueError("frozen research protocol lacks a freeze timestamp")
    return FrozenResearchProtocol(
        path=Path(path),
        protocol=protocol,
        protocol_sha256=stored_hash,
        frozen_at_utc=frozen_at,
    )
