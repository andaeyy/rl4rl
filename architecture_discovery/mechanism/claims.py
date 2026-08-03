"""Falsifiable mechanism claims and evidence-complete assessment rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

from mechanism.validation import (
    require_bool,
    require_identifier,
    require_sha256,
    require_text,
)
from study.serialization import content_hash


class EvidenceKind(StrEnum):
    ABLATION = "ablation"
    INTERVENTION = "intervention"
    RESCUE = "rescue"
    COUNTERFACTUAL = "counterfactual"
    SCALING = "scaling"
    REPLICATION = "replication"


class ClaimVerdict(StrEnum):
    PLANNED = "planned"
    SUPPORTED = "supported"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class DiscriminatingTest:
    test_id: str
    description: str
    prediction_if_claim_true: str
    prediction_if_alternative_true: str

    def __post_init__(self) -> None:
        require_identifier(self.test_id, "test_id")
        require_text(self.description, "description")
        require_text(self.prediction_if_claim_true, "prediction_if_claim_true")
        require_text(
            self.prediction_if_alternative_true,
            "prediction_if_alternative_true",
        )
        if self.prediction_if_claim_true == self.prediction_if_alternative_true:
            raise ValueError("a discriminating test must make different predictions")

    def to_dict(self) -> dict[str, str]:
        return {
            "test_id": self.test_id,
            "description": self.description,
            "prediction_if_claim_true": self.prediction_if_claim_true,
            "prediction_if_alternative_true": self.prediction_if_alternative_true,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscriminatingTest":
        return cls(
            test_id=require_identifier(payload["test_id"], "test_id"),
            description=require_text(payload["description"], "description"),
            prediction_if_claim_true=require_text(
                payload["prediction_if_claim_true"], "prediction_if_claim_true"
            ),
            prediction_if_alternative_true=require_text(
                payload["prediction_if_alternative_true"],
                "prediction_if_alternative_true",
            ),
        )


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    kind: EvidenceKind
    description: str

    def __post_init__(self) -> None:
        require_identifier(self.requirement_id, "requirement_id")
        require_text(self.description, "description")

    def to_dict(self) -> dict[str, str]:
        return {
            "requirement_id": self.requirement_id,
            "kind": self.kind.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceRequirement":
        return cls(
            requirement_id=require_identifier(
                payload["requirement_id"], "requirement_id"
            ),
            kind=EvidenceKind(payload["kind"]),
            description=require_text(payload["description"], "description"),
        )


@dataclass(frozen=True)
class MechanismClaim:
    claim_id: str
    candidate_snapshot_id: str
    candidate_snapshot_sha256: str
    proposed_mechanism: str
    causal_claim: str
    falsifiable_prediction: str
    nearest_alternative: str
    discriminating_tests: tuple[DiscriminatingTest, ...]
    required_evidence: tuple[EvidenceRequirement, ...]
    schema_name: str = field(default="MechanismClaim", init=False)
    schema_version: str = field(default="1.0", init=False)

    SCHEMA_NAME: ClassVar[str] = "MechanismClaim"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        require_identifier(self.claim_id, "claim_id")
        require_identifier(self.candidate_snapshot_id, "candidate_snapshot_id")
        require_sha256(self.candidate_snapshot_sha256, "candidate_snapshot_sha256")
        require_text(self.proposed_mechanism, "proposed_mechanism")
        require_text(self.causal_claim, "causal_claim")
        require_text(self.falsifiable_prediction, "falsifiable_prediction")
        require_text(self.nearest_alternative, "nearest_alternative")
        object.__setattr__(self, "discriminating_tests", tuple(self.discriminating_tests))
        object.__setattr__(self, "required_evidence", tuple(self.required_evidence))
        if not self.discriminating_tests:
            raise ValueError("a mechanism claim needs at least one discriminating test")
        if not self.required_evidence:
            raise ValueError("a mechanism claim needs explicit evidence requirements")
        test_ids = [test.test_id for test in self.discriminating_tests]
        requirement_ids = [item.requirement_id for item in self.required_evidence]
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("discriminating test IDs must be unique")
        if len(requirement_ids) != len(set(requirement_ids)):
            raise ValueError("evidence requirement IDs must be unique")

    @property
    def claim_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "candidate_snapshot_id": self.candidate_snapshot_id,
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "proposed_mechanism": self.proposed_mechanism,
            "causal_claim": self.causal_claim,
            "falsifiable_prediction": self.falsifiable_prediction,
            "nearest_alternative": self.nearest_alternative,
            "discriminating_tests": [item.to_dict() for item in self.discriminating_tests],
            "required_evidence": [item.to_dict() for item in self.required_evidence],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MechanismClaim":
        if payload.get("schema_name") != cls.SCHEMA_NAME:
            raise ValueError("not a MechanismClaim record")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported MechanismClaim schema version")
        return cls(
            claim_id=require_identifier(payload["claim_id"], "claim_id"),
            candidate_snapshot_id=require_identifier(
                payload["candidate_snapshot_id"], "candidate_snapshot_id"
            ),
            candidate_snapshot_sha256=require_sha256(
                payload["candidate_snapshot_sha256"],
                "candidate_snapshot_sha256",
            ),
            proposed_mechanism=require_text(
                payload["proposed_mechanism"], "proposed_mechanism"
            ),
            causal_claim=require_text(payload["causal_claim"], "causal_claim"),
            falsifiable_prediction=require_text(
                payload["falsifiable_prediction"], "falsifiable_prediction"
            ),
            nearest_alternative=require_text(
                payload["nearest_alternative"], "nearest_alternative"
            ),
            discriminating_tests=tuple(
                DiscriminatingTest.from_dict(item)
                for item in payload["discriminating_tests"]
            ),
            required_evidence=tuple(
                EvidenceRequirement.from_dict(item)
                for item in payload["required_evidence"]
            ),
        )


@dataclass(frozen=True)
class ClaimEvidence:
    evidence_id: str
    requirement_id: str
    discriminating_test_ids: tuple[str, ...]
    artifact_sha256: str
    supports_prediction: bool
    summary: str

    def __post_init__(self) -> None:
        require_bool(self.supports_prediction, "supports_prediction")
        require_identifier(self.evidence_id, "evidence_id")
        require_identifier(self.requirement_id, "requirement_id")
        object.__setattr__(
            self, "discriminating_test_ids", tuple(self.discriminating_test_ids)
        )
        if not self.discriminating_test_ids:
            raise ValueError("evidence must address a discriminating test")
        for test_id in self.discriminating_test_ids:
            require_identifier(test_id, "discriminating_test_id")
        require_sha256(self.artifact_sha256, "artifact_sha256")
        require_text(self.summary, "summary")


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    claim_hash: str
    verdict: ClaimVerdict
    evidence_ids: tuple[str, ...]
    missing_test_ids: tuple[str, ...]
    missing_requirement_ids: tuple[str, ...]
    unsupported_evidence_ids: tuple[str, ...]


def assess_claim(
    claim: MechanismClaim,
    evidence: tuple[ClaimEvidence, ...],
    *,
    requested_verdict: ClaimVerdict,
) -> ClaimAssessment:
    """Assess a claim, refusing a supported verdict with incomplete evidence."""

    evidence = tuple(evidence)
    known_tests = {item.test_id for item in claim.discriminating_tests}
    known_requirements = {item.requirement_id for item in claim.required_evidence}
    evidence_ids: set[str] = set()
    covered_tests: set[str] = set()
    covered_requirements: set[str] = set()
    unsupported: set[str] = set()
    for item in evidence:
        if item.evidence_id in evidence_ids:
            raise ValueError(f"duplicate evidence ID {item.evidence_id!r}")
        evidence_ids.add(item.evidence_id)
        if item.requirement_id not in known_requirements:
            raise ValueError(f"unknown evidence requirement {item.requirement_id!r}")
        unknown_tests = set(item.discriminating_test_ids) - known_tests
        if unknown_tests:
            raise ValueError(f"evidence refers to unknown tests {sorted(unknown_tests)}")
        if item.supports_prediction:
            covered_requirements.add(item.requirement_id)
            covered_tests.update(item.discriminating_test_ids)
        else:
            unsupported.add(item.evidence_id)

    missing_tests = tuple(sorted(known_tests - covered_tests))
    missing_requirements = tuple(sorted(known_requirements - covered_requirements))
    if requested_verdict is ClaimVerdict.SUPPORTED and (
        missing_tests or missing_requirements or unsupported
    ):
        raise ValueError(
            "claim cannot be supported without all declared discriminating evidence"
        )
    if requested_verdict is ClaimVerdict.PLANNED and evidence:
        raise ValueError("a planned assessment cannot already contain outcome evidence")
    return ClaimAssessment(
        claim_id=claim.claim_id,
        claim_hash=claim.claim_hash,
        verdict=requested_verdict,
        evidence_ids=tuple(sorted(evidence_ids)),
        missing_test_ids=missing_tests,
        missing_requirement_ids=missing_requirements,
        unsupported_evidence_ids=tuple(sorted(unsupported)),
    )
