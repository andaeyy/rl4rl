"""Typed, immutable records for hypotheses and their evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from common.evaluation_profiles import EvaluationLayer
from study.serialization import content_hash


SCHEMA_VERSION = "1.0"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_EXPECTED_EVALUATION_SCHEMA = {
    EvaluationLayer.SEARCH: "search_evaluation",
    EvaluationLayer.QUALIFICATION: "qualification_evaluation",
    EvaluationLayer.CONFIRMATION: "confirmation_evaluation",
}


def require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field_name} must be a stable identifier")
    return value


def require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be non-empty, trimmed text")
    return value


def require_sha256(value: str, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


class ConfidenceLevel(StrEnum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class HypothesisStatus(StrEnum):
    PLANNED = "planned"
    ACTIVE = "active"
    WEAKENED = "weakened"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"
    SUPPORTED = "supported"


class EvidenceDirection(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class EvidenceUse(StrEnum):
    ADAPTIVE_SEARCH = "adaptive_search"
    POSTSEARCH_ASSESSMENT = "postsearch_assessment"


class LedgerPhase(StrEnum):
    SEARCH = "search"
    POSTSEARCH = "postsearch"
    SEALED = "sealed"


@dataclass(frozen=True)
class PredictionSpec:
    prediction_id: str
    statement: str
    falsification_condition: str

    def __post_init__(self) -> None:
        require_identifier(self.prediction_id, "prediction_id")
        require_text(self.statement, "prediction statement")
        require_text(self.falsification_condition, "falsification_condition")

    def to_dict(self) -> dict[str, str]:
        return {
            "prediction_id": self.prediction_id,
            "statement": self.statement,
            "falsification_condition": self.falsification_condition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PredictionSpec":
        return cls(
            prediction_id=require_identifier(
                payload["prediction_id"], "prediction_id"
            ),
            statement=require_text(payload["statement"], "prediction statement"),
            falsification_condition=require_text(
                payload["falsification_condition"], "falsification_condition"
            ),
        )


@dataclass(frozen=True)
class DiscriminatingTestSpec:
    test_id: str
    description: str
    prediction_if_claim_true: str
    prediction_if_alternative_true: str

    def __post_init__(self) -> None:
        require_identifier(self.test_id, "test_id")
        require_text(self.description, "test description")
        require_text(self.prediction_if_claim_true, "prediction_if_claim_true")
        require_text(
            self.prediction_if_alternative_true,
            "prediction_if_alternative_true",
        )
        if self.prediction_if_claim_true == self.prediction_if_alternative_true:
            raise ValueError("a discriminating test must separate the two explanations")

    def to_dict(self) -> dict[str, str]:
        return {
            "test_id": self.test_id,
            "description": self.description,
            "prediction_if_claim_true": self.prediction_if_claim_true,
            "prediction_if_alternative_true": self.prediction_if_alternative_true,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DiscriminatingTestSpec":
        return cls(
            test_id=require_identifier(payload["test_id"], "test_id"),
            description=require_text(payload["description"], "test description"),
            prediction_if_claim_true=require_text(
                payload["prediction_if_claim_true"], "prediction_if_claim_true"
            ),
            prediction_if_alternative_true=require_text(
                payload["prediction_if_alternative_true"],
                "prediction_if_alternative_true",
            ),
        )


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    hypothesis: str
    causal_claim: str
    predictions: tuple[PredictionSpec, ...]
    nearest_alternative: str
    discriminating_tests: tuple[DiscriminatingTestSpec, ...]
    initial_confidence: ConfidenceLevel
    initial_status: HypothesisStatus = HypothesisStatus.PLANNED

    def __post_init__(self) -> None:
        require_identifier(self.hypothesis_id, "hypothesis_id")
        require_text(self.hypothesis, "hypothesis")
        require_text(self.causal_claim, "causal_claim")
        require_text(self.nearest_alternative, "nearest_alternative")
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "discriminating_tests", tuple(self.discriminating_tests))
        if not self.predictions or not self.discriminating_tests:
            raise ValueError("hypotheses require predictions and discriminating tests")
        prediction_ids = [item.prediction_id for item in self.predictions]
        test_ids = [item.test_id for item in self.discriminating_tests]
        if len(set(prediction_ids)) != len(prediction_ids):
            raise ValueError("prediction IDs must be unique within a hypothesis")
        if len(set(test_ids)) != len(test_ids):
            raise ValueError("discriminating test IDs must be unique")
        if self.initial_status not in {HypothesisStatus.PLANNED, HypothesisStatus.ACTIVE}:
            raise ValueError("a frozen protocol can begin only planned or active")

    @property
    def spec_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis": self.hypothesis,
            "causal_claim": self.causal_claim,
            "predictions": [item.to_dict() for item in self.predictions],
            "nearest_alternative": self.nearest_alternative,
            "discriminating_tests": [
                item.to_dict() for item in self.discriminating_tests
            ],
            "initial_confidence": self.initial_confidence.value,
            "initial_status": self.initial_status.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HypothesisSpec":
        return cls(
            hypothesis_id=require_identifier(
                payload["hypothesis_id"], "hypothesis_id"
            ),
            hypothesis=require_text(payload["hypothesis"], "hypothesis"),
            causal_claim=require_text(payload["causal_claim"], "causal_claim"),
            predictions=tuple(
                PredictionSpec.from_dict(item) for item in payload["predictions"]
            ),
            nearest_alternative=require_text(
                payload["nearest_alternative"], "nearest_alternative"
            ),
            discriminating_tests=tuple(
                DiscriminatingTestSpec.from_dict(item)
                for item in payload["discriminating_tests"]
            ),
            initial_confidence=ConfidenceLevel(payload["initial_confidence"]),
            initial_status=HypothesisStatus(payload["initial_status"]),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    hypothesis_id: str
    intended_use: EvidenceUse
    source_layer: EvaluationLayer
    source_schema_name: str
    source_record_id: str
    source_record_sha256: str
    direction: EvidenceDirection
    prediction_ids: tuple[str, ...]
    discriminating_test_ids: tuple[str, ...]
    summary: str
    observed_at_utc: str
    schema_name: str = field(default="ResearchEvidenceRecord", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        require_identifier(self.evidence_id, "evidence_id")
        require_identifier(self.hypothesis_id, "hypothesis_id")
        require_identifier(self.source_record_id, "source_record_id")
        require_sha256(self.source_record_sha256, "source_record_sha256")
        require_text(self.summary, "evidence summary")
        expected_schema = _EXPECTED_EVALUATION_SCHEMA[self.source_layer]
        if self.source_schema_name != expected_schema:
            raise ValueError(
                f"{self.source_layer.value} evidence must reference {expected_schema}"
            )
        if (
            self.intended_use is EvidenceUse.ADAPTIVE_SEARCH
            and self.source_layer is not EvaluationLayer.SEARCH
        ):
            raise ValueError("adaptive evidence is restricted to Layer A")
        if (
            self.intended_use is EvidenceUse.POSTSEARCH_ASSESSMENT
            and self.source_layer is EvaluationLayer.SEARCH
        ):
            raise ValueError("post-search sealed evidence must come from Layer B or C")
        predictions = tuple(sorted(set(self.prediction_ids)))
        tests = tuple(sorted(set(self.discriminating_test_ids)))
        if not predictions and not tests:
            raise ValueError("evidence must address a prediction or discriminating test")
        for value in predictions:
            require_identifier(value, "prediction_id")
        for value in tests:
            require_identifier(value, "discriminating_test_id")
        object.__setattr__(self, "prediction_ids", predictions)
        object.__setattr__(self, "discriminating_test_ids", tests)

    @property
    def evidence_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "hypothesis_id": self.hypothesis_id,
            "intended_use": self.intended_use.value,
            "source_layer": self.source_layer.value,
            "source_schema_name": self.source_schema_name,
            "source_record_id": self.source_record_id,
            "source_record_sha256": self.source_record_sha256,
            "direction": self.direction.value,
            "prediction_ids": list(self.prediction_ids),
            "discriminating_test_ids": list(self.discriminating_test_ids),
            "summary": self.summary,
            "observed_at_utc": self.observed_at_utc,
        }


@dataclass(frozen=True)
class FailedPredictionRecord:
    failure_id: str
    hypothesis_id: str
    prediction_id: str
    evidence_ids: tuple[str, ...]
    explanation: str
    failed_at_utc: str
    schema_name: str = field(default="FailedPredictionRecord", init=False)
    schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        require_identifier(self.failure_id, "failure_id")
        require_identifier(self.hypothesis_id, "hypothesis_id")
        require_identifier(self.prediction_id, "prediction_id")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not evidence_ids:
            raise ValueError("a failed prediction requires linked evidence")
        for evidence_id in evidence_ids:
            require_identifier(evidence_id, "evidence_id")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        require_text(self.explanation, "failed-prediction explanation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "failure_id": self.failure_id,
            "hypothesis_id": self.hypothesis_id,
            "prediction_id": self.prediction_id,
            "evidence_ids": list(self.evidence_ids),
            "explanation": self.explanation,
            "failed_at_utc": self.failed_at_utc,
        }


@dataclass(frozen=True)
class HypothesisState:
    hypothesis_id: str
    hypothesis_spec_sha256: str
    confidence: ConfidenceLevel
    status: HypothesisStatus
    evidence_ids: tuple[str, ...]
    failed_prediction_ids: tuple[str, ...]
    revision: int

    @property
    def state_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "hypothesis_spec_sha256": self.hypothesis_spec_sha256,
            "confidence": self.confidence.value,
            "status": self.status.value,
            "evidence_ids": list(self.evidence_ids),
            "failed_prediction_ids": list(self.failed_prediction_ids),
            "revision": self.revision,
        }


@dataclass(frozen=True)
class AdaptiveHypothesisUpdate:
    update_id: str
    hypothesis_id: str
    protocol_sha256: str
    expected_prior_state_sha256: str
    evidence_ids: tuple[str, ...]
    failed_prediction_ids: tuple[str, ...]
    new_confidence: ConfidenceLevel
    new_status: HypothesisStatus
    rationale: str

    def __post_init__(self) -> None:
        require_identifier(self.update_id, "update_id")
        require_identifier(self.hypothesis_id, "hypothesis_id")
        require_sha256(self.protocol_sha256, "protocol_sha256")
        require_sha256(self.expected_prior_state_sha256, "expected_prior_state_sha256")
        if self.new_status is HypothesisStatus.SUPPORTED:
            raise ValueError("Layer A adaptation cannot establish scientific support")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not evidence_ids:
            raise ValueError("an adaptive update requires Layer A evidence")
        for evidence_id in evidence_ids:
            require_identifier(evidence_id, "evidence_id")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        failed_prediction_ids = tuple(sorted(set(self.failed_prediction_ids)))
        for failure_id in failed_prediction_ids:
            require_identifier(failure_id, "failed_prediction_id")
        object.__setattr__(self, "failed_prediction_ids", failed_prediction_ids)
        require_text(self.rationale, "update rationale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "hypothesis_id": self.hypothesis_id,
            "protocol_sha256": self.protocol_sha256,
            "expected_prior_state_sha256": self.expected_prior_state_sha256,
            "evidence_ids": list(self.evidence_ids),
            "failed_prediction_ids": list(self.failed_prediction_ids),
            "new_confidence": self.new_confidence.value,
            "new_status": self.new_status.value,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class PostsearchAssessment:
    assessment_id: str
    hypothesis_id: str
    protocol_sha256: str
    search_close_event_sha256: str
    evidence_ids: tuple[str, ...]
    confidence: ConfidenceLevel
    status: HypothesisStatus
    rationale: str

    def __post_init__(self) -> None:
        require_identifier(self.assessment_id, "assessment_id")
        require_identifier(self.hypothesis_id, "hypothesis_id")
        require_sha256(self.protocol_sha256, "protocol_sha256")
        require_sha256(self.search_close_event_sha256, "search_close_event_sha256")
        evidence_ids = tuple(sorted(set(self.evidence_ids)))
        if not evidence_ids:
            raise ValueError("post-search assessment requires sealed evidence")
        for evidence_id in evidence_ids:
            require_identifier(evidence_id, "evidence_id")
        object.__setattr__(self, "evidence_ids", evidence_ids)
        if self.status in {HypothesisStatus.PLANNED, HypothesisStatus.ACTIVE}:
            raise ValueError("post-search assessment must reach a terminal interpretation")
        require_text(self.rationale, "assessment rationale")

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "hypothesis_id": self.hypothesis_id,
            "protocol_sha256": self.protocol_sha256,
            "search_close_event_sha256": self.search_close_event_sha256,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence.value,
            "status": self.status.value,
            "rationale": self.rationale,
        }
