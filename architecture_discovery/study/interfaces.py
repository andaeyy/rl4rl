"""Narrow generator and evaluator interfaces used by the common study engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from study.budget import OpportunityOutcome
from study.contracts import ConditionSpec
from study.serialization import require_int, require_str


class RetryableProviderError(RuntimeError):
    """Provider transport failed before a usable response was stored."""


class RetryableEvaluationError(RuntimeError):
    """Predeclared infrastructure failure that may retry inside the opportunity."""


@dataclass(frozen=True)
class ProposalContext:
    study_id: str
    block_id: str
    run_id: str
    run_seed: int
    condition: ConditionSpec
    opportunity_index: int
    provider_attempt: int
    parent_ids: tuple[str, ...]
    transition_active: bool
    repair: bool = False
    previous_response: str | None = None


@dataclass(frozen=True)
class ProposalResult:
    response_text: str
    candidate_source: str | None
    prompt_tokens: int | None
    completion_tokens: int | None

    def __post_init__(self) -> None:
        require_str(self.response_text, "response_text")
        if self.candidate_source is not None:
            require_str(self.candidate_source, "candidate_source")
        for field_name in ("prompt_tokens", "completion_tokens"):
            value = getattr(self, field_name)
            if value is not None:
                require_int(value, field_name)
                if value < 0:
                    raise ValueError(f"{field_name} cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "response_text": self.response_text,
            "candidate_source": self.candidate_source,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> ProposalResult:
        expected = {
            "response_text",
            "candidate_source",
            "prompt_tokens",
            "completion_tokens",
        }
        if set(payload) != expected:
            raise ValueError("proposal result has invalid fields")
        return cls(
            response_text=require_str(payload["response_text"], "response_text"),
            candidate_source=(
                None
                if payload.get("candidate_source") is None
                else require_str(payload["candidate_source"], "candidate_source")
            ),
            prompt_tokens=(
                None
                if payload.get("prompt_tokens") is None
                else require_int(payload["prompt_tokens"], "prompt_tokens")
            ),
            completion_tokens=(
                None
                if payload.get("completion_tokens") is None
                else require_int(payload["completion_tokens"], "completion_tokens")
            ),
        )


@dataclass(frozen=True, init=False)
class EvaluationResult:
    outcome: OpportunityOutcome
    score: float
    training_attempts: int
    training_steps: int
    training_examples: int
    accelerator_kind: str
    accelerator_seconds: float
    evaluation_cases: int
    infrastructure_retries: int = 0
    failure_stage: str = ""
    schema_name: str = field(default="EvaluationResult", init=False)
    schema_version: str = field(default="2.0", init=False)

    def __init__(
        self,
        outcome: OpportunityOutcome,
        score: float,
        training_attempts: int,
        training_steps: int,
        training_examples: int,
        evaluation_cases: int,
        infrastructure_retries: int = 0,
        failure_stage: str = "",
        *,
        accelerator_kind: str | None = None,
        accelerator_seconds: float | None = None,
        mps_seconds: float | None = None,
    ) -> None:
        """Create a v2 result while accepting the old source-code argument."""

        if mps_seconds is not None:
            self._validate_seconds(mps_seconds, "mps_seconds")
            if accelerator_kind is not None and accelerator_kind != "mps":
                raise ValueError(
                    "mps_seconds cannot be combined with a non-MPS accelerator_kind"
                )
            if (
                accelerator_seconds is not None
                and accelerator_seconds != mps_seconds
            ):
                raise ValueError(
                    "mps_seconds and accelerator_seconds must agree when both are supplied"
                )
            accelerator_kind = "mps"
            accelerator_seconds = mps_seconds
        if accelerator_kind is None or accelerator_seconds is None:
            raise ValueError(
                "accelerator_kind and accelerator_seconds are required"
            )
        values = {
            "outcome": outcome,
            "score": score,
            "training_attempts": training_attempts,
            "training_steps": training_steps,
            "training_examples": training_examples,
            "accelerator_kind": accelerator_kind,
            "accelerator_seconds": accelerator_seconds,
            "evaluation_cases": evaluation_cases,
            "infrastructure_retries": infrastructure_retries,
            "failure_stage": failure_stage,
            "schema_name": "EvaluationResult",
            "schema_version": "2.0",
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    @staticmethod
    def _validate_seconds(value: object, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field_name} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
        if value < 0:
            raise ValueError(f"{field_name} cannot be negative")

    def __post_init__(self) -> None:
        if self.outcome is OpportunityOutcome.INVALID:
            raise ValueError("invalid-source outcomes occur before candidate evaluation")
        for name in (
            "training_attempts",
            "training_steps",
            "training_examples",
            "evaluation_cases",
            "infrastructure_retries",
        ):
            value = require_int(getattr(self, name), name)
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        require_str(self.accelerator_kind, "accelerator_kind")
        if (
            not self.accelerator_kind
            or self.accelerator_kind != self.accelerator_kind.strip()
            or self.accelerator_kind != self.accelerator_kind.lower()
        ):
            raise ValueError(
                "accelerator_kind must be a non-empty lowercase string without surrounding whitespace"
            )
        self._validate_seconds(self.accelerator_seconds, "accelerator_seconds")
        require_str(self.failure_stage, "failure_stage")

    @property
    def mps_seconds(self) -> float:
        """Deprecated source-code alias for pre-v2 consumers."""

        return self.accelerator_seconds

    def to_dict(self) -> dict[str, object]:
        common = {
            "outcome": self.outcome.value,
            "score": self.score,
            "training_attempts": self.training_attempts,
            "training_steps": self.training_steps,
            "training_examples": self.training_examples,
        }
        if self.schema_version == "1.0":
            compute = {"mps_seconds": self.accelerator_seconds}
            schema = {}
        elif self.schema_version == "2.0":
            compute = {
                "accelerator_kind": self.accelerator_kind,
                "accelerator_seconds": self.accelerator_seconds,
            }
            schema = {
                "schema_name": self.schema_name,
                "schema_version": self.schema_version,
            }
        else:  # pragma: no cover - guarded at construction/loading boundaries
            raise ValueError("unsupported EvaluationResult schema version")
        return {
            **schema,
            **common,
            **compute,
            "evaluation_cases": self.evaluation_cases,
            "infrastructure_retries": self.infrastructure_retries,
            "failure_stage": self.failure_stage,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EvaluationResult:
        legacy_fields = {
            "outcome",
            "score",
            "training_attempts",
            "training_steps",
            "training_examples",
            "mps_seconds",
            "evaluation_cases",
            "infrastructure_retries",
            "failure_stage",
        }
        v2_fields = (
            legacy_fields
            - {"mps_seconds"}
            | {
                "schema_name",
                "schema_version",
                "accelerator_kind",
                "accelerator_seconds",
            }
        )
        if set(payload) == legacy_fields:
            result = cls(
                outcome=OpportunityOutcome(require_str(payload["outcome"], "outcome")),
                score=payload["score"],
                training_attempts=require_int(
                    payload["training_attempts"], "training_attempts"
                ),
                training_steps=require_int(payload["training_steps"], "training_steps"),
                training_examples=require_int(
                    payload["training_examples"], "training_examples"
                ),
                mps_seconds=payload["mps_seconds"],
                evaluation_cases=require_int(
                    payload["evaluation_cases"], "evaluation_cases"
                ),
                infrastructure_retries=require_int(
                    payload["infrastructure_retries"], "infrastructure_retries"
                ),
                failure_stage=require_str(payload["failure_stage"], "failure_stage"),
            )
            object.__setattr__(result, "schema_version", "1.0")
            return result
        if set(payload) != v2_fields:
            raise ValueError("evaluation result has invalid fields")
        if payload.get("schema_name") != "EvaluationResult":
            raise ValueError("expected EvaluationResult schema")
        if payload.get("schema_version") != "2.0":
            raise ValueError("unsupported EvaluationResult schema version")
        return cls(
            outcome=OpportunityOutcome(require_str(payload["outcome"], "outcome")),
            score=payload["score"],
            training_attempts=require_int(
                payload["training_attempts"], "training_attempts"
            ),
            training_steps=require_int(payload["training_steps"], "training_steps"),
            training_examples=require_int(
                payload["training_examples"], "training_examples"
            ),
            accelerator_kind=require_str(
                payload["accelerator_kind"], "accelerator_kind"
            ),
            accelerator_seconds=payload["accelerator_seconds"],
            evaluation_cases=require_int(
                payload["evaluation_cases"], "evaluation_cases"
            ),
            infrastructure_retries=require_int(
                payload["infrastructure_retries"], "infrastructure_retries"
            ),
            failure_stage=require_str(payload["failure_stage"], "failure_stage"),
        )


class ProposalGenerator(Protocol):
    def generate(self, context: ProposalContext) -> ProposalResult: ...


class CandidateEvaluator(Protocol):
    def evaluate_seed(self, initial_candidate_id: str, run_seed: int) -> EvaluationResult: ...

    def evaluate_candidate(
        self,
        candidate_source: str,
        *,
        candidate_id: str,
        opportunity_index: int,
        run_seed: int,
    ) -> EvaluationResult: ...
