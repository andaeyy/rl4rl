"""Narrow generator and evaluator interfaces used by the common study engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
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


@dataclass(frozen=True)
class EvaluationResult:
    outcome: OpportunityOutcome
    score: float
    training_attempts: int
    training_steps: int
    training_examples: int
    mps_seconds: float
    evaluation_cases: int
    infrastructure_retries: int = 0
    failure_stage: str = ""

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
        if isinstance(self.mps_seconds, bool) or not isinstance(
            self.mps_seconds, (int, float)
        ):
            raise ValueError("mps_seconds must be numeric")
        if not math.isfinite(self.mps_seconds):
            raise ValueError("mps_seconds must be finite")
        if self.mps_seconds < 0:
            raise ValueError("mps_seconds cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "score": self.score,
            "training_attempts": self.training_attempts,
            "training_steps": self.training_steps,
            "training_examples": self.training_examples,
            "mps_seconds": self.mps_seconds,
            "evaluation_cases": self.evaluation_cases,
            "infrastructure_retries": self.infrastructure_retries,
            "failure_stage": self.failure_stage,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> EvaluationResult:
        expected = {
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
        if set(payload) != expected:
            raise ValueError("evaluation result has invalid fields")
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
            mps_seconds=payload["mps_seconds"],
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
