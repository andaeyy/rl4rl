"""Deterministic provider-free fixtures for infrastructure tests and smoke runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from study.budget import OpportunityOutcome
from study.interfaces import (
    EvaluationResult,
    ProposalContext,
    ProposalResult,
    RetryableProviderError,
)
from study.serialization import content_hash


@dataclass
class DeterministicFakeGenerator:
    """Emits synthetic text only. It never imports or contacts a model provider."""

    fail_first_attempts: dict[int, int] = field(default_factory=dict)
    parse_failures: set[int] = field(default_factory=set)
    calls: list[ProposalContext] = field(default_factory=list)

    def generate(self, context: ProposalContext) -> ProposalResult:
        self.calls.append(context)
        if context.provider_attempt <= self.fail_first_attempts.get(
            context.opportunity_index, 0
        ):
            raise RetryableProviderError(
                f"synthetic provider failure at opportunity {context.opportunity_index}"
            )
        transition = "transition" if context.transition_active else "ordinary"
        prompt_key = {
            "run_seed": context.run_seed,
            "condition": context.condition.to_dict(),
            "opportunity": context.opportunity_index,
            "parents": list(context.parent_ids),
            "proposal_mode": transition,
            "repair": context.repair,
        }
        prompt_digest = content_hash(prompt_key)
        response_text = (
            f"OFFLINE PROPOSAL {context.opportunity_index} {transition} "
            f"{prompt_digest[:16]}"
        )
        if context.opportunity_index in self.parse_failures:
            candidate_source = None
        else:
            candidate_source = (
                "# Synthetic architecture candidate; never executed.\n"
                f"OFFLINE_CANDIDATE = '{prompt_digest}'\n"
            )
        return ProposalResult(
            response_text=response_text,
            candidate_source=candidate_source,
            prompt_tokens=10 + len(context.parent_ids),
            completion_tokens=12,
        )


@dataclass
class DeterministicFakeEvaluator:
    """Produces resource records without importing Torch or training a model."""

    accepted_opportunities: set[int] | None = None
    seed_calls: list[tuple[str, int]] = field(default_factory=list)
    candidate_calls: list[tuple[str, int, int]] = field(default_factory=list)

    def evaluate_seed(self, initial_candidate_id: str, run_seed: int) -> EvaluationResult:
        self.seed_calls.append((initial_candidate_id, run_seed))
        return EvaluationResult(
            outcome=OpportunityOutcome.ACCEPTED,
            score=0.5,
            training_attempts=1,
            training_steps=4,
            training_examples=16,
            mps_seconds=0.001,
            evaluation_cases=8,
        )

    def evaluate_candidate(
        self,
        candidate_source: str,
        *,
        candidate_id: str,
        opportunity_index: int,
        run_seed: int,
    ) -> EvaluationResult:
        self.candidate_calls.append((candidate_id, opportunity_index, run_seed))
        accepted = (
            opportunity_index % 2 == 1
            if self.accepted_opportunities is None
            else opportunity_index in self.accepted_opportunities
        )
        return EvaluationResult(
            outcome=(
                OpportunityOutcome.ACCEPTED
                if accepted
                else OpportunityOutcome.REJECTED
            ),
            score=0.75 if accepted else 0.25,
            training_attempts=1,
            training_steps=4,
            training_examples=16,
            mps_seconds=0.001,
            evaluation_cases=8,
        )
