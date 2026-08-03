from dataclasses import fields, replace

import pytest

from baselines.no_search import (
    DeterministicFakeBackend,
    GeneratorBudgetMismatch,
    IndependentOpportunity,
    NoSearchProposalGenerator,
    NoSearchSpec,
    assert_matched_generator_budget,
)
from baselines.prompt_placebo import PromptPlaceboSpec
from study.budget import BudgetExceeded, BudgetLedger, BudgetSpec, OpportunityOutcome
from study.contracts import ConditionSpec
from study.interfaces import ProposalContext


def _context(
    *,
    parents: tuple[str, ...] = ("secret-parent",),
    transition: bool = False,
    repair: bool = False,
    opportunity: int = 1,
) -> ProposalContext:
    return ProposalContext(
        study_id="study",
        block_id="block",
        run_id="run",
        run_seed=19,
        condition=ConditionSpec.for_id("C3"),
        opportunity_index=opportunity,
        provider_attempt=1,
        parent_ids=parents,
        transition_active=transition,
        repair=repair,
    )


def _spec() -> NoSearchSpec:
    return NoSearchSpec(
        system_prompt="Propose one architecture.",
        task_prompt="Return a self-contained candidate.",
        max_completion_tokens=256,
    )


def test_no_search_information_boundary_drops_all_adaptive_state() -> None:
    opportunity_fields = {item.name for item in fields(IndependentOpportunity)}
    forbidden = {
        "parent_ids",
        "scores",
        "candidate_history",
        "feedback",
        "transition_active",
        "repair",
        "condition",
    }
    assert opportunity_fields.isdisjoint(forbidden)

    backend = DeterministicFakeBackend()
    generator = NoSearchProposalGenerator(
        spec=_spec(), backend=backend, scientific=False
    )
    generator.generate(
        _context(
            parents=("parent-A", "parent-B"),
            transition=True,
            repair=True,
        )
    )
    generator.generate(_context(parents=("different-parent",)))

    assert backend.requests[0].model_input == backend.requests[1].model_input
    visible = repr(backend.requests[0].model_input)
    assert "parent-A" not in visible
    assert "different-parent" not in visible
    assert "transition" not in visible
    assert "repair" not in visible


def test_each_no_search_opportunity_is_independent_but_reproducible() -> None:
    first_backend = DeterministicFakeBackend()
    second_backend = DeterministicFakeBackend()
    first = NoSearchProposalGenerator(
        spec=_spec(), backend=first_backend, scientific=False
    )
    second = NoSearchProposalGenerator(
        spec=_spec(), backend=second_backend, scientific=False
    )

    first_results = [first.generate(_context(opportunity=index)) for index in (1, 2)]
    second_results = [second.generate(_context(opportunity=index)) for index in (1, 2)]

    assert first_results == second_results
    assert first_results[0].candidate_source != first_results[1].candidate_source
    assert first_backend.requests[0].model_input == first_backend.requests[1].model_input


def test_fake_backend_cannot_be_used_for_a_scientific_run() -> None:
    with pytest.raises(ValueError, match="cannot run scientifically"):
        NoSearchProposalGenerator(
            spec=_spec(),
            backend=DeterministicFakeBackend(),
            scientific=True,
        )


def test_prompt_placebo_is_a_distinct_control_contract() -> None:
    placebo = PromptPlaceboSpec(
        placebo_id="placebo-1", prompt_template="Adaptive but neutral prompt"
    )
    assert placebo.retains_search_feedback is True
    with pytest.raises(TypeError, match="not a prompt-placebo"):
        NoSearchProposalGenerator(
            spec=placebo,  # type: ignore[arg-type]
            backend=DeterministicFakeBackend(),
            scientific=False,
        )


def test_generator_budget_match_includes_opportunities_tokens_and_repairs() -> None:
    reference = BudgetSpec.toy(3)
    assert_matched_generator_budget(reference, BudgetSpec.from_dict(reference.to_dict()))

    for field_name, value in (
        ("proposal_opportunities", 4),
        ("prompt_tokens", reference.prompt_tokens + 1),
        ("completion_tokens", reference.completion_tokens + 1),
        ("repairs", reference.repairs + 1),
    ):
        with pytest.raises(GeneratorBudgetMismatch, match=field_name):
            assert_matched_generator_budget(
                replace(reference, **{field_name: value}), reference
            )


def test_no_search_usage_is_charged_to_the_common_ledger() -> None:
    budget = BudgetSpec.toy(1)
    ledger = BudgetLedger(budget)
    ledger.record_seed_evaluation(
        training_attempts=1,
        training_steps=4,
        training_examples=16,
        mps_seconds=1.0,
        evaluation_cases=8,
    )
    ledger.begin_opportunity(1)
    ledger.start_provider_attempt()
    result = NoSearchProposalGenerator(
        spec=_spec(),
        backend=DeterministicFakeBackend(prompt_tokens=23, completion_tokens=17),
        scientific=False,
    ).generate(_context())
    ledger.record_provider_usage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
    ledger.finish_opportunity(OpportunityOutcome.ACCEPTED)
    assert ledger.prompt_tokens == 23
    assert ledger.completion_tokens == 17
    assert ledger.proposal_opportunities == 1

    too_small = replace(budget, prompt_tokens=22)
    failing = BudgetLedger(too_small)
    failing.record_seed_evaluation(
        training_attempts=1,
        training_steps=4,
        training_examples=16,
        mps_seconds=1.0,
        evaluation_cases=8,
    )
    failing.begin_opportunity(1)
    failing.start_provider_attempt()
    with pytest.raises(BudgetExceeded, match="prompt_tokens"):
        failing.record_provider_usage(prompt_tokens=23, completion_tokens=17)
