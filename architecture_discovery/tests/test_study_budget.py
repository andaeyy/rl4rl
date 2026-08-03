import pytest

from study.budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetSpec,
    OpportunityOutcome,
    OpportunityStateError,
)


def _record_seed(ledger: BudgetLedger) -> None:
    ledger.record_seed_evaluation(
        training_attempts=1,
        training_steps=4,
        training_examples=16,
        mps_seconds=0.1,
        evaluation_cases=8,
    )


def test_seed_is_separate_and_next_opportunity_cannot_start_early() -> None:
    ledger = BudgetLedger(BudgetSpec.toy(2))
    with pytest.raises(OpportunityStateError, match="seed"):
        ledger.begin_opportunity(1)
    _record_seed(ledger)
    ledger.begin_opportunity(1)
    with pytest.raises(OpportunityStateError, match="not terminal"):
        ledger.begin_opportunity(2)
    ledger.finish_opportunity(OpportunityOutcome.INVALID)
    ledger.begin_opportunity(2)
    ledger.finish_opportunity(OpportunityOutcome.REJECTED)

    assert ledger.seed_evaluations == 1
    assert ledger.proposal_opportunities == 2
    assert ledger.terminal_opportunities == 2


def test_provider_retries_stay_inside_one_opportunity() -> None:
    ledger = BudgetLedger(BudgetSpec.toy(1))
    _record_seed(ledger)
    ledger.begin_opportunity(1)
    assert ledger.start_provider_attempt() == 1
    ledger.record_infrastructure_retry()
    assert ledger.start_provider_attempt() == 2
    ledger.record_provider_usage(prompt_tokens=12, completion_tokens=9)
    ledger.finish_opportunity(OpportunityOutcome.ACCEPTED)

    assert ledger.proposal_opportunities == 1
    assert ledger.provider_attempts == 2
    assert ledger.infrastructure_retries == 1
    assert ledger.accepted == 1


def test_ledger_round_trip_reconstructs_unique_sources_and_attempts() -> None:
    ledger = BudgetLedger(BudgetSpec.toy(1))
    _record_seed(ledger)
    ledger.begin_opportunity(1)
    ledger.start_provider_attempt()
    ledger.record_provider_usage(prompt_tokens=None, completion_tokens=None)
    ledger.record_candidate_source("candidate-hash")
    ledger.record_candidate_source("candidate-hash")
    ledger.record_training(attempts=1, steps=4, examples=16, mps_seconds=0.1)
    ledger.record_evaluation(8)
    ledger.finish_opportunity(OpportunityOutcome.ACCEPTED)

    restored = BudgetLedger.from_dict(ledger.to_dict())
    assert restored.to_dict() == ledger.to_dict()
    assert restored.unique_candidate_sources == 1
    assert restored.unknown_provider_usage == 1


def test_repairs_have_separate_total_and_per_opportunity_ceilings() -> None:
    ledger = BudgetLedger(BudgetSpec.toy(1))
    _record_seed(ledger)
    ledger.begin_opportunity(1)
    ledger.record_repair()
    with pytest.raises(BudgetExceeded, match="per-opportunity"):
        ledger.record_repair()
    ledger.finish_opportunity(OpportunityOutcome.INVALID)

    assert ledger.repairs == 1
    assert ledger.to_dict()["repairs_by_opportunity"] == {"1": 1}
