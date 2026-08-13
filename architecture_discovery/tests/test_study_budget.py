import pytest

from study.budget import (
    BudgetExceeded,
    BudgetLedger,
    BudgetSpec,
    OpportunityOutcome,
    OpportunityStateError,
)
from study.interfaces import EvaluationResult
from study.serialization import content_hash


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


def _as_v1_budget(payload: dict[str, object]) -> dict[str, object]:
    legacy = dict(payload)
    legacy["schema_version"] = "1.0"
    legacy["mps_seconds"] = legacy.pop("accelerator_seconds")
    legacy.pop("accelerator_kind")
    return legacy


def test_new_budget_and_ledger_records_are_accelerator_neutral_v2() -> None:
    spec = BudgetSpec.toy(1)
    ledger = BudgetLedger(spec)
    _record_seed(ledger)
    ledger.begin_opportunity(1)
    ledger.finish_opportunity(OpportunityOutcome.INVALID)

    spec_payload = spec.to_dict()
    ledger_payload = ledger.to_dict()
    assert spec_payload["schema_version"] == "2.0"
    assert spec_payload["accelerator_kind"] == "cpu"
    assert "mps_seconds" not in spec_payload
    assert ledger_payload["schema_version"] == "2.0"
    assert ledger_payload["accelerator_kind"] == "cpu"
    assert "mps_seconds" not in ledger_payload


def test_v1_budget_and_ledger_round_trip_without_hash_or_field_rewrite() -> None:
    legacy_spec = _as_v1_budget(BudgetSpec.toy(1).to_dict())
    # Schema v1 was intrinsically MPS; the scalar ceiling is otherwise unchanged.
    restored_spec = BudgetSpec.from_dict(legacy_spec)
    assert restored_spec.accelerator_kind == "mps"
    assert restored_spec.to_dict() == legacy_spec
    assert content_hash(restored_spec.to_dict()) == content_hash(legacy_spec)

    active = BudgetLedger(BudgetSpec.toy(1))
    _record_seed(active)
    active.begin_opportunity(1)
    active.finish_opportunity(OpportunityOutcome.INVALID)
    legacy_ledger = active.to_dict()
    legacy_ledger["schema_version"] = "1.0"
    legacy_ledger["spec"] = legacy_spec
    legacy_ledger["mps_seconds"] = legacy_ledger.pop("accelerator_seconds")
    legacy_ledger.pop("accelerator_kind")

    restored_ledger = BudgetLedger.from_dict(legacy_ledger)
    assert restored_ledger.to_dict() == legacy_ledger
    assert content_hash(restored_ledger.to_dict()) == content_hash(legacy_ledger)


def test_v1_evaluation_result_round_trips_without_hash_or_field_rewrite() -> None:
    legacy = {
        "outcome": "accepted",
        "score": 0.75,
        "training_attempts": 1,
        "training_steps": 4,
        "training_examples": 16,
        "mps_seconds": 0.25,
        "evaluation_cases": 8,
        "infrastructure_retries": 0,
        "failure_stage": "",
    }
    restored = EvaluationResult.from_dict(legacy)
    assert restored.accelerator_kind == "mps"
    assert restored.to_dict() == legacy
    assert content_hash(restored.to_dict()) == content_hash(legacy)


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -0.1])
def test_v2_accelerator_seconds_require_a_finite_nonnegative_number(
    invalid: object,
) -> None:
    payload = BudgetSpec.toy(1).to_dict()
    payload["accelerator_seconds"] = invalid
    with pytest.raises(ValueError, match="numeric|finite|non-negative"):
        BudgetSpec.from_dict(payload)
