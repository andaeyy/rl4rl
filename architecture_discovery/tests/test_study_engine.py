import json
from pathlib import Path

import pytest

from study.contracts import ConditionId, StudySpec
from study.engine import CommonStudyEngine, RunStateError
from study.fakes import DeterministicFakeEvaluator, DeterministicFakeGenerator
from study.randomization import generate_plan


def _run_for_condition(plan, condition_id: ConditionId):
    return next(
        run for run in plan.runs if run.condition.condition_id is condition_id
    )


def test_all_conditions_receive_equal_seed_and_descendant_opportunities(tmp_path) -> None:
    spec = StudySpec.toy(
        study_id="equal-opportunities",
        block_count=1,
        proposal_opportunities=3,
    )
    plan = generate_plan(spec, tmp_path)
    states = {}
    seed_calls = {}
    for condition_id in ConditionId:
        evaluator = DeterministicFakeEvaluator()
        state = CommonStudyEngine(
            study=spec,
            run=_run_for_condition(plan, condition_id),
            generator=DeterministicFakeGenerator(),
            evaluator=evaluator,
        ).execute()
        states[condition_id] = state
        seed_calls[condition_id] = evaluator.seed_calls

    assert len({tuple(calls) for calls in seed_calls.values()}) == 1
    for state in states.values():
        assert state.status == "completed"
        assert state.ledger["seed_evaluations"] == 1
        assert state.ledger["proposal_opportunities"] == 3
        assert state.ledger["terminal_opportunities"] == 3
        assert state.ledger["candidate_training_attempts"] == 4


def test_only_treatment_policies_change_parent_and_transition_context(tmp_path) -> None:
    spec = StudySpec.toy(
        study_id="treatment-context",
        proposal_opportunities=3,
    )
    plan = generate_plan(spec, tmp_path)
    calls = {}
    for condition_id in ConditionId:
        generator = DeterministicFakeGenerator()
        CommonStudyEngine(
            study=spec,
            run=_run_for_condition(plan, condition_id),
            generator=generator,
            evaluator=DeterministicFakeEvaluator(accepted_opportunities={1, 2, 3}),
        ).execute()
        calls[condition_id] = generator.calls

    assert [len(call.parent_ids) for call in calls[ConditionId.C0]] == [1, 1, 1]
    assert [len(call.parent_ids) for call in calls[ConditionId.C1]] == [1, 1, 1]
    assert [len(call.parent_ids) for call in calls[ConditionId.C2]] == [1, 2, 2]
    assert [len(call.parent_ids) for call in calls[ConditionId.C3]] == [1, 2, 2]
    assert [call.transition_active for call in calls[ConditionId.C0]] == [
        False,
        False,
        False,
    ]
    assert [call.transition_active for call in calls[ConditionId.C1]] == [
        False,
        True,
        False,
    ]


def test_parse_failure_and_provider_retry_consume_opportunities(tmp_path) -> None:
    spec = StudySpec.toy(study_id="failure-accounting", proposal_opportunities=3)
    plan = generate_plan(spec, tmp_path)
    generator = DeterministicFakeGenerator(
        fail_first_attempts={1: 1},
        parse_failures={2},
    )
    state = CommonStudyEngine(
        study=spec,
        run=_run_for_condition(plan, ConditionId.C0),
        generator=generator,
        evaluator=DeterministicFakeEvaluator(),
    ).execute()

    assert state.ledger["proposal_opportunities"] == 3
    assert state.ledger["provider_attempts"] == 5
    assert state.ledger["infrastructure_retries"] == 1
    assert state.ledger["parse_failures"] == 2
    assert state.ledger["repairs"] == 1
    assert state.ledger["invalid"] == 1
    assert state.ledger["candidate_training_attempts"] == 3


def test_resume_reuses_stored_proposal_instead_of_calling_generator_again(tmp_path) -> None:
    class CrashOnceEvaluator(DeterministicFakeEvaluator):
        def __init__(self) -> None:
            super().__init__()
            self.crashed = False

        def evaluate_candidate(self, *args, **kwargs):
            if not self.crashed:
                self.crashed = True
                raise RuntimeError("synthetic process interruption")
            return super().evaluate_candidate(*args, **kwargs)

    spec = StudySpec.toy(study_id="resume-proposal", proposal_opportunities=1)
    run = _run_for_condition(generate_plan(spec, tmp_path), ConditionId.C0)
    generator = DeterministicFakeGenerator()
    evaluator = CrashOnceEvaluator()
    engine = CommonStudyEngine(
        study=spec,
        run=run,
        generator=generator,
        evaluator=evaluator,
    )

    with pytest.raises(RuntimeError, match="interruption"):
        engine.execute()
    stored = Path(run.run_directory, "run_state.json").read_text()
    assert "OFFLINE PROPOSAL" in stored
    assert len(generator.calls) == 1

    state = engine.execute()
    assert state.status == "completed"
    assert len(generator.calls) == 1
    assert state.ledger["provider_attempts"] == 1


@pytest.mark.parametrize(
    ("field_name", "mutated_value", "message"),
    (
        ("transition_active", "false", "transition_active must be boolean"),
        ("transition_active", False, "frozen proposal policy"),
        ("parent_ids", ["candidate:attacker"], "frozen parent policy"),
    ),
)
def test_resume_rejects_tampered_active_treatment_context(
    tmp_path, field_name, mutated_value, message
) -> None:
    class CrashEvaluator(DeterministicFakeEvaluator):
        def evaluate_candidate(self, *args, **kwargs):
            raise RuntimeError("synthetic interruption")

    spec = StudySpec.toy(study_id="resume-treatment-tamper", proposal_opportunities=1)
    run = _run_for_condition(generate_plan(spec, tmp_path), ConditionId.C1)
    engine = CommonStudyEngine(
        study=spec,
        run=run,
        generator=DeterministicFakeGenerator(),
        evaluator=CrashEvaluator(),
    )
    with pytest.raises(RuntimeError, match="interruption"):
        engine.execute()

    state_path = Path(run.run_directory, "run_state.json")
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["active_opportunity"][field_name] = mutated_value
    state_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunStateError, match=message):
        engine.execute()
