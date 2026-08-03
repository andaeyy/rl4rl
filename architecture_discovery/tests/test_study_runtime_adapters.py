from pathlib import Path
from types import SimpleNamespace

from common.gpt56_sol import GPT56SolProfile
from evaluation.records import RecordEnvelope, SearchEvaluationRecord
from study.budget import OpportunityOutcome
from study.contracts import ConditionId, ConditionSpec
from study.interfaces import ProposalContext
from study.runtime_adapters import (
    CandidateSourceStore,
    LayerACandidateEvaluator,
    MatchedCausalProposalGenerator,
)


def _hash(character: str) -> str:
    return character * 64


class _FakeCompletions:
    def create(self, **_request):
        response = (
            "Hypothesis: change the constant.\n"
            "<<<<<<< SEARCH\nVALUE = 1\n=======\nVALUE = 2\n>>>>>>> REPLACE\n"
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=response))],
            usage=SimpleNamespace(prompt_tokens=101, completion_tokens=17),
        )


class _FakeClient:
    chat = SimpleNamespace(completions=_FakeCompletions())


def _profile() -> GPT56SolProfile:
    return GPT56SolProfile(
        model="gpt-5.6-sol",
        reasoning_effort="high",
        max_completion_tokens=1024,
        timeout_seconds=30,
        retries=0,
        retry_delay_seconds=0,
        seed=7,
    )


def _context(condition: ConditionId, parents: tuple[str, ...]) -> ProposalContext:
    return ProposalContext(
        study_id="study",
        block_id="block",
        run_id="run",
        run_seed=7,
        condition=ConditionSpec.for_id(condition),
        opportunity_index=1,
        provider_attempt=1,
        parent_ids=parents,
        transition_active=condition in {ConditionId.C1, ConditionId.C3},
    )


def test_causal_generator_uses_fixed_parent_slots_and_full_source_store(tmp_path):
    store = CandidateSourceStore(tmp_path / "sources")
    store.register("seed", "VALUE = 1\n")
    store.register("inspiration", "VALUE = 9\n")
    generator = MatchedCausalProposalGenerator(
        client=_FakeClient(),
        generation=_profile(),
        source_store=store,
        portfolio_size=3,
        system_prompt="shared",
        request_log_root=tmp_path / "requests",
    )
    single_prompt = generator.build_user_prompt(_context(ConditionId.C0, ("seed",)))
    portfolio_prompt = generator.build_user_prompt(
        _context(ConditionId.C2, ("inspiration", "seed"))
    )
    for index in range(1, 4):
        assert f"PARENT SLOT {index}:" in single_prompt
        assert f"PARENT SLOT {index}:" in portfolio_prompt
    assert single_prompt.count(generator.neutral_slot_text) == 2
    assert portfolio_prompt.count(generator.neutral_slot_text) == 1

    proposal = generator.generate(_context(ConditionId.C0, ("seed",)))
    assert proposal.candidate_source == "VALUE = 2\n"
    assert proposal.prompt_tokens == 101
    assert proposal.completion_tokens == 17
    assert store.path(next(path.stem for path in store.root.glob("*.py") if path.stem not in {"seed", "inspiration"})).is_file()


def test_layer_a_adapter_maps_only_typed_search_result(tmp_path):
    source_store = CandidateSourceStore(tmp_path / "sources")
    source_store.register("seed", "VALUE = 1\n")

    def fake_evaluate(_path: Path, **kwargs):
        context = kwargs["context"]
        return SearchEvaluationRecord(
            envelope=RecordEnvelope.create(
                schema_name="search_evaluation",
                study_id=context.study_id,
                block_id=context.block_id,
                run_id=context.run_id,
                condition_id=context.condition_id,
                writer_component="test",
                code_sha256=_hash("a"),
                config_sha256=_hash("b"),
                environment_sha256=_hash("c"),
            ),
            candidate_id="candidate-test",
            training_record_id="training-test",
            execution_ok=True,
            transformer_valid=True,
            public_accuracy=1.0,
            search_score=1.0,
            eligible_for_parent=True,
            parameter_count_metadata=10**9,
        )

    adapter = LayerACandidateEvaluator(
        study_id="study",
        block_id="block",
        run_id="run",
        condition_id="C0",
        initial_candidate_id="seed",
        source_store=source_store,
        output_root=tmp_path / "run",
        training_profile="smoke_train_v1",
        device="cpu",
        allow_cpu_for_tests=True,
        evaluation_profile="smoke_eval_v1",
        evaluation_case_count=8,
        pi_decision_record_id=None,
        eligibility_threshold=0.99,
        evaluate_function=fake_evaluate,
    )
    result = adapter.evaluate_seed("seed", 7)
    assert result.outcome is OpportunityOutcome.ACCEPTED
    assert result.score == 1.0
    assert result.evaluation_cases == 8
    assert not hasattr(result, "parameter_count_metadata")
