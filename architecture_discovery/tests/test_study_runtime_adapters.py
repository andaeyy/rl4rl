import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from architecture_ir import validate_ir_candidate_json
from common.evaluator import file_hash
from common.gpt56_sol import GPT56SolProfile
from evaluation.records import RecordEnvelope, SearchEvaluationRecord
from study.budget import OpportunityOutcome
from study.contracts import ConditionId, ConditionSpec
from study.interfaces import ProposalContext
from study.runtime_adapters import (
    CandidateSourceStore,
    LayerACandidateEvaluator,
    MatchedCausalProposalGenerator,
    canonicalize_architecture_ir,
)
from study.serialization import content_hash


_INITIAL_PATH = (
    Path(__file__).resolve().parents[1] / "common" / "initial_candidate.ir.json"
)


def _ir(label: str) -> str:
    payload = json.loads(_INITIAL_PATH.read_text(encoding="utf-8"))
    payload["graph_id"] = f"runtime_adapter_{label}"
    payload["metadata"]["mechanism_hypothesis"] = f"test mechanism {label}"
    return canonicalize_architecture_ir(
        json.dumps(payload),
        require_hypothesis=True,
    )


def _hash(character: str) -> str:
    return character * 64


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content

    def create(self, **_request):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=101, completion_tokens=17),
        )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


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
    seed = _ir("seed")
    inspiration = _ir("inspiration")
    proposal_ir = _ir("proposal")
    seed_id = content_hash(seed)
    inspiration_id = content_hash(inspiration)
    store.register(seed_id, seed)
    store.register(inspiration_id, inspiration)
    generator = MatchedCausalProposalGenerator(
        client=_FakeClient(f"```json\n{proposal_ir}\n```"),
        generation=_profile(),
        source_store=store,
        portfolio_size=3,
        system_prompt="shared",
        request_log_root=tmp_path / "requests",
    )
    single_prompt = generator.build_user_prompt(
        _context(ConditionId.C0, (seed_id,))
    )
    portfolio_prompt = generator.build_user_prompt(
        _context(ConditionId.C2, (inspiration_id, seed_id))
    )
    for index in range(1, 4):
        assert f"PARENT SLOT {index}:" in single_prompt
        assert f"PARENT SLOT {index}:" in portfolio_prompt
    assert single_prompt.count(generator.neutral_slot_text) == 2
    assert portfolio_prompt.count(generator.neutral_slot_text) == 1
    assert "```json" in single_prompt
    assert "SEARCH/REPLACE" not in single_prompt
    assert "```python" not in single_prompt

    proposal = generator.generate(_context(ConditionId.C0, (seed_id,)))
    assert proposal.candidate_source == proposal_ir
    assert proposal.prompt_tokens == 101
    assert proposal.completion_tokens == 17
    assert store.path(content_hash(proposal_ir)).suffix == ".json"
    assert not tuple(store.root.glob("*.py"))
    assert validate_ir_candidate_json(proposal.candidate_source).valid


def test_causal_generator_rejects_python_or_diff_output(tmp_path) -> None:
    store = CandidateSourceStore(tmp_path / "sources")
    seed = _ir("seed-invalid-output")
    seed_id = content_hash(seed)
    store.register(seed_id, seed)
    generator = MatchedCausalProposalGenerator(
        client=_FakeClient(
            "<<<<<<< SEARCH\nVALUE = 1\n=======\nVALUE = 2\n>>>>>>> REPLACE"
        ),
        generation=_profile(),
        source_store=store,
        portfolio_size=2,
        system_prompt="shared",
        request_log_root=tmp_path / "requests",
    )

    proposal = generator.generate(_context(ConditionId.C0, (seed_id,)))

    assert proposal.candidate_source is None
    assert tuple(store.root.glob("*.json")) == (store.path(seed_id),)


def test_source_store_canonicalizes_and_binds_engine_identity(tmp_path) -> None:
    raw = _INITIAL_PATH.read_text(encoding="utf-8")
    canonical = canonicalize_architecture_ir(raw, require_hypothesis=False)
    store = CandidateSourceStore(tmp_path / "sources")

    path = store.register(content_hash(canonical), raw)

    assert path.read_text(encoding="utf-8") == canonical
    with pytest.raises(ValueError, match="canonical architecture IR"):
        store.register("not-the-canonical-id", canonical)


def test_layer_a_adapter_maps_only_typed_search_result(tmp_path):
    source_store = CandidateSourceStore(tmp_path / "sources")
    seed = _ir("layer-a-seed")
    seed_id = content_hash(seed)
    source_store.register(seed_id, seed)

    def fake_evaluate(candidate_path: Path, **kwargs):
        context = kwargs["context"]
        training_output = kwargs["training_output_dir"]
        training_output.mkdir(parents=True)
        (training_output / "training_summary.json").write_text(
            json.dumps(
                {
                    "candidate_source_hash": file_hash(candidate_path),
                    "profile_name": kwargs["training_profile"],
                    "steps_completed": 4,
                    "examples_processed": 16,
                    "train_seconds": 0.25,
                }
            ),
            encoding="utf-8",
        )
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
            candidate_id=f"candidate-{file_hash(candidate_path)}",
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
        initial_candidate_id=seed_id,
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
    result = adapter.evaluate_seed(seed_id, 7)
    assert result.outcome is OpportunityOutcome.ACCEPTED
    assert result.score == 1.0
    assert result.training_attempts == 1
    assert result.training_steps == 4
    assert result.training_examples == 16
    assert result.accelerator_seconds == 0.25
    assert result.evaluation_cases == 8
    assert not hasattr(result, "parameter_count_metadata")


def test_layer_a_adapter_rejects_mismatched_controller_binding(tmp_path) -> None:
    source_store = CandidateSourceStore(tmp_path / "sources")
    seed = _ir("layer-a-binding")
    seed_id = content_hash(seed)
    source_store.register(seed_id, seed)

    def fake_evaluate(_candidate_path: Path, **kwargs):
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
            candidate_id=f"candidate-{'0' * 64}",
            training_record_id="training-test",
            execution_ok=True,
            transformer_valid=True,
            public_accuracy=1.0,
            search_score=1.0,
            eligible_for_parent=True,
        )

    adapter = LayerACandidateEvaluator(
        study_id="study",
        block_id="block",
        run_id="run",
        condition_id="C0",
        initial_candidate_id=seed_id,
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

    with pytest.raises(ValueError, match="candidate source hash"):
        adapter.evaluate_seed(seed_id, 7)
