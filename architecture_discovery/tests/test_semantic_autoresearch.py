from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import agents.semantic_autoresearch.run as semantic_run
from agents.semantic_autoresearch.run import (
    FROZEN_DESCRIPTOR_AXES,
    FrozenSemanticArchive,
    IRProposalError,
    PilotPreflightError,
    ProposalResponse,
    RunOptions,
    run_semantic_autoresearch,
)
from architecture_ir import decode_graph_json, encode_graph_json
from common.evaluator import file_hash
from evaluation.records import ControllerSearchView, SCHEMA_VERSION


INITIAL_IR = Path(semantic_run.ROOT / "common" / "initial_candidate.ir.json")


def _ir(index: int, *, hypothesis: bool = True) -> str:
    base = decode_graph_json(INITIAL_IR.read_text(encoding="utf-8"))
    metadata = dict(base.metadata)
    if hypothesis:
        metadata["mechanism_hypothesis"] = f"semantic mechanism {index}"
    else:
        metadata.pop("mechanism_hypothesis", None)
    normalization = next(
        node for node in base.nodes if node.kind.value == "normalization"
    )
    nodes = tuple(
        replace(
            normalization,
            attributes={
                **normalization.attributes,
                "epsilon": 1e-5 + index * 1e-7,
            },
        )
        if node.node_id == normalization.node_id
        else node
        for node in base.nodes
    )
    return encode_graph_json(
        replace(
            base,
            graph_id=f"semantic.candidate.{index}",
            nodes=nodes,
            metadata=metadata,
        )
    )


def _signature(*, attention_organization: int = 2):
    values = {
        "semantic_token_representation": 1,
        "semantic_positional_integration": 1,
        "semantic_attention_projection": 1,
        "semantic_attention_organization": attention_organization,
        "semantic_feedforward_mechanism": 1,
        "semantic_normalization": 1,
        "semantic_depth_topology": 2,
        "semantic_output_readout": 2,
        "semantic_tokenization": 1,
    }
    assert tuple(values) == FROZEN_DESCRIPTOR_AXES
    return tuple((name, float(values[name])) for name in FROZEN_DESCRIPTOR_AXES)


def _view(
    label: str,
    *,
    eligible: bool = True,
    score: float = 0.0,
    attention_organization: int = 2,
) -> ControllerSearchView:
    return ControllerSearchView(
        schema_name="search_evaluation",
        schema_version=SCHEMA_VERSION,
        record_id=f"record-{label}",
        run_id="placeholder-run",
        condition_id="placeholder-condition",
        candidate_id=f"candidate-{label}",
        execution_ok=True,
        transformer_valid=True,
        public_accuracy=score,
        search_score=score,
        eligible_for_parent=eligible,
        failure_stage="" if eligible else "public_accuracy",
        infrastructure_failure=False,
        online_descriptor_codes=(
            _signature(attention_organization=attention_organization)
            if eligible
            else ()
        ),
    )


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.generate_calls = 0
        self.preflight_calls = 0
        self.messages = []

    def preflight(self):
        self.preflight_calls += 1

    def generate(self, messages):
        self.generate_calls += 1
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return ProposalResponse(response, input_tokens=13, output_tokens=9)

    def manifest_fields(self):
        return {"model": "fake-semantic-ir-provider", "paid_calls": False}


class FakeEvaluator:
    controller_only_test_double = True

    def __init__(self, results=None, *, zero_accuracy=False, bind=True):
        self.results = list(results) if results is not None else None
        self.zero_accuracy = zero_accuracy
        self.bind = bind
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        request.candidate_path.read_text(encoding="utf-8")
        index = len(self.requests)
        if self.results is None:
            score = 0.0 if self.zero_accuracy else min(1.0, 0.5 + index / 100)
            result = _view(
                str(index),
                score=score,
                attention_organization=2 if index % 2 else 3,
            )
        else:
            result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if self.bind:
            source_hash = file_hash(request.candidate_path)
            result = replace(
                result,
                record_id=f"record-{index}-{source_hash}",
                run_id=request.context.run_id,
                condition_id=request.context.condition_id,
                candidate_id=f"candidate-{source_hash}",
            )
        return result


def _options(tmp_path: Path, *, iterations: int = 1) -> RunOptions:
    initial = tmp_path / "initial.ir.json"
    initial.write_text(_ir(0, hypothesis=False), encoding="utf-8")
    return RunOptions(
        iterations=iterations,
        seed=3,
        output_dir=tmp_path / "run",
        initial_candidate=initial,
        training_profile="smoke_train_v1",
        evaluation_profile="smoke_eval_v1",
        evaluation_case_count=64,
        device="mps",
        allow_cpu_for_tests=False,
        pi_decision_record_id=None,
        eligibility_threshold=0.0,
        engineering_pilot=True,
    )


def _lineage(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_ten_zero_accuracy_ir_opportunities_are_accounted_without_api_or_mps(tmp_path):
    provider = FakeProvider([_ir(index) for index in range(1, 11)])
    evaluator = FakeEvaluator(zero_accuracy=True)

    summary = run_semantic_autoresearch(
        _options(tmp_path, iterations=10),
        provider=provider,
        evaluator=evaluator,
    )

    assert provider.preflight_calls == 1
    assert provider.generate_calls == 10
    assert len(evaluator.requests) == 11
    assert summary["proposal_opportunities_requested"] == 10
    assert summary["proposal_opportunities_terminal"] == 10
    records = _lineage(tmp_path / "run" / "lineage.jsonl")
    assert len(records) == 11
    assert [record["opportunity_index"] for record in records] == list(range(11))
    assert all(record["public_accuracy"] == 0.0 for record in records)
    assert all(request.candidate_path.suffix == ".json" for request in evaluator.requests)
    assert not list((tmp_path / "run" / "artifacts").glob("*.py"))

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["run_mode"] == "engineering_pilot"
    assert manifest["training"]["profile"] == "smoke_train_v1"
    assert manifest["evaluation"]["profile"] == "smoke_eval_v1"
    assert manifest["training"]["device"] == "mps"
    assert manifest["evaluation"]["eligibility_threshold"] == 0.0
    assert manifest["authoritative_scientific_evidence"] is False
    assert manifest["candidate_format"] == "architecture_tensor_graph@1.0"
    assert manifest["architecture_hash_schema"] == "architecture_executable_v2"
    assert manifest["selection_semantics"] == "mechanics_only_transformer_validity"
    assert manifest["architecture_deduplication"]["duplicate_proposals_train"] is False
    assert (
        manifest["trusted_executable_component_hashes"]
        == semantic_run.trusted_component_hashes()
    )
    assert (
        manifest["trusted_component_set_sha256"]
        == semantic_run.trusted_component_set_sha256(
            manifest["trusted_executable_component_hashes"]
        )
    )
    assert "architecture_ir_contract" in {
        component["name"] for component in manifest["prompt_protocol"]["components"]
    }
    assert "softmax_mix" in provider.messages[0][0]["content"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("import subprocess", "invalid architecture IR"),
        ("before\n```json\n{}\n```", "malformed JSON fence"),
        (_ir(1, hypothesis=False), "mechanism_hypothesis"),
    ],
)
def test_non_ir_or_malformed_response_never_reaches_evaluator(
    tmp_path,
    response,
    message,
):
    provider = FakeProvider([response])
    evaluator = FakeEvaluator()

    run_semantic_autoresearch(
        _options(tmp_path),
        provider=provider,
        evaluator=evaluator,
    )

    assert len(evaluator.requests) == 1
    failure = _lineage(tmp_path / "run" / "lineage.jsonl")[-1]
    assert failure["failure_stage"] == "ir_validation"
    assert message in failure["error"]
    assert not list((tmp_path / "run" / "artifacts").glob("0001_*.ir.json"))


def test_oversized_response_is_rejected_before_training():
    with pytest.raises(IRProposalError, match="controller limit"):
        semantic_run._validated_ir_text(
            " " * 257,
            max_ir_bytes=256,
            require_hypothesis=True,
        )


def test_json_fence_is_accepted_and_saved_as_canonical_ir(tmp_path):
    provider = FakeProvider([f"```json\n{_ir(1)}\n```"])
    evaluator = FakeEvaluator()

    run_semantic_autoresearch(
        _options(tmp_path),
        provider=provider,
        evaluator=evaluator,
    )

    child = next((tmp_path / "run" / "artifacts").glob("0001_*.ir.json"))
    assert child.read_text() == _ir(1)
    assert len(evaluator.requests) == 2


def test_metadata_only_proposal_is_rejected_without_training(tmp_path):
    parent = decode_graph_json(_ir(0, hypothesis=False))
    response = encode_graph_json(
        replace(
            parent,
            graph_id="semantic.metadata.only",
            metadata={
                **parent.metadata,
                "mechanism_hypothesis": "prose only",
            },
        )
    )
    evaluator = FakeEvaluator()

    run_semantic_autoresearch(
        _options(tmp_path),
        provider=FakeProvider([response]),
        evaluator=evaluator,
    )

    assert len(evaluator.requests) == 1
    failure = _lineage(tmp_path / "run" / "lineage.jsonl")[-1]
    assert failure["failure_stage"] == "mutation_no_change"


def test_invalid_noncausal_graph_fails_before_candidate_evaluation(tmp_path):
    graph = decode_graph_json(_ir(9))
    attention = next(node for node in graph.nodes if node.kind.value == "attention")
    graph = replace(
        graph,
        graph_id="semantic.invalid.noncausal",
        nodes=tuple(
            replace(attention, attributes={**attention.attributes, "causal": False})
            if node.node_id == attention.node_id
            else node
            for node in graph.nodes
        ),
    )
    provider = FakeProvider([encode_graph_json(graph)])
    evaluator = FakeEvaluator()

    run_semantic_autoresearch(
        _options(tmp_path),
        provider=provider,
        evaluator=evaluator,
    )

    assert len(evaluator.requests) == 1
    failure = _lineage(tmp_path / "run" / "lineage.jsonl")[-1]
    assert failure["failure_stage"] == "ir_validation"
    assert "attention_not_causal" in failure["error"]


def test_ineligible_candidate_never_becomes_a_semantic_parent(tmp_path):
    provider = FakeProvider([_ir(1), _ir(2)])
    evaluator = FakeEvaluator(
        [
            _view("seed", score=0.5),
            _view("rejected", eligible=False),
            _view("accepted", score=0.6, attention_organization=3),
        ]
    )

    run_semantic_autoresearch(
        _options(tmp_path, iterations=2),
        provider=provider,
        evaluator=evaluator,
    )

    second_prompt = provider.messages[1][-1]["content"]
    assert _ir(0, hypothesis=False) in second_prompt
    assert _ir(1) not in second_prompt
    records = _lineage(tmp_path / "run" / "lineage.jsonl")
    assert records[1]["retention_decision"] == "reject"
    assert records[2]["parent_id"] == records[0]["candidate_id"]


def test_run_wide_duplicate_is_charged_without_retraining(tmp_path):
    provider = FakeProvider([_ir(1), _ir(2), _ir(1)])
    evaluator = FakeEvaluator()

    summary = run_semantic_autoresearch(
        _options(tmp_path, iterations=3),
        provider=provider,
        evaluator=evaluator,
    )

    assert summary["proposal_opportunities_terminal"] == 3
    assert len(evaluator.requests) == 3  # seed plus two unique children
    duplicate = _lineage(tmp_path / "run" / "lineage.jsonl")[-1]
    assert duplicate["failure_stage"] == "duplicate_architecture"
    assert duplicate["retention_decision"] == "duplicate_rejected"


def test_provider_backed_engineering_options_require_mps_but_injected_cpu_is_allowed(
    tmp_path,
):
    options = replace(
        _options(tmp_path),
        device="cpu",
        allow_cpu_for_tests=True,
    )
    with pytest.raises(ValueError, match="provider-backed engineering pilots require"):
        semantic_run._validate_options(options)

    summary = run_semantic_autoresearch(
        options,
        provider=FakeProvider([_ir(1)]),
        evaluator=FakeEvaluator(),
    )
    assert summary["proposal_opportunities_terminal"] == 1


def test_seed_must_be_eligible_before_any_provider_call(tmp_path):
    provider = FakeProvider([_ir(1)])
    evaluator = FakeEvaluator([_view("seed", eligible=False)])

    with pytest.raises(PilotPreflightError, match="no paid proposal call"):
        run_semantic_autoresearch(
            _options(tmp_path),
            provider=provider,
            evaluator=evaluator,
        )

    assert provider.generate_calls == 0


def test_stale_evaluation_binding_cannot_enter_archive(tmp_path):
    provider = FakeProvider([_ir(1)])
    evaluator = FakeEvaluator([_view("seed"), _view("stale")])
    evaluator.bind = False

    with pytest.raises(PilotPreflightError, match="before any paid proposal call"):
        run_semantic_autoresearch(
            _options(tmp_path),
            provider=provider,
            evaluator=evaluator,
        )

    assert provider.generate_calls == 0


def test_archive_parent_policy_uses_coverage_and_accuracy_not_category_order(tmp_path):
    archive = FrozenSemanticArchive()
    first = _view("first", score=0.6, attention_organization=3)
    second = _view("second", score=0.9, attention_organization=2)
    archive.consider(
        candidate_id="candidate-first",
        lineage_record_id="lineage-first",
        source_path=tmp_path / "first.ir.json",
        view=first,
        opportunity=1,
    )
    archive.consider(
        candidate_id="candidate-second",
        lineage_record_id="lineage-second",
        source_path=tmp_path / "second.ir.json",
        view=second,
        opportunity=2,
    )

    assert archive.select_parent().candidate_id == "candidate-second"
    assert archive.select_parent().candidate_id == "candidate-first"


def test_injected_evaluator_requires_controller_only_test_boundary(tmp_path):
    with pytest.raises(PilotPreflightError, match="controller-only test double"):
        run_semantic_autoresearch(
            _options(tmp_path),
            provider=FakeProvider([_ir(1)]),
            evaluator=lambda request: pytest.fail("must not evaluate"),
        )


def test_cli_engineering_pilot_forces_smoke_profiles_zero_threshold_and_mps(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = []
    provider = FakeProvider([])
    monkeypatch.setenv("DISCOVERY_LAYER_A_CASES", "10000")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semantic",
            "--iterations",
            "1",
            "--engineering-pilot",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )
    monkeypatch.setattr(
        semantic_run,
        "_preflight_default_evaluator",
        lambda options: captured.append(options),
    )
    monkeypatch.setattr(
        semantic_run,
        "_provider_from_environment",
        lambda config, seed, **kwargs: provider,
    )
    monkeypatch.setattr(
        semantic_run,
        "run_semantic_autoresearch",
        lambda options, provider: captured.append(options) or {"ok": True},
    )

    semantic_run.main()

    assert len(captured) == 2
    assert all(option.engineering_pilot for option in captured)
    assert all(option.training_profile == "smoke_train_v1" for option in captured)
    assert all(option.evaluation_profile == "smoke_eval_v1" for option in captured)
    assert all(option.device == "mps" for option in captured)
    assert all(option.eligibility_threshold == 0.0 for option in captured)
    assert all(option.evaluation_case_count is None for option in captured)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_engineering_profiles_require_explicit_pilot_flag(tmp_path):
    with pytest.raises(ValueError, match="explicit --engineering-pilot"):
        semantic_run._validate_options(
            replace(_options(tmp_path), engineering_pilot=False)
        )


def test_cli_scientific_mode_preserves_frozen_point99_threshold(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = []
    provider = FakeProvider([])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semantic",
            "--iterations",
            "1",
            "--evaluation-cases",
            "10000",
            "--output-dir",
            str(tmp_path / "scientific"),
        ],
    )
    monkeypatch.setattr(
        semantic_run,
        "_preflight_default_evaluator",
        lambda options: captured.append(options),
    )
    monkeypatch.setattr(
        semantic_run,
        "_provider_from_environment",
        lambda config, seed, **kwargs: provider,
    )
    monkeypatch.setattr(
        semantic_run,
        "run_semantic_autoresearch",
        lambda options, provider: captured.append(options) or {"ok": True},
    )

    semantic_run.main()

    assert all(not option.engineering_pilot for option in captured)
    assert all(option.training_profile == "full_train_v1" for option in captured)
    assert all(option.evaluation_profile == "scientific_layer_a_v1" for option in captured)
    assert all(option.eligibility_threshold == 0.99 for option in captured)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_no_generated_python_containment_bypass_remains():
    source = Path(semantic_run.__file__).read_text(encoding="utf-8")
    assert "_require_generated_code_boundary" not in source
    assert "containment.audit" not in source
    assert "SEARCH/REPLACE" not in source
    config = (Path(semantic_run.AGENT_DIR) / "config.yaml").read_text(encoding="utf-8")
    assert "candidate_path: common/initial_candidate.ir.json" in config
