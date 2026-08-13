from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agents.greedy_autoresearch import run
from agents.greedy_autoresearch.run import (
    IRProposalError,
    PilotPreflightError,
    ProposalResponse,
    RunOptions,
    run_greedy_autoresearch,
)
from architecture_ir import decode_graph_json, encode_graph_json
from common.evaluator import file_hash
from evaluation.records import ControllerSearchView, SCHEMA_VERSION


INITIAL_IR = Path(run.ROOT / "common" / "initial_candidate.ir.json")


def _ir(index: int, *, hypothesis: bool = True) -> str:
    base = decode_graph_json(INITIAL_IR.read_text(encoding="utf-8"))
    metadata = dict(base.metadata)
    if hypothesis:
        metadata["mechanism_hypothesis"] = f"mechanism {index}"
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
    graph = replace(
        base,
        graph_id=f"greedy.candidate.{index}",
        nodes=nodes,
        metadata=metadata,
    )
    return encode_graph_json(graph)


def _options(tmp_path: Path, *, iterations: int = 1) -> RunOptions:
    initial = tmp_path / "initial.ir.json"
    initial.write_text(_ir(0, hypothesis=False), encoding="utf-8")
    return RunOptions(
        iterations=iterations,
        seed=3,
        output_dir=tmp_path / "run",
        initial_candidate=initial,
        training_profile="smoke_train_cuda_v2",
        evaluation_profile="smoke_eval_v1",
        evaluation_case_count=64,
        device="cuda",
        allow_cpu_for_tests=False,
        pi_decision_record_id=None,
        eligibility_threshold=0.0,
        engineering_pilot=True,
    )


class FakeProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.generate_calls = 0
        self.messages = []
        self.preflight_calls = 0

    def preflight(self):
        self.preflight_calls += 1

    def generate(self, messages):
        self.generate_calls += 1
        self.messages.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return ProposalResponse(response, input_tokens=11, output_tokens=7)

    def manifest_fields(self):
        return {"model": "fake-ir-provider", "paid_calls": False}


class FakeEvaluator:
    controller_only_test_double = True

    def __init__(self, *, eligible=True, scores=None):
        self.eligible = eligible
        self.scores = list(scores) if scores is not None else None
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        request.candidate_path.read_text(encoding="utf-8")
        source_hash = file_hash(request.candidate_path)
        index = len(self.requests)
        eligible = self.eligible if isinstance(self.eligible, bool) else self.eligible[index - 1]
        score = 0.0 if self.scores is None else self.scores[index - 1]
        return ControllerSearchView(
            schema_name="search_evaluation",
            schema_version=SCHEMA_VERSION,
            record_id=f"record-{index}-{source_hash}",
            run_id=request.context.run_id,
            condition_id=request.context.condition_id,
            candidate_id=f"candidate-{source_hash}",
            execution_ok=True,
            transformer_valid=True,
            public_accuracy=score,
            search_score=score,
            eligible_for_parent=eligible,
            failure_stage="" if eligible else "public_accuracy",
            infrastructure_failure=False,
            online_descriptor_codes=(),
        )


def _lineage(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_ten_ir_opportunities_use_ten_provider_calls_and_eleven_evaluations(tmp_path):
    provider = FakeProvider([_ir(index) for index in range(1, 11)])
    evaluator = FakeEvaluator()

    summary = run_greedy_autoresearch(
        _options(tmp_path, iterations=10),
        provider=provider,
        evaluator=evaluator,
    )

    assert provider.preflight_calls == 1
    assert provider.generate_calls == 10
    assert len(evaluator.requests) == 11
    assert summary["proposal_opportunities_requested"] == 10
    assert summary["proposal_opportunities_terminal"] == 10
    assert summary["lineage_path"] == "lineage.jsonl"
    assert summary["incumbent_path"] == "incumbent.ir.json"
    records = _lineage(tmp_path / "run" / "lineage.jsonl")
    assert len(records) == 11
    assert [record["proposal_opportunity"] for record in records] == list(range(11))
    assert all(record["public_accuracy"] == 0.0 for record in records)
    assert all(request.candidate_path.suffix == ".json" for request in evaluator.requests)
    assert not list((tmp_path / "run" / "artifacts").glob("*.py"))

    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["schema_name"] == "ControllerRunManifest"
    assert manifest["schema_version"] == "2.0"
    assert summary["schema_name"] == "ControllerRunSummary"
    assert summary["schema_version"] == "2.0"
    assert manifest["run_mode"] == "engineering_pilot"
    assert manifest["training"]["profile"] == "smoke_train_cuda_v2"
    assert manifest["evaluation"]["profile"] == "smoke_eval_v1"
    assert manifest["training"]["device"] == "cuda"
    assert manifest["authoritative_scientific_evidence"] is False
    assert manifest["candidate_format"] == "architecture_tensor_graph@1.0"
    assert manifest["architecture_hash_schema"] == "architecture_executable_v2"
    assert manifest["selection_semantics"] == "mechanics_only_transformer_validity"
    assert manifest["greedy_retention"]["rejects_search_score_regressions"] is True
    assert manifest["architecture_deduplication"]["duplicate_proposals_train"] is False
    assert manifest["trusted_executable_component_hashes"] == run.trusted_component_hashes()
    assert manifest["trusted_component_set_sha256"] == run.trusted_component_set_sha256(
        manifest["trusted_executable_component_hashes"]
    )
    assert "architecture_ir_contract" in {
        component["name"] for component in manifest["prompt_protocol"]["components"]
    }
    assert "softmax_mix" in provider.messages[0][0]["content"]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ("import os\nos.system('id')", "invalid architecture IR"),
        ("prose\n```json\n{}\n```", "malformed JSON fence"),
        (_ir(1, hypothesis=False), "mechanism_hypothesis"),
    ],
)
def test_malformed_or_non_ir_response_fails_before_candidate_evaluation(
    tmp_path,
    response,
    message,
):
    provider = FakeProvider([response])
    evaluator = FakeEvaluator()

    summary = run_greedy_autoresearch(
        _options(tmp_path),
        provider=provider,
        evaluator=evaluator,
    )

    assert summary["proposal_opportunities_terminal"] == 1
    assert len(evaluator.requests) == 1
    record = _lineage(tmp_path / "run" / "lineage.jsonl")[-1]
    assert record["failure_stage"] == "ir_validation"
    assert message in record["error"]
    assert not list((tmp_path / "run" / "artifacts").glob("*.ir.json"))


def test_oversized_ir_response_fails_closed_before_training(tmp_path):
    options = replace(_options(tmp_path), max_ir_bytes=128)
    # The seed is larger than 128 bytes, so use the parser directly to isolate
    # the untrusted-response byte gate.
    with pytest.raises(IRProposalError, match="controller limit"):
        run._validated_ir_text(
            " " * 129,
            max_ir_bytes=128,
            require_hypothesis=True,
        )
    assert options.max_ir_bytes == 128


def test_single_json_fence_is_canonicalized_and_evaluated(tmp_path):
    provider = FakeProvider([f"```json\n{_ir(1)}\n```"])
    evaluator = FakeEvaluator()

    run_greedy_autoresearch(_options(tmp_path), provider=provider, evaluator=evaluator)

    child = next((tmp_path / "run" / "artifacts").glob("*.ir.json"))
    assert child.read_text() == _ir(1)
    assert len(evaluator.requests) == 2


def test_metadata_only_proposal_is_rejected_without_training(tmp_path):
    parent = decode_graph_json(_ir(0, hypothesis=False))
    response = encode_graph_json(
        replace(
            parent,
            graph_id="greedy.metadata.only",
            metadata={
                **parent.metadata,
                "mechanism_hypothesis": "prose only",
            },
        )
    )
    evaluator = FakeEvaluator()

    run_greedy_autoresearch(
        _options(tmp_path),
        provider=FakeProvider([response]),
        evaluator=evaluator,
    )

    assert len(evaluator.requests) == 1
    failure = _lineage(tmp_path / "run" / "lineage.jsonl")[-1]
    assert failure["failure_stage"] == "mutation_no_change"


def test_invalid_graph_fails_static_validation_before_evaluator(tmp_path):
    graph = decode_graph_json(_ir(9))
    attention = next(node for node in graph.nodes if node.kind.value == "attention")
    graph = replace(
        graph,
        graph_id="greedy.invalid.noncausal",
        nodes=tuple(
            replace(attention, attributes={**attention.attributes, "causal": False})
            if node.node_id == attention.node_id
            else node
            for node in graph.nodes
        ),
    )
    provider = FakeProvider([encode_graph_json(graph)])
    evaluator = FakeEvaluator()

    run_greedy_autoresearch(_options(tmp_path), provider=provider, evaluator=evaluator)

    assert len(evaluator.requests) == 1
    failure = _lineage(tmp_path / "run" / "lineage.jsonl")[-1]
    assert failure["failure_stage"] == "ir_validation"
    assert "attention_not_causal" in failure["error"]


def test_ineligible_ir_never_replaces_the_incumbent(tmp_path):
    provider = FakeProvider([_ir(1), _ir(2)])
    evaluator = FakeEvaluator(eligible=[True, False, True])

    run_greedy_autoresearch(
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


def test_eligible_score_regression_does_not_replace_incumbent(tmp_path):
    provider = FakeProvider([_ir(1), _ir(2)])
    evaluator = FakeEvaluator(scores=[0.8, 0.7, 0.9])

    run_greedy_autoresearch(
        _options(tmp_path, iterations=2),
        provider=provider,
        evaluator=evaluator,
    )

    records = _lineage(tmp_path / "run" / "lineage.jsonl")
    assert records[1]["retention_decision"] == "reject_score_regression"
    assert records[2]["parent_id"] == records[0]["candidate_id"]
    assert _ir(0, hypothesis=False) in provider.messages[1][-1]["content"]


def test_run_wide_duplicate_is_charged_without_retraining(tmp_path):
    provider = FakeProvider([_ir(1), _ir(2), _ir(1)])
    evaluator = FakeEvaluator()

    summary = run_greedy_autoresearch(
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
        run._validate_options(options)

    summary = run_greedy_autoresearch(
        options,
        provider=FakeProvider([_ir(1)]),
        evaluator=FakeEvaluator(),
    )
    assert summary["proposal_opportunities_terminal"] == 1


def test_seed_failure_stops_before_provider_call(tmp_path):
    provider = FakeProvider([_ir(1)])
    evaluator = FakeEvaluator(eligible=False)

    with pytest.raises(PilotPreflightError, match="no provider call"):
        run_greedy_autoresearch(
            _options(tmp_path),
            provider=provider,
            evaluator=evaluator,
        )

    assert provider.generate_calls == 0


def test_injected_evaluator_requires_explicit_controller_only_boundary(tmp_path):
    provider = FakeProvider([_ir(1)])

    with pytest.raises(PilotPreflightError, match="controller-only test double"):
        run_greedy_autoresearch(
            _options(tmp_path),
            provider=provider,
            evaluator=lambda request: pytest.fail("must not evaluate"),
        )


def test_cli_engineering_pilot_forces_smoke_profiles_and_cuda(
    tmp_path,
    monkeypatch,
    capsys,
):
    captured = []
    fake_provider = FakeProvider([])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "greedy",
            "--iterations",
            "1",
            "--engineering-pilot",
            "--output-dir",
            str(tmp_path / "run"),
        ],
    )
    monkeypatch.setattr(run, "_preflight_default_evaluator", lambda options: captured.append(options))
    monkeypatch.setattr(
        run,
        "_provider_from_environment",
        lambda config, seed, **kwargs: fake_provider,
    )
    monkeypatch.setattr(
        run,
        "run_greedy_autoresearch",
        lambda options, provider: captured.append(options) or {"ok": True},
    )

    run.main()

    assert len(captured) == 2
    assert all(option.engineering_pilot for option in captured)
    assert all(
        option.training_profile == "smoke_train_cuda_v2" for option in captured
    )
    assert all(option.evaluation_profile == "smoke_eval_v1" for option in captured)
    assert all(option.device == "cuda" for option in captured)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_engineering_profiles_require_explicit_pilot_flag(tmp_path):
    options = replace(_options(tmp_path), engineering_pilot=False)
    with pytest.raises(ValueError, match="explicit --engineering-pilot"):
        run._validate_options(options)


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
            "greedy",
            "--iterations",
            "1",
            "--evaluation-cases",
            "10000",
            "--output-dir",
            str(tmp_path / "scientific"),
        ],
    )
    monkeypatch.setattr(run, "_preflight_default_evaluator", lambda options: captured.append(options))
    monkeypatch.setattr(
        run,
        "_provider_from_environment",
        lambda config, seed, **kwargs: provider,
    )
    monkeypatch.setattr(
        run,
        "run_greedy_autoresearch",
        lambda options, provider: captured.append(options) or {"ok": True},
    )

    run.main()

    assert all(not option.engineering_pilot for option in captured)
    assert all(
        option.training_profile == "full_train_cuda_v2" for option in captured
    )
    assert all(option.evaluation_profile == "scientific_layer_a_v1" for option in captured)
    assert all(option.eligibility_threshold == 0.99 for option in captured)
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_no_generated_python_containment_bypass_remains():
    source = Path(run.__file__).read_text(encoding="utf-8")
    assert "_require_generated_code_boundary" not in source
    assert "containment.audit" not in source
    assert "SEARCH/REPLACE" not in source
    config = (Path(run.AGENT_DIR) / "config.yaml").read_text(encoding="utf-8")
    assert "candidate_path: common/initial_candidate.ir.json" in config
