import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from architecture_ir import (
    DEFAULT_LIMITS,
    IRInterpreterError,
    RuntimeBindings,
    load_and_build_ir_candidate,
    probe_fresh_build,
    probe_runtime_validity,
    validate_ir_candidate_json,
    validate_ir_candidate_path,
)
from architecture_ir.codec import MAX_IR_JSON_BYTES


ROOT = Path(__file__).parents[1]
INITIAL_IR = ROOT / "common" / "initial_candidate.ir.json"


def _initial_payload() -> dict:
    return json.loads(INITIAL_IR.read_text(encoding="utf-8"))


def _node(payload: dict, node_id: str) -> dict:
    return next(node for node in payload["nodes"] if node["node_id"] == node_id)


def _variation_payload() -> dict:
    hidden = ["Batch", "Time", 8]
    return {
        "schema_name": "architecture_tensor_graph",
        "schema_version": "1.0",
        "graph_id": "trusted_variation_v1",
        "input_node_id": "tokens",
        "output_node_id": "readout",
        "nodes": [
            {
                "node_id": "tokens",
                "kind": "input",
                "input_shapes": [],
                "output_shape": ["Batch", "Time"],
                "attributes": {},
            },
            {
                "node_id": "embedding",
                "kind": "token_embedding",
                "input_shapes": [["Batch", "Time"]],
                "output_shape": hidden,
                "attributes": {"vocab": 15},
            },
            {
                "node_id": "position",
                "kind": "positional",
                "input_shapes": [hidden],
                "output_shape": hidden,
                "attributes": {"mechanism": "sinusoidal"},
            },
            {
                "node_id": "norm",
                "kind": "normalization",
                "input_shapes": [hidden],
                "output_shape": hidden,
                "attributes": {
                    "mechanism": "rms_norm",
                    "epsilon": 1e-6,
                    "affine": True,
                },
            },
            {
                "node_id": "attention",
                "kind": "attention",
                "input_shapes": [hidden],
                "output_shape": hidden,
                "attributes": {"causal": True, "heads": 2, "bias": True},
            },
            {
                "node_id": "gated_ff",
                "kind": "feed_forward",
                "input_shapes": [hidden],
                "output_shape": hidden,
                "attributes": {
                    "mechanism": "gated",
                    "hidden_dimension": 16,
                    "activation": "silu",
                    "bias": True,
                },
            },
            {
                "node_id": "router",
                "kind": "routing",
                "input_shapes": [hidden, hidden],
                "output_shape": hidden,
                "attributes": {"mechanism": "softmax_mix", "temperature": 0.75},
            },
            {
                "node_id": "gate",
                "kind": "algebraic",
                "input_shapes": [hidden, hidden],
                "output_shape": hidden,
                "attributes": {"mechanism": "learned_gate"},
            },
            {
                "node_id": "composition",
                "kind": "composition",
                "input_shapes": [hidden, hidden],
                "output_shape": hidden,
                "attributes": {"mechanism": "concat_project", "bias": True},
            },
            {
                "node_id": "readout",
                "kind": "readout",
                "input_shapes": [hidden],
                "output_shape": ["Batch", "Time", 15],
                "attributes": {"vocab": 15, "bias": True},
            },
        ],
        "edges": [
            {"source": "tokens", "target": "embedding", "target_port": 0, "kind": "data"},
            {"source": "embedding", "target": "position", "target_port": 0, "kind": "data"},
            {"source": "position", "target": "norm", "target_port": 0, "kind": "data"},
            {"source": "norm", "target": "attention", "target_port": 0, "kind": "data"},
            {"source": "norm", "target": "gated_ff", "target_port": 0, "kind": "data"},
            {"source": "attention", "target": "router", "target_port": 0, "kind": "routing"},
            {"source": "gated_ff", "target": "router", "target_port": 1, "kind": "routing"},
            {"source": "position", "target": "gate", "target_port": 0, "kind": "residual"},
            {"source": "router", "target": "gate", "target_port": 1, "kind": "routing"},
            {"source": "gate", "target": "composition", "target_port": 0, "kind": "data"},
            {"source": "norm", "target": "composition", "target_port": 1, "kind": "data"},
            {"source": "composition", "target": "readout", "target_port": 0, "kind": "data"},
        ],
        "metadata": {
            "max_sequence_length": 35,
            "vocab_size": 15,
            "mechanism_hypothesis": "exercise evaluator-owned variation primitives",
        },
    }


def test_initial_ir_builds_equivalent_conventional_decoder_on_cpu():
    validation = validate_ir_candidate_path(INITIAL_IR)
    assert validation.valid, validation.to_dict()
    assert validation.estimated_parameter_count == 6080
    assert (
        validation.graph.legacy_architecture_hash
        == "2a767d7b5c3110ce04db2a3a87f2808f1618e0a6649655a7e6d5a19853de0dd8"
    )
    assert validation.graph.architecture_hash != validation.graph.legacy_architecture_hash

    candidate = load_and_build_ir_candidate(INITIAL_IR, 17)
    assert candidate.graph is validation.graph or candidate.graph.graph_hash == validation.graph_hash
    assert candidate.validation.valid
    assert candidate.metadata["parameter_count"] == 6080
    assert candidate.metadata["parameter_count_role"] == "descriptive_metadata_only"
    assert candidate.metadata["execution_provenance"] == "trusted_ir_interpreter"
    assert (
        candidate.metadata["architecture_ir_architecture_hash_schema"]
        == candidate.graph.architecture_hash_schema
    )
    assert not candidate.bindings.validate()
    assert set(candidate.bindings.attention_modules) == {
        "block1_attention",
        "block2_attention",
    }
    assert {parameter.device.type for parameter in candidate.model.parameters()} == {"cpu"}
    assert {buffer.device.type for buffer in candidate.model.buffers()} == {"cpu"}

    token_ids = torch.randint(0, 15, (2, 35), dtype=torch.long)
    logits = candidate.model(token_ids)
    assert logits.shape == (2, 35, 15)


def test_interpreter_build_is_seed_deterministic_and_preserves_global_rng():
    before = torch.get_rng_state().clone()
    first = load_and_build_ir_candidate(INITIAL_IR, 123).model
    after = torch.get_rng_state()
    repeated = load_and_build_ir_candidate(INITIAL_IR, 123).model
    changed = load_and_build_ir_candidate(INITIAL_IR, 124).model

    assert torch.equal(before, after)
    first_state = first.state_dict()
    repeated_state = repeated.state_dict()
    changed_state = changed.state_dict()
    assert first is not repeated
    assert all(torch.equal(first_state[key], repeated_state[key]) for key in first_state)
    assert any(not torch.equal(first_state[key], changed_state[key]) for key in first_state)

    evidence = probe_fresh_build(
        lambda seed: load_and_build_ir_candidate(INITIAL_IR, seed).model,
        seed=91,
    )
    assert evidence.passed, evidence.to_dict()


def test_initial_ir_is_causal_and_produces_runtime_attention_bindings():
    candidate = load_and_build_ir_candidate(INITIAL_IR, 5)
    candidate.model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    second = torch.tensor([[1, 2, 3, 4, 6]], dtype=torch.long)
    with torch.no_grad():
        first_logits = candidate.model(first)
        second_logits = candidate.model(second)
    assert torch.allclose(first_logits[:, :-1], second_logits[:, :-1], atol=1e-6)

    runtime = probe_runtime_validity(
        candidate.model,
        bindings=candidate.bindings,
        token_ids=first,
        expected_device="cpu",
    )
    assert runtime.passed, runtime.to_dict()
    assert runtime.checks["causal_mask_buffer_observed"]
    assert set(runtime.attention_intervention_max_deltas) == {
        "block1_attention",
        "block2_attention",
    }
    assert all(
        delta > 1e-8
        for delta in runtime.attention_intervention_max_deltas.values()
    )


def test_runtime_binding_graph_hash_must_match_interpreted_model():
    candidate = load_and_build_ir_candidate(INITIAL_IR, 5)
    forged_bindings = replace(candidate.bindings, graph_hash="0" * 64)
    evidence = probe_runtime_validity(
        candidate.model,
        bindings=forged_bindings,
        token_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        expected_device="cpu",
    )

    assert not evidence.passed
    assert not evidence.checks["graph_identity"]
    assert evidence.observed_model_graph_hash == candidate.graph.graph_hash
    assert any("does not match" in error for error in evidence.errors)


def test_variation_primitives_build_train_and_preserve_logits_contract(tmp_path):
    path = tmp_path / "variation.json"
    path.write_text(json.dumps(_variation_payload()), encoding="utf-8")
    validation = validate_ir_candidate_path(path)
    assert validation.valid, validation.to_dict()

    candidate = load_and_build_ir_candidate(path, 44)
    token_ids = torch.randint(0, 15, (3, 8), dtype=torch.long)
    logits = candidate.model(token_ids)
    assert logits.shape == (3, 8, 15)
    logits.square().mean().backward()
    influenced = [
        parameter
        for parameter in candidate.model.parameters()
        if parameter.grad is not None and torch.count_nonzero(parameter.grad).item()
    ]
    assert influenced


def test_fixed_routing_sigmoid_gate_and_identity_are_supported():
    payload = _variation_payload()
    router = _node(payload, "router")
    router["attributes"] = {"mechanism": "fixed_mix", "weights": [0.25, 0.75]}
    gate = _node(payload, "gate")
    gate["attributes"] = {"mechanism": "sigmoid_gate"}
    gate["input_shapes"].append(["Batch", "Time", 8])
    payload["edges"].append(
        {"source": "norm", "target": "gate", "target_port": 2, "kind": "data"}
    )
    composition = _node(payload, "composition")
    composition["attributes"] = {"mechanism": "identity"}
    composition["input_shapes"] = [["Batch", "Time", 8]]
    payload["edges"] = [
        edge
        for edge in payload["edges"]
        if not (edge["target"] == "composition" and edge["target_port"] == 1)
    ]
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert validation.valid, validation.to_dict()


def test_proportional_fixed_mix_weights_share_executable_identity(tmp_path):
    first_payload = _variation_payload()
    _node(first_payload, "router")["attributes"] = {
        "mechanism": "fixed_mix",
        "weights": [1, 3],
    }
    scaled_payload = json.loads(json.dumps(first_payload))
    _node(scaled_payload, "router")["attributes"]["weights"] = [2.0, 6.0]

    first_validation = validate_ir_candidate_json(json.dumps(first_payload))
    scaled_validation = validate_ir_candidate_json(json.dumps(scaled_payload))
    assert first_validation.valid, first_validation.to_dict()
    assert scaled_validation.valid, scaled_validation.to_dict()
    assert first_validation.graph_hash != scaled_validation.graph_hash
    assert first_validation.architecture_hash == scaled_validation.architecture_hash

    first_path = tmp_path / "first-fixed-mix.json"
    scaled_path = tmp_path / "scaled-fixed-mix.json"
    first_path.write_text(json.dumps(first_payload), encoding="utf-8")
    scaled_path.write_text(json.dumps(scaled_payload), encoding="utf-8")
    first_state = load_and_build_ir_candidate(first_path, 13).model.state_dict()
    scaled_state = load_and_build_ir_candidate(scaled_path, 13).model.state_dict()
    assert first_state.keys() == scaled_state.keys()
    assert all(
        torch.equal(first_state[name], scaled_state[name]) for name in first_state
    )


@pytest.mark.parametrize(
    "weights",
    (
        [1e-50, 1.0],
        [3e38, 3e38],
        [1e-45, 1e38],
        [10**400, 1],
    ),
)
def test_fixed_mix_rejects_unsafe_float32_normalization(weights):
    payload = _variation_payload()
    _node(payload, "router")["attributes"] = {
        "mechanism": "fixed_mix",
        "weights": weights,
    }

    validation = validate_ir_candidate_json(json.dumps(payload))

    assert not validation.valid
    assert any(
        issue.code == "invalid_primitive_attribute" and "fixed_mix" in issue.message
        for issue in validation.issues
    )


def test_valid_build_contains_only_finite_parameters_and_buffers(tmp_path):
    payload = _variation_payload()
    _node(payload, "router")["attributes"] = {
        "mechanism": "fixed_mix",
        "weights": [0.25, 0.75],
    }
    path = tmp_path / "finite-fixed-mix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    model = load_and_build_ir_candidate(path, 7).model

    assert all(
        not (torch.is_floating_point(tensor) or torch.is_complex(tensor))
        or bool(torch.isfinite(tensor).all().item())
        for tensor in (*model.parameters(), *model.buffers())
    )


@pytest.mark.parametrize("kind", ["recurrent", "state", "custom"])
def test_stateful_and_custom_primitives_fail_closed(kind):
    payload = _initial_payload()
    node = _node(payload, "block1_feed_forward")
    node["kind"] = kind
    if kind == "custom":
        node["attributes"] = {
            "trusted_primitive": "candidate_owned",
            "primitive_version": "1",
        }
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "unsupported_primitive" in {issue.code for issue in validation.issues}


def test_state_and_recurrent_edges_fail_closed():
    payload = _initial_payload()
    edge = next(edge for edge in payload["edges"] if edge["target"] == "block1_attention")
    edge["kind"] = "state"
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "unsupported_edge" in {issue.code for issue in validation.issues}


def test_unknown_attributes_mechanisms_and_dynamic_payload_fields_are_rejected():
    payload = _initial_payload()
    _node(payload, "block1_attention")["attributes"]["module_path"] = "evil.Attention"
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "executable_attribute" in {issue.code for issue in validation.issues}

    payload = _initial_payload()
    _node(payload, "block1_feed_forward")["attributes"]["dropout"] = 0.5
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "unknown_primitive_attribute" in {issue.code for issue in validation.issues}

    payload = _initial_payload()
    payload["candidate_python"] = "exec('unsafe')"
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert validation.issues[0].code == "json_decode"


def test_candidate_strings_are_inert_data_and_never_executed(tmp_path):
    marker = tmp_path / "must_not_exist"
    payload = _initial_payload()
    payload["metadata"]["mechanism_hypothesis"] = (
        f"__import__('pathlib').Path({str(marker)!r}).write_text('owned')"
    )
    path = tmp_path / "inert.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    candidate = load_and_build_ir_candidate(path, 1)
    assert not marker.exists()
    assert "__import__" in candidate.metadata["mechanism_hypothesis"]


def test_fixed_vocabulary_and_logits_shape_are_enforced():
    payload = _initial_payload()
    _node(payload, "token_embedding")["attributes"]["vocab"] = 16
    _node(payload, "readout")["attributes"]["vocab"] = 16
    _node(payload, "readout")["output_shape"][-1] = 16
    payload["metadata"]["vocab_size"] = 16
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    codes = {issue.code for issue in validation.issues}
    assert "fixed_vocabulary" in codes
    assert "logits_shape" in codes


def test_invalid_tied_embedding_is_rejected_before_build():
    payload = _initial_payload()
    _node(payload, "readout")["attributes"]["tie_embedding"] = "block1_attention"
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "invalid_tied_embedding" in {issue.code for issue in validation.issues}


def test_resource_caps_are_checked_without_module_allocation(monkeypatch):
    limits = replace(DEFAULT_LIMITS, max_parameters=6079, max_buffer_elements=2449)

    def forbidden_allocation(*_args, **_kwargs):
        raise AssertionError("validation allocated a module")

    monkeypatch.setattr(torch.nn, "Embedding", forbidden_allocation)
    validation = validate_ir_candidate_path(INITIAL_IR, limits=limits)
    assert not validation.valid
    codes = {issue.code for issue in validation.issues}
    assert "parameter_limit" in codes
    assert "buffer_limit" in codes


def test_full_profile_workspace_cap_rejects_multi_gigabyte_graph_before_build(
    monkeypatch,
):
    payload = _initial_payload()
    for node in payload["nodes"]:
        for shape in [*node["input_shapes"], node["output_shape"]]:
            if len(shape) == 3 and shape[-1] == 16:
                shape[-1] = DEFAULT_LIMITS.max_hidden_dimension

    def forbidden_allocation(*_args, **_kwargs):
        raise AssertionError("workspace validation allocated a module")

    monkeypatch.setattr(torch.nn, "Embedding", forbidden_allocation)
    validation = validate_ir_candidate_json(json.dumps(payload))

    assert not validation.valid
    codes = {issue.code for issue in validation.issues}
    assert "training_workspace_limit" in codes
    assert "parameter_limit" not in codes
    assert validation.estimated_training_workspace_bytes is not None
    assert (
        validation.estimated_training_workspace_bytes
        > DEFAULT_LIMITS.max_training_workspace_bytes
    )


def test_sequence_hidden_head_and_fan_in_caps_fail_before_build():
    payload = _initial_payload()
    payload["metadata"]["max_sequence_length"] = 34
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "sequence_length_limit" in {issue.code for issue in validation.issues}

    payload = _initial_payload()
    payload["metadata"]["max_sequence_length"] = DEFAULT_LIMITS.max_sequence_length + 1
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "sequence_length_limit" in {issue.code for issue in validation.issues}

    payload = _initial_payload()
    attention = _node(payload, "block1_attention")
    attention["attributes"]["heads"] = 3
    validation = validate_ir_candidate_json(json.dumps(payload))
    assert not validation.valid
    assert "attention_heads" in {issue.code for issue in validation.issues}


def test_invalid_seed_input_dtype_token_range_and_sequence_length_fail_closed():
    with pytest.raises(IRInterpreterError, match="seed"):
        load_and_build_ir_candidate(INITIAL_IR, True)

    model = load_and_build_ir_candidate(INITIAL_IR, 1).model
    with pytest.raises(TypeError, match="long"):
        model(torch.zeros((1, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match="token IDs"):
        model(torch.tensor([[0, 15]], dtype=torch.long))
    with pytest.raises(ValueError, match="sequence length"):
        model(torch.zeros((1, 36), dtype=torch.long))


def test_path_loader_rejects_non_json_oversized_and_non_utf8_files(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"graph_id":"one","graph_id":"two"}', encoding="utf-8")
    validation = validate_ir_candidate_path(duplicate)
    assert not validation.valid
    assert validation.issues[0].code == "json_decode"

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_IR_JSON_BYTES + 1))
    validation = validate_ir_candidate_path(oversized)
    assert not validation.valid
    assert validation.issues[0].code == "file_read"

    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe")
    with pytest.raises(IRInterpreterError, match="UTF-8"):
        load_and_build_ir_candidate(binary, 0)


def test_deeply_nested_json_returns_invalid_instead_of_recursion_error():
    deeply_nested = "[" * 2_000 + "0" + "]" * 2_000
    validation = validate_ir_candidate_json(
        '{"metadata":' + deeply_nested + "}"
    )

    assert not validation.valid
    assert validation.graph is None
    assert validation.issues[0].code == "json_nesting_limit"


def test_each_bound_attention_must_individually_influence_logits():
    class BoundAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(8, 8, bias=False)
            mask = torch.tril(torch.ones(8, 8, dtype=torch.bool))
            self.register_buffer("causal_mask", mask.view(1, 1, 8, 8))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.projection(value)

    class OneBypassedAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(15, 8)
            self.live_attention = BoundAttention()
            self.bypassed_attention = BoundAttention()
            self.readout = nn.Linear(8, 15)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            causal_state = torch.cumsum(self.embedding(token_ids), dim=1)
            live = self.live_attention(causal_state)
            bypassed = self.bypassed_attention(causal_state)
            return self.readout(live + 0.0 * bypassed)

    model = OneBypassedAttention()
    bindings = RuntimeBindings(
        graph_hash="a" * 64,
        attention_modules={
            "live": "live_attention",
            "bypassed": "bypassed_attention",
        },
    )
    evidence = probe_runtime_validity(
        model,
        bindings=bindings,
        token_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long),
        expected_device="cpu",
    )

    assert not evidence.passed
    assert evidence.attention_calls["live"] > 0
    assert evidence.attention_calls["bypassed"] > 0
    assert evidence.attention_intervention_max_deltas["live"] > 1e-8
    assert evidence.attention_intervention_max_deltas["bypassed"] == 0.0
    assert not evidence.checks["each_attention_influences_output"]
    assert not evidence.checks["attention_influences_output"]
