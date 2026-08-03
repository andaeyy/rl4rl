from dataclasses import replace

import pytest

from architecture_ir.graph import (
    ArchitectureGraph,
    CustomPrimitiveSpec,
    EdgeKind,
    IREdge,
    IRNode,
    PrimitiveKind,
    TensorShape,
    validate_graph,
)


TOKENS = TensorShape(("Batch", "Time"))
HIDDEN = TensorShape(("Batch", "Time", "Hidden"))
LOGITS = TensorShape(("Batch", "Time", "Vocab"))


def _node(
    node_id: str,
    kind: PrimitiveKind,
    inputs: tuple[TensorShape, ...],
    output: TensorShape,
    **attributes,
) -> IRNode:
    return IRNode(node_id, kind, inputs, output, attributes)


def valid_graph(*, metadata=None) -> ArchitectureGraph:
    nodes = (
        _node("tokens", PrimitiveKind.INPUT, (), TOKENS),
        _node("embedding", PrimitiveKind.TOKEN_EMBEDDING, (TOKENS,), HIDDEN, vocab=15),
        _node("position", PrimitiveKind.POSITIONAL, (HIDDEN,), HIDDEN, mechanism="learned"),
        _node("attention", PrimitiveKind.ATTENTION, (HIDDEN,), HIDDEN, causal=True, heads=2),
        _node("norm", PrimitiveKind.NORMALIZATION, (HIDDEN,), HIDDEN, mechanism="layer_norm"),
        _node("ff", PrimitiveKind.FEED_FORWARD, (HIDDEN,), HIDDEN, mechanism="gated"),
        _node("readout", PrimitiveKind.READOUT, (HIDDEN,), LOGITS, vocab=15),
    )
    edges = tuple(
        IREdge(source, target)
        for source, target in (
            ("tokens", "embedding"),
            ("embedding", "position"),
            ("position", "attention"),
            ("attention", "norm"),
            ("norm", "ff"),
            ("ff", "readout"),
        )
    )
    return ArchitectureGraph(
        graph_id="novel.architecture.v1",
        input_node_id="tokens",
        output_node_id="readout",
        nodes=nodes,
        edges=edges,
        metadata={} if metadata is None else metadata,
    )


def test_valid_typed_transformer_graph_has_stable_canonical_hash():
    graph = valid_graph(metadata={"parameter_count": 10_000_000})
    result = validate_graph(graph)
    assert result.valid, result.to_dict()
    assert result.attention_node_ids == ("attention",)
    assert len(graph.graph_hash) == 64
    assert graph.graph_hash == replace(graph, nodes=tuple(reversed(graph.nodes))).graph_hash


def test_parameter_count_is_metadata_not_a_validation_or_selection_objective():
    small_metadata = validate_graph(valid_graph(metadata={"parameter_count": 1})).valid
    large_metadata = validate_graph(valid_graph(metadata={"parameter_count": 10**12})).valid
    absent_metadata = validate_graph(valid_graph()).valid
    assert small_metadata and large_metadata and absent_metadata


def test_malformed_shape_and_unbound_port_are_rejected_before_runtime():
    graph = valid_graph()
    malformed_readout = replace(
        next(node for node in graph.nodes if node.node_id == "readout"),
        input_shapes=(TensorShape(("Batch", "WrongTime", "Hidden")),),
    )
    graph = replace(
        graph,
        nodes=tuple(
            malformed_readout if node.node_id == "readout" else node for node in graph.nodes
        ),
        edges=tuple(edge for edge in graph.edges if edge.target != "norm"),
    )
    result = validate_graph(graph)
    assert not result.valid
    codes = {issue.code for issue in result.issues}
    assert "edge_shape_mismatch" in codes
    assert "unbound_input_port" in codes


def test_missing_attention_and_noncausal_attention_are_rejected():
    graph = valid_graph()
    no_attention = replace(
        graph,
        nodes=tuple(
            replace(node, kind=PrimitiveKind.COMPOSITION)
            if node.node_id == "attention"
            else node
            for node in graph.nodes
        ),
    )
    assert "missing_attention" in {
        issue.code for issue in validate_graph(no_attention).issues
    }

    noncausal = replace(
        graph,
        nodes=tuple(
            replace(node, attributes={"causal": False, "heads": 2})
            if node.node_id == "attention"
            else node
            for node in graph.nodes
        ),
    )
    assert "attention_not_causal" in {
        issue.code for issue in validate_graph(noncausal).issues
    }


def test_dead_attention_cannot_satisfy_transformer_graph_contract():
    graph = valid_graph()
    dead = _node("decorative_attention", PrimitiveKind.ATTENTION, (HIDDEN,), HIDDEN, causal=True, heads=1)
    graph = replace(graph, nodes=graph.nodes + (dead,))
    result = validate_graph(graph)
    assert not result.valid
    assert any(
        issue.code in {"dead_node", "unbound_input_port"}
        and issue.node_id == "decorative_attention"
        for issue in result.issues
    )


def test_instantaneous_cycles_are_rejected_but_typed_state_edges_are_representable():
    graph = valid_graph()
    cycle_graph = replace(graph, edges=graph.edges + (IREdge("ff", "attention"),))
    result = validate_graph(cycle_graph)
    assert not result.valid
    assert any(issue.code in {"duplicate_target_port", "instantaneous_cycle"} for issue in result.issues)

    state_node = _node(
        "memory",
        PrimitiveKind.RECURRENT,
        (HIDDEN, HIDDEN),
        HIDDEN,
        mechanism="learned_state_transition",
    )
    state_nodes = tuple(
        state_node if node.node_id == "ff" else node for node in graph.nodes
    )
    state_edges = tuple(
        IREdge(edge.source, "memory" if edge.target == "ff" else edge.target, edge.target_port, edge.kind)
        if edge.source != "ff"
        else IREdge("memory", edge.target, edge.target_port, edge.kind)
        for edge in graph.edges
    ) + (IREdge("memory", "memory", target_port=1, kind=EdgeKind.STATE),)
    state_graph = replace(graph, nodes=state_nodes, edges=state_edges)
    state_result = validate_graph(state_graph)
    assert state_result.valid, state_result.to_dict()


def test_custom_mechanisms_require_versioned_trusted_contracts_without_code_payloads():
    graph = valid_graph()
    custom = _node(
        "novel_state_router",
        PrimitiveKind.CUSTOM,
        (HIDDEN,),
        HIDDEN,
        trusted_primitive="state_router",
        primitive_version="1",
        temperature=0.5,
    )
    nodes = tuple(
        custom if node.node_id == "ff" else node for node in graph.nodes
    )
    edges = tuple(
        IREdge(edge.source, "novel_state_router" if edge.target == "ff" else edge.target, edge.target_port, edge.kind)
        if edge.source != "ff"
        else IREdge("novel_state_router", edge.target, edge.target_port, edge.kind)
        for edge in graph.edges
    )
    custom_graph = replace(graph, nodes=nodes, edges=edges)
    assert not validate_graph(custom_graph).valid

    spec = CustomPrimitiveSpec(
        name="state_router",
        version="1",
        input_arity=1,
        allowed_attributes=frozenset({"temperature"}),
        input_ranks=(3,),
        output_rank=3,
        permits_state_cycle=True,
    )
    assert validate_graph(custom_graph, custom_primitives=(spec,)).valid

    executable = replace(custom, attributes={"source": "exec('bad')"})
    executable_graph = replace(
        custom_graph,
        nodes=tuple(
            executable if node.node_id == "novel_state_router" else node
            for node in custom_graph.nodes
        ),
    )
    result = validate_graph(executable_graph, custom_primitives=(spec,))
    assert not result.valid
    assert "executable_attribute" in {issue.code for issue in result.issues}


def test_ir_attributes_reject_python_objects_and_invalid_shapes():
    with pytest.raises(TypeError, match="JSON scalar"):
        _node("bad", PrimitiveKind.ALGEBRAIC, (HIDDEN,), HIDDEN, callback=lambda x: x)
    with pytest.raises(ValueError, match="positive"):
        TensorShape(("Batch", 0, "Hidden"))
    with pytest.raises(ValueError, match="finite"):
        valid_graph(metadata={"unstable": float("nan")})
    with pytest.raises(ValueError, match="nonnegative integer"):
        IREdge("tokens", "embedding", target_port=0.5)
