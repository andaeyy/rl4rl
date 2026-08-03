from __future__ import annotations

import pytest

from architecture_ir import (
    ArchitectureGraph,
    EdgeKind,
    IREdge,
    IRNode,
    PrimitiveKind,
    TensorShape,
)
from novelty.signatures import MechanismSignature, ProbeSignature


def build_graph(
    *,
    prefix: str = "a",
    width: int = 16,
    heads: int = 2,
    wrapper: bool = False,
    reverse_nodes: bool = False,
    wording: str | None = None,
    depth: int = 1,
) -> ArchitectureGraph:
    token_shape = TensorShape(("B", "T"))
    hidden_shape = TensorShape(("B", "T", width))
    output_shape = TensorShape(("B", "T", 10))
    input_id = f"{prefix}input"
    embedding_id = f"{prefix}embed"
    attention_ids = [f"{prefix}attention{index}" for index in range(depth)]
    wrapper_id = f"{prefix}wrapper"
    output_id = f"{prefix}output"
    nodes = [
        IRNode(input_id, PrimitiveKind.INPUT, (), token_shape),
        IRNode(
            embedding_id,
            PrimitiveKind.TOKEN_EMBEDDING,
            (token_shape,),
            hidden_shape,
            {"vocab_size": 20, "embedding_width": width},
        ),
        IRNode(
            output_id,
            PrimitiveKind.READOUT,
            (hidden_shape,),
            output_shape,
            {"vocab_size": 10},
        ),
    ]
    nodes.extend(
        IRNode(
            attention_id,
            PrimitiveKind.ATTENTION,
            (hidden_shape,),
            hidden_shape,
            {
                "causal": True,
                "heads": heads,
                "projection": "dense",
                **({"description": wording} if wording is not None else {}),
            },
        )
        for attention_id in attention_ids
    )
    # Keep container ordering irrelevant to the canonicalizer.
    output_node = nodes.pop(2)
    nodes.append(output_node)
    edges = [IREdge(input_id, embedding_id), IREdge(embedding_id, attention_ids[0])]
    edges.extend(
        IREdge(source, target)
        for source, target in zip(attention_ids, attention_ids[1:], strict=False)
    )
    last_attention_id = attention_ids[-1]
    if wrapper:
        nodes.append(
            IRNode(
                wrapper_id,
                PrimitiveKind.COMPOSITION,
                (hidden_shape,),
                hidden_shape,
                {"operation": "wrapper"},
            )
        )
        edges.extend(
            [IREdge(last_attention_id, wrapper_id), IREdge(wrapper_id, output_id)]
        )
    else:
        edges.append(IREdge(last_attention_id, output_id, kind=EdgeKind.DATA))
    if reverse_nodes:
        nodes.reverse()
        edges.reverse()
    return ArchitectureGraph(
        graph_id=f"graph-{prefix}",
        input_node_id=input_id,
        output_node_id=output_id,
        nodes=tuple(nodes),
        edges=tuple(edges),
        metadata={"candidate_label": prefix},
    )


def evidence(*, intervention: str = "large_effect") -> tuple[ProbeSignature, ProbeSignature]:
    return (
        ProbeSignature.behavior(
            "behavior-v1",
            {"prefix_dependence": "present", "length_shift": "stable"},
        ),
        ProbeSignature.intervention(
            "intervention-v1",
            {"attention_zeroing": intervention, "routing_swap": "direction_reversed"},
        ),
    )


def signature(graph: ArchitectureGraph, *, intervention: str = "large_effect") -> MechanismSignature:
    behavior, intervention_signature = evidence(intervention=intervention)
    return MechanismSignature.create(
        graph,
        behavior=behavior,
        intervention=intervention_signature,
    )


def test_node_renaming_ordering_and_source_wording_do_not_change_signature() -> None:
    first = signature(build_graph(prefix="alpha"))
    renamed = signature(
        build_graph(
            prefix="zeta",
            reverse_nodes=True,
        )
    )
    reworded = signature(
        build_graph(prefix="words", wording="Claims a completely novel mechanism")
    )

    assert first.graph.mechanism_hash == renamed.graph.mechanism_hash
    assert first.graph.parameterization_hash == renamed.graph.parameterization_hash
    assert first.cluster_key == renamed.cluster_key
    assert first.cluster_key == reworded.cluster_key
    assert first.graph.parameterization_hash != reworded.graph.parameterization_hash
    # No source text or descriptive wording is accepted by the signature API.
    assert "source" not in first.to_dict()
    assert "descriptor" not in first.to_dict()


def test_parameter_variants_share_mechanism_but_retain_audit_hash() -> None:
    narrow = signature(build_graph(prefix="narrow", width=16, heads=2))
    wide = signature(build_graph(prefix="wide", width=64, heads=8))

    assert narrow.graph.mechanism_hash == wide.graph.mechanism_hash
    assert narrow.cluster_key == wide.cluster_key
    assert narrow.graph.parameterization_hash != wide.graph.parameterization_hash
    assert narrow.signature_hash != wide.signature_hash


def test_repeated_linear_depth_is_a_scale_variant_not_a_new_cluster() -> None:
    shallow = signature(build_graph(prefix="shallow", depth=1))
    deep = signature(build_graph(prefix="deep", depth=3))

    assert shallow.graph.mechanism_hash == deep.graph.mechanism_hash
    assert shallow.cluster_key == deep.cluster_key
    assert shallow.graph.parameterization_hash != deep.graph.parameterization_hash


def test_transparent_composition_refactor_is_contracted() -> None:
    direct = signature(build_graph(prefix="direct"))
    wrapped = signature(build_graph(prefix="wrapped", wrapper=True))

    assert direct.graph.mechanism_hash == wrapped.graph.mechanism_hash
    assert direct.cluster_key == wrapped.cluster_key
    assert direct.graph.parameterization_hash != wrapped.graph.parameterization_hash


def test_intervention_evidence_distinguishes_mechanisms() -> None:
    affected = signature(build_graph(prefix="affected"), intervention="large_effect")
    unaffected = signature(build_graph(prefix="unaffected"), intervention="no_effect")

    assert affected.graph.mechanism_hash == unaffected.graph.mechanism_hash
    assert affected.cluster_key != unaffected.cluster_key


def test_raw_performance_metrics_cannot_be_mechanism_probes() -> None:
    with pytest.raises(ValueError, match="cannot encode search"):
        ProbeSignature.behavior("behavior-v1", {"public_accuracy": "high"})


def test_deserialized_signature_rejects_tampered_normalized_graph() -> None:
    original = signature(build_graph(prefix="tamper"))
    payload = original.to_dict()
    payload["graph"]["primitive_classes"][0]["kind"] = "routing"

    with pytest.raises(ValueError, match="mechanism hash|edge motif"):
        MechanismSignature.from_dict(payload)
