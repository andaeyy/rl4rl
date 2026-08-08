"""Independent static and runtime architecture descriptor extraction."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import ModuleType

import torch.nn as nn

from architecture_ir import ArchitectureGraph, PrimitiveKind
from common.descriptor_schema import SEMANTIC_METRIC_NAMES, encode


EVALUATOR_OWNED_TOKENIZATION_DESCRIPTOR = "digit_reversed_output"


@dataclass(frozen=True)
class DescriptorResult:
    categories: dict[str, str]
    codes: dict[str, float]
    confidence: dict[str, float]


def _semantic_result(categories: dict[str, str]) -> DescriptorResult:
    codes = {
        SEMANTIC_METRIC_NAMES[axis]: float(encode(axis, category))
        for axis, category in categories.items()
    }
    confidence = {
        axis: 0.9 if category != "unknown_or_other" else 0.2
        for axis, category in categories.items()
    }
    return DescriptorResult(categories, codes, confidence)


def extract_ir_descriptors(graph: ArchitectureGraph) -> DescriptorResult:
    """Derive public semantic descriptors from validated declarative fields.

    Candidate prose and graph IDs do not influence these categories.  Unknown
    mechanisms remain explicit instead of being guessed from names.
    """

    nodes_by_kind: dict[PrimitiveKind, list] = {}
    for node in graph.nodes:
        nodes_by_kind.setdefault(node.kind, []).append(node)

    token_nodes = nodes_by_kind.get(PrimitiveKind.TOKEN_EMBEDDING, [])
    token_mechanisms = {
        str(node.attributes.get("mechanism", "learned_lookup"))
        for node in token_nodes
    }
    token = (
        next(iter(token_mechanisms))
        if len(token_mechanisms) == 1
        and next(iter(token_mechanisms))
        in {"learned_lookup", "factorized_lookup", "parametric_function"}
        else "unknown_or_other"
    )

    positional_nodes = nodes_by_kind.get(PrimitiveKind.POSITIONAL, [])
    positional_mechanisms = {
        str(node.attributes.get("mechanism", "unknown_or_other"))
        for node in positional_nodes
    }
    positional_map = {
        "learned": "learned_additive",
        "learned_additive": "learned_additive",
        "sinusoidal": "fixed_additive",
        "fixed_additive": "fixed_additive",
        "rotary": "rotary",
        "relative_bias": "relative_bias",
    }
    position = (
        positional_map.get(next(iter(positional_mechanisms)), "unknown_or_other")
        if len(positional_mechanisms) == 1
        else "unknown_or_other"
    )

    attention_nodes = nodes_by_kind.get(PrimitiveKind.ATTENTION, [])
    projections = {
        str(node.attributes.get("projection", "independent_dense"))
        for node in attention_nodes
    }
    projection = (
        next(iter(projections))
        if len(projections) == 1
        and next(iter(projections))
        in {"independent_dense", "factorized", "tied", "deterministic_algebraic"}
        else "unknown_or_other"
    )
    head_counts = {
        int(node.attributes["heads"])
        for node in attention_nodes
        if isinstance(node.attributes.get("heads"), int)
        and not isinstance(node.attributes.get("heads"), bool)
    }
    organizations = {
        str(node.attributes.get("organization", "")) for node in attention_nodes
    }
    if organizations.intersection({"grouped", "multiquery", "grouped_or_multiquery"}):
        attention_org = "grouped_or_multiquery"
    elif organizations.intersection({"routed", "algebraic", "routed_or_algebraic"}):
        attention_org = "routed_or_algebraic"
    elif head_counts and max(head_counts) > 1:
        attention_org = "standard_multihead"
    elif head_counts == {1}:
        attention_org = "single_head"
    else:
        attention_org = "unknown_or_other"

    feedforward_nodes = nodes_by_kind.get(PrimitiveKind.FEED_FORWARD, [])
    feedforward_values = {
        str(node.attributes.get("mechanism", "gelu"))
        for node in feedforward_nodes
    }
    feedforward_map = {
        "gelu": "gelu_mlp",
        "gelu_mlp": "gelu_mlp",
        "relu": "relu_mlp",
        "relu_mlp": "relu_mlp",
        "gated": "gated_glu",
        "glu": "gated_glu",
        "swiglu": "gated_glu",
        "algebraic_gate": "algebraic_gate",
    }
    feedforward = (
        feedforward_map.get(next(iter(feedforward_values)), "unknown_or_other")
        if len(feedforward_values) == 1
        else "unknown_or_other"
    )

    normalization_nodes = nodes_by_kind.get(PrimitiveKind.NORMALIZATION, [])
    norm_values = {
        str(node.attributes.get("mechanism", "layer_norm"))
        for node in normalization_nodes
    }
    norm_map = {
        "layer_norm": "layernorm",
        "layernorm": "layernorm",
        "rms_norm": "rmsnorm",
        "rmsnorm": "rmsnorm",
        "parameter_free": "parameter_free",
    }
    normalization = (
        norm_map.get(next(iter(norm_values)), "unknown_or_other")
        if len(norm_values) == 1
        else "unknown_or_other"
    )

    if len(attention_nodes) > 1:
        topology = "sequential_blocks"
    elif len(attention_nodes) == 1:
        topology = "single_block"
    else:
        topology = "unknown_or_other"
    if nodes_by_kind.get(PrimitiveKind.RECURRENT):
        topology = "shared_recurrent_block"

    readout_nodes = nodes_by_kind.get(PrimitiveKind.READOUT, [])
    tied = {
        bool(
            node.attributes.get("tie_token_embedding", False)
            or node.attributes.get("tie_embedding")
        )
        for node in readout_nodes
    }
    readout = (
        "tied_embedding"
        if tied == {True}
        else "independent_linear" if tied == {False} else "unknown_or_other"
    )

    return _semantic_result(
        {
            "token_representation": token,
            "positional_integration": position,
            "attention_projection": projection,
            "attention_organization": attention_org,
            "feedforward_mechanism": feedforward,
            "normalization": normalization,
            "depth_topology": topology,
            "output_readout": readout,
            # Tokenization is frozen by the evaluator-owned Phase-1 task
            # adapter. Candidate metadata cannot alter this descriptor.
            "tokenization": EVALUATOR_OWNED_TOKENIZATION_DESCRIPTOR,
        }
    )


def _source(module: ModuleType) -> str:
    try:
        return inspect.getsource(module).lower()
    except (OSError, TypeError):
        return ""


def extract_descriptors(module: ModuleType, model: nn.Module) -> DescriptorResult:
    text = _source(module)
    classes = [child.__class__.__name__.lower() for child in model.modules()]

    token = "learned_lookup" if any(isinstance(child, nn.Embedding) for child in model.modules()) else "unknown_or_other"
    if "factorized" in text and "embed" in text:
        token = "factorized_lookup"
    if "parametric" in text and "embed" in text:
        token = "parametric_function"

    if "rope" in text or "rotary" in text:
        position = "rotary"
    elif "alibi" in text or "relative_bias" in text:
        position = "relative_bias"
    elif hasattr(model, "pos_emb") and isinstance(model.pos_emb, nn.Embedding):
        position = "learned_additive"
    elif "sinus" in text:
        position = "fixed_additive"
    else:
        position = "unknown_or_other"

    if "low_rank" in text or "factorized" in text:
        projection = "factorized"
    elif "tie_q" in text or "tied_q" in text or "shared_q" in text:
        projection = "tied"
    elif "rotation(" in text or "algebraic" in text:
        projection = "deterministic_algebraic"
    elif any("attention" in name for name in classes):
        projection = "independent_dense"
    else:
        projection = "unknown_or_other"

    head_count = 0
    for child in model.modules():
        head_count = max(head_count, int(getattr(child, "n_heads", 0)))
    if "grouped" in text or "multiquery" in text or "multi_query" in text:
        attention_org = "grouped_or_multiquery"
    elif "routed" in text or "algebraic prefix" in text:
        attention_org = "routed_or_algebraic"
    elif head_count > 1:
        attention_org = "standard_multihead"
    elif head_count == 1:
        attention_org = "single_head"
    else:
        attention_org = "unknown_or_other"

    if any(isinstance(child, nn.GELU) for child in model.modules()):
        feedforward = "gelu_mlp"
    elif any(isinstance(child, nn.ReLU) for child in model.modules()):
        feedforward = "relu_mlp"
    elif "swiglu" in text or "glu" in text:
        feedforward = "gated_glu"
    elif "algebraic_gate" in text:
        feedforward = "algebraic_gate"
    else:
        feedforward = "unknown_or_other"

    if any(isinstance(child, nn.LayerNorm) for child in model.modules()):
        normalization = "layernorm"
    elif any("rmsnorm" in name for name in classes):
        normalization = "rmsnorm"
    elif "parameter_free_norm" in text:
        normalization = "parameter_free"
    else:
        normalization = "unknown_or_other"

    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, nn.ModuleList) and len(blocks) > 1:
        topology = "sequential_blocks"
    elif isinstance(blocks, nn.ModuleList) and len(blocks) == 1:
        topology = "single_block"
    elif "recurrent" in text or "shared_block" in text:
        topology = "shared_recurrent_block"
    else:
        topology = "unknown_or_other"

    if (
        hasattr(model, "head")
        and hasattr(model, "token_emb")
        and getattr(model.head, "weight", None) is getattr(model.token_emb, "weight", None)
    ):
        readout = "tied_embedding"
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        readout = "independent_linear"
    else:
        readout = "unknown_or_other"

    if "reversed" in text and "digit" in text:
        tokenization = "digit_reversed_output"
    elif "digit_pair" in text or "pair token" in text:
        tokenization = "digit_pair"
    else:
        tokenization = "unknown_or_other"

    categories = {
        "token_representation": token,
        "positional_integration": position,
        "attention_projection": projection,
        "attention_organization": attention_org,
        "feedforward_mechanism": feedforward,
        "normalization": normalization,
        "depth_topology": topology,
        "output_readout": readout,
        "tokenization": tokenization,
    }
    return _semantic_result(categories)
