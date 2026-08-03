"""Independent static and runtime architecture descriptor extraction."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import ModuleType

import torch.nn as nn

from common.descriptor_schema import SEMANTIC_METRIC_NAMES, encode


@dataclass(frozen=True)
class DescriptorResult:
    categories: dict[str, str]
    codes: dict[str, float]
    confidence: dict[str, float]


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
    codes = {
        SEMANTIC_METRIC_NAMES[axis]: float(encode(axis, category))
        for axis, category in categories.items()
    }
    confidence = {
        axis: 0.9 if category != "unknown_or_other" else 0.2
        for axis, category in categories.items()
    }
    return DescriptorResult(categories, codes, confidence)

