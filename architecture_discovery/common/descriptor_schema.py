"""Frozen architecture-family categories for the semantic archive."""

from __future__ import annotations


CATEGORY_CODES: dict[str, dict[str, int]] = {
    "token_representation": {
        "unknown_or_other": 0,
        "learned_lookup": 1,
        "factorized_lookup": 2,
        "parametric_function": 3,
        "algebraic_map": 4,
    },
    "positional_integration": {
        "unknown_or_other": 0,
        "learned_additive": 1,
        "fixed_additive": 2,
        "relative_bias": 3,
        "rotary": 4,
        "none": 5,
    },
    "attention_projection": {
        "unknown_or_other": 0,
        "independent_dense": 1,
        "factorized": 2,
        "tied": 3,
        "deterministic_algebraic": 4,
    },
    "attention_organization": {
        "unknown_or_other": 0,
        "single_head": 1,
        "standard_multihead": 2,
        "grouped_or_multiquery": 3,
        "routed_or_algebraic": 4,
    },
    "feedforward_mechanism": {
        "unknown_or_other": 0,
        "gelu_mlp": 1,
        "relu_mlp": 2,
        "gated_glu": 3,
        "algebraic_gate": 4,
        "absent": 5,
    },
    "normalization": {
        "unknown_or_other": 0,
        "layernorm": 1,
        "rmsnorm": 2,
        "shared_norm": 3,
        "parameter_free": 4,
        "none": 5,
    },
    "depth_topology": {
        "unknown_or_other": 0,
        "single_block": 1,
        "sequential_blocks": 2,
        "shared_recurrent_block": 3,
        "cross_layer_tied": 4,
    },
    "output_readout": {
        "unknown_or_other": 0,
        "independent_linear": 1,
        "tied_embedding": 2,
        "factorized": 3,
        "parametric_decoder": 4,
    },
    "tokenization": {
        "unknown_or_other": 0,
        "digit_reversed_output": 1,
        "digit_forward_output": 2,
        "digit_pair": 3,
        "alternative_base": 4,
    },
}

SEMANTIC_METRIC_NAMES = {
    axis: f"semantic_{axis}" for axis in CATEGORY_CODES
}


def encode(axis: str, category: str) -> int:
    return CATEGORY_CODES[axis].get(category, CATEGORY_CODES[axis]["unknown_or_other"])


def bin_counts() -> dict[str, int]:
    return {
        SEMANTIC_METRIC_NAMES[axis]: max(codes.values()) + 1
        for axis, codes in CATEGORY_CODES.items()
    }

