"""Trusted evaluator-owned PyTorch interpreter for architecture IR candidates.

The only untrusted input accepted by this module is strict JSON decoded by the
architecture IR codec.  Candidate data selects from a small, statically
dispatched set of evaluator-owned primitives; it can never supply Python,
callables, import paths, modules, checkpoints, or executable expressions.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture_ir.codec import MAX_IR_JSON_BYTES, IRDecodeError, decode_graph_json
from architecture_ir.graph import (
    ArchitectureGraph,
    EdgeKind,
    IRNode,
    NodeAttribute,
    PrimitiveKind,
    ValidationResult,
    validate_graph,
)
from architecture_ir.runtime_evidence import RuntimeBindings


PHASE1_VOCAB_SIZE = 15
PHASE1_TRAINING_BATCH_SIZE = 512
PHASE1_TRAINING_SEQUENCE_LENGTH = 35
FLOAT32_BYTES = 4
AUTOGRAD_WORKSPACE_MULTIPLIER = 4
MAX_IR_JSON_NESTING = 64


@dataclass(frozen=True)
class InterpreterLimits:
    """Hard allocation/topology ceilings, never fitness or tie-break criteria."""

    max_nodes: int = 128
    max_edges: int = 512
    max_fan_in: int = 16
    max_sequence_length: int = 512
    max_hidden_dimension: int = 2_048
    max_feed_forward_dimension: int = 8_192
    max_attention_heads: int = 128
    max_parameters: int = 64_000_000
    max_buffer_elements: int = 16_000_000
    # This is a safety ceiling for the frozen Phase-1 full-training profile,
    # not an optimization objective.  Parameter count remains descriptive.
    max_training_workspace_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_LIMITS = InterpreterLimits()


@dataclass(frozen=True)
class InterpreterIssue:
    code: str
    message: str
    node_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "node_id": self.node_id}


@dataclass(frozen=True)
class IRCandidateValidation:
    """Schema, primitive-contract, and pre-allocation resource validation."""

    valid: bool
    graph: ArchitectureGraph | None
    graph_validation: ValidationResult | None
    issues: tuple[InterpreterIssue, ...]
    estimated_parameter_count: int | None
    estimated_buffer_elements: int | None
    estimated_training_workspace_bytes: int | None

    @property
    def graph_hash(self) -> str | None:
        return None if self.graph is None else self.graph.graph_hash

    @property
    def architecture_hash(self) -> str | None:
        return None if self.graph is None else self.graph.architecture_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "graph_hash": self.graph_hash,
            "architecture_hash": self.architecture_hash,
            "graph_validation": (
                None if self.graph_validation is None else self.graph_validation.to_dict()
            ),
            "issues": [issue.to_dict() for issue in self.issues],
            "estimated_parameter_count": self.estimated_parameter_count,
            "estimated_buffer_elements": self.estimated_buffer_elements,
            "estimated_training_workspace_bytes": self.estimated_training_workspace_bytes,
        }


@dataclass(frozen=True)
class InterpretedCandidate:
    graph: ArchitectureGraph
    validation: IRCandidateValidation
    model: nn.Module
    metadata: dict[str, Any]
    bindings: RuntimeBindings


class IRInterpreterError(ValueError):
    pass


def _issue(
    issues: list[InterpreterIssue],
    code: str,
    message: str,
    node: IRNode | None = None,
) -> None:
    issues.append(InterpreterIssue(code, message, None if node is None else node.node_id))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _as_float32(value: float) -> float | None:
    """Round through IEEE float32 without creating a tensor."""

    try:
        converted = struct.unpack("!f", struct.pack("!f", float(value)))[0]
    except (OverflowError, struct.error, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _normalized_fixed_mix_weights(
    weights: Sequence[Any],
) -> tuple[tuple[float, ...] | None, str | None]:
    """Validate exactly the numeric representation stored by ``_FixedMix``.

    JSON numbers are Python doubles, while the trusted module stores float32.
    A finite nonzero double can therefore become zero or infinity during module
    construction.  Reject those lossy cases before any module is allocated.
    """

    converted: list[float] = []
    for value in weights:
        if not _is_number(value):
            return None, "fixed_mix weights must be finite numbers"
        original = float(value)
        rounded = _as_float32(original)
        if rounded is None:
            return None, "fixed_mix weights must be representable as finite float32"
        if original != 0.0 and rounded == 0.0:
            return None, "fixed_mix weight underflows to zero in float32"
        converted.append(rounded)

    denominator_exact = math.fsum(abs(value) for value in converted)
    denominator = _as_float32(denominator_exact)
    if denominator is None or denominator <= 0.0:
        return None, "fixed_mix float32 normalization denominator must be finite and positive"

    normalized: list[float] = []
    for original, value in zip(weights, converted, strict=True):
        rounded = _as_float32(value / denominator)
        if rounded is None:
            return None, "fixed_mix normalization produced a non-finite float32 weight"
        if float(original) != 0.0 and rounded == 0.0:
            return None, "fixed_mix normalized weight underflows to zero in float32"
        normalized.append(rounded)
    if not any(value != 0.0 for value in normalized):
        return None, "fixed_mix normalization produced only zero float32 weights"
    return tuple(normalized), None


def _json_nesting_exceeds(text: str, maximum: int = MAX_IR_JSON_NESTING) -> bool:
    """Bound JSON container nesting before recursive decoder/model helpers run."""

    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > maximum:
                return True
        elif character in "]}":
            depth = max(0, depth - 1)
    return False


def _training_workspace_bytes(graph: ArchitectureGraph) -> int:
    """Conservative pre-allocation bound for frozen full-profile training.

    This models the evaluator-owned Phase-1 batch (512), its fixed 35-token
    sequence, forward intermediates, and an autograd/backward safety multiple.
    It is deliberately a hard memory-safety estimate, never a fitness metric.
    """

    batch = PHASE1_TRAINING_BATCH_SIZE
    steps = PHASE1_TRAINING_SEQUENCE_LENGTH
    sequence_factor = batch * steps
    forward_elements = 0

    for node in graph.nodes:
        if node.output_shape.rank != 3:
            continue
        width_value = node.output_shape.dimensions[-1]
        if not _is_int(width_value) or width_value < 1:
            continue
        width = int(width_value)
        sequence_elements = sequence_factor * width

        if node.kind is PrimitiveKind.TOKEN_EMBEDDING:
            forward_elements += sequence_elements
        elif node.kind is PrimitiveKind.POSITIONAL:
            forward_elements += 2 * sequence_elements
        elif node.kind is PrimitiveKind.ATTENTION:
            heads_value = node.attributes.get("heads")
            heads = int(heads_value) if _is_int(heads_value) and heads_value > 0 else 1
            # output + q/k/v + attended content, plus scores and probabilities
            forward_elements += 5 * sequence_elements
            forward_elements += 2 * batch * heads * steps * steps
        elif node.kind is PrimitiveKind.NORMALIZATION:
            forward_elements += 3 * sequence_elements
        elif node.kind is PrimitiveKind.FEED_FORWARD:
            hidden_value = node.attributes.get("hidden_dimension")
            hidden = (
                int(hidden_value)
                if _is_int(hidden_value) and hidden_value > 0
                else width
            )
            hidden_elements = sequence_factor * hidden
            multiplier = 2 if node.attributes.get("mechanism") == "gelu" else 4
            forward_elements += sequence_elements + multiplier * hidden_elements
        elif node.kind is PrimitiveKind.ALGEBRAIC:
            multiplier = 2 if node.attributes.get("mechanism") == "sigmoid_gate" else 1
            forward_elements += multiplier * sequence_elements
        elif node.kind is PrimitiveKind.ROUTING:
            forward_elements += sequence_elements
        elif node.kind is PrimitiveKind.COMPOSITION:
            input_width = sum(
                int(shape.dimensions[-1])
                for shape in node.input_shapes
                if shape.rank == 3 and _is_int(shape.dimensions[-1])
            )
            if node.attributes.get("mechanism") == "concat_project":
                forward_elements += sequence_elements + sequence_factor * input_width
            else:
                forward_elements += sequence_elements
        elif node.kind is PrimitiveKind.READOUT:
            forward_elements += sequence_elements

    token_input_bytes = batch * steps * 8  # evaluator inputs are torch.long
    return (
        forward_elements
        * AUTOGRAD_WORKSPACE_MULTIPLIER
        * FLOAT32_BYTES
        + token_input_bytes
    )


def _attributes(
    node: IRNode,
    issues: list[InterpreterIssue],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> bool:
    keys = set(node.attributes)
    missing = required.difference(keys)
    unknown = keys.difference(required.union(optional))
    if missing:
        _issue(
            issues,
            "missing_primitive_attribute",
            f"{node.kind.value} is missing required attributes {sorted(missing)}",
            node,
        )
    if unknown:
        _issue(
            issues,
            "unknown_primitive_attribute",
            f"{node.kind.value} has unsupported attributes {sorted(unknown)}",
            node,
        )
    return not missing and not unknown


def _positive_int_attribute(
    node: IRNode,
    key: str,
    issues: list[InterpreterIssue],
    *,
    maximum: int,
) -> int | None:
    value = node.attributes.get(key)
    if not _is_int(value) or value < 1 or value > maximum:
        _issue(
            issues,
            "invalid_primitive_attribute",
            f"{key} must be an integer in [1, {maximum}]",
            node,
        )
        return None
    return value


def _bool_attribute(
    node: IRNode,
    key: str,
    issues: list[InterpreterIssue],
    *,
    default: bool | None = None,
) -> bool | None:
    value = node.attributes.get(key, default)
    if not isinstance(value, bool):
        _issue(issues, "invalid_primitive_attribute", f"{key} must be boolean", node)
        return None
    return value


def _sequence_width(node: IRNode, issues: list[InterpreterIssue]) -> int | None:
    if node.output_shape.rank != 3:
        return None
    width = node.output_shape.dimensions[-1]
    if not _is_int(width):
        _issue(
            issues,
            "symbolic_hidden_dimension",
            "interpreter requires a concrete positive hidden dimension",
            node,
        )
        return None
    return width


def _input_width(node: IRNode, port: int, issues: list[InterpreterIssue]) -> int | None:
    if port >= len(node.input_shapes) or node.input_shapes[port].rank != 3:
        return None
    width = node.input_shapes[port].dimensions[-1]
    if not _is_int(width):
        _issue(
            issues,
            "symbolic_hidden_dimension",
            "interpreter requires concrete input hidden dimensions",
            node,
        )
        return None
    return width


def _validate_prefixes(
    graph: ArchitectureGraph,
    issues: list[InterpreterIssue],
) -> None:
    nodes = {node.node_id: node for node in graph.nodes}
    input_node = nodes.get(graph.input_node_id)
    if input_node is None or input_node.output_shape.rank != 2:
        return
    sequence_prefix = input_node.output_shape.dimensions
    for node in graph.nodes:
        shapes = (*node.input_shapes, node.output_shape)
        for shape in shapes:
            if shape.rank == 3 and shape.dimensions[:2] != sequence_prefix:
                _issue(
                    issues,
                    "sequence_prefix_mismatch",
                    "all sequence tensors must preserve the graph input batch/time axes",
                    node,
                )
                break


def _validate_node(
    node: IRNode,
    *,
    max_sequence_length: int,
    limits: InterpreterLimits,
    nodes: Mapping[str, IRNode],
    issues: list[InterpreterIssue],
) -> tuple[int, int]:
    """Return conservative (parameter count, buffer elements) estimates."""

    width = _sequence_width(node, issues)
    parameter_count = 0
    buffer_elements = 0

    if node.kind is PrimitiveKind.INPUT:
        _attributes(node, issues, required=frozenset())
        if node.output_shape.rank != 2:
            _issue(issues, "input_shape", "input must have rank [batch, time]", node)
        return 0, 0

    if node.kind in {PrimitiveKind.RECURRENT, PrimitiveKind.STATE, PrimitiveKind.CUSTOM}:
        _issue(
            issues,
            "unsupported_primitive",
            f"{node.kind.value} is intentionally unsupported by interpreter version 1",
            node,
        )
        return 0, 0

    if node.kind is PrimitiveKind.TOKEN_EMBEDDING:
        if _attributes(node, issues, required=frozenset({"vocab"})):
            vocab = node.attributes["vocab"]
            if vocab != PHASE1_VOCAB_SIZE or isinstance(vocab, bool):
                _issue(
                    issues,
                    "fixed_vocabulary",
                    f"token embedding vocab must equal {PHASE1_VOCAB_SIZE}",
                    node,
                )
        if width is not None:
            parameter_count = PHASE1_VOCAB_SIZE * width

    elif node.kind is PrimitiveKind.POSITIONAL:
        if _attributes(node, issues, required=frozenset({"mechanism"})):
            mechanism = node.attributes["mechanism"]
            if mechanism not in {"learned", "sinusoidal"}:
                _issue(
                    issues,
                    "unsupported_mechanism",
                    "positional mechanism must be learned or sinusoidal",
                    node,
                )
            elif width is not None:
                if mechanism == "learned":
                    parameter_count = max_sequence_length * width
                else:
                    buffer_elements = max_sequence_length * width

    elif node.kind is PrimitiveKind.ATTENTION:
        valid_keys = _attributes(
            node,
            issues,
            required=frozenset({"causal", "heads"}),
            optional=frozenset({"bias"}),
        )
        heads = _positive_int_attribute(
            node, "heads", issues, maximum=limits.max_attention_heads
        )
        causal = _bool_attribute(node, "causal", issues)
        bias = _bool_attribute(node, "bias", issues, default=False)
        input_width = _input_width(node, 0, issues)
        if causal is not True:
            _issue(issues, "attention_not_causal", "attention must set causal=true", node)
        if width is not None and input_width is not None and input_width != width:
            _issue(issues, "attention_width", "attention must preserve hidden width", node)
        if width is not None and heads is not None and width % heads:
            _issue(issues, "attention_heads", "hidden width must divide evenly by heads", node)
        if valid_keys and width is not None and bias is not None:
            parameter_count = 4 * width * width + (4 * width if bias else 0)
            buffer_elements = max_sequence_length * max_sequence_length

    elif node.kind is PrimitiveKind.NORMALIZATION:
        if _attributes(
            node,
            issues,
            required=frozenset({"mechanism"}),
            optional=frozenset({"epsilon", "affine"}),
        ):
            mechanism = node.attributes["mechanism"]
            epsilon = node.attributes.get("epsilon", 1e-5)
            affine = node.attributes.get("affine", True)
            if mechanism not in {"layer_norm", "rms_norm"}:
                _issue(
                    issues,
                    "unsupported_mechanism",
                    "normalization mechanism must be layer_norm or rms_norm",
                    node,
                )
            if not _is_number(epsilon) or not 0.0 < float(epsilon) <= 0.1:
                _issue(
                    issues,
                    "invalid_primitive_attribute",
                    "epsilon must be finite and in (0, 0.1]",
                    node,
                )
            if not isinstance(affine, bool):
                _issue(issues, "invalid_primitive_attribute", "affine must be boolean", node)
            input_width = _input_width(node, 0, issues)
            if width is not None and input_width is not None and input_width != width:
                _issue(issues, "normalization_width", "normalization must preserve width", node)
            if width is not None and isinstance(affine, bool) and affine:
                parameter_count = 2 * width if mechanism == "layer_norm" else width

    elif node.kind is PrimitiveKind.FEED_FORWARD:
        if _attributes(
            node,
            issues,
            required=frozenset({"mechanism", "hidden_dimension"}),
            optional=frozenset({"bias", "activation"}),
        ):
            mechanism = node.attributes["mechanism"]
            hidden = _positive_int_attribute(
                node,
                "hidden_dimension",
                issues,
                maximum=limits.max_feed_forward_dimension,
            )
            bias = _bool_attribute(node, "bias", issues, default=False)
            activation = node.attributes.get("activation", "gelu")
            if mechanism not in {"gelu", "gated"}:
                _issue(
                    issues,
                    "unsupported_mechanism",
                    "feed-forward mechanism must be gelu or gated",
                    node,
                )
            if mechanism == "gelu" and "activation" in node.attributes:
                _issue(
                    issues,
                    "irrelevant_primitive_attribute",
                    "activation is only valid for gated feed-forward",
                    node,
                )
            if mechanism == "gated" and activation not in {"gelu", "silu"}:
                _issue(
                    issues,
                    "unsupported_mechanism",
                    "gated activation must be gelu or silu",
                    node,
                )
            input_width = _input_width(node, 0, issues)
            if width is not None and input_width is not None and input_width != width:
                _issue(issues, "feed_forward_width", "feed-forward must preserve width", node)
            if width is not None and hidden is not None and bias is not None:
                multiplier = 2 if mechanism == "gelu" else 3
                parameter_count = multiplier * width * hidden
                if bias:
                    parameter_count += (hidden + width) if mechanism == "gelu" else 2 * hidden + width

    elif node.kind is PrimitiveKind.ALGEBRAIC:
        if _attributes(node, issues, required=frozenset({"mechanism"})):
            mechanism = node.attributes["mechanism"]
            if mechanism not in {"add", "learned_gate", "sigmoid_gate"}:
                _issue(
                    issues,
                    "unsupported_mechanism",
                    "algebraic mechanism must be add, learned_gate, or sigmoid_gate",
                    node,
                )
            expected_arity = 3 if mechanism == "sigmoid_gate" else 2
            if mechanism == "add":
                if not 2 <= len(node.input_shapes) <= limits.max_fan_in:
                    _issue(issues, "algebraic_arity", "add requires 2..max_fan_in inputs", node)
            elif len(node.input_shapes) != expected_arity:
                _issue(
                    issues,
                    "algebraic_arity",
                    f"{mechanism} requires exactly {expected_arity} inputs",
                    node,
                )
            widths = [_input_width(node, port, issues) for port in range(len(node.input_shapes))]
            if width is not None and any(item is not None and item != width for item in widths):
                _issue(issues, "algebraic_width", "algebraic inputs/output must have equal width", node)
            if mechanism == "learned_gate" and width is not None:
                parameter_count = width

    elif node.kind is PrimitiveKind.ROUTING:
        if _attributes(
            node,
            issues,
            required=frozenset({"mechanism"}),
            optional=frozenset({"temperature", "weights"}),
        ):
            mechanism = node.attributes["mechanism"]
            if mechanism not in {"softmax_mix", "fixed_mix"}:
                _issue(
                    issues,
                    "unsupported_mechanism",
                    "routing mechanism must be softmax_mix or fixed_mix",
                    node,
                )
            if not 2 <= len(node.input_shapes) <= limits.max_fan_in:
                _issue(issues, "routing_arity", "routing requires 2..max_fan_in inputs", node)
            widths = [_input_width(node, port, issues) for port in range(len(node.input_shapes))]
            if width is not None and any(item is not None and item != width for item in widths):
                _issue(issues, "routing_width", "routing inputs/output must have equal width", node)
            if mechanism == "softmax_mix":
                if "weights" in node.attributes:
                    _issue(
                        issues,
                        "irrelevant_primitive_attribute",
                        "weights is only valid for fixed_mix",
                        node,
                    )
                temperature = node.attributes.get("temperature", 1.0)
                if not _is_number(temperature) or not 0.01 <= float(temperature) <= 100.0:
                    _issue(
                        issues,
                        "invalid_primitive_attribute",
                        "temperature must be finite and in [0.01, 100]",
                        node,
                    )
                parameter_count = len(node.input_shapes)
            else:
                if "temperature" in node.attributes:
                    _issue(
                        issues,
                        "irrelevant_primitive_attribute",
                        "temperature is only valid for softmax_mix",
                        node,
                    )
                weights = node.attributes.get("weights")
                if not isinstance(weights, tuple) or len(weights) != len(node.input_shapes):
                    _issue(
                        issues,
                        "invalid_primitive_attribute",
                        "fixed_mix weights must match input arity",
                        node,
                    )
                else:
                    _normalized, numeric_error = _normalized_fixed_mix_weights(weights)
                    if numeric_error is not None:
                        _issue(
                            issues,
                            "invalid_primitive_attribute",
                            numeric_error,
                            node,
                        )
                    else:
                        buffer_elements = len(weights)

    elif node.kind is PrimitiveKind.COMPOSITION:
        if _attributes(
            node,
            issues,
            required=frozenset({"mechanism"}),
            optional=frozenset({"bias"}),
        ):
            mechanism = node.attributes["mechanism"]
            if mechanism not in {"identity", "concat_project"}:
                _issue(
                    issues,
                    "unsupported_mechanism",
                    "composition mechanism must be identity or concat_project",
                    node,
                )
            bias = _bool_attribute(node, "bias", issues, default=False)
            if mechanism == "identity":
                if len(node.input_shapes) != 1:
                    _issue(issues, "composition_arity", "identity requires one input", node)
                input_width = _input_width(node, 0, issues)
                if width is not None and input_width is not None and input_width != width:
                    _issue(
                        issues,
                        "composition_width",
                        "identity composition must preserve hidden width",
                        node,
                    )
                if "bias" in node.attributes:
                    _issue(
                        issues,
                        "irrelevant_primitive_attribute",
                        "bias is only valid for concat_project",
                        node,
                    )
            else:
                if not 2 <= len(node.input_shapes) <= limits.max_fan_in:
                    _issue(
                        issues,
                        "composition_arity",
                        "concat_project requires 2..max_fan_in inputs",
                        node,
                    )
                input_widths = [
                    _input_width(node, port, issues) for port in range(len(node.input_shapes))
                ]
                if width is not None and all(item is not None for item in input_widths):
                    parameter_count = sum(int(item) for item in input_widths) * width
                    if bias:
                        parameter_count += width

    elif node.kind is PrimitiveKind.READOUT:
        if _attributes(
            node,
            issues,
            required=frozenset({"vocab"}),
            optional=frozenset({"bias", "tie_embedding"}),
        ):
            vocab = node.attributes["vocab"]
            bias = _bool_attribute(node, "bias", issues, default=False)
            if vocab != PHASE1_VOCAB_SIZE or isinstance(vocab, bool):
                _issue(
                    issues,
                    "fixed_vocabulary",
                    f"readout vocab must equal {PHASE1_VOCAB_SIZE}",
                    node,
                )
            if width != PHASE1_VOCAB_SIZE:
                _issue(
                    issues,
                    "logits_shape",
                    f"readout output width must equal {PHASE1_VOCAB_SIZE}",
                    node,
                )
            input_width = _input_width(node, 0, issues)
            tie = node.attributes.get("tie_embedding")
            if tie is not None:
                if not isinstance(tie, str):
                    _issue(
                        issues,
                        "invalid_primitive_attribute",
                        "tie_embedding must be a token-embedding node ID",
                        node,
                    )
                else:
                    embedding = nodes.get(tie)
                    if embedding is None or embedding.kind is not PrimitiveKind.TOKEN_EMBEDDING:
                        _issue(
                            issues,
                            "invalid_tied_embedding",
                            "tie_embedding must reference a token_embedding node",
                            node,
                        )
                    else:
                        embedding_width = _sequence_width(embedding, issues)
                        if input_width is not None and embedding_width != input_width:
                            _issue(
                                issues,
                                "invalid_tied_embedding",
                                "tied embedding width must match readout input width",
                                node,
                            )
            if input_width is not None and tie is None:
                parameter_count = input_width * PHASE1_VOCAB_SIZE
            if bias:
                parameter_count += PHASE1_VOCAB_SIZE

    if width is not None and node.kind is not PrimitiveKind.READOUT:
        if width < 1 or width > limits.max_hidden_dimension:
            _issue(
                issues,
                "hidden_dimension_limit",
                f"hidden dimension must be in [1, {limits.max_hidden_dimension}]",
                node,
            )
    return parameter_count, buffer_elements


def validate_ir_candidate_json(
    text: str,
    *,
    limits: InterpreterLimits = DEFAULT_LIMITS,
) -> IRCandidateValidation:
    """Validate strict untrusted JSON without allocating a PyTorch model."""

    if isinstance(text, str) and _json_nesting_exceeds(text):
        return IRCandidateValidation(
            False,
            None,
            None,
            (
                InterpreterIssue(
                    "json_nesting_limit",
                    f"architecture IR JSON nesting exceeds hard limit {MAX_IR_JSON_NESTING}",
                ),
            ),
            None,
            None,
            None,
        )
    try:
        graph = decode_graph_json(text)
    except (IRDecodeError, TypeError, ValueError, RecursionError) as error:
        return IRCandidateValidation(
            False,
            None,
            None,
            (InterpreterIssue("json_decode", str(error)),),
            None,
            None,
            None,
        )

    graph_validation = validate_graph(graph)
    issues = [
        InterpreterIssue(issue.code, issue.message, issue.node_id)
        for issue in graph_validation.issues
    ]
    if len(graph.nodes) > limits.max_nodes:
        _issue(
            issues,
            "node_limit",
            f"graph has {len(graph.nodes)} nodes; limit is {limits.max_nodes}",
        )
    if len(graph.edges) > limits.max_edges:
        _issue(
            issues,
            "edge_limit",
            f"graph has {len(graph.edges)} edges; limit is {limits.max_edges}",
        )
    max_sequence_length = graph.metadata.get("max_sequence_length")
    if (
        not _is_int(max_sequence_length)
        or max_sequence_length < PHASE1_TRAINING_SEQUENCE_LENGTH
        or max_sequence_length > limits.max_sequence_length
    ):
        _issue(
            issues,
            "sequence_length_limit",
            "metadata.max_sequence_length must be an integer in "
            f"[{PHASE1_TRAINING_SEQUENCE_LENGTH}, "
            f"{limits.max_sequence_length}] so every validated candidate can "
            "consume the frozen Phase-1 training sequence",
        )
        safe_sequence_length = PHASE1_TRAINING_SEQUENCE_LENGTH
    else:
        safe_sequence_length = max_sequence_length
    declared_vocab = graph.metadata.get("vocab_size", PHASE1_VOCAB_SIZE)
    if declared_vocab != PHASE1_VOCAB_SIZE or isinstance(declared_vocab, bool):
        _issue(
            issues,
            "fixed_vocabulary",
            f"metadata.vocab_size must equal {PHASE1_VOCAB_SIZE}",
        )
    _validate_prefixes(graph, issues)

    nodes = {node.node_id: node for node in graph.nodes}
    parameters = 0
    buffers = 0
    for node in graph.nodes:
        node_parameters, node_buffers = _validate_node(
            node,
            max_sequence_length=safe_sequence_length,
            limits=limits,
            nodes=nodes,
            issues=issues,
        )
        parameters += node_parameters
        buffers += node_buffers
    training_workspace_bytes = _training_workspace_bytes(graph)

    for edge in graph.edges:
        if edge.kind in {EdgeKind.STATE, EdgeKind.RECURRENT}:
            _issue(
                issues,
                "unsupported_edge",
                f"{edge.kind.value} edges are intentionally unsupported by interpreter version 1",
            )

    if parameters > limits.max_parameters:
        _issue(
            issues,
            "parameter_limit",
            f"estimated parameters {parameters} exceed hard limit {limits.max_parameters}",
        )
    if buffers > limits.max_buffer_elements:
        _issue(
            issues,
            "buffer_limit",
            f"estimated buffer elements {buffers} exceed hard limit {limits.max_buffer_elements}",
        )
    if training_workspace_bytes > limits.max_training_workspace_bytes:
        _issue(
            issues,
            "training_workspace_limit",
            "estimated frozen-profile training workspace "
            f"{training_workspace_bytes} bytes exceeds hard limit "
            f"{limits.max_training_workspace_bytes} bytes "
            f"(batch={PHASE1_TRAINING_BATCH_SIZE}, "
            f"sequence={PHASE1_TRAINING_SEQUENCE_LENGTH}, float32)",
        )

    deduplicated = tuple(
        {
            (issue.code, issue.message, issue.node_id): issue
            for issue in issues
        }.values()
    )
    return IRCandidateValidation(
        not deduplicated,
        graph,
        graph_validation,
        deduplicated,
        parameters,
        buffers,
        training_workspace_bytes,
    )


def _read_ir_text(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_IR_JSON_BYTES + 1)
    except OSError as error:
        raise IRInterpreterError(f"could not read architecture IR: {error}") from error
    if len(payload) > MAX_IR_JSON_BYTES:
        raise IRInterpreterError(
            f"architecture IR exceeds {MAX_IR_JSON_BYTES} byte input limit"
        )
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise IRInterpreterError("architecture IR must be valid UTF-8 JSON") from error


def validate_ir_candidate_path(
    path: str | Path,
    *,
    limits: InterpreterLimits = DEFAULT_LIMITS,
) -> IRCandidateValidation:
    """Read and validate an IR artifact without constructing modules."""

    candidate_path = Path(path)
    try:
        text = _read_ir_text(candidate_path)
    except IRInterpreterError as error:
        return IRCandidateValidation(
            False,
            None,
            None,
            (InterpreterIssue("file_read", str(error)),),
            None,
            None,
            None,
        )
    return validate_ir_candidate_json(text, limits=limits)


class _CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, max_sequence_length: int, bias: bool) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=bias)
        self.output = nn.Linear(width, width, bias=bias)
        mask = torch.tril(
            torch.ones(max_sequence_length, max_sequence_length, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask.view(1, 1, max_sequence_length, max_sequence_length))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, steps, width = value.shape
        if steps > self.causal_mask.shape[-1]:
            raise ValueError("input sequence exceeds the architecture maximum")
        qkv = self.qkv(value).reshape(batch, steps, 3, self.heads, self.head_width)
        query, key, content = qkv.permute(2, 0, 3, 1, 4)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~self.causal_mask[:, :, :steps, :steps], float("-inf"))
        attended = F.softmax(scores, dim=-1) @ content
        attended = attended.transpose(1, 2).contiguous().reshape(batch, steps, width)
        return self.output(attended)


class _LearnedPosition(nn.Module):
    def __init__(self, max_sequence_length: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_sequence_length, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        steps = value.shape[1]
        if steps > self.embedding.num_embeddings:
            raise ValueError("input sequence exceeds the architecture maximum")
        positions = torch.arange(steps, device=value.device)
        return value + self.embedding(positions)


class _SinusoidalPosition(nn.Module):
    def __init__(self, max_sequence_length: int, width: int) -> None:
        super().__init__()
        positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        even_indices = torch.arange(0, width, 2, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10_000.0) * even_indices / max(1, width))
        encoding = torch.zeros(max_sequence_length, width, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if width > 1:
            encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        steps = value.shape[1]
        if steps > self.encoding.shape[0]:
            raise ValueError("input sequence exceeds the architecture maximum")
        return value + self.encoding[:steps].to(dtype=value.dtype)


class _RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float, affine: bool) -> None:
        super().__init__()
        self.epsilon = epsilon
        if affine:
            self.weight = nn.Parameter(torch.ones(width))
        else:
            self.register_parameter("weight", None)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + self.epsilon)
        normalized = normalized.to(dtype=value.dtype)
        return normalized if self.weight is None else normalized * self.weight


class _GeluFeedForward(nn.Module):
    def __init__(self, width: int, hidden: int, bias: bool) -> None:
        super().__init__()
        self.input = nn.Linear(width, hidden, bias=bias)
        self.output = nn.Linear(hidden, width, bias=bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.input(value)))


class _GatedFeedForward(nn.Module):
    def __init__(self, width: int, hidden: int, bias: bool, activation: str) -> None:
        super().__init__()
        self.input = nn.Linear(width, 2 * hidden, bias=bias)
        self.output = nn.Linear(hidden, width, bias=bias)
        self.activation = activation

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        content, gate = self.input(value).chunk(2, dim=-1)
        activated = F.gelu(gate) if self.activation == "gelu" else F.silu(gate)
        return self.output(content * activated)


class _Add(nn.Module):
    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result


class _LearnedGate(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(width))

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.logits)
        return gate * first + (1.0 - gate) * second


class _SigmoidGate(nn.Module):
    def forward(
        self, first: torch.Tensor, second: torch.Tensor, gate_logits: torch.Tensor
    ) -> torch.Tensor:
        gate = torch.sigmoid(gate_logits)
        return gate * first + (1.0 - gate) * second


class _SoftmaxMix(nn.Module):
    def __init__(self, arity: int, temperature: float) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(arity))
        self.temperature = temperature

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.logits / self.temperature, dim=0)
        result = torch.zeros_like(values[0])
        for weight, value in zip(weights, values, strict=True):
            result = result + weight * value
        return result


class _FixedMix(nn.Module):
    def __init__(self, weights: Sequence[float]) -> None:
        super().__init__()
        normalized, error = _normalized_fixed_mix_weights(weights)
        if normalized is None or error is not None:
            raise IRInterpreterError(error or "invalid fixed_mix weights")
        self.register_buffer(
            "weights", torch.tensor(normalized, dtype=torch.float32)
        )

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(values[0])
        for weight, value in zip(self.weights, values, strict=True):
            result = result + weight.to(dtype=value.dtype) * value
        return result


class _Identity(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value


class _ConcatProject(nn.Module):
    def __init__(self, input_width: int, output_width: int, bias: bool) -> None:
        super().__init__()
        self.projection = nn.Linear(input_width, output_width, bias=bias)

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat(values, dim=-1))


class _Readout(nn.Module):
    def __init__(
        self,
        input_width: int,
        *,
        bias: bool,
        tied_weight: nn.Parameter | None,
    ) -> None:
        super().__init__()
        if tied_weight is None:
            self.weight = nn.Parameter(torch.empty(PHASE1_VOCAB_SIZE, input_width))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = tied_weight
        if bias:
            self.bias = nn.Parameter(torch.empty(PHASE1_VOCAB_SIZE))
            bound = 1 / math.sqrt(input_width)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value, self.weight, self.bias)


def _topological_order(graph: ArchitectureGraph) -> tuple[str, ...]:
    node_ids = {node.node_id for node in graph.nodes}
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for edge in graph.edges:
        adjacency[edge.source].append(edge.target)
        indegree[edge.target] += 1
    ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for target in sorted(adjacency[current]):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(ordered) != len(node_ids):
        raise IRInterpreterError("validated graph unexpectedly contains a cycle")
    return tuple(ordered)


def _node_module(
    node: IRNode,
    *,
    max_sequence_length: int,
    embeddings: Mapping[str, nn.Embedding],
) -> nn.Module:
    width = int(node.output_shape.dimensions[-1])
    attributes = node.attributes
    if node.kind is PrimitiveKind.TOKEN_EMBEDDING:
        return embeddings[node.node_id]
    if node.kind is PrimitiveKind.POSITIONAL:
        return (
            _LearnedPosition(max_sequence_length, width)
            if attributes["mechanism"] == "learned"
            else _SinusoidalPosition(max_sequence_length, width)
        )
    if node.kind is PrimitiveKind.ATTENTION:
        return _CausalSelfAttention(
            width,
            int(attributes["heads"]),
            max_sequence_length,
            bool(attributes.get("bias", False)),
        )
    if node.kind is PrimitiveKind.NORMALIZATION:
        input_width = int(node.input_shapes[0].dimensions[-1])
        epsilon = float(attributes.get("epsilon", 1e-5))
        affine = bool(attributes.get("affine", True))
        if attributes["mechanism"] == "layer_norm":
            return nn.LayerNorm(input_width, eps=epsilon, elementwise_affine=affine)
        return _RMSNorm(input_width, epsilon, affine)
    if node.kind is PrimitiveKind.FEED_FORWARD:
        input_width = int(node.input_shapes[0].dimensions[-1])
        hidden = int(attributes["hidden_dimension"])
        bias = bool(attributes.get("bias", False))
        if attributes["mechanism"] == "gelu":
            return _GeluFeedForward(input_width, hidden, bias)
        return _GatedFeedForward(
            input_width, hidden, bias, str(attributes.get("activation", "gelu"))
        )
    if node.kind is PrimitiveKind.ALGEBRAIC:
        mechanism = attributes["mechanism"]
        if mechanism == "add":
            return _Add()
        if mechanism == "learned_gate":
            return _LearnedGate(width)
        return _SigmoidGate()
    if node.kind is PrimitiveKind.ROUTING:
        if attributes["mechanism"] == "softmax_mix":
            return _SoftmaxMix(len(node.input_shapes), float(attributes.get("temperature", 1.0)))
        return _FixedMix(tuple(float(value) for value in attributes["weights"]))
    if node.kind is PrimitiveKind.COMPOSITION:
        if attributes["mechanism"] == "identity":
            return _Identity()
        input_width = sum(int(shape.dimensions[-1]) for shape in node.input_shapes)
        return _ConcatProject(input_width, width, bool(attributes.get("bias", False)))
    if node.kind is PrimitiveKind.READOUT:
        input_width = int(node.input_shapes[0].dimensions[-1])
        tied_name = attributes.get("tie_embedding")
        tied_weight = None if tied_name is None else embeddings[str(tied_name)].weight
        return _Readout(
            input_width,
            bias=bool(attributes.get("bias", False)),
            tied_weight=tied_weight,
        )
    raise IRInterpreterError(f"unsupported primitive reached construction: {node.kind.value}")


class _InterpretedArchitecture(nn.Module):
    """Immutable execution plan whose operations are all evaluator-owned modules."""

    def __init__(self, graph: ArchitectureGraph) -> None:
        super().__init__()
        self.graph_hash = graph.graph_hash
        self.input_node_id = graph.input_node_id
        self.output_node_id = graph.output_node_id
        self.max_sequence_length = int(graph.metadata["max_sequence_length"])
        node_by_id = {node.node_id: node for node in graph.nodes}
        order = _topological_order(graph)
        incoming: dict[str, dict[int, str]] = {node_id: {} for node_id in order}
        for edge in graph.edges:
            incoming[edge.target][edge.target_port] = edge.source

        operators: list[nn.Module] = []
        execution_nodes: list[str] = []
        sources: list[tuple[str, ...]] = []
        module_paths: dict[str, str] = {}
        # Pre-create embeddings in topological order so readout tying is safe
        # even when a graph has multiple legal embedding branches.
        embeddings: dict[str, nn.Embedding] = {
            node_id: nn.Embedding(
                PHASE1_VOCAB_SIZE,
                int(node_by_id[node_id].output_shape.dimensions[-1]),
            )
            for node_id in order
            if node_by_id[node_id].kind is PrimitiveKind.TOKEN_EMBEDDING
        }
        for node_id in order:
            node = node_by_id[node_id]
            if node.kind is PrimitiveKind.INPUT:
                continue
            module = _node_module(
                node,
                max_sequence_length=self.max_sequence_length,
                embeddings=embeddings,
            )
            index = len(operators)
            operators.append(module)
            execution_nodes.append(node_id)
            sources.append(tuple(incoming[node_id][port] for port in range(len(node.input_shapes))))
            module_paths[node_id] = f"operators.{index}"
            if node.kind is PrimitiveKind.TOKEN_EMBEDDING:
                if not isinstance(module, nn.Embedding):
                    raise IRInterpreterError("trusted token embedding construction failed")

        self.operators = nn.ModuleList(operators)
        self._execution_nodes = tuple(execution_nodes)
        self._sources = tuple(sources)
        self._attention_paths = {
            node.node_id: module_paths[node.node_id]
            for node in graph.nodes
            if node.kind is PrimitiveKind.ATTENTION
        }

    @property
    def attention_module_paths(self) -> Mapping[str, str]:
        return dict(self._attention_paths)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("model input must be a torch.Tensor")
        if token_ids.ndim != 2:
            raise ValueError("model input must have shape [batch, sequence]")
        if token_ids.dtype != torch.long:
            raise TypeError("token IDs must use torch.long dtype")
        if token_ids.shape[1] < 1 or token_ids.shape[1] > self.max_sequence_length:
            raise ValueError("input sequence length is outside the architecture bounds")
        if token_ids.numel() and (
            int(token_ids.min().detach().cpu()) < 0
            or int(token_ids.max().detach().cpu()) >= PHASE1_VOCAB_SIZE
        ):
            raise ValueError(f"token IDs must be in [0, {PHASE1_VOCAB_SIZE - 1}]")

        values: dict[str, torch.Tensor] = {self.input_node_id: token_ids}
        for node_id, sources, operator in zip(
            self._execution_nodes, self._sources, self.operators, strict=True
        ):
            values[node_id] = operator(*(values[source] for source in sources))
        logits = values[self.output_node_id]
        expected = (token_ids.shape[0], token_ids.shape[1], PHASE1_VOCAB_SIZE)
        if logits.shape != expected:
            raise RuntimeError(
                f"trusted interpreter produced logits shape {tuple(logits.shape)}, expected {expected}"
            )
        return logits


def _json_metadata(value: NodeAttribute) -> Any:
    if isinstance(value, tuple):
        return [_json_metadata(item) for item in value]
    return value


def _build_validated_candidate(
    validation: IRCandidateValidation,
    *,
    seed: int,
) -> InterpretedCandidate:
    if not validation.valid or validation.graph is None:
        detail = "; ".join(
            f"[{issue.code}] {issue.message}" for issue in validation.issues
        )
        raise IRInterpreterError(f"architecture IR validation failed: {detail}")
    if not _is_int(seed) or not -(2**63) <= seed < 2**63:
        raise IRInterpreterError("initialization seed must be a signed 64-bit integer")

    graph = validation.graph
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _InterpretedArchitecture(graph)

    nonfinite_state = [
        name
        for name, tensor in (
            *model.named_parameters(),
            *model.named_buffers(),
        )
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor))
        and not bool(torch.isfinite(tensor.detach()).all().item())
    ]
    if nonfinite_state:
        raise IRInterpreterError(
            "trusted IR construction produced non-finite parameters/buffers: "
            + ", ".join(sorted(nonfinite_state))
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != validation.estimated_parameter_count:
        raise IRInterpreterError(
            "trusted parameter estimate mismatch: "
            f"estimated {validation.estimated_parameter_count}, built {parameter_count}"
        )
    if any(parameter.device.type != "cpu" for parameter in model.parameters()) or any(
        buffer.device.type != "cpu" for buffer in model.buffers()
    ):
        raise IRInterpreterError("fresh IR model was not constructed entirely on CPU")

    candidate_metadata = {
        key: _json_metadata(value) for key, value in graph.metadata.items()
    }
    metadata = dict(candidate_metadata)
    metadata.update(
        {
            "name": str(candidate_metadata.get("name", graph.graph_id)),
            "architecture_ir_graph_id": graph.graph_id,
            "architecture_ir_graph_hash": graph.graph_hash,
            "architecture_ir_architecture_hash": graph.architecture_hash,
            "architecture_ir_architecture_hash_schema": graph.architecture_hash_schema,
            "architecture_ir_schema": f"{graph.schema_name}@{graph.schema_version}",
            "parameter_count": parameter_count,
            "params": parameter_count,
            "parameter_count_role": "descriptive_metadata_only",
            "initialization_seed": seed,
            "initial_device": "cpu",
            "execution_provenance": "trusted_ir_interpreter",
        }
    )
    bindings = RuntimeBindings(
        graph_hash=graph.graph_hash,
        attention_modules=model.attention_module_paths,
    )
    binding_problems = bindings.validate()
    if binding_problems:
        raise IRInterpreterError("invalid trusted runtime bindings: " + "; ".join(binding_problems))
    return InterpretedCandidate(graph, validation, model, metadata, bindings)


def load_and_build_ir_candidate(
    path: str | Path,
    seed: int,
    *,
    limits: InterpreterLimits = DEFAULT_LIMITS,
) -> InterpretedCandidate:
    """Strictly load, validate, and deterministically build one CPU candidate."""

    text = _read_ir_text(Path(path))
    validation = validate_ir_candidate_json(text, limits=limits)
    return _build_validated_candidate(validation, seed=seed)
