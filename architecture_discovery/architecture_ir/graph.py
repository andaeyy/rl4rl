"""Typed tensor/module graph for architecture-only candidate proposals.

The IR contains data, shapes, and trusted primitive references.  It contains no
Python source, import paths, shell commands, checkpoints, or callables.  New
mechanisms are introduced by versioned primitive contracts, not by executing
candidate-supplied code.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, TypeAlias


SCHEMA_NAME = "architecture_tensor_graph"
SCHEMA_VERSION = "1.0"
NodeAttribute: TypeAlias = str | int | float | bool | None | tuple["NodeAttribute", ...]
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_FORBIDDEN_ATTRIBUTE_KEYS = {
    "bytecode",
    "callable",
    "checkpoint",
    "command",
    "eval",
    "exec",
    "file",
    "import",
    "module_path",
    "pickle",
    "python",
    "source",
    "state_dict",
}


class PrimitiveKind(StrEnum):
    INPUT = "input"
    TOKEN_EMBEDDING = "token_embedding"
    POSITIONAL = "positional"
    ATTENTION = "attention"
    NORMALIZATION = "normalization"
    FEED_FORWARD = "feed_forward"
    RECURRENT = "recurrent"
    ROUTING = "routing"
    STATE = "state"
    ALGEBRAIC = "algebraic"
    COMPOSITION = "composition"
    CUSTOM = "custom"
    READOUT = "readout"


class EdgeKind(StrEnum):
    DATA = "data"
    RESIDUAL = "residual"
    ROUTING = "routing"
    STATE = "state"
    RECURRENT = "recurrent"


@dataclass(frozen=True)
class TensorShape:
    """A tensor shape with positive concrete or stable symbolic dimensions."""

    dimensions: tuple[int | str, ...]

    def __post_init__(self) -> None:
        normalized = tuple(self.dimensions)
        object.__setattr__(self, "dimensions", normalized)
        if not normalized:
            raise ValueError("tensor shape must have at least one dimension")
        for dimension in normalized:
            if isinstance(dimension, bool):
                raise ValueError("boolean is not a valid tensor dimension")
            if isinstance(dimension, int) and dimension <= 0:
                raise ValueError("concrete tensor dimensions must be positive")
            if isinstance(dimension, str) and not _IDENTIFIER.fullmatch(dimension):
                raise ValueError(f"invalid symbolic tensor dimension {dimension!r}")
            if not isinstance(dimension, (int, str)):
                raise TypeError("tensor dimensions must be integers or symbolic strings")

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    def compatible_with(self, expected: "TensorShape") -> bool:
        if self.rank != expected.rank:
            return False
        for actual_dimension, expected_dimension in zip(
            self.dimensions, expected.dimensions, strict=True
        ):
            if actual_dimension == expected_dimension:
                continue
            if actual_dimension == "Any" or expected_dimension == "Any":
                continue
            return False
        return True

    def to_list(self) -> list[int | str]:
        return list(self.dimensions)


def _freeze_attribute(value: Any) -> NodeAttribute:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("IR floating-point attributes must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_attribute(item) for item in value)
    raise TypeError(
        "IR attributes must be JSON scalar values or sequences of scalar values; "
        f"received {type(value).__name__}"
    )


def _freeze_attributes(attributes: Mapping[str, Any]) -> Mapping[str, NodeAttribute]:
    frozen: dict[str, NodeAttribute] = {}
    for key, value in attributes.items():
        if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key):
            raise ValueError(f"invalid IR attribute key {key!r}")
        frozen[key] = _freeze_attribute(value)
    return MappingProxyType(frozen)


def _attribute_json(value: NodeAttribute) -> Any:
    if isinstance(value, tuple):
        return [_attribute_json(item) for item in value]
    return value


@dataclass(frozen=True)
class IRNode:
    node_id: str
    kind: PrimitiveKind
    input_shapes: tuple[TensorShape, ...]
    output_shape: TensorShape
    attributes: Mapping[str, NodeAttribute] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.node_id):
            raise ValueError(f"invalid node ID {self.node_id!r}")
        if not isinstance(self.kind, PrimitiveKind):
            raise TypeError("node kind must be a PrimitiveKind")
        object.__setattr__(self, "input_shapes", tuple(self.input_shapes))
        if not all(isinstance(shape, TensorShape) for shape in self.input_shapes):
            raise TypeError("node input shapes must be TensorShape instances")
        if not isinstance(self.output_shape, TensorShape):
            raise TypeError("node output shape must be a TensorShape")
        object.__setattr__(self, "attributes", _freeze_attributes(self.attributes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "input_shapes": [shape.to_list() for shape in self.input_shapes],
            "output_shape": self.output_shape.to_list(),
            "attributes": {
                key: _attribute_json(value) for key, value in sorted(self.attributes.items())
            },
        }


@dataclass(frozen=True)
class IREdge:
    source: str
    target: str
    target_port: int = 0
    kind: EdgeKind = EdgeKind.DATA

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.source) or not _IDENTIFIER.fullmatch(self.target):
            raise ValueError("edge endpoints must be valid node IDs")
        if (
            not isinstance(self.target_port, int)
            or isinstance(self.target_port, bool)
            or self.target_port < 0
        ):
            raise ValueError("target port must be a nonnegative integer")
        if not isinstance(self.kind, EdgeKind):
            raise TypeError("edge kind must be an EdgeKind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "target_port": self.target_port,
            "kind": self.kind.value,
        }


@dataclass(frozen=True)
class CustomPrimitiveSpec:
    """Versioned contract for one trusted, evaluator-owned custom primitive."""

    name: str
    version: str
    input_arity: int | None
    allowed_attributes: frozenset[str]
    input_ranks: tuple[int, ...] | None
    output_rank: int
    permits_state_cycle: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_attributes", frozenset(self.allowed_attributes))
        if self.input_ranks is not None:
            object.__setattr__(self, "input_ranks", tuple(self.input_ranks))
        if not _IDENTIFIER.fullmatch(self.name):
            raise ValueError(f"invalid custom primitive name {self.name!r}")
        if not self.version:
            raise ValueError("custom primitive version is required")
        if self.input_arity is not None and (
            not isinstance(self.input_arity, int)
            or isinstance(self.input_arity, bool)
            or self.input_arity < 1
        ):
            raise ValueError("custom primitive input arity must be positive or None")
        if (
            not isinstance(self.output_rank, int)
            or isinstance(self.output_rank, bool)
            or self.output_rank < 1
        ):
            raise ValueError("custom primitive output rank must be positive")
        if not all(
            isinstance(key, str) and _IDENTIFIER.fullmatch(key)
            for key in self.allowed_attributes
        ):
            raise ValueError("custom primitive attribute names must be valid identifiers")
        if self.input_ranks is not None and any(
            not isinstance(rank, int) or isinstance(rank, bool) or rank < 1
            for rank in self.input_ranks
        ):
            raise ValueError("custom primitive input ranks must be positive")

    @property
    def registry_key(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class ArchitectureGraph:
    graph_id: str
    input_node_id: str
    output_node_id: str
    nodes: tuple[IRNode, ...]
    edges: tuple[IREdge, ...]
    metadata: Mapping[str, NodeAttribute] = field(default_factory=dict)
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.graph_id):
            raise ValueError(f"invalid graph ID {self.graph_id!r}")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        if not all(isinstance(node, IRNode) for node in self.nodes):
            raise TypeError("graph nodes must be IRNode instances")
        if not all(isinstance(edge, IREdge) for edge in self.edges):
            raise TypeError("graph edges must be IREdge instances")
        object.__setattr__(self, "metadata", _freeze_attributes(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "graph_id": self.graph_id,
            "input_node_id": self.input_node_id,
            "output_node_id": self.output_node_id,
            "nodes": [node.to_dict() for node in sorted(self.nodes, key=lambda item: item.node_id)],
            "edges": [
                edge.to_dict()
                for edge in sorted(
                    self.edges,
                    key=lambda item: (item.target, item.target_port, item.source, item.kind),
                )
            ],
            "metadata": {
                key: _attribute_json(value) for key, value in sorted(self.metadata.items())
            },
        }

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def graph_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    node_id: str | None = None
    edge_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "node_id": self.node_id,
            "edge_index": self.edge_index,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    graph_hash: str
    issues: tuple[ValidationIssue, ...]
    attention_node_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "graph_hash": self.graph_hash,
            "issues": [issue.to_dict() for issue in self.issues],
            "attention_node_ids": list(self.attention_node_ids),
        }


def _issue(
    issues: list[ValidationIssue],
    code: str,
    message: str,
    *,
    node_id: str | None = None,
    edge_index: int | None = None,
) -> None:
    issues.append(ValidationIssue(code, message, node_id, edge_index))


def _validate_primitive(
    node: IRNode,
    custom_registry: Mapping[str, CustomPrimitiveSpec],
    issues: list[ValidationIssue],
) -> None:
    inputs = node.input_shapes
    if node.kind is PrimitiveKind.INPUT:
        if inputs:
            _issue(issues, "input_has_inputs", "input node must not declare inputs", node_id=node.node_id)
        return
    if not inputs:
        _issue(issues, "missing_input_shape", "non-input node must declare an input", node_id=node.node_id)
        return

    if node.kind is PrimitiveKind.TOKEN_EMBEDDING:
        if len(inputs) != 1 or inputs[0].rank != 2 or node.output_shape.rank != 3:
            _issue(issues, "embedding_shape", "token embedding requires rank-2 input and rank-3 output", node_id=node.node_id)
    elif node.kind in {
        PrimitiveKind.POSITIONAL,
        PrimitiveKind.ATTENTION,
        PrimitiveKind.NORMALIZATION,
    }:
        if len(inputs) != 1 or inputs[0].rank != 3 or not inputs[0].compatible_with(node.output_shape):
            _issue(issues, "sequence_shape", f"{node.kind.value} must preserve one rank-3 sequence shape", node_id=node.node_id)
        if node.kind is PrimitiveKind.ATTENTION and node.attributes.get("causal") is not True:
            _issue(issues, "attention_not_causal", "attention primitive must declare causal=true", node_id=node.node_id)
        if node.kind is PrimitiveKind.ATTENTION:
            heads = node.attributes.get("heads")
            if not isinstance(heads, int) or isinstance(heads, bool) or heads < 1:
                _issue(issues, "invalid_attention_heads", "attention heads must be a positive integer", node_id=node.node_id)
    elif node.kind in {
        PrimitiveKind.FEED_FORWARD,
        PrimitiveKind.READOUT,
    }:
        if len(inputs) != 1 or inputs[0].rank != 3 or node.output_shape.rank != 3:
            _issue(issues, "rank3_sequence_transform", f"{node.kind.value} requires one rank-3 input and rank-3 output", node_id=node.node_id)
    elif node.kind in {PrimitiveKind.RECURRENT, PrimitiveKind.STATE}:
        if any(shape.rank != 3 for shape in inputs) or node.output_shape.rank != 3:
            _issue(issues, "stateful_rank3_transform", f"{node.kind.value} requires rank-3 sequence tensors", node_id=node.node_id)
    elif node.kind in {
        PrimitiveKind.ROUTING,
        PrimitiveKind.ALGEBRAIC,
        PrimitiveKind.COMPOSITION,
    }:
        if any(shape.rank != 3 for shape in inputs) or node.output_shape.rank != 3:
            _issue(issues, "multi_input_rank", f"{node.kind.value} requires rank-3 sequence tensors", node_id=node.node_id)
    elif node.kind is PrimitiveKind.CUSTOM:
        primitive = node.attributes.get("trusted_primitive")
        version = node.attributes.get("primitive_version")
        key = f"{primitive}@{version}"
        spec = custom_registry.get(key)
        if spec is None:
            _issue(issues, "untrusted_custom_primitive", f"custom primitive {key!r} is not registered", node_id=node.node_id)
            return
        if spec.input_arity is not None and len(inputs) != spec.input_arity:
            _issue(issues, "custom_input_arity", f"custom primitive {key} expects {spec.input_arity} inputs", node_id=node.node_id)
        if spec.input_ranks is not None and tuple(shape.rank for shape in inputs) != spec.input_ranks:
            _issue(issues, "custom_input_rank", f"custom primitive {key} input ranks do not match its contract", node_id=node.node_id)
        if node.output_shape.rank != spec.output_rank:
            _issue(issues, "custom_output_rank", f"custom primitive {key} output rank does not match its contract", node_id=node.node_id)
        contract_keys = {"trusted_primitive", "primitive_version"}.union(spec.allowed_attributes)
        unexpected = set(node.attributes).difference(contract_keys)
        if unexpected:
            _issue(issues, "custom_attribute", f"custom primitive has undeclared attributes: {sorted(unexpected)}", node_id=node.node_id)


def _reachable(start: str, adjacency: Mapping[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(adjacency.get(current, ()))
    return seen


def _instantaneous_cycle(nodes: Iterable[str], edges: Iterable[IREdge]) -> bool:
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    indegree: dict[str, int] = {node: 0 for node in nodes}
    for edge in edges:
        if edge.kind in {EdgeKind.STATE, EdgeKind.RECURRENT}:
            continue
        if edge.target not in adjacency[edge.source]:
            adjacency[edge.source].add(edge.target)
            indegree[edge.target] += 1
    ready = [node for node, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for target in adjacency[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    return visited != len(indegree)


def validate_graph(
    graph: ArchitectureGraph,
    *,
    custom_primitives: Iterable[CustomPrimitiveSpec] = (),
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    if graph.schema_name != SCHEMA_NAME or graph.schema_version != SCHEMA_VERSION:
        _issue(issues, "schema_version", "unsupported architecture IR schema")

    node_by_id: dict[str, IRNode] = {}
    for node in graph.nodes:
        if node.node_id in node_by_id:
            _issue(issues, "duplicate_node", f"duplicate node ID {node.node_id}", node_id=node.node_id)
        node_by_id[node.node_id] = node
        forbidden = set(node.attributes).intersection(_FORBIDDEN_ATTRIBUTE_KEYS)
        if forbidden:
            _issue(issues, "executable_attribute", f"IR contains forbidden executable/state attributes: {sorted(forbidden)}", node_id=node.node_id)

    if graph.input_node_id not in node_by_id:
        _issue(issues, "missing_input_node", "graph input node does not exist")
    elif node_by_id[graph.input_node_id].kind is not PrimitiveKind.INPUT:
        _issue(issues, "wrong_input_kind", "graph input node must use input primitive", node_id=graph.input_node_id)
    if graph.output_node_id not in node_by_id:
        _issue(issues, "missing_output_node", "graph output node does not exist")
    elif node_by_id[graph.output_node_id].kind is not PrimitiveKind.READOUT:
        _issue(issues, "wrong_output_kind", "graph output node must use readout primitive", node_id=graph.output_node_id)

    registry = {spec.registry_key: spec for spec in custom_primitives}
    for node in graph.nodes:
        _validate_primitive(node, registry, issues)

    assigned_ports: set[tuple[str, int]] = set()
    valid_edges: list[IREdge] = []
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    reverse: dict[str, set[str]] = {node_id: set() for node_id in node_by_id}
    for index, edge in enumerate(graph.edges):
        source = node_by_id.get(edge.source)
        target = node_by_id.get(edge.target)
        if source is None or target is None:
            _issue(issues, "unknown_edge_endpoint", "edge references an unknown node", edge_index=index)
            continue
        if edge.target_port >= len(target.input_shapes):
            _issue(issues, "invalid_target_port", "edge target port is outside declared inputs", edge_index=index)
            continue
        port = (edge.target, edge.target_port)
        if port in assigned_ports:
            _issue(issues, "duplicate_target_port", "multiple edges supply one target port", edge_index=index)
            continue
        assigned_ports.add(port)
        expected = target.input_shapes[edge.target_port]
        if not source.output_shape.compatible_with(expected):
            _issue(
                issues,
                "edge_shape_mismatch",
                f"{edge.source} output {source.output_shape.dimensions} does not match {edge.target} port {edge.target_port} {expected.dimensions}",
                edge_index=index,
            )
        if edge.kind in {EdgeKind.STATE, EdgeKind.RECURRENT} and target.kind not in {
            PrimitiveKind.RECURRENT,
            PrimitiveKind.STATE,
            PrimitiveKind.CUSTOM,
        }:
            _issue(issues, "invalid_state_edge", "state/recurrent edges must target a stateful primitive", edge_index=index)
        if edge.kind in {EdgeKind.STATE, EdgeKind.RECURRENT} and target.kind is PrimitiveKind.CUSTOM:
            primitive_key = (
                f"{target.attributes.get('trusted_primitive')}@"
                f"{target.attributes.get('primitive_version')}"
            )
            spec = registry.get(primitive_key)
            if spec is None or not spec.permits_state_cycle:
                _issue(
                    issues,
                    "custom_state_edge_not_permitted",
                    "custom primitive contract does not permit state/recurrent edges",
                    edge_index=index,
                )
        valid_edges.append(edge)
        adjacency[edge.source].add(edge.target)
        reverse[edge.target].add(edge.source)

    for node in graph.nodes:
        for port in range(len(node.input_shapes)):
            if (node.node_id, port) not in assigned_ports:
                _issue(issues, "unbound_input_port", f"input port {port} is not connected", node_id=node.node_id)

    if node_by_id and _instantaneous_cycle(node_by_id, valid_edges):
        _issue(issues, "instantaneous_cycle", "data/residual/routing edges contain an instantaneous cycle")

    if graph.input_node_id in node_by_id and graph.output_node_id in node_by_id:
        forward = _reachable(graph.input_node_id, adjacency)
        backward = _reachable(graph.output_node_id, reverse)
        if graph.output_node_id not in forward:
            _issue(issues, "no_input_output_path", "graph output is not reachable from graph input")
        for node_id in node_by_id:
            if node_id not in forward or node_id not in backward:
                _issue(issues, "dead_node", "node is not on an input-to-output path", node_id=node_id)

    attention_nodes = tuple(
        sorted(node.node_id for node in graph.nodes if node.kind is PrimitiveKind.ATTENTION)
    )
    if not attention_nodes:
        _issue(issues, "missing_attention", "transformer-valid graph must contain an attention primitive")

    deduplicated: list[ValidationIssue] = []
    seen: set[tuple[Any, ...]] = set()
    for issue in issues:
        key = (issue.code, issue.message, issue.node_id, issue.edge_index)
        if key not in seen:
            seen.add(key)
            deduplicated.append(issue)
    return ValidationResult(not deduplicated, graph.graph_hash, tuple(deduplicated), attention_nodes)
