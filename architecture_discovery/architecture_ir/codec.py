"""Strict JSON codec for untrusted declarative architecture proposals."""

from __future__ import annotations

import json
from typing import Any, Mapping

from architecture_ir.graph import (
    ArchitectureGraph,
    EdgeKind,
    IREdge,
    IRNode,
    PrimitiveKind,
    TensorShape,
)


MAX_IR_JSON_BYTES = 2 * 1024 * 1024
MAX_IR_NODES = 10_000
MAX_IR_EDGES = 100_000


class IRDecodeError(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IRDecodeError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise IRDecodeError(f"non-finite JSON number {value!r} is forbidden")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise IRDecodeError(f"{context} must be a JSON object")
    return value


def _sequence(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise IRDecodeError(f"{context} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = required.difference(value)
    extra = set(value).difference(required)
    if missing:
        raise IRDecodeError(f"{context} is missing keys: {sorted(missing)}")
    if extra:
        raise IRDecodeError(f"{context} has unknown keys: {sorted(extra)}")


def _shape(value: Any, context: str) -> TensorShape:
    dimensions = _sequence(value, context)
    try:
        return TensorShape(tuple(dimensions))
    except (TypeError, ValueError) as error:
        raise IRDecodeError(f"invalid {context}: {error}") from error


def _node(value: Any, index: int) -> IRNode:
    context = f"nodes[{index}]"
    node = _mapping(value, context)
    _exact_keys(
        node,
        required={"node_id", "kind", "input_shapes", "output_shape", "attributes"},
        context=context,
    )
    input_shapes = tuple(
        _shape(shape, f"{context}.input_shapes[{shape_index}]")
        for shape_index, shape in enumerate(_sequence(node["input_shapes"], f"{context}.input_shapes"))
    )
    try:
        return IRNode(
            node_id=node["node_id"],
            kind=PrimitiveKind(node["kind"]),
            input_shapes=input_shapes,
            output_shape=_shape(node["output_shape"], f"{context}.output_shape"),
            attributes=_mapping(node["attributes"], f"{context}.attributes"),
        )
    except (TypeError, ValueError) as error:
        raise IRDecodeError(f"invalid {context}: {error}") from error


def _edge(value: Any, index: int) -> IREdge:
    context = f"edges[{index}]"
    edge = _mapping(value, context)
    _exact_keys(
        edge,
        required={"source", "target", "target_port", "kind"},
        context=context,
    )
    try:
        return IREdge(
            source=edge["source"],
            target=edge["target"],
            target_port=edge["target_port"],
            kind=EdgeKind(edge["kind"]),
        )
    except (TypeError, ValueError) as error:
        raise IRDecodeError(f"invalid {context}: {error}") from error


def decode_graph_json(text: str) -> ArchitectureGraph:
    if not isinstance(text, str):
        raise TypeError("architecture IR input must be text")
    encoded_size = len(text.encode("utf-8"))
    if encoded_size > MAX_IR_JSON_BYTES:
        raise IRDecodeError(
            f"architecture IR exceeds {MAX_IR_JSON_BYTES} byte input limit"
        )
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except IRDecodeError:
        raise
    except json.JSONDecodeError as error:
        raise IRDecodeError(f"invalid architecture IR JSON: {error.msg}") from error

    graph = _mapping(raw, "graph")
    required = {
        "schema_name",
        "schema_version",
        "graph_id",
        "input_node_id",
        "output_node_id",
        "nodes",
        "edges",
        "metadata",
    }
    _exact_keys(graph, required=required, context="graph")
    raw_nodes = _sequence(graph["nodes"], "graph.nodes")
    raw_edges = _sequence(graph["edges"], "graph.edges")
    if len(raw_nodes) > MAX_IR_NODES:
        raise IRDecodeError(f"graph exceeds {MAX_IR_NODES} node limit")
    if len(raw_edges) > MAX_IR_EDGES:
        raise IRDecodeError(f"graph exceeds {MAX_IR_EDGES} edge limit")
    try:
        return ArchitectureGraph(
            graph_id=graph["graph_id"],
            input_node_id=graph["input_node_id"],
            output_node_id=graph["output_node_id"],
            nodes=tuple(_node(value, index) for index, value in enumerate(raw_nodes)),
            edges=tuple(_edge(value, index) for index, value in enumerate(raw_edges)),
            metadata=_mapping(graph["metadata"], "graph.metadata"),
            schema_name=graph["schema_name"],
            schema_version=graph["schema_version"],
        )
    except IRDecodeError:
        raise
    except (TypeError, ValueError) as error:
        raise IRDecodeError(f"invalid graph: {error}") from error


def encode_graph_json(graph: ArchitectureGraph) -> str:
    if not isinstance(graph, ArchitectureGraph):
        raise TypeError("encode_graph_json requires ArchitectureGraph")
    return graph.canonical_json
