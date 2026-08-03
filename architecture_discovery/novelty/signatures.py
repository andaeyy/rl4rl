"""Scale-invariant mechanism signatures from trusted architecture evidence.

The clustering signature deliberately ignores graph IDs, node names, concrete
tensor widths, and numeric hyperparameter values. It retains primitive types,
semantic categorical attributes, local graph motifs, and categorical behavior
and intervention probes. A separate parameterization hash preserves the more
specific graph for audit without turning scale changes into new mechanisms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable, Mapping

from architecture_ir import ArchitectureGraph, EdgeKind, PrimitiveKind, validate_graph
from architecture_ir.graph import CustomPrimitiveSpec, IREdge, IRNode

from novelty.serialization import (
    content_sha256,
    require_int,
    require_sha256,
    require_str,
)


SIGNATURE_SCHEMA_NAME = "MechanismSignature"
SIGNATURE_SCHEMA_VERSION = "1.0"
_TOKEN = re.compile(r"^[a-z][a-z0-9_.:-]*$")
_SCALE_ATTRIBUTE_PARTS = {
    "depth",
    "dim",
    "dropout",
    "expansion",
    "heads",
    "hidden",
    "layers",
    "parameter",
    "rank",
    "scale",
    "size",
    "vocab",
    "width",
}
_PRESENTATION_ATTRIBUTES = {
    "class_name",
    "comment",
    "description",
    "display_name",
    "label",
    "notes",
    "source_note",
}
_PERFORMANCE_PROBE_PARTS = {
    "accuracy",
    "eval_loss",
    "official",
    "public_score",
    "reward",
    "search_score",
    "shadow",
    "validation_loss",
}


class ProbeKind(StrEnum):
    BEHAVIOR = "behavior"
    INTERVENTION = "intervention"


def _normal_token(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string category, not a raw measurement")
    token = re.sub(r"[^a-z0-9_.:-]+", "_", value.strip().lower()).strip("_")
    if not _TOKEN.fullmatch(token):
        raise ValueError(f"{field_name} must normalize to a stable lowercase token")
    return token


@dataclass(frozen=True, order=True)
class ProbeObservation:
    probe_id: str
    outcome: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_id", _normal_token(self.probe_id, "probe_id"))
        object.__setattr__(self, "outcome", _normal_token(self.outcome, "outcome"))
        if any(part in self.probe_id for part in _PERFORMANCE_PROBE_PARTS):
            raise ValueError(
                "mechanism probes cannot encode search, accuracy, reward, or sealed outcomes"
            )

    def to_dict(self) -> dict[str, str]:
        return {"probe_id": self.probe_id, "response_category": self.outcome}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProbeObservation":
        return cls(
            probe_id=require_str(payload["probe_id"], "probe_id"),
            outcome=require_str(payload["response_category"], "response_category"),
        )


@dataclass(frozen=True)
class ProbeSignature:
    kind: ProbeKind
    protocol_id: str
    observations: tuple[ProbeObservation, ...]
    schema_version: str = SIGNATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_id", _normal_token(self.protocol_id, "protocol_id"))
        if any(part in self.protocol_id for part in _PERFORMANCE_PROBE_PARTS):
            raise ValueError("probe protocols cannot be named after performance outcomes")
        ordered = tuple(sorted(self.observations))
        if not ordered:
            raise ValueError("a probe signature requires at least one categorical observation")
        if len({item.probe_id for item in ordered}) != len(ordered):
            raise ValueError("probe IDs must be unique within a signature")
        object.__setattr__(self, "observations", ordered)
        if self.schema_version != SIGNATURE_SCHEMA_VERSION:
            raise ValueError("unsupported probe-signature schema version")

    @classmethod
    def behavior(
        cls, protocol_id: str, observations: Mapping[str, str]
    ) -> "ProbeSignature":
        return cls(
            kind=ProbeKind.BEHAVIOR,
            protocol_id=protocol_id,
            observations=tuple(
                ProbeObservation(probe_id, outcome)
                for probe_id, outcome in observations.items()
            ),
        )

    @classmethod
    def intervention(
        cls, protocol_id: str, observations: Mapping[str, str]
    ) -> "ProbeSignature":
        return cls(
            kind=ProbeKind.INTERVENTION,
            protocol_id=protocol_id,
            observations=tuple(
                ProbeObservation(probe_id, outcome)
                for probe_id, outcome in observations.items()
            ),
        )

    @property
    def signature_hash(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind.value,
            "protocol_id": self.protocol_id,
            "observations": [item.to_dict() for item in self.observations],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProbeSignature":
        return cls(
            kind=ProbeKind(require_str(payload["kind"], "kind")),
            protocol_id=require_str(payload["protocol_id"], "protocol_id"),
            observations=tuple(
                ProbeObservation.from_dict(item) for item in payload["observations"]
            ),
            schema_version=require_str(
                payload.get("schema_version", SIGNATURE_SCHEMA_VERSION),
                "schema_version",
            ),
        )


def _is_scale_attribute(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in _SCALE_ATTRIBUTE_PARTS)


def _normal_attribute(value: Any, *, retain_numbers: bool) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if retain_numbers else "<numeric>"
    if isinstance(value, str):
        return re.sub(r"\s+", "_", value.strip().lower())
    if isinstance(value, (tuple, list)):
        return tuple(_normal_attribute(item, retain_numbers=retain_numbers) for item in value)
    raise TypeError(f"unsupported IR attribute value {type(value).__name__}")


def _attributes(node: IRNode, *, retain_parameters: bool) -> tuple[tuple[str, Any], ...]:
    values: list[tuple[str, Any]] = []
    for key, value in sorted(node.attributes.items()):
        if not retain_parameters and key.lower() in _PRESENTATION_ATTRIBUTES:
            continue
        if not retain_parameters and _is_scale_attribute(key):
            continue
        values.append(
            (key.lower(), _normal_attribute(value, retain_numbers=retain_parameters))
        )
    return tuple(values)


@dataclass(frozen=True, order=True)
class PrimitiveClass:
    kind: str
    input_ranks: tuple[int, ...]
    output_rank: int
    semantic_attributes: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        require_str(self.kind, "kind")
        for rank in self.input_ranks:
            require_int(rank, "input rank")
        require_int(self.output_rank, "output_rank")
        if not self.kind or any(rank < 0 for rank in self.input_ranks):
            raise ValueError("primitive kind and input ranks must be valid")
        if self.output_rank < 1:
            raise ValueError("primitive output rank must be positive")
        normalized: list[tuple[str, Any]] = []
        for key, value in self.semantic_attributes:
            if _is_scale_attribute(key):
                raise ValueError("normalized primitive classes cannot retain scale attributes")
            frozen = _normal_attribute(value, retain_numbers=False)
            normalized.append((str(key).lower(), frozen))
        ordered = tuple(sorted(normalized))
        if len({key for key, _ in ordered}) != len(ordered):
            raise ValueError("normalized primitive attribute keys must be unique")
        object.__setattr__(self, "input_ranks", tuple(self.input_ranks))
        object.__setattr__(self, "semantic_attributes", ordered)

    @property
    def class_hash(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "input_ranks": list(self.input_ranks),
            "output_rank": self.output_rank,
            "semantic_attributes": {
                key: value for key, value in self.semantic_attributes
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveClass":
        attributes = payload.get("semantic_attributes", {})
        return cls(
            kind=require_str(payload["kind"], "kind"),
            input_ranks=tuple(
                require_int(value, "input rank") for value in payload["input_ranks"]
            ),
            output_rank=require_int(payload["output_rank"], "output_rank"),
            semantic_attributes=tuple(sorted(dict(attributes).items())),
        )


@dataclass(frozen=True, order=True)
class EdgeMotif:
    source_class_hash: str
    edge_kind: str
    target_port: int
    target_class_hash: str

    def __post_init__(self) -> None:
        require_sha256(self.source_class_hash, "source_class_hash")
        require_str(self.edge_kind, "edge_kind")
        require_int(self.target_port, "target_port")
        require_sha256(self.target_class_hash, "target_class_hash")
        if not self.edge_kind or self.target_port < 0:
            raise ValueError("edge kind and target port must be valid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_class_hash": self.source_class_hash,
            "edge_kind": self.edge_kind,
            "target_port": self.target_port,
            "target_class_hash": self.target_class_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EdgeMotif":
        return cls(
            source_class_hash=require_str(
                payload["source_class_hash"], "source_class_hash"
            ),
            edge_kind=require_str(payload["edge_kind"], "edge_kind"),
            target_port=require_int(payload["target_port"], "target_port"),
            target_class_hash=require_str(
                payload["target_class_hash"], "target_class_hash"
            ),
        )


@dataclass(frozen=True)
class NormalizedMechanismGraph:
    primitive_classes: tuple[PrimitiveClass, ...]
    edge_motifs: tuple[EdgeMotif, ...]
    path_motif_hashes: tuple[str, ...]
    input_class_hash: str
    output_class_hash: str
    mechanism_hash: str
    parameterization_hash: str
    normalization_version: str = "1.0"

    def __post_init__(self) -> None:
        if self.normalization_version != "1.0":
            raise ValueError("unsupported mechanism-normalization version")
        primitive_classes = tuple(
            sorted(self.primitive_classes, key=lambda item: item.class_hash)
        )
        if len({item.class_hash for item in primitive_classes}) != len(primitive_classes):
            raise ValueError("normalized primitive classes must be unique")
        edge_motifs = tuple(sorted(self.edge_motifs))
        if len(set(edge_motifs)) != len(edge_motifs):
            raise ValueError("normalized edge motifs must be unique")
        paths = tuple(sorted(set(self.path_motif_hashes)))
        for path_hash in paths:
            require_sha256(path_hash, "path_motif_hash")
        known_classes = {item.class_hash for item in primitive_classes}
        for edge in edge_motifs:
            require_sha256(edge.source_class_hash, "source_class_hash")
            require_sha256(edge.target_class_hash, "target_class_hash")
            if edge.source_class_hash not in known_classes or edge.target_class_hash not in known_classes:
                raise ValueError("edge motif refers to an unknown primitive class")
        if self.input_class_hash not in known_classes or self.output_class_hash not in known_classes:
            raise ValueError("input or output refers to an unknown primitive class")
        require_sha256(self.parameterization_hash, "parameterization_hash")
        mechanism_payload = {
            "normalization_version": self.normalization_version,
            "primitive_classes": [item.to_dict() for item in primitive_classes],
            "edge_motifs": [item.to_dict() for item in edge_motifs],
            "path_motif_hashes": list(paths),
            "input_class_hash": self.input_class_hash,
            "output_class_hash": self.output_class_hash,
        }
        if content_sha256(mechanism_payload) != self.mechanism_hash:
            raise ValueError("mechanism hash does not match the normalized graph")
        object.__setattr__(self, "primitive_classes", primitive_classes)
        object.__setattr__(self, "edge_motifs", edge_motifs)
        object.__setattr__(self, "path_motif_hashes", paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalization_version": self.normalization_version,
            "primitive_classes": [item.to_dict() for item in self.primitive_classes],
            "edge_motifs": [item.to_dict() for item in self.edge_motifs],
            "path_motif_hashes": list(self.path_motif_hashes),
            "input_class_hash": self.input_class_hash,
            "output_class_hash": self.output_class_hash,
            "mechanism_hash": self.mechanism_hash,
            "parameterization_hash": self.parameterization_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "NormalizedMechanismGraph":
        return cls(
            primitive_classes=tuple(
                PrimitiveClass.from_dict(item) for item in payload["primitive_classes"]
            ),
            edge_motifs=tuple(EdgeMotif.from_dict(item) for item in payload["edge_motifs"]),
            path_motif_hashes=tuple(
                require_str(value, "path_motif_hash")
                for value in payload["path_motif_hashes"]
            ),
            input_class_hash=require_str(
                payload["input_class_hash"], "input_class_hash"
            ),
            output_class_hash=require_str(
                payload["output_class_hash"], "output_class_hash"
            ),
            mechanism_hash=require_str(payload["mechanism_hash"], "mechanism_hash"),
            parameterization_hash=require_str(
                payload["parameterization_hash"], "parameterization_hash"
            ),
            normalization_version=require_str(
                payload.get("normalization_version", "1.0"),
                "normalization_version",
            ),
        )


def _transparent_composition(
    node: IRNode, incoming: list[IREdge], outgoing: list[IREdge]
) -> bool:
    if node.kind is not PrimitiveKind.COMPOSITION or len(incoming) != 1 or len(outgoing) != 1:
        return False
    if incoming[0].kind in {EdgeKind.STATE, EdgeKind.RECURRENT}:
        return False
    if outgoing[0].kind in {EdgeKind.STATE, EdgeKind.RECURRENT}:
        return False
    if not node.attributes:
        return True
    return set(node.attributes) == {"operation"} and str(
        node.attributes["operation"]
    ).lower() in {"compose", "identity", "sequential", "wrapper"}


def _contract_transparent_nodes(
    graph: ArchitectureGraph,
) -> tuple[dict[str, IRNode], list[IREdge], str, str]:
    nodes = {node.node_id: node for node in graph.nodes}
    edges = list(graph.edges)
    input_id = graph.input_node_id
    output_id = graph.output_node_id
    changed = True
    while changed:
        changed = False
        for node_id, node in sorted(nodes.items()):
            if node_id in {input_id, output_id}:
                continue
            incoming = [edge for edge in edges if edge.target == node_id]
            outgoing = [edge for edge in edges if edge.source == node_id]
            if not _transparent_composition(node, incoming, outgoing):
                continue
            before = incoming[0]
            after = outgoing[0]
            bridge_kind = after.kind if after.kind is not EdgeKind.DATA else before.kind
            bridge = IREdge(
                source=before.source,
                target=after.target,
                target_port=after.target_port,
                kind=bridge_kind,
            )
            edges = [
                edge
                for edge in edges
                if edge.source != node_id and edge.target != node_id
            ]
            edges.append(bridge)
            del nodes[node_id]
            changed = True
            break
    return nodes, edges, input_id, output_id


def _partition_labels(payload_by_node: Mapping[str, Any]) -> dict[str, str]:
    hashes = {node_id: content_sha256(payload) for node_id, payload in payload_by_node.items()}
    ordered = {value: f"class-{index:04d}" for index, value in enumerate(sorted(set(hashes.values())))}
    return {node_id: ordered[value] for node_id, value in hashes.items()}


def _parameterization_hash(graph: ArchitectureGraph) -> str:
    nodes = {node.node_id: node for node in graph.nodes}
    base = {
        node_id: {
            "kind": node.kind.value,
            "input_shapes": [shape.to_list() for shape in node.input_shapes],
            "output_shape": node.output_shape.to_list(),
            "attributes": {key: value for key, value in _attributes(node, retain_parameters=True)},
        }
        for node_id, node in nodes.items()
    }
    labels = _partition_labels(base)
    for _ in range(max(1, len(nodes) + 1)):
        refined: dict[str, Any] = {}
        for node_id in nodes:
            incoming = sorted(
                (edge.kind.value, edge.target_port, labels[edge.source])
                for edge in graph.edges
                if edge.target == node_id
            )
            outgoing = sorted(
                (edge.kind.value, edge.target_port, labels[edge.target])
                for edge in graph.edges
                if edge.source == node_id
            )
            refined[node_id] = {
                "base": base[node_id],
                "incoming": incoming,
                "outgoing": outgoing,
            }
        new_labels = _partition_labels(refined)
        if new_labels == labels:
            break
        labels = new_labels
    payload = {
        "nodes": sorted((labels[node_id], base[node_id]) for node_id in nodes),
        "edges": sorted(
            (
                labels[edge.source],
                edge.kind.value,
                edge.target_port,
                labels[edge.target],
            )
            for edge in graph.edges
        ),
        "input": labels[graph.input_node_id],
        "output": labels[graph.output_node_id],
    }
    return content_sha256(payload)


def _collapse_adjacent_repetitions(values: list[str]) -> list[str]:
    """Collapse repeated contiguous motifs in a linear depth stack."""

    current = list(values)
    while True:
        candidates: list[list[str]] = [current]
        for start in range(len(current)):
            remaining = len(current) - start
            for period in range(1, remaining // 2 + 1):
                pattern = current[start : start + period]
                repeats = 1
                while (
                    start + (repeats + 1) * period <= len(current)
                    and current[
                        start + repeats * period : start + (repeats + 1) * period
                    ]
                    == pattern
                ):
                    repeats += 1
                if repeats >= 2:
                    candidates.append(
                        current[:start]
                        + pattern
                        + current[start + repeats * period :]
                    )
        best = min(candidates, key=lambda item: (len(item), item))
        if len(best) == len(current):
            return current
        current = best


def _linear_mechanism_edges(
    *,
    nodes: Mapping[str, IRNode],
    edges: list[IREdge],
    node_classes: Mapping[str, PrimitiveClass],
    input_id: str,
    output_id: str,
) -> tuple[list[IREdge], dict[str, str]] | None:
    """Return synthetic scale-collapsed edges for a pure sequential stack."""

    if len(edges) != len(nodes) - 1 or any(edge.kind is not EdgeKind.DATA for edge in edges):
        return None
    incoming: dict[str, list[IREdge]] = {node_id: [] for node_id in nodes}
    outgoing: dict[str, list[IREdge]] = {node_id: [] for node_id in nodes}
    for edge in edges:
        incoming[edge.target].append(edge)
        outgoing[edge.source].append(edge)
    if incoming[input_id] or outgoing[output_id]:
        return None
    if any(
        len(incoming[node_id]) != 1 or len(outgoing[node_id]) != 1
        for node_id in nodes
        if node_id not in {input_id, output_id}
    ):
        return None
    sequence = [input_id]
    seen = {input_id}
    while sequence[-1] != output_id:
        next_edges = outgoing[sequence[-1]]
        if len(next_edges) != 1 or next_edges[0].target in seen:
            return None
        sequence.append(next_edges[0].target)
        seen.add(sequence[-1])
    if seen != set(nodes):
        return None
    class_sequence = [node_classes[node_id].class_hash for node_id in sequence]
    collapsed = [
        class_sequence[0],
        *_collapse_adjacent_repetitions(class_sequence[1:-1]),
        class_sequence[-1],
    ]
    if len(collapsed) == len(class_sequence):
        return None
    synthetic_nodes = [f"linear{index}" for index in range(len(collapsed))]
    return [
        IREdge(synthetic_nodes[index], synthetic_nodes[index + 1])
        for index in range(len(synthetic_nodes) - 1)
    ], dict(zip(synthetic_nodes, collapsed, strict=True))


def normalize_mechanism_graph(graph: ArchitectureGraph) -> NormalizedMechanismGraph:
    nodes, edges, input_id, output_id = _contract_transparent_nodes(graph)
    node_classes = {
        node_id: PrimitiveClass(
            kind=node.kind.value,
            input_ranks=tuple(shape.rank for shape in node.input_shapes),
            output_rank=node.output_shape.rank,
            semantic_attributes=_attributes(node, retain_parameters=False),
        )
        for node_id, node in nodes.items()
    }
    primitive_classes = tuple(
        sorted(
            {item.class_hash: item for item in node_classes.values()}.values(),
            key=lambda item: item.class_hash,
        )
    )
    linear_projection = _linear_mechanism_edges(
        nodes=nodes,
        edges=edges,
        node_classes=node_classes,
        input_id=input_id,
        output_id=output_id,
    )
    if linear_projection is None:
        projected_edges = edges
        projected_classes = {
            node_id: item.class_hash for node_id, item in node_classes.items()
        }
    else:
        projected_edges, projected_classes = linear_projection
    edge_motifs = tuple(
        sorted(
            {
                EdgeMotif(
                    source_class_hash=projected_classes[edge.source],
                    edge_kind=edge.kind.value,
                    target_port=edge.target_port,
                    target_class_hash=projected_classes[edge.target],
                )
                for edge in projected_edges
            }
        )
    )
    outgoing: dict[str, list[IREdge]] = {
        node_id: [] for node_id in projected_classes
    }
    for edge in projected_edges:
        outgoing[edge.source].append(edge)
    path_motifs: set[str] = set()
    for first in projected_edges:
        for second in outgoing[first.target]:
            path_motifs.add(
                content_sha256(
                    {
                        "source": projected_classes[first.source],
                        "first_edge": first.kind.value,
                        "middle": projected_classes[first.target],
                        "second_edge": second.kind.value,
                        "target": projected_classes[second.target],
                    }
                )
            )
    mechanism_payload = {
        "normalization_version": "1.0",
        "primitive_classes": [item.to_dict() for item in primitive_classes],
        "edge_motifs": [item.to_dict() for item in edge_motifs],
        "path_motif_hashes": sorted(path_motifs),
        "input_class_hash": node_classes[input_id].class_hash,
        "output_class_hash": node_classes[output_id].class_hash,
    }
    return NormalizedMechanismGraph(
        primitive_classes=primitive_classes,
        edge_motifs=edge_motifs,
        path_motif_hashes=tuple(sorted(path_motifs)),
        input_class_hash=node_classes[input_id].class_hash,
        output_class_hash=node_classes[output_id].class_hash,
        mechanism_hash=content_sha256(mechanism_payload),
        parameterization_hash=_parameterization_hash(graph),
    )


@dataclass(frozen=True)
class MechanismSignature:
    graph: NormalizedMechanismGraph
    behavior: ProbeSignature
    intervention: ProbeSignature
    signature_hash: str
    cluster_key: str
    schema_name: str = SIGNATURE_SCHEMA_NAME
    schema_version: str = SIGNATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_name != SIGNATURE_SCHEMA_NAME or self.schema_version != SIGNATURE_SCHEMA_VERSION:
            raise ValueError("unsupported mechanism-signature schema")
        require_sha256(self.signature_hash, "signature_hash")
        require_sha256(self.cluster_key, "cluster_key")
        if self.behavior.kind is not ProbeKind.BEHAVIOR:
            raise ValueError("behavior evidence has the wrong probe kind")
        if self.intervention.kind is not ProbeKind.INTERVENTION:
            raise ValueError("intervention evidence has the wrong probe kind")
        expected_cluster = content_sha256(
            {
                "normalization_version": self.graph.normalization_version,
                "mechanism_graph_hash": self.graph.mechanism_hash,
                "behavior_hash": self.behavior.signature_hash,
                "intervention_hash": self.intervention.signature_hash,
            }
        )
        if self.cluster_key != expected_cluster:
            raise ValueError("mechanism cluster key does not match its evidence")
        expected_signature = content_sha256(
            {
                "cluster_key": self.cluster_key,
                "parameterization_hash": self.graph.parameterization_hash,
            }
        )
        if self.signature_hash != expected_signature:
            raise ValueError("mechanism signature hash does not match its payload")

    @classmethod
    def create(
        cls,
        graph: ArchitectureGraph,
        *,
        behavior: ProbeSignature,
        intervention: ProbeSignature,
        custom_primitives: Iterable[CustomPrimitiveSpec] = (),
    ) -> "MechanismSignature":
        validation = validate_graph(graph, custom_primitives=custom_primitives)
        if not validation.valid:
            codes = ", ".join(issue.code for issue in validation.issues)
            raise ValueError(f"cannot sign an invalid architecture graph: {codes}")
        normalized = normalize_mechanism_graph(graph)
        cluster_key = content_sha256(
            {
                "normalization_version": normalized.normalization_version,
                "mechanism_graph_hash": normalized.mechanism_hash,
                "behavior_hash": behavior.signature_hash,
                "intervention_hash": intervention.signature_hash,
            }
        )
        signature_hash = content_sha256(
            {
                "cluster_key": cluster_key,
                "parameterization_hash": normalized.parameterization_hash,
            }
        )
        return cls(
            graph=normalized,
            behavior=behavior,
            intervention=intervention,
            signature_hash=signature_hash,
            cluster_key=cluster_key,
        )

    def review_payload(self) -> dict[str, Any]:
        """Return only scale-free, outcome-free mechanism evidence."""

        return {
            "normalization_version": self.graph.normalization_version,
            "mechanism_graph_hash": self.graph.mechanism_hash,
            "primitive_classes": [item.to_dict() for item in self.graph.primitive_classes],
            "edge_motifs": [item.to_dict() for item in self.graph.edge_motifs],
            "path_motif_hashes": list(self.graph.path_motif_hashes),
            "behavior": self.behavior.to_dict(),
            "intervention": self.intervention.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "graph": self.graph.to_dict(),
            "behavior": self.behavior.to_dict(),
            "intervention": self.intervention.to_dict(),
            "signature_hash": self.signature_hash,
            "cluster_key": self.cluster_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MechanismSignature":
        return cls(
            graph=NormalizedMechanismGraph.from_dict(payload["graph"]),
            behavior=ProbeSignature.from_dict(payload["behavior"]),
            intervention=ProbeSignature.from_dict(payload["intervention"]),
            signature_hash=require_str(payload["signature_hash"], "signature_hash"),
            cluster_key=require_str(payload["cluster_key"], "cluster_key"),
            schema_name=require_str(
                payload.get("schema_name", SIGNATURE_SCHEMA_NAME), "schema_name"
            ),
            schema_version=require_str(
                payload.get("schema_version", SIGNATURE_SCHEMA_VERSION),
                "schema_version",
            ),
        )
