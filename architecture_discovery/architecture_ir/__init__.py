"""Extensible, declarative architecture intermediate representation."""

from architecture_ir.codec import IRDecodeError, decode_graph_json, encode_graph_json
from architecture_ir.graph import (
    ArchitectureGraph,
    CustomPrimitiveSpec,
    EdgeKind,
    IREdge,
    IRNode,
    PrimitiveKind,
    TensorShape,
    ValidationIssue,
    ValidationResult,
    validate_graph,
)
from architecture_ir.runtime_evidence import (
    FreshBuildEvidence,
    RuntimeBindings,
    RuntimeValidityEvidence,
    probe_fresh_build,
    probe_runtime_validity,
)

__all__ = [
    "ArchitectureGraph",
    "CustomPrimitiveSpec",
    "EdgeKind",
    "FreshBuildEvidence",
    "IREdge",
    "IRDecodeError",
    "IRNode",
    "PrimitiveKind",
    "RuntimeBindings",
    "RuntimeValidityEvidence",
    "TensorShape",
    "ValidationIssue",
    "ValidationResult",
    "decode_graph_json",
    "encode_graph_json",
    "probe_fresh_build",
    "probe_runtime_validity",
    "validate_graph",
]
