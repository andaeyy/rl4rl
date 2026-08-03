"""Post-search mechanism novelty evidence.

This package is deliberately outside controller-visible search code. Online
semantic descriptors are search heuristics and are never novelty evidence.
"""

from novelty.clustering import (
    CandidateMechanism,
    MechanismClusterRecord,
    cluster_candidates,
    unique_cluster_counts_by_run,
)
from novelty.corpus import (
    CORPUS_POPULATION_REQUIRED,
    CorpusPopulationRequired,
    CorpusSignatureMatch,
    CorpusVerification,
    ReferenceCorpusManifest,
    ReferenceMechanism,
    freeze_corpus,
    query_corpus,
    verify_frozen_corpus,
)
from novelty.signatures import (
    MechanismSignature,
    NormalizedMechanismGraph,
    ProbeKind,
    ProbeObservation,
    ProbeSignature,
    normalize_mechanism_graph,
)
from novelty.taxonomy import NOVELTY_DEFINITIONS, NoveltyLabel, definition

__all__ = [
    "CORPUS_POPULATION_REQUIRED",
    "CandidateMechanism",
    "CorpusPopulationRequired",
    "CorpusSignatureMatch",
    "CorpusVerification",
    "MechanismClusterRecord",
    "MechanismSignature",
    "NOVELTY_DEFINITIONS",
    "NormalizedMechanismGraph",
    "NoveltyLabel",
    "ProbeKind",
    "ProbeObservation",
    "ProbeSignature",
    "ReferenceCorpusManifest",
    "ReferenceMechanism",
    "cluster_candidates",
    "definition",
    "freeze_corpus",
    "query_corpus",
    "normalize_mechanism_graph",
    "unique_cluster_counts_by_run",
    "verify_frozen_corpus",
]
