"""Immutable research artifacts independent of controller retention policies."""

from artifacts.failures import (
    DEFAULT_RERUNNABLE_INFRASTRUCTURE_CLASSES,
    FailureClass,
    FailureDomain,
    FailureRecord,
    RerunAuthorization,
    RerunNotAuthorized,
    RerunPolicy,
    authorize_rerun,
)
from artifacts.index import ArtifactIndex, ArtifactIndexEntry, INDEX_CATEGORIES
from artifacts.records import (
    GENESIS_EVENT_SHA256,
    ArtifactContext,
    EventKind,
    EventRecord,
)
from artifacts.store import (
    ArtifactIntegrityError,
    ContentAddressedObjectStore,
    FrozenIndexReference,
    IntegrityFinding,
    IntegrityReport,
    IntegritySeverity,
    ObjectReference,
    RunArtifactStore,
    RunLock,
)
from artifacts.study_sink import (
    ArtifactEmittingStudyEngine,
    ImmutableStudyEventSink,
    SinkObservation,
    StudyEventSinkError,
    StudyStateEventSink,
)

__all__ = [
    "DEFAULT_RERUNNABLE_INFRASTRUCTURE_CLASSES",
    "GENESIS_EVENT_SHA256",
    "INDEX_CATEGORIES",
    "ArtifactContext",
    "ArtifactIndex",
    "ArtifactIndexEntry",
    "ArtifactEmittingStudyEngine",
    "ArtifactIntegrityError",
    "ContentAddressedObjectStore",
    "EventKind",
    "EventRecord",
    "FailureClass",
    "FailureDomain",
    "FailureRecord",
    "FrozenIndexReference",
    "IntegrityFinding",
    "IntegrityReport",
    "IntegritySeverity",
    "ImmutableStudyEventSink",
    "ObjectReference",
    "RerunAuthorization",
    "RerunNotAuthorized",
    "RerunPolicy",
    "RunArtifactStore",
    "RunLock",
    "SinkObservation",
    "StudyEventSinkError",
    "StudyStateEventSink",
    "authorize_rerun",
]
