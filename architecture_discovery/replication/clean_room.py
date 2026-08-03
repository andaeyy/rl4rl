"""Evidence that a replication implementation is independent of candidate weights/code."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mechanism.validation import require_bool, require_identifier, require_sha256
from replication.policy import ReplicationPolicy
from study.serialization import content_hash


@dataclass(frozen=True)
class CleanRoomReimplementationRecord:
    record_id: str
    candidate_snapshot_id: str
    candidate_snapshot_sha256: str
    architecture_spec_sha256: str
    implementation_sha256: str
    original_checkpoint_sha256: str
    builder_id: str
    implementer_id: str
    protocol_id: str
    original_candidate_source_accessed: bool
    original_checkpoint_accessed: bool
    initialization_checkpoint_id: str | None = None
    schema_name: str = field(default="CleanRoomReimplementationRecord", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_bool(
            self.original_candidate_source_accessed,
            "original_candidate_source_accessed",
        )
        require_bool(self.original_checkpoint_accessed, "original_checkpoint_accessed")
        for field_name in (
            "record_id",
            "candidate_snapshot_id",
            "builder_id",
            "implementer_id",
            "protocol_id",
        ):
            require_identifier(getattr(self, field_name), field_name)
        for field_name in (
            "candidate_snapshot_sha256",
            "architecture_spec_sha256",
            "implementation_sha256",
            "original_checkpoint_sha256",
        ):
            require_sha256(getattr(self, field_name), field_name)
        if self.original_candidate_source_accessed:
            raise ValueError(
                "clean-room implementer cannot access original candidate source"
            )
        if self.original_checkpoint_accessed:
            raise ValueError("clean-room implementer cannot access original checkpoint")
        if self.initialization_checkpoint_id is not None:
            raise ValueError("clean-room replication cannot initialize from a checkpoint")

    @property
    def record_hash(self) -> str:
        return content_hash(self.to_dict())

    def assert_compatible(self, policy: ReplicationPolicy) -> None:
        if self.candidate_snapshot_id != policy.frozen_snapshot_id:
            raise ValueError("clean-room record refers to the wrong snapshot")
        if self.candidate_snapshot_sha256 != policy.frozen_snapshot_sha256:
            raise ValueError("clean-room record snapshot hash does not match policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "candidate_snapshot_id": self.candidate_snapshot_id,
            "candidate_snapshot_sha256": self.candidate_snapshot_sha256,
            "architecture_spec_sha256": self.architecture_spec_sha256,
            "implementation_sha256": self.implementation_sha256,
            "original_checkpoint_sha256": self.original_checkpoint_sha256,
            "builder_id": self.builder_id,
            "implementer_id": self.implementer_id,
            "protocol_id": self.protocol_id,
            "original_candidate_source_accessed": self.original_candidate_source_accessed,
            "original_checkpoint_accessed": self.original_checkpoint_accessed,
            "initialization_checkpoint_id": self.initialization_checkpoint_id,
        }
