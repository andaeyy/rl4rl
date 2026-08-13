"""Immutable completed-run snapshots accepted by sealed evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from evaluation.records import content_sha256, require_bool, require_sha256, utc_now

_FROZEN_CANDIDATE_FIELDS = frozenset(
    {"candidate_id", "source_sha256", "checkpoint_sha256"}
)
_FROZEN_SNAPSHOT_FIELDS = frozenset(
    {
        "snapshot_id",
        "run_id",
        "budget_checkpoint_id",
        "terminal_event_sha256",
        "frozen_at_utc",
        "run_complete",
        "frozen",
        "candidates",
        "snapshot_sha256",
        "schema_name",
        "schema_version",
    }
)


def _identifier(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character in value for character in ("/", "\\", "\x00"))
    ):
        raise ValueError(f"{field_name} must be a safe non-empty identifier")
    return value


@dataclass(frozen=True)
class FrozenCandidateArtifact:
    candidate_id: str
    source_sha256: str
    checkpoint_sha256: str

    def validate(self) -> None:
        _identifier(self.candidate_id, "candidate_id")
        require_sha256(self.source_sha256, "candidate source_sha256")
        require_sha256(self.checkpoint_sha256, "candidate checkpoint_sha256")

    def to_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrozenCandidateArtifact:
        if not isinstance(payload, Mapping) or set(payload) != _FROZEN_CANDIDATE_FIELDS:
            raise ValueError("frozen candidate artifact has invalid fields")
        candidate = cls(
            candidate_id=payload["candidate_id"],
            source_sha256=payload["source_sha256"],
            checkpoint_sha256=payload["checkpoint_sha256"],
        )
        candidate.validate()
        return candidate

    @property
    def artifact_sha256(self) -> str:
        return content_sha256(asdict(self))


@dataclass(frozen=True)
class FrozenRunSnapshot:
    snapshot_id: str
    run_id: str
    budget_checkpoint_id: str
    terminal_event_sha256: str
    frozen_at_utc: str
    run_complete: bool
    frozen: bool
    candidates: tuple[FrozenCandidateArtifact, ...]
    snapshot_sha256: str
    schema_name: str = "FrozenRunSnapshot"
    schema_version: str = "1.0"

    def _hash_payload(self) -> dict[str, object]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "budget_checkpoint_id": self.budget_checkpoint_id,
            "terminal_event_sha256": self.terminal_event_sha256,
            "frozen_at_utc": self.frozen_at_utc,
            "run_complete": self.run_complete,
            "frozen": self.frozen,
            "candidates": [asdict(candidate) for candidate in self.candidates],
        }

    def validate(self, *, require_completed: bool = True) -> None:
        if self.schema_name != "FrozenRunSnapshot" or self.schema_version != "1.0":
            raise ValueError("unsupported frozen run snapshot schema")
        _identifier(self.snapshot_id, "snapshot_id")
        _identifier(self.run_id, "run_id")
        _identifier(self.budget_checkpoint_id, "budget_checkpoint_id")
        if not isinstance(self.frozen_at_utc, str) or not self.frozen_at_utc.endswith(
            "Z"
        ):
            raise ValueError("frozen_at_utc must be an explicit UTC timestamp")
        require_sha256(self.terminal_event_sha256, "terminal_event_sha256")
        require_bool(self.run_complete, "run_complete")
        require_bool(self.frozen, "frozen")
        if require_completed and (
            self.run_complete is not True or self.frozen is not True
        ):
            raise ValueError(
                "sealed evaluation requires a completed, frozen run snapshot"
            )
        if not self.candidates:
            raise ValueError("a frozen run snapshot must contain candidates")
        identifiers: set[str] = set()
        for candidate in self.candidates:
            candidate.validate()
            if candidate.candidate_id in identifiers:
                raise ValueError("duplicate candidate in frozen run snapshot")
            identifiers.add(candidate.candidate_id)
        require_sha256(self.snapshot_sha256, "snapshot_sha256")
        if content_sha256(self._hash_payload()) != self.snapshot_sha256:
            raise ValueError("frozen run snapshot hash mismatch")

    def to_dict(self) -> dict[str, object]:
        self.validate(require_completed=False)
        return {
            **self._hash_payload(),
            "snapshot_sha256": self.snapshot_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FrozenRunSnapshot:
        if not isinstance(payload, Mapping) or set(payload) != _FROZEN_SNAPSHOT_FIELDS:
            raise ValueError("frozen run snapshot has invalid fields")
        raw_candidates = payload["candidates"]
        if not isinstance(raw_candidates, list):
            raise ValueError("frozen run snapshot candidates must be an array")
        snapshot = cls(
            snapshot_id=payload["snapshot_id"],
            run_id=payload["run_id"],
            budget_checkpoint_id=payload["budget_checkpoint_id"],
            terminal_event_sha256=payload["terminal_event_sha256"],
            frozen_at_utc=payload["frozen_at_utc"],
            run_complete=payload["run_complete"],
            frozen=payload["frozen"],
            candidates=tuple(
                FrozenCandidateArtifact.from_dict(candidate)
                for candidate in raw_candidates
            ),
            snapshot_sha256=payload["snapshot_sha256"],
            schema_name=payload["schema_name"],
            schema_version=payload["schema_version"],
        )
        snapshot.validate()
        return snapshot

    def candidate(self, candidate_id: str) -> FrozenCandidateArtifact:
        self.validate()
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise ValueError(f"candidate {candidate_id!r} is absent from frozen snapshot")


def freeze_completed_run(
    *,
    snapshot_id: str,
    run_id: str,
    budget_checkpoint_id: str,
    terminal_event_sha256: str,
    candidate_artifacts: Mapping[str, tuple[str, str]],
    run_complete: bool,
) -> FrozenRunSnapshot:
    """Create a snapshot only after a terminal run event exists."""

    require_bool(run_complete, "run_complete")
    if run_complete is not True:
        raise ValueError("cannot freeze an incomplete run")
    candidates = tuple(
        FrozenCandidateArtifact(
            candidate_id=candidate_id,
            source_sha256=hashes[0],
            checkpoint_sha256=hashes[1],
        )
        for candidate_id, hashes in sorted(candidate_artifacts.items())
    )
    frozen_at = utc_now()
    payload: dict[str, object] = {
        "schema_name": "FrozenRunSnapshot",
        "schema_version": "1.0",
        "snapshot_id": snapshot_id,
        "run_id": run_id,
        "budget_checkpoint_id": budget_checkpoint_id,
        "terminal_event_sha256": terminal_event_sha256,
        "frozen_at_utc": frozen_at,
        "run_complete": True,
        "frozen": True,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
    snapshot = FrozenRunSnapshot(
        snapshot_id=snapshot_id,
        run_id=run_id,
        budget_checkpoint_id=budget_checkpoint_id,
        terminal_event_sha256=terminal_event_sha256,
        frozen_at_utc=frozen_at,
        run_complete=True,
        frozen=True,
        candidates=candidates,
        snapshot_sha256=content_sha256(payload),
    )
    snapshot.validate()
    return snapshot
