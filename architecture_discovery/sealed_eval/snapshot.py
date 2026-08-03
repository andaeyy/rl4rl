"""Immutable completed-run snapshots accepted by sealed evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from evaluation.records import content_sha256, require_sha256, utc_now


@dataclass(frozen=True)
class FrozenCandidateArtifact:
    candidate_id: str
    source_sha256: str
    checkpoint_sha256: str

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id is required")
        require_sha256(self.source_sha256, "candidate source_sha256")
        require_sha256(self.checkpoint_sha256, "candidate checkpoint_sha256")

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

    def _hash_payload(self) -> dict[str, object]:
        return {
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
        if not self.snapshot_id or not self.run_id or not self.budget_checkpoint_id:
            raise ValueError("snapshot, run, and budget-checkpoint IDs are required")
        require_sha256(self.terminal_event_sha256, "terminal_event_sha256")
        if require_completed and (not self.run_complete or not self.frozen):
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

    if not run_complete:
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

