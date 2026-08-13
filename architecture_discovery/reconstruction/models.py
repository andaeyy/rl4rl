"""Typed products reconstructed from the immutable event ledger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from analysis.outcomes import RunOutcome
from artifacts.records import ArtifactContext, content_sha256
from artifacts.store import IntegrityFinding


@dataclass(frozen=True)
class ReconstructedRun:
    context: ArtifactContext
    status: str
    last_sequence: int
    last_event_sha256: str
    event_record_ids: tuple[str, ...]
    budget_totals: Mapping[str, int | float]
    accelerator_kind: str | None
    ancestry: Mapping[str, tuple[str, ...]]
    qualifying_mechanism_cluster_keys: tuple[str, ...]
    parent_selection_history: tuple[tuple[str, ...], ...]
    promotion_ids: tuple[str, ...]
    review_ids: tuple[str, ...]
    rerun_attempt_ids: tuple[str, ...]
    failure_domain: str
    failure_class: str
    integrity_findings: tuple[IntegrityFinding, ...]
    outcome: RunOutcome | None
    schema_name: str = "ReconstructedRun"
    schema_version: str = "2.0"

    @property
    def reconstruction_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "context": self.context.to_dict(),
            "status": self.status,
            "last_sequence": self.last_sequence,
            "last_event_sha256": self.last_event_sha256,
            "event_record_ids": list(self.event_record_ids),
            "budget_totals": {
                key: self.budget_totals[key] for key in sorted(self.budget_totals)
            },
            "ancestry": {
                candidate_id: list(self.ancestry[candidate_id])
                for candidate_id in sorted(self.ancestry)
            },
            "qualifying_mechanism_cluster_keys": list(
                self.qualifying_mechanism_cluster_keys
            ),
            "parent_selection_history": [
                list(parent_ids) for parent_ids in self.parent_selection_history
            ],
            "promotion_ids": list(self.promotion_ids),
            "review_ids": list(self.review_ids),
            "rerun_attempt_ids": list(self.rerun_attempt_ids),
            "failure_domain": self.failure_domain,
            "failure_class": self.failure_class,
            "integrity_findings": [item.to_dict() for item in self.integrity_findings],
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }
        if self.schema_version == "2.0":
            payload["accelerator_kind"] = self.accelerator_kind
        elif self.schema_version != "1.0":
            raise ValueError("unsupported ReconstructedRun schema version")
        return payload

    def analysis_ready_rows(self) -> tuple[dict[str, Any], ...]:
        """Return zero or one run-level rows, never candidate-level pseudo-replicates."""

        return () if self.outcome is None else (self.outcome.to_dict(),)
