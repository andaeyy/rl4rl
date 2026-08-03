"""Independent reviewer records that preserve every raw novelty label."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from novelty.serialization import require_identifier, require_sha256, utc_now
from novelty.taxonomy import NoveltyLabel
from review.blinding import assert_blind_text


class ReviewConfidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class ReviewerAssessment:
    record_id: str
    packet_id: str
    reviewer_pseudonym: str
    corpus_sha256: str
    label: NoveltyLabel
    confidence: ReviewConfidence
    rationale: str
    nearest_reference_ids: tuple[str, ...]
    created_at_utc: str
    schema_name: str = "IndependentNoveltyAssessment"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        require_identifier(self.record_id, "record_id")
        require_identifier(self.packet_id, "packet_id")
        require_identifier(self.reviewer_pseudonym, "reviewer_pseudonym")
        require_sha256(self.corpus_sha256, "corpus_sha256")
        if not self.rationale.strip():
            raise ValueError("review rationale cannot be empty")
        assert_blind_text(self.rationale, "rationale")
        references = tuple(sorted(set(self.nearest_reference_ids)))
        for reference_id in references:
            require_identifier(reference_id, "nearest_reference_id")
        object.__setattr__(self, "nearest_reference_ids", references)

    @classmethod
    def create(
        cls,
        *,
        packet_id: str,
        reviewer_pseudonym: str,
        corpus_sha256: str,
        label: NoveltyLabel | str,
        confidence: ReviewConfidence | str,
        rationale: str,
        nearest_reference_ids: tuple[str, ...] = (),
    ) -> "ReviewerAssessment":
        return cls(
            record_id=f"novelty-review-{uuid.uuid4().hex}",
            packet_id=packet_id,
            reviewer_pseudonym=reviewer_pseudonym,
            corpus_sha256=corpus_sha256,
            label=NoveltyLabel(label),
            confidence=ReviewConfidence(confidence),
            rationale=rationale,
            nearest_reference_ids=nearest_reference_ids,
            created_at_utc=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "packet_id": self.packet_id,
            "reviewer_pseudonym": self.reviewer_pseudonym,
            "corpus_sha256": self.corpus_sha256,
            "label": self.label.value,
            "confidence": self.confidence.value,
            "rationale": self.rationale,
            "nearest_reference_ids": list(self.nearest_reference_ids),
            "created_at_utc": self.created_at_utc,
        }
