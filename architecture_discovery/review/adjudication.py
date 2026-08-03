"""Three-reviewer adjudication and inter-reviewer agreement metrics."""

from __future__ import annotations

import math
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterable

from novelty.serialization import require_identifier, require_sha256, utc_now
from novelty.taxonomy import NoveltyLabel
from review.records import ReviewerAssessment


class AdjudicationStatus(StrEnum):
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    NEEDS_ADDITIONAL_REVIEW = "needs_additional_review"


@dataclass(frozen=True)
class NoveltyReviewRecord:
    record_id: str
    packet_id: str
    corpus_sha256: str
    final_label: NoveltyLabel
    status: AdjudicationStatus
    raw_review_record_ids: tuple[str, ...]
    raw_labels: tuple[tuple[str, NoveltyLabel], ...]
    label_counts: tuple[tuple[NoveltyLabel, int], ...]
    created_at_utc: str
    schema_name: str = "NoveltyReviewRecord"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        require_identifier(self.record_id, "record_id")
        require_identifier(self.packet_id, "packet_id")
        require_sha256(self.corpus_sha256, "corpus_sha256")
        if len(set(self.raw_review_record_ids)) != len(self.raw_review_record_ids):
            raise ValueError("raw review record IDs must be unique")
        if len({reviewer for reviewer, _ in self.raw_labels}) != len(self.raw_labels):
            raise ValueError("raw labels must come from independent reviewers")
        if self.final_label in {NoveltyLabel.N3, NoveltyLabel.N4} and len(self.raw_labels) < 3:
            raise ValueError("N3 and N4 decisions require at least three reviewers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "record_id": self.record_id,
            "packet_id": self.packet_id,
            "corpus_sha256": self.corpus_sha256,
            "final_label": self.final_label.value,
            "status": self.status.value,
            "raw_review_record_ids": list(self.raw_review_record_ids),
            "raw_labels": [
                {"reviewer_pseudonym": reviewer, "label": label.value}
                for reviewer, label in self.raw_labels
            ],
            "label_counts": [
                {"label": label.value, "count": count}
                for label, count in self.label_counts
            ],
            "created_at_utc": self.created_at_utc,
        }


def adjudicate(assessments: Iterable[ReviewerAssessment]) -> NoveltyReviewRecord:
    records = tuple(assessments)
    if len(records) < 2:
        raise ValueError("adjudication requires at least two independent reviews")
    packet_ids = {record.packet_id for record in records}
    corpus_hashes = {record.corpus_sha256 for record in records}
    reviewers = {record.reviewer_pseudonym for record in records}
    record_ids = {record.record_id for record in records}
    if len(packet_ids) != 1 or len(corpus_hashes) != 1:
        raise ValueError("all reviews in an adjudication must concern one packet and corpus")
    if len(reviewers) != len(records):
        raise ValueError("a reviewer may contribute only one label per packet")
    if len(record_ids) != len(records):
        raise ValueError("duplicate raw review records are not allowed")

    counts = Counter(record.label for record in records)
    high_novelty_seen = bool({NoveltyLabel.N3, NoveltyLabel.N4}.intersection(counts))
    if high_novelty_seen and len(records) < 3:
        final_label = NoveltyLabel.X
        status = AdjudicationStatus.NEEDS_ADDITIONAL_REVIEW
    else:
        most_common = counts.most_common()
        winner, winner_count = most_common[0]
        tied = len(most_common) > 1 and most_common[1][1] == winner_count
        strict_majority = winner_count > len(records) / 2
        if tied or not strict_majority or winner is NoveltyLabel.X:
            final_label = NoveltyLabel.X
            status = AdjudicationStatus.UNRESOLVED
        else:
            final_label = winner
            status = AdjudicationStatus.RESOLVED
    ordered = tuple(sorted(records, key=lambda item: item.reviewer_pseudonym))
    return NoveltyReviewRecord(
        record_id=f"novelty-adjudication-{uuid.uuid4().hex}",
        packet_id=next(iter(packet_ids)),
        corpus_sha256=next(iter(corpus_hashes)),
        final_label=final_label,
        status=status,
        raw_review_record_ids=tuple(record.record_id for record in ordered),
        raw_labels=tuple(
            (record.reviewer_pseudonym, record.label) for record in ordered
        ),
        label_counts=tuple(
            (label, counts[label]) for label in NoveltyLabel if counts[label]
        ),
        created_at_utc=utc_now(),
    )


@dataclass(frozen=True)
class AgreementMetrics:
    packet_count: int
    rating_count: int
    observed_pairwise_agreement: float
    expected_chance_agreement: float
    fleiss_kappa: float | None
    label_totals: tuple[tuple[NoveltyLabel, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "packet_count": self.packet_count,
            "rating_count": self.rating_count,
            "observed_pairwise_agreement": self.observed_pairwise_agreement,
            "expected_chance_agreement": self.expected_chance_agreement,
            "fleiss_kappa": self.fleiss_kappa,
            "label_totals": [
                {"label": label.value, "count": count}
                for label, count in self.label_totals
            ],
        }


def agreement_metrics(
    assessments: Iterable[ReviewerAssessment],
) -> AgreementMetrics:
    records = tuple(assessments)
    grouped: dict[str, list[ReviewerAssessment]] = defaultdict(list)
    for record in records:
        grouped[record.packet_id].append(record)
    usable = [items for items in grouped.values() if len(items) >= 2]
    if not usable:
        raise ValueError("agreement requires at least one packet with two reviews")
    agreements: list[float] = []
    totals: Counter[NoveltyLabel] = Counter()
    rating_count = 0
    for items in usable:
        reviewer_ids = [item.reviewer_pseudonym for item in items]
        if len(set(reviewer_ids)) != len(reviewer_ids):
            raise ValueError("agreement input contains duplicate reviewers for a packet")
        counts = Counter(item.label for item in items)
        ratings = len(items)
        agreeing_pairs = sum(count * (count - 1) for count in counts.values())
        agreements.append(agreeing_pairs / (ratings * (ratings - 1)))
        totals.update(counts)
        rating_count += ratings
    observed = sum(agreements) / len(agreements)
    expected = sum((count / rating_count) ** 2 for count in totals.values())
    if math.isclose(expected, 1.0):
        kappa = 1.0 if math.isclose(observed, 1.0) else None
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return AgreementMetrics(
        packet_count=len(usable),
        rating_count=rating_count,
        observed_pairwise_agreement=observed,
        expected_chance_agreement=expected,
        fleiss_kappa=kappa,
        label_totals=tuple(
            (label, totals[label]) for label in NoveltyLabel if totals[label]
        ),
    )
