from __future__ import annotations

import hashlib

import pytest

from novelty.taxonomy import NoveltyLabel
from review.adjudication import (
    AdjudicationStatus,
    adjudicate,
    agreement_metrics,
)
from review.records import ReviewConfidence, ReviewerAssessment


CORPUS_HASH = hashlib.sha256(b"frozen corpus").hexdigest()


def assessment(reviewer: str, label: NoveltyLabel) -> ReviewerAssessment:
    return ReviewerAssessment.create(
        packet_id="packet-one",
        reviewer_pseudonym=reviewer,
        corpus_sha256=CORPUS_HASH,
        label=label,
        confidence=ReviewConfidence.HIGH,
        rationale="The categorical mechanism evidence supports this corpus comparison.",
        nearest_reference_ids=("reference-one",),
    )


def test_n3_n4_require_three_independent_reviewers_and_preserve_raw_labels() -> None:
    two = [assessment("reviewer-a", NoveltyLabel.N4), assessment("reviewer-b", NoveltyLabel.N4)]
    pending = adjudicate(two)
    assert pending.status is AdjudicationStatus.NEEDS_ADDITIONAL_REVIEW
    assert pending.final_label is NoveltyLabel.X

    three = [*two, assessment("reviewer-c", NoveltyLabel.N3)]
    resolved = adjudicate(three)
    assert resolved.status is AdjudicationStatus.RESOLVED
    assert resolved.final_label is NoveltyLabel.N4
    assert resolved.raw_labels == (
        ("reviewer-a", NoveltyLabel.N4),
        ("reviewer-b", NoveltyLabel.N4),
        ("reviewer-c", NoveltyLabel.N3),
    )


def test_duplicate_reviewer_is_not_independent() -> None:
    with pytest.raises(ValueError, match="only one label"):
        adjudicate(
            [
                assessment("reviewer-a", NoveltyLabel.N2),
                assessment("reviewer-a", NoveltyLabel.N2),
            ]
        )


def test_agreement_metrics_retain_label_totals() -> None:
    records = [
        assessment("reviewer-a", NoveltyLabel.N4),
        assessment("reviewer-b", NoveltyLabel.N4),
        assessment("reviewer-c", NoveltyLabel.N3),
    ]
    metrics = agreement_metrics(records)

    assert metrics.packet_count == 1
    assert metrics.rating_count == 3
    assert metrics.observed_pairwise_agreement == pytest.approx(1 / 3)
    assert dict(metrics.label_totals) == {NoveltyLabel.N3: 1, NoveltyLabel.N4: 2}
