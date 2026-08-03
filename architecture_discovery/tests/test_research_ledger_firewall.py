from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from common.evaluation_profiles import EvaluationLayer
from research_ledger import (
    AdaptiveHypothesisUpdate,
    ConfidenceLevel,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceUse,
    FailedPredictionRecord,
    HypothesisStatus,
    PostsearchAssessment,
    ResearchLedger,
    ResearchLedgerBoundaryError,
    freeze_protocol,
)
from test_research_ledger_protocol import toy_protocol


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def layer_a_evidence(*, direction: EvidenceDirection = EvidenceDirection.SUPPORTS) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="evidence-layer-a-one",
        hypothesis_id="hypothesis-routing",
        intended_use=EvidenceUse.ADAPTIVE_SEARCH,
        source_layer=EvaluationLayer.SEARCH,
        source_schema_name="search_evaluation",
        source_record_id="search-record-one",
        source_record_sha256=digest("search record"),
        direction=direction,
        prediction_ids=("prediction-carry",),
        discriminating_test_ids=("test-route-zeroing",),
        summary="The public probe produced the preregistered categorical response.",
        observed_at_utc="2026-07-31T12:00:00Z",
    )


def layer_b_evidence() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="evidence-layer-b-one",
        hypothesis_id="hypothesis-routing",
        intended_use=EvidenceUse.POSTSEARCH_ASSESSMENT,
        source_layer=EvaluationLayer.QUALIFICATION,
        source_schema_name="qualification_evaluation",
        source_record_id="qualification-record-one",
        source_record_sha256=digest("qualification record"),
        direction=EvidenceDirection.SUPPORTS,
        prediction_ids=("prediction-carry",),
        discriminating_test_ids=("test-route-zeroing",),
        summary="The sealed qualification evidence supports the prediction.",
        observed_at_utc="2026-07-31T13:00:00Z",
    )


def test_adaptive_update_accepts_only_layer_a_and_cannot_mutate_protocol(tmp_path: Path) -> None:
    protocol = toy_protocol()
    ledger = ResearchLedger(freeze_protocol(protocol, tmp_path / "protocol.json"))
    prior = ledger.state("hypothesis-routing")
    evidence = layer_a_evidence()
    update = AdaptiveHypothesisUpdate(
        update_id="update-one",
        hypothesis_id="hypothesis-routing",
        protocol_sha256=protocol.protocol_hash,
        expected_prior_state_sha256=prior.state_hash,
        evidence_ids=(evidence.evidence_id,),
        failed_prediction_ids=(),
        new_confidence=ConfidenceLevel.HIGH,
        new_status=HypothesisStatus.ACTIVE,
        rationale="The public result increased confidence without changing the claim.",
    )

    revised = ledger.apply_adaptive_update(update, evidence=(evidence,))

    assert revised.revision == 1
    assert revised.hypothesis_spec_sha256 == prior.hypothesis_spec_sha256
    assert protocol.hypothesis("hypothesis-routing").causal_claim == (
        "The routed path causes improved propagation of carry state."
    )
    assert ledger.to_dict()["adaptive_evidence"][0]["source_layer"] == "layer_a"

    with pytest.raises(ValueError, match="restricted to Layer A"):
        replace(layer_b_evidence(), intended_use=EvidenceUse.ADAPTIVE_SEARCH)


def test_layer_b_cannot_enter_search_and_is_accepted_only_after_close(tmp_path: Path) -> None:
    protocol = toy_protocol()
    ledger = ResearchLedger(freeze_protocol(protocol, tmp_path / "protocol.json"))
    sealed = layer_b_evidence()

    with pytest.raises(ResearchLedgerBoundaryError, match="only after search closes"):
        ledger.record_postsearch_evidence((sealed,))

    close = ledger.close_search(
        closure_id="search-close-one",
        reason="The frozen proposal budget reached its terminal event.",
    )
    ledger.record_postsearch_evidence((sealed,))
    assessment = PostsearchAssessment(
        assessment_id="assessment-one",
        hypothesis_id="hypothesis-routing",
        protocol_sha256=protocol.protocol_hash,
        search_close_event_sha256=close.event_sha256,
        evidence_ids=(sealed.evidence_id,),
        confidence=ConfidenceLevel.HIGH,
        status=HypothesisStatus.SUPPORTED,
        rationale="The sealed evidence supports the frozen prediction.",
    )
    ledger.record_postsearch_assessment(assessment)

    assert ledger.state("hypothesis-routing").revision == 0
    assert ledger.to_dict()["postsearch_assessments"][0]["status"] == "supported"

    public = layer_a_evidence()
    update = AdaptiveHypothesisUpdate(
        update_id="late-update",
        hypothesis_id="hypothesis-routing",
        protocol_sha256=protocol.protocol_hash,
        expected_prior_state_sha256=ledger.state("hypothesis-routing").state_hash,
        evidence_ids=(public.evidence_id,),
        failed_prediction_ids=(),
        new_confidence=ConfidenceLevel.HIGH,
        new_status=HypothesisStatus.ACTIVE,
        rationale="This update is deliberately too late.",
    )
    with pytest.raises(ResearchLedgerBoundaryError, match="after the search closes"):
        ledger.apply_adaptive_update(update, evidence=(public,))


def test_failed_prediction_is_explicit_and_requires_contradicting_evidence(tmp_path: Path) -> None:
    protocol = toy_protocol()
    ledger = ResearchLedger(freeze_protocol(protocol, tmp_path / "protocol.json"))
    evidence = layer_a_evidence(direction=EvidenceDirection.CONTRADICTS)
    failure = FailedPredictionRecord(
        failure_id="failed-prediction-one",
        hypothesis_id="hypothesis-routing",
        prediction_id="prediction-carry",
        evidence_ids=(evidence.evidence_id,),
        explanation="The public probe met the frozen falsification condition.",
        failed_at_utc="2026-07-31T12:05:00Z",
    )
    prior = ledger.state("hypothesis-routing")
    update = AdaptiveHypothesisUpdate(
        update_id="update-falsification",
        hypothesis_id="hypothesis-routing",
        protocol_sha256=protocol.protocol_hash,
        expected_prior_state_sha256=prior.state_hash,
        evidence_ids=(evidence.evidence_id,),
        failed_prediction_ids=(failure.failure_id,),
        new_confidence=ConfidenceLevel.LOW,
        new_status=HypothesisStatus.WEAKENED,
        rationale="The failed prediction weakens the hypothesis.",
    )

    state = ledger.apply_adaptive_update(
        update,
        evidence=(evidence,),
        failed_predictions=(failure,),
    )

    assert state.failed_prediction_ids == ("failed-prediction-one",)
    assert ledger.to_dict()["failed_predictions"][0]["prediction_id"] == (
        "prediction-carry"
    )
