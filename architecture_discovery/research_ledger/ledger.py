"""Append-only research ledger with a hard Layer A adaptation firewall."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from common.evaluation_profiles import EvaluationLayer
from research_ledger.protocol import FrozenResearchProtocol
from research_ledger.records import (
    AdaptiveHypothesisUpdate,
    EvidenceDirection,
    EvidenceRecord,
    EvidenceUse,
    FailedPredictionRecord,
    HypothesisState,
    HypothesisStatus,
    LedgerPhase,
    PostsearchAssessment,
    require_identifier,
    require_sha256,
    require_text,
)
from study.serialization import content_hash


class ResearchLedgerBoundaryError(RuntimeError):
    """Raised before sealed evidence or protocol changes can affect search."""


class LedgerEventKind(StrEnum):
    ADAPTIVE_UPDATE = "adaptive_update"
    SEARCH_CLOSED = "search_closed"
    POSTSEARCH_EVIDENCE = "postsearch_evidence"
    POSTSEARCH_ASSESSMENT = "postsearch_assessment"
    LEDGER_SEALED = "ledger_sealed"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported ledger event payload {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class ResearchLedgerEvent:
    sequence: int
    event_kind: LedgerEventKind
    created_at_utc: str
    protocol_sha256: str
    previous_event_sha256: str | None
    payload: Mapping[str, Any]
    event_sha256: str
    schema_name: str = "ResearchLedgerEvent"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))

    def hash_payload(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "event_kind": self.event_kind.value,
            "created_at_utc": self.created_at_utc,
            "protocol_sha256": self.protocol_sha256,
            "previous_event_sha256": self.previous_event_sha256,
            "payload": _thaw(self.payload),
        }

    def validate(self) -> None:
        if self.sequence < 0:
            raise ValueError("ledger event sequence cannot be negative")
        require_sha256(self.protocol_sha256, "protocol_sha256")
        if self.previous_event_sha256 is not None:
            require_sha256(self.previous_event_sha256, "previous_event_sha256")
        require_sha256(self.event_sha256, "event_sha256")
        if content_hash(self.hash_payload()) != self.event_sha256:
            raise ValueError("research ledger event hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {**self.hash_payload(), "event_sha256": self.event_sha256}


class ResearchLedger:
    """Mutable append interface over immutable protocol and event records.

    The search-state projection can change only through
    :meth:`apply_adaptive_update`, which accepts exact Layer A evidence. Layer B
    and C evidence can be recorded only after a terminal search-close event and
    is stored outside the adaptive state projection.
    """

    def __init__(self, frozen_protocol: FrozenResearchProtocol) -> None:
        if not isinstance(frozen_protocol, FrozenResearchProtocol):
            raise TypeError("research ledger requires a frozen protocol receipt")
        frozen_protocol.verify()
        self._frozen_protocol = frozen_protocol
        self._phase = LedgerPhase.SEARCH
        self._states = {
            item.hypothesis_id: HypothesisState(
                hypothesis_id=item.hypothesis_id,
                hypothesis_spec_sha256=item.spec_hash,
                confidence=item.initial_confidence,
                status=item.initial_status,
                evidence_ids=(),
                failed_prediction_ids=(),
                revision=0,
            )
            for item in frozen_protocol.protocol.hypotheses
        }
        self._adaptive_evidence: dict[str, EvidenceRecord] = {}
        self._postsearch_evidence: dict[str, EvidenceRecord] = {}
        self._failed_predictions: dict[str, FailedPredictionRecord] = {}
        self._adaptive_updates: dict[str, AdaptiveHypothesisUpdate] = {}
        self._postsearch_assessments: dict[str, PostsearchAssessment] = {}
        self._events: list[ResearchLedgerEvent] = []
        self._search_close_event_sha256: str | None = None

    @property
    def protocol_sha256(self) -> str:
        return self._frozen_protocol.protocol_sha256

    @property
    def phase(self) -> LedgerPhase:
        return self._phase

    @property
    def events(self) -> tuple[ResearchLedgerEvent, ...]:
        return tuple(self._events)

    @property
    def search_close_event_sha256(self) -> str | None:
        return self._search_close_event_sha256

    def state(self, hypothesis_id: str) -> HypothesisState:
        try:
            return self._states[hypothesis_id]
        except KeyError as error:
            raise ValueError(f"unknown frozen hypothesis {hypothesis_id!r}") from error

    def _new_event(
        self, event_kind: LedgerEventKind, payload: dict[str, Any]
    ) -> ResearchLedgerEvent:
        created_at = _utc_now()
        previous = self._events[-1].event_sha256 if self._events else None
        event = ResearchLedgerEvent(
            sequence=len(self._events),
            event_kind=event_kind,
            created_at_utc=created_at,
            protocol_sha256=self.protocol_sha256,
            previous_event_sha256=previous,
            payload=payload,
            event_sha256=content_hash(
                {
                    "schema_name": "ResearchLedgerEvent",
                    "schema_version": "1.0",
                    "sequence": len(self._events),
                    "event_kind": event_kind.value,
                    "created_at_utc": created_at,
                    "protocol_sha256": self.protocol_sha256,
                    "previous_event_sha256": previous,
                    "payload": payload,
                }
            ),
        )
        event.validate()
        return event

    def _verify_protocol(self) -> None:
        try:
            self._frozen_protocol.verify()
        except ValueError as error:
            raise ResearchLedgerBoundaryError(
                "frozen protocol verification failed; no ledger mutation was applied"
            ) from error

    def apply_adaptive_update(
        self,
        update: AdaptiveHypothesisUpdate,
        *,
        evidence: Iterable[EvidenceRecord],
        failed_predictions: Iterable[FailedPredictionRecord] = (),
    ) -> HypothesisState:
        self._verify_protocol()
        if self._phase is not LedgerPhase.SEARCH:
            raise ResearchLedgerBoundaryError(
                "adaptive updates are prohibited after the search closes"
            )
        if update.protocol_sha256 != self.protocol_sha256:
            raise ResearchLedgerBoundaryError("update refers to a different protocol")
        if update.update_id in self._adaptive_updates:
            raise ValueError("adaptive update ID already exists")
        hypothesis = self._frozen_protocol.protocol.hypothesis(update.hypothesis_id)
        prior = self.state(update.hypothesis_id)
        if update.expected_prior_state_sha256 != prior.state_hash:
            raise ValueError("adaptive update was built from a stale hypothesis state")

        new_evidence = tuple(evidence)
        if not new_evidence:
            raise ValueError("adaptive update must provide new Layer A evidence")
        if len({item.evidence_id for item in new_evidence}) != len(new_evidence):
            raise ValueError("adaptive evidence IDs must be unique")
        known_predictions = {item.prediction_id for item in hypothesis.predictions}
        known_tests = {item.test_id for item in hypothesis.discriminating_tests}
        for item in new_evidence:
            if item.evidence_id in self._adaptive_evidence or item.evidence_id in self._postsearch_evidence:
                raise ValueError("evidence ID already exists in the research ledger")
            if item.hypothesis_id != update.hypothesis_id:
                raise ValueError("adaptive evidence refers to a different hypothesis")
            if item.intended_use is not EvidenceUse.ADAPTIVE_SEARCH:
                raise ResearchLedgerBoundaryError("post-search evidence cannot drive adaptation")
            if item.source_layer is not EvaluationLayer.SEARCH:
                raise ResearchLedgerBoundaryError("Layer B/C evidence cannot enter search")
            if not set(item.prediction_ids).issubset(known_predictions):
                raise ValueError("adaptive evidence refers to an unknown prediction")
            if not set(item.discriminating_test_ids).issubset(known_tests):
                raise ValueError("adaptive evidence refers to an unknown test")

        available_evidence = {
            **self._adaptive_evidence,
            **{item.evidence_id: item for item in new_evidence},
        }
        expected_evidence_ids = tuple(
            sorted(set(prior.evidence_ids).union(item.evidence_id for item in new_evidence))
        )
        if update.evidence_ids != expected_evidence_ids:
            raise ValueError("adaptive state must retain every prior and new evidence ID")

        failures = tuple(failed_predictions)
        if len({item.failure_id for item in failures}) != len(failures):
            raise ValueError("failed-prediction IDs must be unique")
        for failure in failures:
            if failure.failure_id in self._failed_predictions:
                raise ValueError("failed-prediction ID already exists")
            if failure.hypothesis_id != update.hypothesis_id:
                raise ValueError("failed prediction refers to a different hypothesis")
            if failure.prediction_id not in known_predictions:
                raise ValueError("failed prediction is absent from the frozen protocol")
            if not set(failure.evidence_ids).issubset(available_evidence):
                raise ValueError("failed prediction refers to unavailable evidence")
            if not any(
                available_evidence[evidence_id].direction
                is EvidenceDirection.CONTRADICTS
                for evidence_id in failure.evidence_ids
            ):
                raise ValueError("failed prediction needs contradicting Layer A evidence")
        expected_failures = tuple(
            sorted(
                set(prior.failed_prediction_ids).union(
                    item.failure_id for item in failures
                )
            )
        )
        if update.failed_prediction_ids != expected_failures:
            raise ValueError("adaptive state must retain every failed prediction")
        if failures and update.new_status in {
            HypothesisStatus.PLANNED,
            HypothesisStatus.ACTIVE,
        }:
            raise ValueError("a failed prediction must weaken or close the hypothesis")
        if prior.status is HypothesisStatus.REFUTED and update.new_status not in {
            HypothesisStatus.REFUTED,
            HypothesisStatus.INCONCLUSIVE,
        }:
            raise ValueError("search adaptation cannot reactivate a refuted hypothesis")

        revised = HypothesisState(
            hypothesis_id=prior.hypothesis_id,
            hypothesis_spec_sha256=prior.hypothesis_spec_sha256,
            confidence=update.new_confidence,
            status=update.new_status,
            evidence_ids=expected_evidence_ids,
            failed_prediction_ids=expected_failures,
            revision=prior.revision + 1,
        )
        event = self._new_event(
            LedgerEventKind.ADAPTIVE_UPDATE,
            {
                "update": update.to_dict(),
                "new_evidence": [item.to_dict() for item in new_evidence],
                "failed_predictions": [item.to_dict() for item in failures],
                "prior_state_sha256": prior.state_hash,
                "revised_state": revised.to_dict(),
                "revised_state_sha256": revised.state_hash,
            },
        )
        self._adaptive_evidence.update(
            {item.evidence_id: item for item in new_evidence}
        )
        self._failed_predictions.update(
            {item.failure_id: item for item in failures}
        )
        self._adaptive_updates[update.update_id] = update
        self._states[update.hypothesis_id] = revised
        self._events.append(event)
        return revised

    def close_search(self, *, closure_id: str, reason: str) -> ResearchLedgerEvent:
        self._verify_protocol()
        if self._phase is not LedgerPhase.SEARCH:
            raise ResearchLedgerBoundaryError("search is already closed")
        require_identifier(closure_id, "closure_id")
        require_text(reason, "search-close reason")
        event = self._new_event(
            LedgerEventKind.SEARCH_CLOSED,
            {
                "closure_id": closure_id,
                "reason": reason,
                "frozen_search_states": {
                    hypothesis_id: state.to_dict()
                    for hypothesis_id, state in sorted(self._states.items())
                },
                "adaptive_evidence_ids": sorted(self._adaptive_evidence),
            },
        )
        self._events.append(event)
        self._search_close_event_sha256 = event.event_sha256
        self._phase = LedgerPhase.POSTSEARCH
        return event

    def record_postsearch_evidence(
        self, evidence: Iterable[EvidenceRecord]
    ) -> tuple[EvidenceRecord, ...]:
        self._verify_protocol()
        if self._phase is not LedgerPhase.POSTSEARCH:
            raise ResearchLedgerBoundaryError(
                "sealed evidence is accepted only after search closes and before sealing"
            )
        records = tuple(evidence)
        if not records:
            raise ValueError("post-search evidence batch cannot be empty")
        if len({item.evidence_id for item in records}) != len(records):
            raise ValueError("post-search evidence IDs must be unique")
        for item in records:
            if item.evidence_id in self._adaptive_evidence or item.evidence_id in self._postsearch_evidence:
                raise ValueError("evidence ID already exists")
            self._frozen_protocol.protocol.hypothesis(item.hypothesis_id)
            if item.intended_use is not EvidenceUse.POSTSEARCH_ASSESSMENT:
                raise ResearchLedgerBoundaryError("adaptive evidence belongs in the search ledger")
            if item.source_layer not in {
                EvaluationLayer.QUALIFICATION,
                EvaluationLayer.CONFIRMATION,
            }:
                raise ResearchLedgerBoundaryError("post-search records require Layer B/C")
        event = self._new_event(
            LedgerEventKind.POSTSEARCH_EVIDENCE,
            {"evidence": [item.to_dict() for item in records]},
        )
        self._postsearch_evidence.update({item.evidence_id: item for item in records})
        self._events.append(event)
        return records

    def record_postsearch_assessment(
        self, assessment: PostsearchAssessment
    ) -> PostsearchAssessment:
        self._verify_protocol()
        if self._phase is not LedgerPhase.POSTSEARCH:
            raise ResearchLedgerBoundaryError("post-search assessment is not currently open")
        if assessment.assessment_id in self._postsearch_assessments:
            raise ValueError("post-search assessment ID already exists")
        if assessment.protocol_sha256 != self.protocol_sha256:
            raise ResearchLedgerBoundaryError("assessment refers to another protocol")
        if assessment.search_close_event_sha256 != self._search_close_event_sha256:
            raise ResearchLedgerBoundaryError("assessment bypasses the search-close boundary")
        self._frozen_protocol.protocol.hypothesis(assessment.hypothesis_id)
        if not set(assessment.evidence_ids).issubset(self._postsearch_evidence):
            raise ValueError("assessment refers to unavailable sealed evidence")
        if any(
            self._postsearch_evidence[evidence_id].hypothesis_id
            != assessment.hypothesis_id
            for evidence_id in assessment.evidence_ids
        ):
            raise ValueError("assessment evidence belongs to a different hypothesis")
        event = self._new_event(
            LedgerEventKind.POSTSEARCH_ASSESSMENT,
            {"assessment": assessment.to_dict()},
        )
        self._postsearch_assessments[assessment.assessment_id] = assessment
        self._events.append(event)
        return assessment

    def seal(self, *, seal_id: str, reason: str) -> ResearchLedgerEvent:
        self._verify_protocol()
        if self._phase is not LedgerPhase.POSTSEARCH:
            raise ResearchLedgerBoundaryError("ledger can seal only after search closes")
        require_identifier(seal_id, "seal_id")
        require_text(reason, "seal reason")
        event = self._new_event(
            LedgerEventKind.LEDGER_SEALED,
            {
                "seal_id": seal_id,
                "reason": reason,
                "postsearch_assessment_ids": sorted(self._postsearch_assessments),
            },
        )
        self._events.append(event)
        self._phase = LedgerPhase.SEALED
        return event

    def verify_integrity(self) -> None:
        self._verify_protocol()
        previous: str | None = None
        for sequence, event in enumerate(self._events):
            event.validate()
            if event.sequence != sequence:
                raise ValueError("research ledger event sequence is not contiguous")
            if event.previous_event_sha256 != previous:
                raise ValueError("research ledger event hash chain is broken")
            if event.protocol_sha256 != self.protocol_sha256:
                raise ValueError("research ledger event refers to another protocol")
            previous = event.event_sha256

    @property
    def ledger_hash(self) -> str:
        self.verify_integrity()
        return content_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_name": "ResearchLedger",
            "schema_version": "1.0",
            "study_id": self._frozen_protocol.protocol.study_id,
            "protocol_id": self._frozen_protocol.protocol.protocol_id,
            "protocol_sha256": self.protocol_sha256,
            "phase": self._phase.value,
            "search_close_event_sha256": self._search_close_event_sha256,
            "hypothesis_states": [
                state.to_dict() for _, state in sorted(self._states.items())
            ],
            "adaptive_evidence": [
                item.to_dict() for _, item in sorted(self._adaptive_evidence.items())
            ],
            "postsearch_evidence": [
                item.to_dict() for _, item in sorted(self._postsearch_evidence.items())
            ],
            "failed_predictions": [
                item.to_dict() for _, item in sorted(self._failed_predictions.items())
            ],
            "adaptive_updates": [
                item.to_dict() for _, item in sorted(self._adaptive_updates.items())
            ],
            "postsearch_assessments": [
                item.to_dict()
                for _, item in sorted(self._postsearch_assessments.items())
            ],
            "events": [event.to_dict() for event in self._events],
        }
        if include_hash:
            payload["ledger_sha256"] = content_hash(payload)
        return payload
