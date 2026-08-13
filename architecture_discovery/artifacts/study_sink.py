"""Idempotent bridge from durable study snapshots to the immutable event ledger.

The bridge deliberately observes only ``RunState``. It cannot call a provider or
evaluator, and it does not participate in parent selection. Calling ``observe``
after each durable state write is enough to capture or backfill every transition
that the common engine has made irreversible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from artifacts.failures import FailureClass, FailureRecord
from artifacts.records import EventKind, EventRecord, content_sha256
from artifacts.store import FrozenIndexReference, RunArtifactStore
from study.contracts import RunState
from study.engine import CommonStudyEngine
from study.serialization import content_hash


ARCHITECTURE_IR_MEDIA_TYPE = "application/vnd.rl4rl.architecture-ir+json"
_ARCHITECTURE_IR_FIELDS = {
    "schema_name",
    "schema_version",
    "graph_id",
    "input_node_id",
    "output_node_id",
    "nodes",
    "edges",
    "metadata",
}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r}")


_BUDGET_ACTUAL_FIELDS = (
    "seed_evaluations",
    "proposal_opportunities",
    "provider_attempts",
    "prompt_tokens",
    "completion_tokens",
    "unknown_provider_usage",
    "parse_failures",
    "unique_candidate_sources",
    "candidate_training_attempts",
    "training_steps",
    "training_examples",
    "evaluation_cases",
    "repairs",
    "infrastructure_retries",
    "accepted",
    "rejected",
    "invalid",
    "scientific_failures",
    "infrastructure_failures",
    "terminal_opportunities",
)


class StudyEventSinkError(RuntimeError):
    pass


@dataclass(frozen=True)
class SinkObservation:
    appended_records: tuple[EventRecord, ...]
    search_completion_index: FrozenIndexReference | None = None


class StudyStateEventSink(Protocol):
    def observe(self, state: RunState) -> SinkObservation: ...


class ImmutableStudyEventSink:
    """Translate state snapshots into an idempotent causal event history."""

    def __init__(
        self,
        store: RunArtifactStore,
        *,
        initial_candidate_source: str | None = None,
    ) -> None:
        self.store = store
        self.initial_candidate_source = initial_candidate_source
        if initial_candidate_source is not None:
            if not self._is_architecture_ir(initial_candidate_source):
                raise StudyEventSinkError(
                    "initial candidate source is not Architecture IR JSON"
                )

    def _validate_state_identity(self, state: RunState) -> None:
        context = self.store.context
        expected = (
            context.study_id,
            context.block_id,
            context.run_id,
            context.condition_id,
            context.assignment_sha256,
        )
        actual = (
            state.study_id,
            state.block_id,
            state.run_id,
            state.condition_id,
            state.assignment_hash,
        )
        if actual != expected:
            raise StudyEventSinkError(
                "RunState does not match the frozen artifact context"
            )

    @staticmethod
    def _transition_events(
        events: tuple[EventRecord, ...]
    ) -> dict[str, EventRecord]:
        transitions: dict[str, EventRecord] = {}
        for event in events:
            raw_key = event.payload.get("transition_key")
            if raw_key is None:
                continue
            key = str(raw_key)
            if not key:
                raise StudyEventSinkError("stored transition key cannot be empty")
            if key in transitions:
                raise StudyEventSinkError(f"duplicate transition key {key!r}")
            transitions[key] = event
        return transitions

    def _emit(
        self,
        *,
        transitions: dict[str, EventRecord],
        appended: list[EventRecord],
        transition_key: str,
        event_kind: EventKind,
        payload: Mapping[str, Any],
    ) -> EventRecord:
        expected_payload = {"transition_key": transition_key, **dict(payload)}
        existing = transitions.get(transition_key)
        if existing is not None:
            if (
                existing.event_kind is not event_kind
                or content_sha256(existing.payload) != content_sha256(expected_payload)
            ):
                raise StudyEventSinkError(
                    f"transition {transition_key!r} conflicts with its immutable event"
                )
            return existing
        record = self.store.append(event_kind, expected_payload)
        transitions[transition_key] = record
        appended.append(record)
        return record

    def _put_text(self, text: str, *, media_type: str) -> str:
        return self.store.objects.put_bytes(
            text.encode("utf-8"), media_type=media_type
        ).sha256

    @staticmethod
    def _is_architecture_ir(text: str) -> bool:
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_json_keys,
                parse_constant=_reject_nonfinite_json,
            )
        except (TypeError, ValueError):
            return False
        return (
            isinstance(payload, dict)
            and set(payload) == _ARCHITECTURE_IR_FIELDS
            and payload.get("schema_name") == "architecture_tensor_graph"
            and payload.get("schema_version") == "1.0"
            and isinstance(payload.get("nodes"), list)
            and isinstance(payload.get("edges"), list)
            and isinstance(payload.get("metadata"), dict)
        )

    def _put_candidate_source(self, text: str) -> tuple[str, str]:
        media_type = (
            ARCHITECTURE_IR_MEDIA_TYPE
            if self._is_architecture_ir(text)
            else "text/plain"
        )
        digest = self._put_text(text, media_type=media_type)
        return digest, media_type

    @staticmethod
    def _resource_payload(evaluation: Mapping[str, Any]) -> dict[str, Any]:
        common = {
            "training_attempts": int(evaluation["training_attempts"]),
            "training_steps": int(evaluation["training_steps"]),
            "training_examples": int(evaluation["training_examples"]),
            "evaluation_cases": int(evaluation["evaluation_cases"]),
            "infrastructure_retries": int(
                evaluation.get("infrastructure_retries", 0)
            ),
        }
        has_legacy = "mps_seconds" in evaluation
        has_v2 = (
            "accelerator_kind" in evaluation
            or "accelerator_seconds" in evaluation
        )
        if has_legacy == has_v2:
            raise StudyEventSinkError(
                "evaluation must contain exactly one accelerator resource schema"
            )
        if has_legacy:
            return {
                **common,
                "mps_seconds": float(evaluation["mps_seconds"]),
            }
        accelerator_kind = evaluation.get("accelerator_kind")
        if not isinstance(accelerator_kind, str) or not accelerator_kind:
            raise StudyEventSinkError(
                "evaluation accelerator_kind must be a non-empty string"
            )
        if "accelerator_seconds" not in evaluation:
            raise StudyEventSinkError("evaluation lacks accelerator_seconds")
        return {
            **common,
            "accelerator_kind": accelerator_kind,
            "accelerator_seconds": float(evaluation["accelerator_seconds"]),
        }

    def _emit_evaluation(
        self,
        *,
        transitions: dict[str, EventRecord],
        appended: list[EventRecord],
        scope_key: str,
        candidate_id: str,
        evaluation: Mapping[str, Any],
        opportunity_index: int | None,
    ) -> None:
        resources = self._resource_payload(evaluation)
        training_id = f"training-{self.store.context.run_id}-{scope_key}"
        self._emit(
            transitions=transitions,
            appended=appended,
            transition_key=f"{scope_key}:training",
            event_kind=EventKind.TRAINING,
            payload={
                "training_id": training_id,
                "candidate_id": candidate_id,
                "opportunity_index": opportunity_index,
                **resources,
            },
        )
        self._emit(
            transitions=transitions,
            appended=appended,
            transition_key=f"{scope_key}:search-evaluation",
            event_kind=EventKind.SEARCH_EVALUATION,
            payload={
                "evaluation_id": (
                    f"search-evaluation-{self.store.context.run_id}-{scope_key}"
                ),
                "training_id": training_id,
                "candidate_id": candidate_id,
                "opportunity_index": opportunity_index,
                "outcome": str(evaluation["outcome"]),
                "score": float(evaluation["score"]),
                "failure_stage": str(evaluation.get("failure_stage", "")),
                "evaluation_cases": resources["evaluation_cases"],
            },
        )

    @staticmethod
    def _failure_class(outcome: str, stage: str) -> FailureClass:
        normalized = stage.lower()
        if outcome == "invalid":
            return FailureClass.PROPOSAL_PARSE
        if outcome == "infrastructure_failure":
            if "provider" in normalized:
                return FailureClass.PROVIDER_TRANSIENT
            if "unavailable" in normalized and "mps" in normalized:
                return FailureClass.MPS_UNAVAILABLE
            if "containment" in normalized:
                return FailureClass.CONTAINMENT_UNAVAILABLE
            if "mps" in normalized:
                return FailureClass.MPS_DRIVER_FAILURE
            cuda_markers = ("cuda", "cudnn", "cublas", "nvidia")
            if any(marker in normalized for marker in cuda_markers):
                if "determin" in normalized and (
                    "kernel" in normalized
                    or "algorithm" in normalized
                    or "operation" in normalized
                    or "unsupported" in normalized
                    or "unavailable" in normalized
                ):
                    return FailureClass.CUDA_DETERMINISTIC_KERNEL_UNAVAILABLE
                if "unavailable" in normalized or "not_available" in normalized:
                    return FailureClass.CUDA_UNAVAILABLE
                return FailureClass.CUDA_DRIVER_FAILURE
            if "modal" in normalized:
                return FailureClass.MODAL_INFRASTRUCTURE_FAILURE
            if "filesystem" in normalized or "io" in normalized:
                return FailureClass.FILESYSTEM_IO
            return FailureClass.WORKER_CRASH
        if "training" in normalized or "diverg" in normalized:
            return FailureClass.TRAINING_DIVERGENCE
        if "transformer" in normalized or "valid" in normalized:
            return FailureClass.INVALID_TRANSFORMER
        return FailureClass.NONQUALIFYING_RESULT

    def _observe_opportunity(
        self,
        opportunity: Mapping[str, Any],
        *,
        terminal: bool,
        transitions: dict[str, EventRecord],
        appended: list[EventRecord],
    ) -> None:
        index = int(opportunity["opportunity_index"])
        scope = f"opportunity-{index}"
        parent_ids = [str(item) for item in opportunity["parent_ids"]]
        self._emit(
            transitions=transitions,
            appended=appended,
            transition_key=f"{scope}:parent-selection",
            event_kind=EventKind.PARENT_SELECTION,
            payload={
                "opportunity_index": index,
                "selected_candidate_ids": parent_ids,
                "transition_active": bool(opportunity["transition_active"]),
            },
        )

        provider_attempts = int(opportunity.get("provider_attempts", 0))
        for attempt in range(1, provider_attempts + 1):
            self._emit(
                transitions=transitions,
                appended=appended,
                transition_key=f"{scope}:provider-attempt-{attempt}:started",
                event_kind=EventKind.PROPOSAL,
                payload={
                    "proposal_id": (
                        f"proposal-{self.store.context.run_id}-{index}-{attempt}"
                    ),
                    "opportunity_index": index,
                    "provider_attempt": attempt,
                    "phase": "provider_attempt_started",
                    "parent_candidate_ids": parent_ids,
                    "transition_active": bool(opportunity["transition_active"]),
                },
            )

        proposal = opportunity.get("proposal")
        repairs = int(opportunity.get("repairs", 0))
        previous_response = opportunity.get("previous_response")
        if repairs:
            response_attempt = provider_attempts if proposal is None else provider_attempts - 1
            if response_attempt > 0 and previous_response is not None:
                response_sha256 = self._put_text(
                    str(previous_response), media_type="text/plain"
                )
                matching_attempts = [
                    int(event.payload["provider_attempt"])
                    for event in transitions.values()
                    if event.event_kind is EventKind.PROPOSAL
                    and event.payload.get("phase") == "provider_response"
                    and event.payload.get("parsed_candidate") is False
                    and event.payload.get("response_object_sha256") == response_sha256
                    and int(event.payload.get("opportunity_index", -1)) == index
                ]
                if matching_attempts:
                    response_attempt = matching_attempts[-1]
                self._emit(
                    transitions=transitions,
                    appended=appended,
                    transition_key=f"{scope}:provider-attempt-{response_attempt}:response",
                    event_kind=EventKind.PROPOSAL,
                    payload={
                        "proposal_id": (
                            f"proposal-{self.store.context.run_id}-{index}-{response_attempt}"
                        ),
                        "opportunity_index": index,
                        "provider_attempt": response_attempt,
                        "phase": "provider_response",
                        "parsed_candidate": False,
                        "response_object_sha256": response_sha256,
                        "object_sha256s": [response_sha256],
                    },
                )
            for repair in range(1, repairs + 1):
                self._emit(
                    transitions=transitions,
                    appended=appended,
                    transition_key=f"{scope}:repair-{repair}",
                    event_kind=EventKind.REPAIR,
                    payload={
                        "repair_id": f"repair-{self.store.context.run_id}-{index}-{repair}",
                        "opportunity_index": index,
                        "repair_number": repair,
                        "failed_provider_attempt": max(1, response_attempt),
                        "used_evaluation_feedback": False,
                    },
                )

        if isinstance(proposal, Mapping):
            attempt = provider_attempts
            response_text = str(proposal["response_text"])
            response_sha256 = self._put_text(response_text, media_type="text/plain")
            object_hashes = [response_sha256]
            source = proposal.get("candidate_source")
            source_sha256 = None
            source_media_type = None
            if source is not None:
                source_sha256, source_media_type = self._put_candidate_source(
                    str(source)
                )
                object_hashes.append(source_sha256)
            proposal_payload: dict[str, Any] = {
                "proposal_id": (
                    f"proposal-{self.store.context.run_id}-{index}-{attempt}"
                ),
                "opportunity_index": index,
                "provider_attempt": attempt,
                "phase": "provider_response",
                "parsed_candidate": source is not None,
                "prompt_tokens": proposal.get("prompt_tokens"),
                "completion_tokens": proposal.get("completion_tokens"),
                "response_object_sha256": response_sha256,
                "candidate_source_object_sha256": source_sha256,
                "object_sha256s": sorted(object_hashes),
            }
            if source_media_type is not None:
                proposal_payload["candidate_source_media_type"] = source_media_type
            self._emit(
                transitions=transitions,
                appended=appended,
                transition_key=f"{scope}:provider-attempt-{attempt}:response",
                event_kind=EventKind.PROPOSAL,
                payload=proposal_payload,
            )

        candidate_id_value = opportunity.get("candidate_id")
        if candidate_id_value is not None:
            candidate_id = str(candidate_id_value)
            if not isinstance(proposal, Mapping) or proposal.get("candidate_source") is None:
                raise StudyEventSinkError("candidate ID lacks its immutable source")
            source_sha256, source_media_type = self._put_candidate_source(
                str(proposal["candidate_source"])
            )
            self._emit(
                transitions=transitions,
                appended=appended,
                transition_key=f"{scope}:candidate",
                event_kind=EventKind.CANDIDATE,
                payload={
                    "candidate_id": candidate_id,
                    "proposal_id": (
                        f"proposal-{self.store.context.run_id}-{index}-{provider_attempts}"
                    ),
                    "opportunity_index": index,
                    "parent_candidate_ids": parent_ids,
                    "source_object_sha256": source_sha256,
                    "source_media_type": source_media_type,
                    "object_sha256s": [source_sha256],
                },
            )

        evaluation = opportunity.get("evaluation")
        if isinstance(evaluation, Mapping):
            if candidate_id_value is None:
                raise StudyEventSinkError("evaluation lacks a candidate ID")
            self._emit_evaluation(
                transitions=transitions,
                appended=appended,
                scope_key=scope,
                candidate_id=str(candidate_id_value),
                evaluation=evaluation,
                opportunity_index=index,
            )

        if terminal:
            outcome = str(opportunity["outcome"])
            failure_stage = str(opportunity.get("failure_stage", ""))
            if outcome == "accepted":
                if candidate_id_value is None:
                    raise StudyEventSinkError("accepted opportunity lacks candidate ID")
                self._emit(
                    transitions=transitions,
                    appended=appended,
                    transition_key=f"{scope}:promotion",
                    event_kind=EventKind.PROMOTION,
                    payload={
                        "promotion_id": f"promotion-{self.store.context.run_id}-{index}",
                        "opportunity_index": index,
                        "candidate_id": str(candidate_id_value),
                        "promotion_target": "parent_pool",
                    },
                )
            elif outcome in {
                "invalid",
                "scientific_failure",
                "infrastructure_failure",
            }:
                failure = FailureRecord.create(
                    attempt_id=f"{self.store.context.run_id}-opportunity-{index}",
                    failure_class=self._failure_class(outcome, failure_stage),
                    stage=failure_stage or f"{outcome}_opportunity",
                    # This terminates one proposal opportunity, not the assigned run.
                    terminal=False,
                )
                self._emit(
                    transitions=transitions,
                    appended=appended,
                    transition_key=f"{scope}:failure",
                    event_kind=EventKind.FAILURE,
                    payload={
                        **failure.to_event_payload(),
                        "opportunity_index": index,
                        "opportunity_outcome": outcome,
                    },
                )
            elif outcome != "rejected":
                raise StudyEventSinkError(f"unknown opportunity outcome {outcome!r}")

    def _budget_totals(
        self, ledger: Mapping[str, Any]
    ) -> tuple[dict[str, int | float], str | None]:
        totals: dict[str, int | float] = {}
        for field in _BUDGET_ACTUAL_FIELDS:
            if field not in ledger:
                raise StudyEventSinkError(f"budget ledger lacks {field!r}")
            value = ledger[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise StudyEventSinkError(f"budget field {field!r} is not numeric")
            totals[field] = value
        has_legacy = "mps_seconds" in ledger
        has_v2 = "accelerator_kind" in ledger or "accelerator_seconds" in ledger
        if has_legacy == has_v2:
            raise StudyEventSinkError(
                "budget ledger must contain exactly one accelerator resource schema"
            )
        if has_legacy:
            value = ledger["mps_seconds"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise StudyEventSinkError("budget field 'mps_seconds' is not numeric")
            totals["mps_seconds"] = value
            return totals, None
        accelerator_kind = ledger.get("accelerator_kind")
        if not isinstance(accelerator_kind, str) or not accelerator_kind:
            raise StudyEventSinkError(
                "budget accelerator_kind must be a non-empty string"
            )
        value = ledger.get("accelerator_seconds")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise StudyEventSinkError(
                "budget field 'accelerator_seconds' is not numeric"
            )
        totals["accelerator_seconds"] = value
        return totals, accelerator_kind

    def observe(self, state: RunState) -> SinkObservation:
        self._validate_state_identity(state)
        report = self.store.scan()
        transitions = self._transition_events(report.events)
        appended: list[EventRecord] = []

        self._emit(
            transitions=transitions,
            appended=appended,
            transition_key="run:started",
            event_kind=EventKind.RUN_STATUS,
            payload={"status": "running"},
        )
        seed_payload: dict[str, Any] = {
            "candidate_id": state.initial_candidate_id,
            "parent_candidate_ids": [],
            "candidate_role": "initial_seed",
        }
        if self.initial_candidate_source is not None:
            if content_hash(self.initial_candidate_source) != state.initial_candidate_id:
                raise StudyEventSinkError(
                    "initial candidate source does not match RunState identity"
                )
            seed_sha256, seed_media_type = self._put_candidate_source(
                self.initial_candidate_source
            )
            seed_payload.update(
                {
                    "source_object_sha256": seed_sha256,
                    "source_media_type": seed_media_type,
                    "object_sha256s": [seed_sha256],
                }
            )
        self._emit(
            transitions=transitions,
            appended=appended,
            transition_key="seed:candidate",
            event_kind=EventKind.CANDIDATE,
            payload=seed_payload,
        )
        if state.seed_evaluation is not None:
            self._emit_evaluation(
                transitions=transitions,
                appended=appended,
                scope_key="seed",
                candidate_id=state.initial_candidate_id,
                evaluation=state.seed_evaluation,
                opportunity_index=None,
            )

        opportunities: dict[int, tuple[Mapping[str, Any], bool]] = {}
        for terminal in state.terminal_opportunities:
            index = int(terminal["opportunity_index"])
            if index in opportunities:
                raise StudyEventSinkError("duplicate terminal opportunity in RunState")
            opportunities[index] = (terminal, True)
        if state.active_opportunity is not None:
            index = int(state.active_opportunity["opportunity_index"])
            if index in opportunities:
                raise StudyEventSinkError("active opportunity is already terminal")
            opportunities[index] = (state.active_opportunity, False)
        for index in sorted(opportunities):
            opportunity, terminal = opportunities[index]
            self._observe_opportunity(
                opportunity,
                terminal=terminal,
                transitions=transitions,
                appended=appended,
            )

        if not isinstance(state.ledger, Mapping):
            raise StudyEventSinkError("RunState ledger must be an object")
        totals, accelerator_kind = self._budget_totals(state.ledger)
        budget_payload: dict[str, Any] = {"totals": totals}
        if accelerator_kind is not None:
            budget_payload["accelerator_kind"] = accelerator_kind
            budget_digest = content_sha256(budget_payload)
        else:
            # Preserve the v1 transition identity for already-emitted ledgers.
            budget_digest = content_sha256(totals)
        self._emit(
            transitions=transitions,
            appended=appended,
            transition_key=f"budget:{budget_digest}",
            event_kind=EventKind.BUDGET,
            payload=budget_payload,
        )

        search_completion_index = None
        if state.status == "completed":
            self._emit(
                transitions=transitions,
                appended=appended,
                transition_key="run:completed",
                event_kind=EventKind.RUN_STATUS,
                payload={"status": "completed"},
            )
            try:
                search_completion_index, _ = self.store.load_frozen_index(
                    "search_completion"
                )
            except FileNotFoundError:
                search_completion_index = self.store.freeze_index(
                    "search_completion"
                )
        elif state.status != "running":
            raise StudyEventSinkError(f"unsupported engine status {state.status!r}")
        return SinkObservation(tuple(appended), search_completion_index)

    def freeze_final_index(self) -> FrozenIndexReference:
        """Freeze the final ledger after all sealed reviews and analyses are appended."""

        events = self.store.scan().events
        if not any(
            event.event_kind is EventKind.RUN_STATUS
            and event.payload.get("transition_key") == "run:completed"
            for event in events
        ):
            raise StudyEventSinkError("cannot freeze final index before search completion")
        return self.store.freeze_index("final")


class ArtifactEmittingStudyEngine(CommonStudyEngine):
    """Instrumentation-only engine adapter requiring no change to study.engine."""

    def __init__(self, *args, artifact_sink: StudyStateEventSink, **kwargs) -> None:
        self.artifact_sink = artifact_sink
        super().__init__(*args, **kwargs)

    def _persist(self, state, ledger) -> None:
        super()._persist(state, ledger)
        self.artifact_sink.observe(state)

    def execute(self) -> RunState:
        state = super().execute()
        # Covers an already-completed state returned before CommonStudyEngine._persist.
        self.artifact_sink.observe(state)
        return state
