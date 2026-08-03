"""Trusted Layer B qualification over completed, frozen run snapshots."""

from __future__ import annotations

from dataclasses import dataclass

from common.evaluation_profiles import EvaluationLayer, EvaluationPlan
from evaluation.records import QualificationEvaluationRecord, RecordEnvelope
from sealed_eval.snapshot import FrozenRunSnapshot


@dataclass(frozen=True)
class QualificationMeasurements:
    exact_match_accuracy: float
    metrics: tuple[tuple[str, float], ...] = ()
    complete: bool = True


@dataclass(frozen=True)
class QualificationPolicy:
    exact_match_threshold: float
    decision_record_id: str

    def validate(self) -> None:
        if not 0.0 <= self.exact_match_threshold <= 1.0:
            raise ValueError("Layer B exact-match threshold must be in [0, 1]")
        if not self.decision_record_id:
            raise ValueError("Layer B policy requires a frozen decision record")


class LayerBQualificationRunner:
    """Construct Layer B records without exposing a controller return channel."""

    def __init__(
        self,
        *,
        evaluation_plan: EvaluationPlan,
        policy: QualificationPolicy,
    ) -> None:
        evaluation_plan.validate()
        if evaluation_plan.layer is not EvaluationLayer.QUALIFICATION:
            raise ValueError("LayerBQualificationRunner requires a Layer B plan")
        if not evaluation_plan.sealed or evaluation_plan.controller_visible:
            raise ValueError("Layer B evaluation must be sealed and controller-hidden")
        policy.validate()
        if (
            evaluation_plan.scientific
            and evaluation_plan.pi_decision_record_id != policy.decision_record_id
        ):
            raise ValueError(
                "scientific Layer B policy must use the plan's frozen PI decision"
            )
        self._plan = evaluation_plan
        self._policy = policy

    def evaluate_frozen_candidate(
        self,
        *,
        snapshot: FrozenRunSnapshot,
        candidate_id: str,
        measurements: QualificationMeasurements,
        envelope: RecordEnvelope,
    ) -> QualificationEvaluationRecord:
        snapshot.validate(require_completed=True)
        snapshot.candidate(candidate_id)
        qualifies = (
            measurements.complete
            and measurements.exact_match_accuracy
            >= self._policy.exact_match_threshold
        )
        return QualificationEvaluationRecord(
            envelope=envelope,
            candidate_id=candidate_id,
            frozen_snapshot_id=snapshot.snapshot_id,
            frozen_snapshot_sha256=snapshot.snapshot_sha256,
            evaluation_plan_sha256=self._plan.plan_hash,
            exact_match_accuracy=measurements.exact_match_accuracy,
            qualifies=qualifies,
            evaluation_complete=measurements.complete,
            sealed_metrics=measurements.metrics,
        )
