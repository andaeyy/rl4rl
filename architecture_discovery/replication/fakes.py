"""Deterministic tiny clean-room builder and trainer for offline tests."""

from __future__ import annotations

from dataclasses import dataclass

from mechanism.plans import TrainingBudget
from replication.clean_room import CleanRoomReimplementationRecord
from replication.execution import (
    CandidateReplicationError,
    InfrastructureReplicationError,
    ReplicationBuild,
    ReplicationTrainingOutcome,
    ScientificReplicationError,
)
from replication.policy import ReplicationSeed
from replication.policy import (
    Comparator,
    FailurePolicy,
    MetricRule,
    PromotionRule,
    ReplicationPolicy,
    SuccessRule,
)
from study.serialization import content_hash


@dataclass
class TinyCleanRoomModel:
    token: str
    trained: bool = False


class DeterministicCleanRoomBuilder:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def build_untrained(
        self,
        record: CleanRoomReimplementationRecord,
        *,
        initialization_seed: int,
    ) -> ReplicationBuild:
        call_index = len(self.calls)
        self.calls.append(initialization_seed)
        token = f"clean:{record.record_id}:{initialization_seed}:{call_index}"
        return ReplicationBuild(
            model=TinyCleanRoomModel(token),
            build_id=f"repbuild:{initialization_seed}:{call_index}",
            clean_room_record_id=record.record_id,
            implementation_sha256=record.implementation_sha256,
            initialization_seed=initialization_seed,
            initial_state_sha256=content_hash(
                {"token": token, "implementation": record.implementation_sha256}
            ),
            parameter_count_metadata=1_000 + call_index,
        )


class DeterministicReplicationTrainer:
    def __init__(
        self,
        *,
        failures: dict[str, list[str]] | None = None,
        metric_values: dict[str, float] | None = None,
    ) -> None:
        self.failures = {key: list(value) for key, value in (failures or {}).items()}
        self.metric_values = metric_values or {}
        self.calls: list[tuple[str, int, int, int]] = []

    def train_from_scratch(
        self,
        build: ReplicationBuild,
        *,
        seeds: ReplicationSeed,
        budget: TrainingBudget,
    ) -> ReplicationTrainingOutcome:
        model = build.model
        if not isinstance(model, TinyCleanRoomModel):
            raise TypeError("tiny replication trainer received the wrong model")
        if model.trained:
            raise RuntimeError("replication model instance was already trained")
        self.calls.append(
            (
                seeds.seed_id,
                seeds.training_seed,
                seeds.data_order_seed,
                seeds.evaluation_seed,
            )
        )
        queue = self.failures.get(seeds.seed_id, [])
        if queue:
            failure = queue.pop(0)
            if failure == "infrastructure":
                raise InfrastructureReplicationError("synthetic infrastructure outage")
            if failure == "scientific":
                raise ScientificReplicationError("synthetic scientific failure")
            if failure == "candidate":
                raise CandidateReplicationError("synthetic candidate failure")
            raise ValueError(f"unknown synthetic failure {failure!r}")
        model.trained = True
        value = self.metric_values.get(seeds.seed_id, 1.0)
        return ReplicationTrainingOutcome(
            build_id=build.build_id,
            final_state_sha256=content_hash(
                {
                    "initial": build.initial_state_sha256,
                    "training_seed": seeds.training_seed,
                    "data_order_seed": seeds.data_order_seed,
                }
            ),
            steps_completed=min(2, budget.max_steps),
            examples_seen=min(8, budget.max_examples),
            metrics=(("replication_accuracy", value),),
        )


def toy_replication_policy(*, seed_count: int = 3) -> ReplicationPolicy:
    return ReplicationPolicy(
        policy_id="replication-policy:toy",
        study_id="study:toy",
        claim_id="claim:toy",
        frozen_snapshot_id="snapshot:toy",
        frozen_snapshot_sha256="a" * 64,
        seeds=tuple(
            ReplicationSeed(
                seed_id=f"replication-seed:{index}",
                initialization_seed=10 + index,
                training_seed=20 + index,
                data_order_seed=30 + index,
                evaluation_seed=40 + index,
            )
            for index in range(seed_count)
        ),
        promotion_rule=PromotionRule(
            layer_b_qualification_required=True,
            mechanism_claim_required=True,
            qualification_metric=MetricRule(
                "layer_b_accuracy", Comparator.GREATER_EQUAL, 0.9
            ),
        ),
        success_rule=SuccessRule(
            replication_metric=MetricRule(
                "replication_accuracy", Comparator.GREATER_EQUAL, 0.9
            ),
            minimum_successful_seeds=max(1, seed_count - 1),
        ),
        failure_policy=FailurePolicy(
            infrastructure_retry_classes=("InfrastructureReplicationError",),
            max_infrastructure_retries=1,
        ),
        training_budget=TrainingBudget(
            dataset_id="dataset:replication-toy",
            optimizer_id="optimizer:toy",
            schedule_id="schedule:toy",
            max_steps=2,
            max_examples=8,
            wall_time_seconds=1.0,
        ),
        clean_room_reimplementation_required=True,
        scientific=False,
    )


def toy_clean_room_record() -> CleanRoomReimplementationRecord:
    return CleanRoomReimplementationRecord(
        record_id="clean-room:toy",
        candidate_snapshot_id="snapshot:toy",
        candidate_snapshot_sha256="a" * 64,
        architecture_spec_sha256="b" * 64,
        implementation_sha256="c" * 64,
        original_checkpoint_sha256="d" * 64,
        builder_id="clean-builder:toy",
        implementer_id="implementer:independent",
        protocol_id="clean-room-protocol:v1",
        original_candidate_source_accessed=False,
        original_checkpoint_accessed=False,
    )
