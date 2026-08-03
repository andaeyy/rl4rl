"""Clean-room, from-scratch replication runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from mechanism.plans import TrainingBudget
from mechanism.validation import (
    require_finite,
    require_identifier,
    require_nonnegative_int,
    require_sha256,
)
from replication.clean_room import CleanRoomReimplementationRecord
from replication.ledger import (
    IntentToTreatLedger,
    ReplicationAttemptRecord,
    ReplicationStatus,
)
from replication.policy import FrozenReplicationPolicy, ReplicationSeed


class ReplicationIntegrityError(RuntimeError):
    """Raised when clean-room or from-scratch evidence is invalid."""


class InfrastructureReplicationError(RuntimeError):
    pass


class CandidateReplicationError(RuntimeError):
    pass


class ScientificReplicationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplicationBuild:
    model: Any
    build_id: str
    clean_room_record_id: str
    implementation_sha256: str
    initialization_seed: int
    initial_state_sha256: str
    parameter_count_metadata: int
    loaded_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            raise ReplicationIntegrityError("clean-room builder returned no model")
        require_identifier(self.build_id, "build_id")
        require_identifier(self.clean_room_record_id, "clean_room_record_id")
        require_sha256(self.implementation_sha256, "implementation_sha256")
        require_nonnegative_int(self.initialization_seed, "initialization_seed")
        require_sha256(self.initial_state_sha256, "initial_state_sha256")
        require_nonnegative_int(
            self.parameter_count_metadata, "parameter_count_metadata"
        )
        if self.loaded_checkpoint_sha256 is not None:
            raise ReplicationIntegrityError(
                "replication build cannot load the candidate or any other checkpoint"
            )


@dataclass(frozen=True)
class ReplicationTrainingOutcome:
    build_id: str
    final_state_sha256: str
    steps_completed: int
    examples_seen: int
    metrics: tuple[tuple[str, float], ...]
    loaded_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.build_id, "build_id")
        require_sha256(self.final_state_sha256, "final_state_sha256")
        require_nonnegative_int(self.steps_completed, "steps_completed")
        require_nonnegative_int(self.examples_seen, "examples_seen")
        names: set[str] = set()
        for name, value in self.metrics:
            require_identifier(name, "metric name")
            require_finite(value, f"metric {name}")
            if name in names:
                raise ValueError(f"duplicate metric {name!r}")
            names.add(name)
        if self.loaded_checkpoint_sha256 is not None:
            raise ReplicationIntegrityError(
                "replication trainer reported loading a checkpoint"
            )

    def metric(self, name: str) -> float:
        for metric_name, value in self.metrics:
            if metric_name == name:
                return float(value)
        raise ScientificReplicationError(
            f"training outcome is missing preregistered metric {name!r}"
        )


class CleanRoomBuilder(Protocol):
    def build_untrained(
        self,
        record: CleanRoomReimplementationRecord,
        *,
        initialization_seed: int,
    ) -> ReplicationBuild:
        ...


class ReplicationTrainer(Protocol):
    def train_from_scratch(
        self,
        build: ReplicationBuild,
        *,
        seeds: ReplicationSeed,
        budget: TrainingBudget,
    ) -> ReplicationTrainingOutcome:
        ...


@dataclass(frozen=True)
class ReplicationResult:
    policy_id: str
    policy_hash: str
    ledger: IntentToTreatLedger
    success: bool


class ReplicationRunner:
    def run(
        self,
        frozen_policy: FrozenReplicationPolicy,
        *,
        clean_room: CleanRoomReimplementationRecord,
        builder: CleanRoomBuilder,
        trainer: ReplicationTrainer,
    ) -> ReplicationResult:
        if not isinstance(frozen_policy, FrozenReplicationPolicy):
            raise TypeError("replication requires a hash-verified frozen policy")
        frozen_policy.verify()
        policy = frozen_policy.policy
        clean_room.assert_compatible(policy)
        ledger = IntentToTreatLedger(policy)
        build_ids: set[str] = set()
        model_object_ids: set[int] = set()
        # Retain each model so runtime object IDs cannot be recycled and look
        # like deliberate instance reuse later in the replication set.
        model_instances: list[Any] = []

        for seeds in policy.seeds:
            while ledger.terminal_status(seeds.seed_id) is None:
                attempt_index = len(ledger.attempts_for(seeds.seed_id))
                build: ReplicationBuild | None = None
                try:
                    build = builder.build_untrained(
                        clean_room,
                        initialization_seed=seeds.initialization_seed,
                    )
                    self._validate_build(
                        build,
                        clean_room=clean_room,
                        seeds=seeds,
                        build_ids=build_ids,
                        model_object_ids=model_object_ids,
                    )
                    build_ids.add(build.build_id)
                    model_object_ids.add(id(build.model))
                    model_instances.append(build.model)
                    outcome = trainer.train_from_scratch(
                        build,
                        seeds=seeds,
                        budget=policy.training_budget,
                    )
                    self._validate_outcome(outcome, build, policy.training_budget)
                    metric_rule = policy.success_rule.replication_metric
                    metric_value = outcome.metric(metric_rule.metric_name)
                except ReplicationIntegrityError:
                    raise
                except InfrastructureReplicationError as error:
                    self._record_failure(
                        ledger,
                        seeds,
                        attempt_index,
                        ReplicationStatus.INFRASTRUCTURE_FAILURE,
                        error,
                        build,
                    )
                    continue
                except ScientificReplicationError as error:
                    self._record_failure(
                        ledger,
                        seeds,
                        attempt_index,
                        ReplicationStatus.SCIENTIFIC_FAILURE,
                        error,
                        build,
                    )
                    continue
                except Exception as error:
                    self._record_failure(
                        ledger,
                        seeds,
                        attempt_index,
                        ReplicationStatus.CANDIDATE_FAILURE,
                        error,
                        build,
                    )
                    continue

                ledger.record(
                    ReplicationAttemptRecord(
                        policy_id=policy.policy_id,
                        seed_id=seeds.seed_id,
                        attempt_index=attempt_index,
                        status=ReplicationStatus.SUCCEEDED,
                        build_id=build.build_id,
                        metric_name=metric_rule.metric_name,
                        metric_value=metric_value,
                        final_state_sha256=outcome.final_state_sha256,
                        error_class=None,
                        error_message=None,
                    )
                )

        return ReplicationResult(
            policy_id=policy.policy_id,
            policy_hash=policy.policy_hash,
            ledger=ledger,
            success=ledger.meets_success_rule(),
        )

    @staticmethod
    def _validate_build(
        build: ReplicationBuild,
        *,
        clean_room: CleanRoomReimplementationRecord,
        seeds: ReplicationSeed,
        build_ids: set[str],
        model_object_ids: set[int],
    ) -> None:
        if not isinstance(build, ReplicationBuild):
            raise ReplicationIntegrityError(
                "clean-room builder must return ReplicationBuild evidence"
            )
        if build.clean_room_record_id != clean_room.record_id:
            raise ReplicationIntegrityError("build refers to a different clean-room record")
        if build.implementation_sha256 != clean_room.implementation_sha256:
            raise ReplicationIntegrityError("build uses an unreviewed implementation")
        if build.initialization_seed != seeds.initialization_seed:
            raise ReplicationIntegrityError(
                "builder did not use the preregistered initialization seed"
            )
        if build.initial_state_sha256 == clean_room.original_checkpoint_sha256:
            raise ReplicationIntegrityError("build state equals the prohibited checkpoint")
        if build.build_id in build_ids:
            raise ReplicationIntegrityError("builder reused a build ID")
        if id(build.model) in model_object_ids:
            raise ReplicationIntegrityError("builder reused a model instance")

    @staticmethod
    def _validate_outcome(
        outcome: ReplicationTrainingOutcome,
        build: ReplicationBuild,
        budget: TrainingBudget,
    ) -> None:
        if not isinstance(outcome, ReplicationTrainingOutcome):
            raise ReplicationIntegrityError(
                "replication trainer must return typed training evidence"
            )
        if outcome.build_id != build.build_id:
            raise ReplicationIntegrityError("training outcome refers to another build")
        if outcome.steps_completed > budget.max_steps:
            raise ReplicationIntegrityError("replication exceeded frozen step budget")
        if outcome.examples_seen > budget.max_examples:
            raise ReplicationIntegrityError("replication exceeded frozen example budget")

    @staticmethod
    def _record_failure(
        ledger: IntentToTreatLedger,
        seeds: ReplicationSeed,
        attempt_index: int,
        status: ReplicationStatus,
        error: Exception,
        build: ReplicationBuild | None,
    ) -> None:
        ledger.record(
            ReplicationAttemptRecord(
                policy_id=ledger.policy.policy_id,
                seed_id=seeds.seed_id,
                attempt_index=attempt_index,
                status=status,
                build_id=None if build is None else build.build_id,
                metric_name=None,
                metric_value=None,
                final_state_sha256=None,
                error_class=type(error).__name__,
                error_message=str(error) or type(error).__name__,
            )
        )
