"""Fail-closed paired execution for from-scratch mechanism arms."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from mechanism.plans import (
    ArmSpec,
    FrozenMechanismPlan,
    SeedBundle,
    TrainingBudget,
)
from mechanism.validation import (
    require_finite,
    require_identifier,
    require_nonnegative_int,
    require_sha256,
)


class FromScratchViolation(RuntimeError):
    """Raised when an arm can no longer prove independent initialization."""


class AttemptStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class FreshBuild:
    model: Any
    build_id: str
    arm_id: str
    initialization_seed: int
    initial_state_sha256: str
    parameter_count_metadata: int
    loaded_checkpoint_id: str | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            raise FromScratchViolation("builder returned no model")
        require_identifier(self.build_id, "build_id")
        require_identifier(self.arm_id, "arm_id")
        require_nonnegative_int(self.initialization_seed, "initialization_seed")
        require_sha256(self.initial_state_sha256, "initial_state_sha256")
        require_nonnegative_int(
            self.parameter_count_metadata, "parameter_count_metadata"
        )
        if self.loaded_checkpoint_id is not None:
            raise FromScratchViolation(
                "a fresh mechanism build cannot load any checkpoint"
            )


@dataclass(frozen=True)
class TrainingOutcome:
    build_id: str
    final_state_sha256: str
    steps_completed: int
    examples_seen: int
    metrics: tuple[tuple[str, float], ...]
    loaded_checkpoint_id: str | None = None

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
        if self.loaded_checkpoint_id is not None:
            raise FromScratchViolation(
                "trainer reported loading a checkpoint during an ablation arm"
            )


class ArmBuilder(Protocol):
    def build_untrained(self, arm: ArmSpec, *, initialization_seed: int) -> FreshBuild:
        ...


class ArmTrainer(Protocol):
    def train_from_scratch(
        self,
        build: FreshBuild,
        *,
        arm: ArmSpec,
        seeds: SeedBundle,
        budget: TrainingBudget,
    ) -> TrainingOutcome:
        ...


@dataclass(frozen=True)
class ArmAttemptRecord:
    plan_id: str
    claim_id: str
    arm_id: str
    arm_kind: str
    seed_bundle_id: str
    initialization_seed: int
    training_seed: int
    data_order_seed: int
    status: AttemptStatus
    build_id: str | None
    initial_state_sha256: str | None
    final_state_sha256: str | None
    parameter_count_metadata: int | None
    steps_completed: int
    examples_seen: int
    metrics: tuple[tuple[str, float], ...]
    failure: str | None
    initialization_policy: str = "from_scratch"


@dataclass(frozen=True)
class MechanismExperimentResult:
    plan_id: str
    plan_hash: str
    attempts: tuple[ArmAttemptRecord, ...]
    expected_attempt_count: int

    @property
    def complete(self) -> bool:
        return len(self.attempts) == self.expected_attempt_count

    @property
    def all_succeeded(self) -> bool:
        return self.complete and all(
            attempt.status is AttemptStatus.SUCCEEDED for attempt in self.attempts
        )

    def attempts_for_bundle(self, bundle_id: str) -> tuple[ArmAttemptRecord, ...]:
        return tuple(
            attempt
            for attempt in self.attempts
            if attempt.seed_bundle_id == bundle_id
        )


class PairedAblationRunner:
    """Build and train every arm independently for every paired seed bundle."""

    def run(
        self,
        frozen_plan: FrozenMechanismPlan,
        *,
        builder: ArmBuilder,
        trainer: ArmTrainer,
    ) -> MechanismExperimentResult:
        if not isinstance(frozen_plan, FrozenMechanismPlan):
            raise TypeError("execution requires a hash-verified FrozenMechanismPlan")
        frozen_plan.verify()
        plan = frozen_plan.plan
        attempts: list[ArmAttemptRecord] = []
        build_ids: set[str] = set()
        model_object_ids: set[int] = set()
        # Keep strong references so CPython cannot recycle an object ID during
        # a long experiment and trigger a false model-reuse violation.
        model_instances: list[Any] = []

        for seeds in plan.seed_bundles:
            for arm in plan.arms:
                build: FreshBuild | None = None
                try:
                    build = builder.build_untrained(
                        arm, initialization_seed=seeds.initialization_seed
                    )
                    if not isinstance(build, FreshBuild):
                        raise FromScratchViolation(
                            "builder must return a FreshBuild evidence record"
                        )
                    if build.arm_id != arm.arm_id:
                        raise FromScratchViolation("fresh build refers to the wrong arm")
                    if build.initialization_seed != seeds.initialization_seed:
                        raise FromScratchViolation(
                            "builder did not use the preregistered initialization seed"
                        )
                    if build.build_id in build_ids:
                        raise FromScratchViolation("builder reused a build ID")
                    if id(build.model) in model_object_ids:
                        raise FromScratchViolation("builder reused a model instance")
                    build_ids.add(build.build_id)
                    model_object_ids.add(id(build.model))
                    model_instances.append(build.model)
                    outcome = trainer.train_from_scratch(
                        build,
                        arm=arm,
                        seeds=seeds,
                        budget=plan.training_budget,
                    )
                    if not isinstance(outcome, TrainingOutcome):
                        raise FromScratchViolation(
                            "trainer must return a TrainingOutcome evidence record"
                        )
                    if outcome.build_id != build.build_id:
                        raise FromScratchViolation(
                            "training outcome refers to a different build"
                        )
                    if outcome.steps_completed > plan.training_budget.max_steps:
                        raise FromScratchViolation("trainer exceeded the frozen step budget")
                    if outcome.examples_seen > plan.training_budget.max_examples:
                        raise FromScratchViolation(
                            "trainer exceeded the frozen example budget"
                        )
                except FromScratchViolation:
                    raise
                except Exception as error:
                    attempts.append(
                        ArmAttemptRecord(
                            plan_id=plan.plan_id,
                            claim_id=plan.claim.claim_id,
                            arm_id=arm.arm_id,
                            arm_kind=arm.kind.value,
                            seed_bundle_id=seeds.bundle_id,
                            initialization_seed=seeds.initialization_seed,
                            training_seed=seeds.training_seed,
                            data_order_seed=seeds.data_order_seed,
                            status=AttemptStatus.FAILED,
                            build_id=None if build is None else build.build_id,
                            initial_state_sha256=(
                                None if build is None else build.initial_state_sha256
                            ),
                            final_state_sha256=None,
                            parameter_count_metadata=(
                                None
                                if build is None
                                else build.parameter_count_metadata
                            ),
                            steps_completed=0,
                            examples_seen=0,
                            metrics=(),
                            failure=f"{type(error).__name__}: {error}",
                        )
                    )
                    continue

                attempts.append(
                    ArmAttemptRecord(
                        plan_id=plan.plan_id,
                        claim_id=plan.claim.claim_id,
                        arm_id=arm.arm_id,
                        arm_kind=arm.kind.value,
                        seed_bundle_id=seeds.bundle_id,
                        initialization_seed=seeds.initialization_seed,
                        training_seed=seeds.training_seed,
                        data_order_seed=seeds.data_order_seed,
                        status=AttemptStatus.SUCCEEDED,
                        build_id=build.build_id,
                        initial_state_sha256=build.initial_state_sha256,
                        final_state_sha256=outcome.final_state_sha256,
                        parameter_count_metadata=build.parameter_count_metadata,
                        steps_completed=outcome.steps_completed,
                        examples_seen=outcome.examples_seen,
                        metrics=outcome.metrics,
                        failure=None,
                    )
                )

        return MechanismExperimentResult(
            plan_id=plan.plan_id,
            plan_hash=plan.plan_hash,
            attempts=tuple(attempts),
            expected_attempt_count=len(plan.arms) * len(plan.seed_bundles),
        )
