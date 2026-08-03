from dataclasses import replace

import pytest

from mechanism.execution import (
    FreshBuild,
    FromScratchViolation,
    PairedAblationRunner,
)
from mechanism.fakes import (
    DeterministicTinyBuilder,
    DeterministicTinyTrainer,
    TinyFakeModel,
    toy_mechanism_plan,
)
from mechanism.plans import freeze_mechanism_plan


def test_every_arm_builds_and_trains_independently_with_paired_seeds(tmp_path):
    plan = toy_mechanism_plan(seed_count=2)
    receipt = freeze_mechanism_plan(plan, tmp_path / "plan.json")
    parameter_counts = {
        "arm:original": 9_000,
        "arm:removed": 500,
        "arm:replaced": 12_000,
        "arm:capacity": 9_000,
        "arm:compute": 20_000,
    }
    builder = DeterministicTinyBuilder(parameter_counts=parameter_counts)
    trainer = DeterministicTinyTrainer()
    result = PairedAblationRunner().run(
        receipt,
        builder=builder,
        trainer=trainer,
    )

    assert result.complete and result.all_succeeded
    assert len(builder.calls) == len(plan.arms) * len(plan.seed_bundles)
    assert len(trainer.calls) == len(builder.calls)
    assert len({attempt.build_id for attempt in result.attempts}) == len(result.attempts)
    for bundle in plan.seed_bundles:
        attempts = result.attempts_for_bundle(bundle.bundle_id)
        assert len(attempts) == len(plan.arms)
        assert {item.initialization_seed for item in attempts} == {
            bundle.initialization_seed
        }
        assert {item.training_seed for item in attempts} == {bundle.training_seed}
        assert {item.data_order_seed for item in attempts} == {
            bundle.data_order_seed
        }
    assert {
        item.arm_id: item.parameter_count_metadata
        for item in result.attempts_for_bundle("bundle:0")
    } == parameter_counts


def test_training_failure_is_retained_without_dropping_the_paired_arm(tmp_path):
    plan = toy_mechanism_plan(seed_count=1)
    receipt = freeze_mechanism_plan(plan, tmp_path / "plan.json")
    result = PairedAblationRunner().run(
        receipt,
        builder=DeterministicTinyBuilder(),
        trainer=DeterministicTinyTrainer(
            fail={("arm:removed", "bundle:0")}
        ),
    )
    assert result.complete
    assert not result.all_succeeded
    failed = [item for item in result.attempts if item.failure]
    assert [item.arm_id for item in failed] == ["arm:removed"]
    assert len(result.attempts) == 5


def test_runner_requires_frozen_plan_before_outcomes(tmp_path):
    plan = toy_mechanism_plan(seed_count=1)
    with pytest.raises(TypeError, match="FrozenMechanismPlan"):
        PairedAblationRunner().run(
            plan,  # type: ignore[arg-type]
            builder=DeterministicTinyBuilder(),
            trainer=DeterministicTinyTrainer(),
        )


def test_checkpoint_bearing_build_is_rejected_fail_closed(tmp_path):
    plan = toy_mechanism_plan(seed_count=1)
    receipt = freeze_mechanism_plan(plan, tmp_path / "plan.json")

    class CheckpointBuilder:
        def build_untrained(self, arm, *, initialization_seed):
            return FreshBuild(
                model=TinyFakeModel("bad", 1),
                build_id=f"build:{arm.arm_id}",
                arm_id=arm.arm_id,
                initialization_seed=initialization_seed,
                initial_state_sha256="b" * 64,
                parameter_count_metadata=1,
                loaded_checkpoint_id="original-best",
            )

    with pytest.raises(FromScratchViolation, match="checkpoint"):
        PairedAblationRunner().run(
            receipt,
            builder=CheckpointBuilder(),
            trainer=DeterministicTinyTrainer(),
        )


def test_reused_model_instance_is_rejected(tmp_path):
    plan = toy_mechanism_plan(seed_count=1)
    receipt = freeze_mechanism_plan(plan, tmp_path / "plan.json")

    class ReusingBuilder:
        def __init__(self):
            self.model = TinyFakeModel("shared", 1)
            self.index = 0

        def build_untrained(self, arm, *, initialization_seed):
            self.index += 1
            return FreshBuild(
                model=self.model,
                build_id=f"build:{self.index}",
                arm_id=arm.arm_id,
                initialization_seed=initialization_seed,
                initial_state_sha256=f"{self.index:064x}",
                parameter_count_metadata=1,
            )

    with pytest.raises(FromScratchViolation, match="model instance"):
        PairedAblationRunner().run(
            receipt,
            builder=ReusingBuilder(),
            trainer=DeterministicTinyTrainer(),
        )
