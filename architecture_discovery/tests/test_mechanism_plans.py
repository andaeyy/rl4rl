import json
from dataclasses import replace

import pytest

from mechanism.fakes import toy_mechanism_plan
from mechanism.plans import (
    ArmKind,
    ArmSpec,
    MechanismExperimentPlan,
    freeze_mechanism_plan,
    load_frozen_mechanism_plan,
)


def test_plan_contains_all_controls_and_parameter_count_is_not_an_objective():
    plan = toy_mechanism_plan()
    assert {arm.kind for arm in plan.arms} == set(ArmKind)
    assert plan.parameter_count_role == "metadata_and_capacity_control_only"
    assert plan.counterfactual_grid.case_count == 32
    assert "parameter_count_objective" not in plan.to_dict()
    capacity = next(arm for arm in plan.arms if arm.kind is ArmKind.CAPACITY_MATCHED)
    assert capacity.target_parameter_count == 1_000
    assert all(
        arm.target_parameter_count is None
        for arm in plan.arms
        if arm.kind is not ArmKind.CAPACITY_MATCHED
    )


def test_arm_spec_rejects_checkpoint_initialization_and_size_target_abuse():
    with pytest.raises(ValueError, match="checkpoint"):
        ArmSpec(
            "arm:bad",
            ArmKind.COMPONENT_REMOVED,
            "builder:bad",
            "remove route",
            "route",
            initial_checkpoint_id="candidate-best",
        )
    with pytest.raises(ValueError, match="capacity-control"):
        ArmSpec(
            "arm:bad-size",
            ArmKind.COMPUTE_MATCHED,
            "builder:bad-size",
            "match compute",
            "route",
            target_parameter_count=10,
        )


def test_plan_requires_exactly_one_of_every_arm_kind():
    plan = toy_mechanism_plan()
    duplicate = replace(plan.arms[-1], kind=ArmKind.ORIGINAL, transformation="none")
    with pytest.raises(ValueError, match="exactly one"):
        replace(plan, arms=plan.arms[:-1] + (duplicate,))


def test_frozen_plan_round_trip_and_tamper_detection(tmp_path):
    plan = toy_mechanism_plan()
    path = tmp_path / "mechanism-plan.json"
    receipt = freeze_mechanism_plan(plan, path)
    assert load_frozen_mechanism_plan(path).plan_hash == plan.plan_hash
    with pytest.raises(FileExistsError):
        freeze_mechanism_plan(plan, path)

    payload = json.loads(path.read_text())
    payload["plan"]["training_budget"]["max_steps"] += 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch|modified"):
        receipt.verify()


def test_plan_serialization_round_trip_preserves_hash():
    plan = toy_mechanism_plan()
    restored = MechanismExperimentPlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.plan_hash == plan.plan_hash
