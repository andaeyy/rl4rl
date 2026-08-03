import json

import pytest

from study.contracts import ConditionId, StudySpec
from study.randomization import generate_plan, load_or_create_plan


def test_blocked_randomization_is_complete_paired_and_unique(tmp_path) -> None:
    spec = StudySpec.toy(block_count=4, proposal_opportunities=2)
    plan = generate_plan(spec, tmp_path)

    assert len(plan.blocks) == 4
    assert len({run.run_id for run in plan.runs}) == 16
    assert len({run.run_directory for run in plan.runs}) == 16
    for block in plan.blocks:
        assert {run.condition.condition_id for run in block.runs} == set(ConditionId)
        assert len({run.run_seed for run in block.runs}) == 1
        assert [run.order_index for run in block.runs] == [0, 1, 2, 3]


def test_frozen_plan_resume_preserves_bytes_hash_and_order(tmp_path) -> None:
    spec = StudySpec.toy(
        study_id="resume-order",
        study_seed=41,
        block_count=3,
        proposal_opportunities=2,
    )
    path = tmp_path / "randomization_plan.json"
    first = load_or_create_plan(spec, output_root=tmp_path, plan_path=path)
    frozen_bytes = path.read_bytes()
    second = load_or_create_plan(spec, output_root=tmp_path, plan_path=path)

    assert path.read_bytes() == frozen_bytes
    assert second.assignment_hash == first.assignment_hash
    assert [run.run_id for run in second.runs] == [run.run_id for run in first.runs]
    assert [run.condition.condition_id for run in second.runs] == [
        run.condition.condition_id for run in first.runs
    ]


def test_changed_study_spec_cannot_reuse_frozen_assignment(tmp_path) -> None:
    path = tmp_path / "randomization_plan.json"
    original = StudySpec.toy(study_id="frozen", study_seed=2)
    load_or_create_plan(original, output_root=tmp_path, plan_path=path)
    changed = StudySpec.toy(study_id="frozen", study_seed=3)
    with pytest.raises(ValueError, match="changed after randomization"):
        load_or_create_plan(changed, output_root=tmp_path, plan_path=path)


def test_tampered_assignment_is_rejected(tmp_path) -> None:
    spec = StudySpec.toy(study_id="tamper")
    path = tmp_path / "randomization_plan.json"
    load_or_create_plan(spec, output_root=tmp_path, plan_path=path)
    payload = json.loads(path.read_text())
    payload["blocks"][0]["runs"][0]["order_index"] = 99
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_or_create_plan(spec, output_root=tmp_path, plan_path=path)
