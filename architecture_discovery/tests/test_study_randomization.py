import json
from dataclasses import asdict
from pathlib import Path

import pytest

from study.contracts import ConditionId, RunSpec, StudySpec
from study.randomization import generate_plan, load_or_create_plan


def test_blocked_randomization_is_complete_paired_and_unique(tmp_path) -> None:
    spec = StudySpec.toy(block_count=4, proposal_opportunities=2)
    plan = generate_plan(spec, tmp_path)

    assert len(plan.blocks) == 4
    assert len({run.run_id for run in plan.runs}) == 16
    assert len({run.run_directory for run in plan.runs}) == 16
    assert plan.schema_version == "2.0"
    assert all(not Path(run.run_directory).is_absolute() for run in plan.runs)
    assert all(run.run_directory == f"runs/{run.run_id}" for run in plan.runs)
    assert all(
        run.execution_directory
        == (tmp_path / spec.study_id / "runs" / run.run_id).resolve()
        for run in plan.runs
    )
    assert all("execution_root" not in asdict(run) for run in plan.runs)
    assert all("_execution_root" not in asdict(run) for run in plan.runs)
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


def test_v2_assignment_hash_is_portable_across_execution_roots(tmp_path) -> None:
    spec = StudySpec.toy(study_id="portable-plan", study_seed=43)
    first = generate_plan(spec, tmp_path / "executor-a")
    second = generate_plan(spec, tmp_path / "executor-b")

    assert first.to_dict() == second.to_dict()
    assert first.assignment_hash == second.assignment_hash
    assert first.runs[0].execution_directory != second.runs[0].execution_directory


def test_v1_plan_round_trips_without_field_or_hash_rewrite(tmp_path) -> None:
    spec = StudySpec.toy(study_id="legacy-plan", study_seed=47)
    legacy = generate_plan(spec, tmp_path, schema_version="1.0")
    payload = legacy.to_dict()
    restored = type(legacy).from_dict(payload)

    assert restored.to_dict() == payload
    assert restored.assignment_hash == legacy.assignment_hash
    assert all(Path(run.run_directory).is_absolute() for run in restored.runs)
    assert all(
        run.execution_directory == Path(run.run_directory) for run in restored.runs
    )


def test_v2_execution_directory_rejects_symlink_escape(tmp_path) -> None:
    spec = StudySpec.toy(study_id="symlink-plan", study_seed=53)
    plan = generate_plan(spec, tmp_path)
    study_root = tmp_path / spec.study_id
    outside = tmp_path / "outside"
    outside.mkdir()
    study_root.mkdir()
    (study_root / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes"):
        _ = plan.runs[0].execution_directory


def test_v2_run_spec_rejects_parent_traversal(tmp_path) -> None:
    spec = StudySpec.toy(study_id="traversal-plan", study_seed=59)
    payload = generate_plan(spec, tmp_path).runs[0].to_dict()
    payload["run_directory"] = "runs/../outside"

    with pytest.raises(ValueError, match="canonical relative POSIX"):
        RunSpec.from_dict(payload, execution_root=tmp_path / spec.study_id)


def test_v2_run_spec_refuses_to_deserialize_an_execution_root(tmp_path) -> None:
    spec = StudySpec.toy(study_id="serialized-root", study_seed=61)
    payload = generate_plan(spec, tmp_path).runs[0].to_dict()
    payload["execution_root"] = "/mnt/discovery/private"

    with pytest.raises(ValueError, match="exact schema"):
        RunSpec.from_dict(payload, execution_root=tmp_path / spec.study_id)


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
