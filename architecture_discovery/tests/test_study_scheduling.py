import os
import subprocess
import sys
from pathlib import Path

import pytest

from study.contracts import StudySpec
from study.randomization import generate_plan
from study.scheduling import (
    AcceleratorLease,
    AcceleratorLeaseBusy,
    MPSLease,
    MPSLeaseBusy,
    NoPendingRuns,
    ScheduleStateError,
    SequentialAcceleratorScheduler,
    SequentialRunScheduler,
)
from study.serialization import atomic_write_json, content_hash, read_json


ROOT = Path(__file__).resolve().parents[1]


def test_mps_lease_excludes_another_process(tmp_path) -> None:
    lease_path = tmp_path / "mps.lock"
    with MPSLease(lease_path, run_id="parent"):
        code = f"""
from study.scheduling import MPSLease, MPSLeaseBusy
try:
    lease = MPSLease({str(lease_path)!r}, run_id='child').acquire()
except MPSLeaseBusy:
    raise SystemExit(23)
else:
    lease.release()
    raise SystemExit(0)
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=environment,
            check=False,
        )
        assert result.returncode == 23
    assert not lease_path.exists()


def test_scheduler_follows_frozen_order_and_creates_unique_directories(tmp_path) -> None:
    spec = StudySpec.toy(study_id="sequential", proposal_opportunities=1)
    plan = generate_plan(spec, tmp_path)
    scheduler = SequentialRunScheduler(
        plan,
        state_path=tmp_path / "schedule.json",
        lease_path=tmp_path / "mps.lock",
    )

    observed = []
    for expected in plan.runs:
        with scheduler.claim_next() as claimed:
            observed.append(claimed.run_id)
            assert claimed.run_id == expected.run_id
            assert (claimed.execution_directory / "run_spec.json").is_file()
            with pytest.raises(MPSLeaseBusy):
                MPSLease(tmp_path / "mps.lock", run_id="competitor").acquire()
    assert observed == [run.run_id for run in plan.runs]
    with pytest.raises(NoPendingRuns):
        scheduler.claim_next()
    assert scheduler.summary()["counts"]["completed"] == 4


def test_interrupted_run_blocks_later_assignments_until_authorized(tmp_path) -> None:
    spec = StudySpec.toy(study_id="interrupted", proposal_opportunities=1)
    plan = generate_plan(spec, tmp_path)
    scheduler = SequentialRunScheduler(
        plan,
        state_path=tmp_path / "schedule.json",
        lease_path=tmp_path / "mps.lock",
    )
    first_id = plan.runs[0].run_id
    with pytest.raises(RuntimeError):
        with scheduler.claim_next():
            raise RuntimeError("synthetic infrastructure interruption")
    with pytest.raises(ScheduleStateError, match="interrupted"):
        scheduler.claim_next()
    scheduler.authorize_infrastructure_resume(first_id)
    with scheduler.claim_next() as resumed:
        assert resumed.run_id == first_id


def test_accelerator_lease_v2_records_remote_metadata_and_aliases(tmp_path) -> None:
    lease_path = tmp_path / "accelerator.lock"
    with AcceleratorLease(
        lease_path,
        run_id="modal-run",
        accelerator_kind="cuda",
        remote_call_id="fc-123",
        artifact_location="modal-volume:/study/modal-run",
    ):
        payload = read_json(lease_path)
        assert payload["schema_name"] == "AcceleratorLease"
        assert payload["schema_version"] == "2.0"
        assert payload["accelerator_kind"] == "cuda"
        assert payload["remote_call_id"] == "fc-123"
        assert payload["artifact_location"] == "modal-volume:/study/modal-run"
        with pytest.raises(AcceleratorLeaseBusy):
            AcceleratorLease(
                lease_path,
                run_id="competitor",
                accelerator_kind="cuda",
            ).acquire()
    assert MPSLease is AcceleratorLease
    assert MPSLeaseBusy is AcceleratorLeaseBusy


def test_accelerator_scheduler_is_sequential_and_persists_run_metadata(
    tmp_path,
) -> None:
    spec = StudySpec.toy(study_id="remote-sequential", proposal_opportunities=1)
    plan = generate_plan(spec, tmp_path)
    scheduler = SequentialAcceleratorScheduler(
        plan,
        state_path=tmp_path / "schedule.json",
        lease_path=tmp_path / "accelerator.lock",
        accelerator_kind="cuda",
    )
    first = plan.runs[0]
    with scheduler.claim_next(
        remote_call_id="fc-first",
        artifact_location="modal-volume:/study/first",
    ) as claimed:
        assert claimed.run_id == first.run_id
        state = read_json(tmp_path / "schedule.json")
        assert state["active_run_id"] == first.run_id
        assert list(state["statuses"].values()).count("running") == 1
        assert state["execution_metadata"][first.run_id] == {
            "remote_call_id": "fc-first",
            "artifact_location": "modal-volume:/study/first",
        }
        with pytest.raises(AcceleratorLeaseBusy):
            AcceleratorLease(
                tmp_path / "accelerator.lock",
                run_id="competitor",
                accelerator_kind="cuda",
            ).acquire()


def test_v1_schedule_is_read_without_rewrite_or_hash_change(tmp_path) -> None:
    spec = StudySpec.toy(study_id="legacy-schedule", proposal_opportunities=1)
    plan = generate_plan(spec, tmp_path)
    state_path = tmp_path / "schedule.json"
    SequentialAcceleratorScheduler(
        plan,
        state_path=state_path,
        lease_path=tmp_path / "accelerator.lock",
    )
    legacy = read_json(state_path)
    legacy["schema_version"] = "1.0"
    legacy.pop("accelerator_kind")
    legacy.pop("execution_metadata")
    atomic_write_json(state_path, legacy)
    before = content_hash(read_json(state_path))

    restored = SequentialRunScheduler(
        plan,
        state_path=state_path,
        lease_path=tmp_path / "accelerator.lock",
    )
    assert restored.summary()["accelerator_kind"] == "mps"
    assert content_hash(read_json(state_path)) == before
