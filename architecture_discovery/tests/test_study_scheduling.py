import os
import subprocess
import sys
from pathlib import Path

import pytest

from study.contracts import StudySpec
from study.randomization import generate_plan
from study.scheduling import (
    MPSLease,
    MPSLeaseBusy,
    NoPendingRuns,
    ScheduleStateError,
    SequentialRunScheduler,
)


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
            assert Path(claimed.run_directory, "run_spec.json").is_file()
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
