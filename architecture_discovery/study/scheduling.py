"""Cross-process MPS lease and frozen-order sequential run scheduler."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any

from study.contracts import RunSpec, utc_now
from study.randomization import RandomizationPlan
from study.serialization import atomic_write_json, create_json_exclusive, read_json


class MPSLeaseBusy(RuntimeError):
    """A different process owns the study-wide MPS execution lease."""


class ScheduleStateError(RuntimeError):
    """The persisted sequential schedule is inconsistent or requires review."""


class NoPendingRuns(RuntimeError):
    """The frozen schedule has no pending run available to claim."""


class MPSLease:
    """Fail-closed exclusive lock shared by every candidate-training process."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> MPSLease:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as error:
            try:
                owner = read_json(self.path)
            except Exception:
                owner = {"error": "lease metadata unreadable"}
            raise MPSLeaseBusy(
                f"MPS lease {self.path} is already held: {owner}"
            ) from error
        try:
            payload = {
                "schema_name": "MPSLease",
                "schema_version": "1.0",
                "run_id": self.run_id,
                "token": self.token,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "acquired_at": utc_now(),
            }
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.acquired = True
            return self
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise

    def release(self) -> None:
        if not self.acquired:
            return
        try:
            owner = read_json(self.path)
        except Exception as error:
            raise ScheduleStateError(
                "refusing to remove unreadable MPS lease metadata"
            ) from error
        if owner.get("token") != self.token:
            raise ScheduleStateError("refusing to remove an MPS lease owned elsewhere")
        self.path.unlink()
        self.acquired = False

    def __enter__(self) -> MPSLease:
        return self.acquire()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


class _RunClaim:
    def __init__(self, scheduler: SequentialRunScheduler, expected_run: RunSpec) -> None:
        self.scheduler = scheduler
        self.expected_run = expected_run
        self.lease = MPSLease(
            scheduler.lease_path,
            run_id=expected_run.run_id,
        )
        self.entered = False

    def __enter__(self) -> RunSpec:
        self.lease.acquire()
        try:
            state = self.scheduler._read_state()
            self.scheduler._validate_state(state)
            if state["active_run_id"] is not None:
                raise ScheduleStateError(
                    f"schedule already has active run {state['active_run_id']}"
                )
            actual = self.scheduler._next_pending(state)
            if actual is None:
                raise NoPendingRuns("the frozen schedule is complete")
            if actual.run_id != self.expected_run.run_id:
                raise ScheduleStateError("frozen schedule advanced during claim")
            self.scheduler._prepare_run_directory(actual)
            state["statuses"][actual.run_id] = "running"
            state["active_run_id"] = actual.run_id
            state["revision"] += 1
            state["updated_at"] = utc_now()
            atomic_write_json(self.scheduler.state_path, state)
            self.entered = True
            return actual
        except BaseException:
            self.lease.release()
            raise

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if self.entered:
                state = self.scheduler._read_state()
                self.scheduler._validate_state(state)
                if state["active_run_id"] != self.expected_run.run_id:
                    raise ScheduleStateError("active run changed while its lease was held")
                state["statuses"][self.expected_run.run_id] = (
                    "completed" if exception_type is None else "interrupted"
                )
                state["active_run_id"] = None
                state["revision"] += 1
                state["updated_at"] = utc_now()
                atomic_write_json(self.scheduler.state_path, state)
        finally:
            self.lease.release()


class SequentialRunScheduler:
    """Claims runs in the frozen order while holding the single MPS lease."""

    def __init__(
        self,
        plan: RandomizationPlan,
        *,
        state_path: str | Path,
        lease_path: str | Path,
    ) -> None:
        self.plan = plan
        self.state_path = Path(state_path)
        self.lease_path = Path(lease_path)
        if self.state_path.exists():
            state = self._read_state()
        else:
            state = {
                "schema_name": "SequentialScheduleState",
                "schema_version": "1.0",
                "study_id": plan.study_id,
                "assignment_hash": plan.assignment_hash,
                "run_order": [run.run_id for run in plan.runs],
                "statuses": {run.run_id: "pending" for run in plan.runs},
                "active_run_id": None,
                "revision": 0,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            try:
                create_json_exclusive(self.state_path, state)
            except FileExistsError:
                state = self._read_state()
        self._validate_state(state)
        if state["active_run_id"] is not None and not self.lease_path.exists():
            raise ScheduleStateError(
                "schedule records an active run but its MPS lease is missing; "
                "operator review is required"
            )

    def _read_state(self) -> dict[str, Any]:
        return read_json(self.state_path)

    def _validate_state(self, state: dict[str, Any]) -> None:
        expected_order = [run.run_id for run in self.plan.runs]
        if state.get("study_id") != self.plan.study_id:
            raise ScheduleStateError("schedule belongs to a different study")
        if state.get("assignment_hash") != self.plan.assignment_hash:
            raise ScheduleStateError("schedule assignment hash does not match the plan")
        if state.get("run_order") != expected_order:
            raise ScheduleStateError("stored run order differs from the frozen plan")
        statuses = state.get("statuses")
        if not isinstance(statuses, dict) or set(statuses) != set(expected_order):
            raise ScheduleStateError("schedule statuses do not cover the frozen runs")
        allowed = {"pending", "running", "completed", "interrupted"}
        if any(status not in allowed for status in statuses.values()):
            raise ScheduleStateError("schedule contains an unknown run status")
        running = [run_id for run_id, status in statuses.items() if status == "running"]
        active = state.get("active_run_id")
        if running != ([] if active is None else [active]):
            raise ScheduleStateError("active-run marker and running status disagree")

    def _next_pending(self, state: dict[str, Any]) -> RunSpec | None:
        by_id = {run.run_id: run for run in self.plan.runs}
        for run_id in state["run_order"]:
            status = state["statuses"][run_id]
            if status == "pending":
                return by_id[run_id]
            if status in {"running", "interrupted"}:
                # Never skip a nonterminal earlier assignment and bias later conditions.
                return None
        return None

    def claim_next(self) -> _RunClaim:
        state = self._read_state()
        self._validate_state(state)
        if state["active_run_id"] is not None:
            raise ScheduleStateError(
                f"schedule already has active run {state['active_run_id']}"
            )
        run = self._next_pending(state)
        if run is None:
            if any(status == "interrupted" for status in state["statuses"].values()):
                raise ScheduleStateError(
                    "an interrupted run blocks later assignments pending operator review"
                )
            raise NoPendingRuns("the frozen schedule is complete")
        return _RunClaim(self, run)

    def authorize_infrastructure_resume(self, run_id: str) -> None:
        """Reset only an explicitly reviewed interrupted run; never a completed run."""

        with MPSLease(self.lease_path, run_id=f"resume-{run_id}"):
            state = self._read_state()
            self._validate_state(state)
            if state["active_run_id"] is not None:
                raise ScheduleStateError("cannot resume while another run is active")
            if state["statuses"].get(run_id) != "interrupted":
                raise ScheduleStateError("only an interrupted run may be authorized")
            state["statuses"][run_id] = "pending"
            state["revision"] += 1
            state["updated_at"] = utc_now()
            atomic_write_json(self.state_path, state)

    def _prepare_run_directory(self, run: RunSpec) -> None:
        directory = Path(run.run_directory)
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / "run_spec.json"
        expected = run.to_dict()
        if marker.exists():
            if read_json(marker) != expected:
                raise ScheduleStateError(
                    f"run directory collision or changed assignment at {directory}"
                )
        else:
            create_json_exclusive(marker, expected)

    def summary(self) -> dict[str, Any]:
        state = self._read_state()
        self._validate_state(state)
        counts = {
            status: list(state["statuses"].values()).count(status)
            for status in ("pending", "running", "completed", "interrupted")
        }
        return {
            "study_id": self.plan.study_id,
            "assignment_hash": self.plan.assignment_hash,
            "counts": counts,
            "active_run_id": state["active_run_id"],
            "revision": state["revision"],
        }
