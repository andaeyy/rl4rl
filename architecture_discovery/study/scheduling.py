"""Cross-process MPS lease and frozen-order sequential run scheduler."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from pathlib import Path
from types import TracebackType
from typing import Any, Mapping

from study.contracts import RunSpec, utc_now
from study.randomization import RandomizationPlan
from study.serialization import atomic_write_json, create_json_exclusive, read_json


class MPSLeaseBusy(RuntimeError):
    """A different process owns the study-wide MPS execution lease."""


class AcceleratorLeaseBusy(RuntimeError):
    """A different process owns the lease for an accelerator allocation."""


class ScheduleStateError(RuntimeError):
    """The persisted sequential schedule is inconsistent or requires review."""


class NoPendingRuns(RuntimeError):
    """The frozen schedule has no pending run available to claim."""


def cuda_accelerator_identity(
    *,
    gpu_uuid: str | None = None,
    environment: Mapping[str, str] | None = None,
    hostname: str | None = None,
) -> str:
    """Resolve a stable allocated-GPU key, preferring the physical GPU UUID.

    ``CUDA_VISIBLE_DEVICES=0`` by itself is intentionally insufficient: index zero
    is process-relative and can refer to different GPUs.  Slurm's physical GPU
    assignment is namespaced by hostname when a UUID is unavailable.
    """

    if gpu_uuid:
        return f"cuda:uuid:{gpu_uuid}"
    values = os.environ if environment is None else environment
    visible = values.get("CUDA_VISIBLE_DEVICES", "")
    visible_tokens = tuple(item.strip() for item in visible.split(",") if item.strip())
    visible_uuids = tuple(item for item in visible_tokens if item.startswith("GPU-"))
    if len(visible_uuids) == 1:
        return f"cuda:uuid:{visible_uuids[0]}"
    assignment = next(
        (
            values[name]
            for name in ("SLURM_STEP_GPUS", "SLURM_JOB_GPUS")
            if values.get(name)
        ),
        None,
    )
    if not assignment and values.get("SLURM_GPUS_ON_NODE"):
        assignment = f"visible:{visible}"
    job_id = values.get("SLURM_JOB_ID") or values.get("SLURM_JOBID")
    if not assignment or not job_id:
        raise ScheduleStateError(
            "CUDA accelerator lease requires a GPU UUID or scheduler GPU assignment"
        )
    resolved_hostname = hostname or socket.gethostname()
    return f"cuda:slurm:{resolved_hostname}:{assignment}"


class AcceleratorLease:
    """Exclusive, allocation-keyed accelerator lock for CUDA and future backends."""

    schema_name = "AcceleratorLease"
    schema_version = "2.0"

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        accelerator_key: str,
        backend: str,
    ) -> None:
        normalized_backend = backend.strip().lower()
        if not normalized_backend:
            raise ValueError("accelerator backend cannot be empty")
        if not accelerator_key.strip():
            raise ValueError("accelerator key cannot be empty")
        self.path = Path(path)
        self.run_id = run_id
        self.accelerator_key = accelerator_key
        self.backend = normalized_backend
        self.token = uuid.uuid4().hex
        self.acquired = False

    @classmethod
    def for_allocation(
        cls,
        lease_directory: str | Path,
        *,
        run_id: str,
        accelerator_key: str,
        backend: str,
    ) -> AcceleratorLease:
        """Build a lease path whose identity is bound to the allocated accelerator."""

        digest = hashlib.sha256(accelerator_key.encode("utf-8")).hexdigest()[:24]
        normalized_backend = backend.strip().lower()
        path = Path(lease_directory) / f"{normalized_backend}-{digest}.lock"
        return cls(
            path,
            run_id=run_id,
            accelerator_key=accelerator_key,
            backend=normalized_backend,
        )

    @classmethod
    def for_cuda_allocation(
        cls,
        lease_directory: str | Path,
        *,
        run_id: str,
        gpu_uuid: str | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> AcceleratorLease:
        key = cuda_accelerator_identity(
            gpu_uuid=gpu_uuid,
            environment=environment,
        )
        return cls.for_allocation(
            lease_directory,
            run_id=run_id,
            accelerator_key=key,
            backend="cuda",
        )

    def acquire(self) -> AcceleratorLease:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
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
            raise AcceleratorLeaseBusy(
                f"accelerator lease {self.path} is already held: {owner}"
            ) from error
        try:
            payload = {
                "schema_name": self.schema_name,
                "schema_version": self.schema_version,
                "backend": self.backend,
                "accelerator_key": self.accelerator_key,
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
                "refusing to remove unreadable accelerator lease metadata"
            ) from error
        if owner.get("token") != self.token:
            raise ScheduleStateError(
                "refusing to remove an accelerator lease owned elsewhere"
            )
        if owner.get("accelerator_key") != self.accelerator_key:
            raise ScheduleStateError("accelerator lease identity changed while held")
        self.path.unlink()
        self.acquired = False

    def __enter__(self) -> AcceleratorLease:
        return self.acquire()

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()


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
