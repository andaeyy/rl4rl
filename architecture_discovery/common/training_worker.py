"""Isolated-mode worker entrypoint.

Credential scrubbing and a network guard are useful hygiene, not a complete
filesystem or operating-system sandbox.
"""

from __future__ import annotations

import json
import hashlib
import os
import socket
import tempfile
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from typing import Iterator

from common.device import CUDA_ALLOCATION_ENVIRONMENT_KEYS
from common.task_adapter import DEFAULT_TASK
from common.evaluation_profiles import evaluation_plan_from_dict
from common.trainer import (
    exclusive_training_lock,
    sha256_file,
    train_candidate_in_process,
)
from common.training_config import (
    TrainingSeedBundle,
    get_training_profile,
)
from common.trusted_candidate import validate_trusted_initial_candidate
from study.budget import (
    AcceleratorBackend,
    AcceleratorResourceLedger,
    AcceleratorResourceSpec,
)
from study.scheduling import AcceleratorLease


_BASE_WORKER_ENVIRONMENT = {
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SYSTEM_VERSION_COMPAT",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "DISCOVERY_TRAIN_DEVICE",
    "DISCOVERY_ALLOW_CPU_TRAINING",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "DISCOVERY_IN_TRAINING_WORKER",
}
_CUDA_WORKER_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES",
    "CUBLAS_WORKSPACE_CONFIG",
    "SLURM_JOB_ID",
    *CUDA_ALLOCATION_ENVIRONMENT_KEYS,
}
_SUPPORTED_WORKER_DEVICES = {"mps", "cuda", "cpu"}
_CUDA_LEASE_ROOT = Path("/tmp") / f"architecture-discovery-cuda-leases-{os.getuid()}"


def _deny_network() -> None:
    def denied(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("network access is disabled in candidate workers")

    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied


def _atomic_response(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _rescrub_worker_environment() -> None:
    """Keep candidate-visible state credential-free even for direct bootstrap use."""

    allowed = _BASE_WORKER_ENVIRONMENT | _CUDA_WORKER_ENVIRONMENT
    retained = {key: value for key, value in os.environ.items() if key in allowed}
    os.environ.clear()
    os.environ.update(retained)


def _resolve_job(job: dict[str, Any]):
    profile = get_training_profile(str(job["profile_name"]))
    if profile.version != str(job["profile_version"]):
        raise ValueError("worker profile version mismatch")
    if profile.profile_hash != str(job["profile_hash"]):
        raise ValueError("worker profile hash mismatch")
    seeds = TrainingSeedBundle(**job["seed_bundle"])
    if seeds.bundle_hash != str(job["seed_bundle_hash"]):
        raise ValueError("worker seed-bundle hash mismatch")
    if DEFAULT_TASK.version != str(job["task_adapter_version"]):
        raise ValueError("worker task-adapter version mismatch")
    if DEFAULT_TASK.config_hash != str(job["task_adapter_hash"]):
        raise ValueError("worker task-adapter hash mismatch")
    candidate = Path(job["candidate_path"]).resolve()
    if profile.name == "smoke_train_cuda_v1":
        validate_trusted_initial_candidate(candidate)
    if sha256_file(candidate) != str(job["candidate_source_hash"]):
        raise ValueError("candidate source hash changed before worker execution")
    requested_device = str(job["requested_device"]).strip().lower()
    if requested_device not in _SUPPORTED_WORKER_DEVICES:
        raise ValueError("worker job requested an unsupported training device")
    if os.environ.get("DISCOVERY_TRAIN_DEVICE") != requested_device:
        raise ValueError("worker job/device environment mismatch")
    allow_cpu_for_tests = bool(job["allow_cpu_for_tests"])
    environment_allows_cpu = (
        os.environ.get("DISCOVERY_ALLOW_CPU_TRAINING", "0") == "1"
    )
    if allow_cpu_for_tests != environment_allows_cpu:
        raise ValueError("worker job/CPU-test permission mismatch")
    if requested_device == "cpu" and not allow_cpu_for_tests:
        raise ValueError("CPU fallback is forbidden in candidate workers")
    if requested_device != "cpu" and profile.device_requirement != requested_device:
        raise ValueError("worker job profile/device requirement mismatch")
    return profile, seeds, candidate


@contextmanager
def _training_lock(profile, job: dict[str, Any]) -> Iterator[AcceleratorLease | None]:
    if profile.device_requirement != "cuda":
        with exclusive_training_lock():
            yield None
        return
    _CUDA_LEASE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    _CUDA_LEASE_ROOT.chmod(0o700)
    lease = AcceleratorLease.for_cuda_allocation(
        _CUDA_LEASE_ROOT,
        run_id=(
            f"slurm-{os.environ.get('SLURM_JOB_ID', 'unknown')}-"
            f"{str(job['candidate_source_hash'])[:16]}"
        ),
        environment=os.environ,
    )
    with lease:
        yield lease


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_cuda_resource_usage(
    *,
    job: dict[str, Any],
    profile,
    training,
    lease: AcceleratorLease,
) -> None:
    hardware_condition = f"nvidia_a40:{profile.name}:{profile.profile_hash}"
    spec = AcceleratorResourceSpec(
        backend=AcceleratorBackend.CUDA,
        hardware_condition=hardware_condition,
        accelerator_seconds=float(profile.maximum_wall_seconds),
    )
    ledger = AcceleratorResourceLedger(spec)
    ledger.record(
        float(training.train_seconds),
        backend=AcceleratorBackend.CUDA,
        hardware_condition=hardware_condition,
    )
    output_dir = Path(job["output_dir"]).resolve()
    summary_path = output_dir / "training_summary.json"
    payload = {
        "schema_name": "AcceleratorTrainingResourceRecord",
        "schema_version": "2.0",
        "accelerator_key": lease.accelerator_key,
        "profile_hash": profile.profile_hash,
        "candidate_source_hash": str(job["candidate_source_hash"]),
        "selected_device": training.device,
        "training_summary_sha256": _sha256_file(summary_path),
        "resource_ledger": ledger.to_dict(),
    }
    _atomic_response(output_dir / "accelerator_resource_usage.json", payload)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    profile, seeds, candidate = _resolve_job(job)
    _deny_network()
    with _training_lock(profile, job) as accelerator_lease:
        training = train_candidate_in_process(
            candidate_path=candidate,
            output_dir=job["output_dir"],
            profile=profile,
            seeds=seeds,
            requested_device=str(job["requested_device"]),
            allow_cpu_for_tests=bool(job["allow_cpu_for_tests"]),
            resume=job.get("resume"),
            task=DEFAULT_TASK,
        )
        if accelerator_lease is not None:
            _record_cuda_resource_usage(
                job=job,
                profile=profile,
                training=training,
                lease=accelerator_lease,
            )
        if job["mode"] == "train" or not training.success:
            return {
                "kind": "training_result",
                "training": training.to_dict(),
            }

        from common.evaluator import (
            SearchEvaluationContext,
            evaluate_trained_candidate_in_process,
        )

        evaluation_plan = evaluation_plan_from_dict(job["evaluation_plan"])
        evaluation_context = SearchEvaluationContext(**job["evaluation_context"])

        evaluation = evaluate_trained_candidate_in_process(
            candidate_path=candidate,
            training=training,
            seeds=seeds,
            requested_device=str(job["requested_device"]),
            allow_cpu_for_tests=bool(job["allow_cpu_for_tests"]),
            evaluation_plan=evaluation_plan,
            context=evaluation_context,
            eligibility_threshold=float(job["eligibility_threshold"]),
        )
        return {
            "kind": "search_evaluation",
            "evaluation": evaluation.to_dict(),
        }


def main(job_path: str, response_path: str) -> None:
    response = Path(response_path).resolve()
    try:
        os.umask(0o077)
        _rescrub_worker_environment()
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        payload = run_job(job)
    except BaseException as error:
        payload = {
            "kind": "worker_failure",
            "failure_stage": "worker_infrastructure",
            "error": f"{type(error).__name__}: {error}"[:2_000],
            "traceback": traceback.format_exc()[-8_000:],
        }
    _atomic_response(response, payload)
