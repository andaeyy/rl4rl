"""Credential-scrubbed subprocess client for candidate training/evaluation."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from common.process_control import (
    OUTER_PROCESS_DEADLINE_ENV,
    capture_isolated_process_group,
    terminate_process,
    terminate_process_group,
)
from common.runtime_context import ExecutionContextV1
from common.task_adapter import DEFAULT_TASK
from common.trainer import sha256_file
from common.training_config import TrainingProfile, TrainingSeedBundle

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "training_worker_bootstrap.py"
_INHERITED_ENV_ALLOWLIST = ("LANG", "LC_ALL", "TMPDIR", "SYSTEM_VERSION_COMPAT")
_TRUTHY = {"1", "true", "yes", "on"}
_CUDA_VISIBILITY_KEYS = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
_EXECUTION_CONTEXT_ENV = "DISCOVERY_EXECUTION_CONTEXT_JSON"
_MAX_EXECUTION_CONTEXT_BYTES = 16_384
_OUTER_CLEANUP_GUARD_SECONDS = 10.0


class WorkerError(RuntimeError):
    pass


def _bounded_worker_timeout_seconds(
    maximum_wall_seconds: float,
    *,
    parent_environment: Mapping[str, str] | None = None,
    monotonic_now: float | None = None,
) -> float:
    """Cap a nested worker so group cleanup precedes its Modal parent deadline."""

    default_timeout = float(maximum_wall_seconds) + 300.0
    parent = os.environ if parent_environment is None else parent_environment
    encoded_deadline = parent.get(OUTER_PROCESS_DEADLINE_ENV)
    if encoded_deadline is None:
        return default_timeout
    try:
        outer_deadline = float(encoded_deadline)
    except (TypeError, ValueError) as error:
        raise WorkerError("outer process deadline is invalid") from error
    if not math.isfinite(outer_deadline):
        raise WorkerError("outer process deadline is invalid")
    now = time.monotonic() if monotonic_now is None else float(monotonic_now)
    bounded_timeout = outer_deadline - now - _OUTER_CLEANUP_GUARD_SECONDS
    if not math.isfinite(bounded_timeout) or bounded_timeout <= 0:
        raise WorkerError(
            "outer process deadline has insufficient worker cleanup guard"
        )
    return min(default_timeout, bounded_timeout)


def _outer_process_group_is_contained(
    *,
    parent_environment: Mapping[str, str] | None = None,
) -> bool:
    """Trust outer containment only in the isolated controller group leader."""

    parent = os.environ if parent_environment is None else parent_environment
    return (
        OUTER_PROCESS_DEADLINE_ENV in parent
        and os.getpgrp() == os.getpid()
    )


def _validated_execution_context(
    value: ExecutionContextV1 | Mapping[str, Any] | None,
    *,
    parent_environment: Mapping[str, str] | None = None,
) -> ExecutionContextV1 | None:
    if value is not None:
        return (
            value
            if isinstance(value, ExecutionContextV1)
            else ExecutionContextV1.from_dict(value)
        )
    parent = os.environ if parent_environment is None else parent_environment
    encoded = parent.get(_EXECUTION_CONTEXT_ENV)
    if not encoded:
        return None
    if len(encoded.encode("utf-8")) > _MAX_EXECUTION_CONTEXT_BYTES:
        raise WorkerError("execution context exceeds its 16 KiB limit")
    try:
        payload = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise WorkerError("execution context is not valid JSON") from error
    if not isinstance(payload, dict):
        raise WorkerError("execution context must be a JSON object")
    try:
        return ExecutionContextV1.from_dict(payload)
    except ValueError as error:
        raise WorkerError(f"invalid execution context: {error}") from error


def build_worker_environment(
    *,
    requested_device: str,
    allow_cpu_for_tests: bool,
    model_seed: int,
    cublas_workspace_config: str | None = None,
    parent_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    parent = os.environ if parent_environment is None else parent_environment
    inherited_fallback = str(
        parent.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    ).strip().lower()
    if requested_device == "mps" and inherited_fallback in _TRUTHY:
        raise WorkerError(
            "PYTORCH_ENABLE_MPS_FALLBACK is enabled in the parent environment; "
            "refusing to launch strict MPS training"
        )
    environment = {
        key: parent[key]
        for key in _INHERITED_ENV_ALLOWLIST
        if key in parent and parent[key]
    }
    if requested_device.startswith("cuda"):
        for key in _CUDA_VISIBILITY_KEYS:
            value = str(parent.get(key, "")).strip()
            if not value:
                continue
            if "," in value or value.lower() in {"all", "void", "none"}:
                raise WorkerError(
                    f"{key} must identify exactly one visible CUDA device"
                )
            environment[key] = value
        if cublas_workspace_config not in {":4096:8", ":16:8"}:
            raise WorkerError(
                "CUDA workers require a pinned deterministic CUBLAS workspace"
            )
        environment["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment.update(
        {
            "PYTHONHASHSEED": str(model_seed % (2**32)),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "DISCOVERY_TRAIN_DEVICE": requested_device,
            "DISCOVERY_ALLOW_CPU_TRAINING": "1"
            if allow_cpu_for_tests
            else "0",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "DISCOVERY_IN_TRAINING_WORKER": "1",
        }
    )
    return environment


def run_worker_job(
    *,
    mode: str,
    candidate_path: str | Path,
    output_dir: str | Path,
    profile: TrainingProfile,
    seeds: TrainingSeedBundle,
    requested_device: str,
    allow_cpu_for_tests: bool,
    resume: str | Path | None = None,
    evaluation_plan: dict[str, Any] | None = None,
    evaluation_context: dict[str, str] | None = None,
    eligibility_threshold: float = 0.99,
    execution_context: ExecutionContextV1 | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    candidate = Path(candidate_path).resolve()
    destination = Path(output_dir).resolve()
    if mode not in {"train", "evaluate"}:
        raise ValueError(f"unsupported worker mode: {mode}")
    if mode == "evaluate" and (
        evaluation_plan is None or evaluation_context is None
    ):
        raise ValueError(
            "evaluate jobs require an explicit Layer A plan and record context"
        )
    validated_context = _validated_execution_context(execution_context)
    job = {
        "mode": mode,
        "candidate_path": str(candidate),
        "candidate_source_hash": sha256_file(candidate),
        "output_dir": str(destination),
        "profile_name": profile.name,
        "profile_version": profile.version,
        "profile_hash": profile.profile_hash,
        "seed_bundle": asdict(seeds),
        "seed_bundle_hash": seeds.bundle_hash,
        "task_adapter_version": DEFAULT_TASK.version,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "requested_device": requested_device,
        "allow_cpu_for_tests": allow_cpu_for_tests,
        "resume": str(Path(resume).resolve()) if resume else None,
        "evaluation_plan": evaluation_plan,
        "evaluation_context": evaluation_context,
        "eligibility_threshold": float(eligibility_threshold),
        # This credential-free, exact schema is data inside the job. The parent
        # environment variable itself is intentionally never inherited.
        "execution_context": (
            validated_context.to_dict() if validated_context is not None else None
        ),
    }
    environment = build_worker_environment(
        requested_device=requested_device,
        allow_cpu_for_tests=allow_cpu_for_tests,
        model_seed=seeds.model_initialization_seed,
        cublas_workspace_config=profile.cublas_workspace_config,
    )
    with tempfile.TemporaryDirectory(prefix="architecture-training-job-") as temporary:
        temporary_path = Path(temporary)
        job_path = temporary_path / "job.json"
        response_path = temporary_path / "response.json"
        job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
        timeout_seconds = _bounded_worker_timeout_seconds(
            profile.maximum_wall_seconds
        )
        stdout_path = temporary_path / "worker.stdout"
        stderr_path = temporary_path / "worker.stderr"
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, (
            stderr_path.open("w", encoding="utf-8")
        ) as stderr_handle:
            outer_group_contained = _outer_process_group_is_contained()
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    str(BOOTSTRAP),
                    str(job_path),
                    str(response_path),
                ],
                env=environment,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                # Modal's outer controller owns the process group and closes it
                # before artifact finalization. Local runs need their own group.
                start_new_session=not outer_group_contained,
            )
            process_group_id = (
                None
                if outer_group_contained
                else process.pid
            )

            def close_worker() -> None:
                if process_group_id is None:
                    terminate_process(process)
                else:
                    terminate_process_group(
                        process,
                        process_group_id=process_group_id,
                    )

            try:
                if not outer_group_contained:
                    process_group_id = capture_isolated_process_group(process)
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                close_worker()
                raise WorkerError(
                    f"candidate worker exceeded hard timeout of {timeout_seconds}s"
                ) from error
            except BaseException:
                close_worker()
                raise
            if process_group_id is not None:
                terminate_process_group(
                    process,
                    process_group_id=process_group_id,
                )
        if not response_path.is_file():
            stderr = (
                stderr_path.read_text(encoding="utf-8", errors="replace")[-2_000:]
                if stderr_path.exists()
                else ""
            )
            raise WorkerError(
                "candidate worker produced no response "
                f"(exit={returncode}, stderr_tail={stderr!r})"
            )
        if response_path.stat().st_size > 2_000_000:
            raise WorkerError("candidate worker response exceeded 2 MB")
        response = json.loads(
            response_path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
        if sha256_file(candidate) != job["candidate_source_hash"]:
            raise WorkerError("candidate source changed during worker execution")
        if not isinstance(response, dict):
            raise WorkerError("candidate worker returned an invalid response schema")
        if response.get("kind") not in {
            "training_result",
            "search_evaluation",
            "worker_failure",
        }:
            raise WorkerError("candidate worker returned an unknown response kind")
        return response
