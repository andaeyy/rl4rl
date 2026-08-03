"""Credential-scrubbed subprocess client for candidate training/evaluation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from common.task_adapter import DEFAULT_TASK
from common.trainer import sha256_file
from common.training_config import TrainingProfile, TrainingSeedBundle


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "training_worker_bootstrap.py"
_INHERITED_ENV_ALLOWLIST = ("LANG", "LC_ALL", "TMPDIR", "SYSTEM_VERSION_COMPAT")
_TRUTHY = {"1", "true", "yes", "on"}


class WorkerError(RuntimeError):
    pass


def build_worker_environment(
    *,
    requested_device: str,
    allow_cpu_for_tests: bool,
    model_seed: int,
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
    }
    environment = build_worker_environment(
        requested_device=requested_device,
        allow_cpu_for_tests=allow_cpu_for_tests,
        model_seed=seeds.model_initialization_seed,
    )
    with tempfile.TemporaryDirectory(prefix="architecture-training-job-") as temporary:
        temporary_path = Path(temporary)
        job_path = temporary_path / "job.json"
        response_path = temporary_path / "response.json"
        job_path.write_text(json.dumps(job, sort_keys=True), encoding="utf-8")
        timeout_seconds = profile.maximum_wall_seconds + 300
        stdout_path = temporary_path / "worker.stdout"
        stderr_path = temporary_path / "worker.stderr"
        try:
            with stdout_path.open("w", encoding="utf-8") as stdout_handle, (
                stderr_path.open("w", encoding="utf-8")
            ) as stderr_handle:
                completed = subprocess.run(
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
                    timeout=timeout_seconds,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            raise WorkerError(
                f"candidate worker exceeded hard timeout of {timeout_seconds}s"
            ) from error
        if not response_path.is_file():
            stderr = (
                stderr_path.read_text(encoding="utf-8", errors="replace")[-2_000:]
                if stderr_path.exists()
                else ""
            )
            raise WorkerError(
                "candidate worker produced no response "
                f"(exit={completed.returncode}, stderr_tail={stderr!r})"
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
