"""Isolated-mode worker entrypoint.

Credential scrubbing and a network guard are useful hygiene, not a complete
filesystem or operating-system sandbox.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from pathlib import Path
from typing import Any

from common.evaluation_profiles import evaluation_plan_from_dict
from common.runtime_context import ExecutionContextV1
from common.task_adapter import DEFAULT_TASK
from common.trainer import (
    exclusive_training_lock,
    sha256_file,
    train_candidate_in_process,
)
from common.training_config import (
    TrainingSeedBundle,
    get_training_profile,
)


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
    if sha256_file(candidate) != str(job["candidate_source_hash"]):
        raise ValueError("candidate source hash changed before worker execution")
    raw_context = job.get("execution_context")
    execution_context = (
        None
        if raw_context is None
        else ExecutionContextV1.from_dict(raw_context)
    )
    return profile, seeds, candidate, execution_context


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    profile, seeds, candidate, execution_context = _resolve_job(job)
    _deny_network()
    with exclusive_training_lock():
        training = train_candidate_in_process(
            candidate_path=candidate,
            output_dir=job["output_dir"],
            profile=profile,
            seeds=seeds,
            requested_device=str(job["requested_device"]),
            allow_cpu_for_tests=bool(job["allow_cpu_for_tests"]),
            resume=job.get("resume"),
            task=DEFAULT_TASK,
            execution_context=execution_context,
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
            artifact_root=job["output_dir"],
        )
        return {
            "kind": "search_evaluation",
            "evaluation": evaluation.to_dict(),
        }


def main(job_path: str, response_path: str) -> None:
    response = Path(response_path).resolve()
    try:
        job = json.loads(Path(job_path).read_text(encoding="utf-8"))
        payload = run_job(job)
    except BaseException as error:
        payload = {
            "kind": "worker_failure",
            "failure_stage": "worker_infrastructure",
            "error_type": type(error).__name__,
            "error": "candidate worker failed; details suppressed",
        }
    _atomic_response(response, payload)
