"""Credential-scrubbed subprocess client for candidate training/evaluation."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.device import (
    CUDA_ALLOCATION_ENVIRONMENT_KEYS,
    CUDA_CUBLAS_WORKSPACE_CONFIG,
)
from common.task_adapter import DEFAULT_TASK
from common.trainer import sha256_file
from common.training_config import (
    TrainingProfile,
    TrainingSeedBundle,
    get_training_profile,
)
from common.trusted_candidate import validate_trusted_initial_candidate


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "scripts" / "training_worker_bootstrap.py"
_INHERITED_ENV_ALLOWLIST = ("LANG", "LC_ALL", "TMPDIR", "SYSTEM_VERSION_COMPAT")
_CUDA_ASSIGNMENT_ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "SLURM_JOB_ID",
    *CUDA_ALLOCATION_ENVIRONMENT_KEYS,
)
_CUDA_VISIBLE_DEVICE_PATTERN = re.compile(r"[A-Za-z0-9_.:/-]+")
_TRUTHY = {"1", "true", "yes", "on"}
SUPPORTED_REQUESTED_DEVICES = ("mps", "cuda", "cpu")


class WorkerError(RuntimeError):
    pass


@dataclass(frozen=True)
class HarnessTrainingSelection:
    """Resolved native-harness training settings and their provenance."""

    profile: TrainingProfile
    requested_device: str
    allow_cpu_for_tests: bool
    profile_source: str
    device_source: str

    def manifest_fields(self) -> dict[str, Any]:
        return {
            "profile": self.profile.name,
            "profile_version": self.profile.version,
            "profile_hash": self.profile.profile_hash,
            "requested_device": self.requested_device,
            # Retain the original field for backward-compatible readers.
            "device": self.requested_device,
            "allow_cpu_for_tests": self.allow_cpu_for_tests,
            "selection": {
                "profile_source": self.profile_source,
                "device_source": self.device_source,
            },
            "actual_selection_evidence": {
                "scope": "per_candidate_worker_manifest",
                "record": "training_manifest.json",
                "required_fields": [
                    "profile_hash",
                    "requested_device",
                    "selected_device",
                ],
            },
        }


def _selection_value(
    *,
    command_line: str | None,
    environment: Mapping[str, str],
    environment_name: str,
    configured: str,
) -> tuple[str, str]:
    if command_line:
        return command_line, "command_line"
    if environment.get(environment_name):
        return str(environment[environment_name]), "environment"
    return configured, "configuration"


def resolve_harness_training_selection(
    training_config: Mapping[str, Any],
    *,
    profile_override: str | None,
    device_override: str | None,
    environment: Mapping[str, str] | None = None,
) -> HarnessTrainingSelection:
    """Resolve profile/device overrides without changing frozen config defaults."""

    parent = os.environ if environment is None else environment
    profile_name, profile_source = _selection_value(
        command_line=profile_override,
        environment=parent,
        environment_name="DISCOVERY_TRAINING_PROFILE",
        configured=str(training_config["profile"]),
    )
    requested_device, device_source = _selection_value(
        command_line=device_override,
        environment=parent,
        environment_name="DISCOVERY_TRAIN_DEVICE",
        configured=str(training_config["device"]),
    )
    requested_device = requested_device.strip().lower()

    selectable_profiles = tuple(
        str(value) for value in training_config.get("selectable_profiles", ())
    )
    if selectable_profiles and profile_name not in selectable_profiles:
        raise ValueError(
            f"training profile {profile_name!r} is not selectable by this harness"
        )
    selectable_devices = tuple(
        str(value).lower()
        for value in training_config.get("selectable_devices", ())
    )
    if selectable_devices and requested_device not in selectable_devices:
        raise ValueError(
            f"training device {requested_device!r} is not selectable by this harness"
        )
    if requested_device not in SUPPORTED_REQUESTED_DEVICES:
        raise ValueError(
            f"unsupported training device {requested_device!r}; choose one of "
            f"{SUPPORTED_REQUESTED_DEVICES}"
        )

    profile = get_training_profile(profile_name)
    if profile_source == "configuration" and profile.version != str(
        training_config["profile_version"]
    ):
        raise ValueError("configured training profile version mismatch")

    allow_cpu_for_tests = bool(training_config["allow_cpu_for_tests"])
    if requested_device == "cpu":
        if profile.scientific:
            raise ValueError("scientific training cannot select CPU")
        if not allow_cpu_for_tests:
            raise ValueError(
                "CPU training is disabled; no accelerator fallback is permitted"
            )
    elif profile.device_requirement != requested_device:
        raise ValueError(
            f"profile {profile.name} requires {profile.device_requirement}; "
            f"requested {requested_device} cannot substitute for that backend"
        )
    return HarnessTrainingSelection(
        profile=profile,
        requested_device=requested_device,
        allow_cpu_for_tests=allow_cpu_for_tests,
        profile_source=profile_source,
        device_source=device_source,
    )


def _normalize_requested_device(
    requested_device: str,
    *,
    allow_cpu_for_tests: bool,
) -> str:
    normalized = requested_device.strip().lower()
    if normalized not in SUPPORTED_REQUESTED_DEVICES:
        raise WorkerError(
            f"unsupported worker device {normalized!r}; choose one of "
            f"{SUPPORTED_REQUESTED_DEVICES}"
        )
    if normalized == "cpu" and not allow_cpu_for_tests:
        raise WorkerError("CPU worker launch requires explicit test-only permission")
    return normalized


def _validated_cuda_assignment(
    parent: Mapping[str, str],
) -> dict[str, str]:
    assignment = {
        key: str(parent[key])
        for key in _CUDA_ASSIGNMENT_ENV_ALLOWLIST
        if key in parent and str(parent[key])
    }
    visible = assignment.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and (
        len(visible) > 128
        or _CUDA_VISIBLE_DEVICE_PATTERN.fullmatch(visible) is None
    ):
        raise WorkerError(
            "CUDA_VISIBLE_DEVICES is not a safe single-device assignment"
        )
    for key, value in assignment.items():
        if key == "CUDA_VISIBLE_DEVICES":
            continue
        if len(value) > 4_096 or "\n" in value or "\r" in value:
            raise WorkerError(f"unsafe scheduler assignment value for {key}")
    return assignment


def build_worker_environment(
    *,
    requested_device: str,
    allow_cpu_for_tests: bool,
    model_seed: int,
    parent_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    parent = os.environ if parent_environment is None else parent_environment
    requested_device = _normalize_requested_device(
        requested_device,
        allow_cpu_for_tests=allow_cpu_for_tests,
    )
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
    if requested_device == "cuda":
        environment.update(_validated_cuda_assignment(parent))
        environment["CUBLAS_WORKSPACE_CONFIG"] = CUDA_CUBLAS_WORKSPACE_CONFIG
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
    requested_device = _normalize_requested_device(
        requested_device,
        allow_cpu_for_tests=allow_cpu_for_tests,
    )
    if requested_device != "cpu" and profile.device_requirement != requested_device:
        raise WorkerError(
            f"profile {profile.name} requires {profile.device_requirement}; "
            f"refusing worker request for {requested_device}"
        )
    if profile.name == "smoke_train_cuda_v1":
        try:
            validate_trusted_initial_candidate(candidate)
        except (OSError, ValueError) as error:
            raise WorkerError(str(error)) from error
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
