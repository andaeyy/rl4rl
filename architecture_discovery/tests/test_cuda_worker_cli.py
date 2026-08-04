from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import common.training_worker as training_worker_module
from common.task_adapter import DEFAULT_TASK
from common.trainer import sha256_file
from common.training_client import (
    WorkerError,
    build_worker_environment,
    run_worker_job,
)
from common.training_config import SMOKE_TRAIN_CUDA_V1, TrainingSeedBundle
from scripts import check_environment, retrain_candidate, train_candidate


ROOT = Path(__file__).resolve().parents[1]


def _cuda_parent_environment() -> dict[str, str]:
    return {
        "LANG": "C.UTF-8",
        "CUDA_VISIBLE_DEVICES": "0",
        "SLURM_JOB_ID": "12345",
        "SLURM_JOB_GPUS": "0",
        "SLURM_STEP_GPUS": "0",
        "SLURM_GPUS_ON_NODE": "1",
        "DISCOVERY_API_KEY": "discovery-secret",
        "OPENAI_API_KEY": "openai-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GITHUB_TOKEN": "github-secret",
        "NVIDIA_VISIBLE_DEVICES": "all",
        "LD_LIBRARY_PATH": "/untrusted/library/path",
        "PYTHONPATH": "/untrusted/python/path",
    }


def test_cuda_worker_environment_is_minimal_deterministic_and_secret_free() -> None:
    parent = _cuda_parent_environment()
    environment = build_worker_environment(
        requested_device="cuda",
        allow_cpu_for_tests=False,
        model_seed=7,
        parent_environment=parent,
    )
    assert environment["DISCOVERY_TRAIN_DEVICE"] == "cuda"
    assert environment["DISCOVERY_ALLOW_CPU_TRAINING"] == "0"
    assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["SLURM_JOB_ID"] == "12345"
    assert environment["SLURM_JOB_GPUS"] == "0"
    assert environment["SLURM_STEP_GPUS"] == "0"
    assert environment["SLURM_GPUS_ON_NODE"] == "1"
    for key, value in parent.items():
        if key in {
            "LANG",
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "SLURM_GPUS_ON_NODE",
        }:
            continue
        assert key not in environment
        assert value not in environment.values()


def test_non_cuda_worker_does_not_inherit_cuda_or_scheduler_assignment() -> None:
    environment = build_worker_environment(
        requested_device="mps",
        allow_cpu_for_tests=False,
        model_seed=1,
        parent_environment=_cuda_parent_environment(),
    )
    assert "CUDA_VISIBLE_DEVICES" not in environment
    assert "CUBLAS_WORKSPACE_CONFIG" not in environment
    assert "SLURM_JOB_ID" not in environment


def test_cuda_worker_rejects_ambiguous_visible_device_assignment() -> None:
    parent = _cuda_parent_environment()
    parent["CUDA_VISIBLE_DEVICES"] = "0,1"
    with pytest.raises(WorkerError, match="single-device assignment"):
        build_worker_environment(
            requested_device="cuda",
            allow_cpu_for_tests=False,
            model_seed=1,
            parent_environment=parent,
        )


def test_worker_forbids_cpu_without_explicit_test_permission() -> None:
    with pytest.raises(WorkerError, match="test-only permission"):
        build_worker_environment(
            requested_device="cpu",
            allow_cpu_for_tests=False,
            model_seed=1,
            parent_environment={},
        )


def test_worker_job_passes_only_sanitized_cuda_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for key, value in _cuda_parent_environment().items():
        monkeypatch.setenv(key, value)
    captured: dict[str, object] = {}

    def fake_run(command, *, env, **_kwargs):
        captured["environment"] = dict(env)
        job_path = Path(command[-2])
        response_path = Path(command[-1])
        captured["job"] = json.loads(job_path.read_text(encoding="utf-8"))
        response_path.write_text(
            json.dumps(
                {
                    "kind": "worker_failure",
                    "failure_stage": "worker_infrastructure",
                    "error": "mocked worker",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("common.training_client.subprocess.run", fake_run)
    response = run_worker_job(
        mode="train",
        candidate_path=ROOT / "common" / "initial_candidate.py",
        output_dir=tmp_path / "training",
        profile=SMOKE_TRAIN_CUDA_V1,
        seeds=TrainingSeedBundle.from_run_seed(1),
        requested_device="cuda",
        allow_cpu_for_tests=False,
    )
    assert response["kind"] == "worker_failure"
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "DISCOVERY_API_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    serialized_job = json.dumps(captured["job"], sort_keys=True)
    assert "discovery-secret" not in serialized_job
    assert "openai-secret" not in serialized_job


def test_cuda_smoke_worker_rejects_an_alternate_candidate_path(tmp_path) -> None:
    alternate = tmp_path / "alternate.py"
    alternate.write_bytes((ROOT / "common" / "initial_candidate.py").read_bytes())

    with pytest.raises(WorkerError, match="restricted"):
        run_worker_job(
            mode="train",
            candidate_path=alternate,
            output_dir=tmp_path / "training",
            profile=SMOKE_TRAIN_CUDA_V1,
            seeds=TrainingSeedBundle.from_run_seed(1),
            requested_device="cuda",
            allow_cpu_for_tests=False,
        )


def test_cuda_worker_uses_accelerator_lease_and_emits_v2_resource_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for key, value in _cuda_parent_environment().items():
        if key in {
            "CUDA_VISIBLE_DEVICES",
            "SLURM_JOB_ID",
            "SLURM_JOB_GPUS",
            "SLURM_STEP_GPUS",
            "SLURM_GPUS_ON_NODE",
        }:
            monkeypatch.setenv(key, value)
    monkeypatch.setenv("DISCOVERY_TRAIN_DEVICE", "cuda")
    monkeypatch.setenv("DISCOVERY_ALLOW_CPU_TRAINING", "0")
    monkeypatch.setattr(
        training_worker_module,
        "_CUDA_LEASE_ROOT",
        tmp_path / "leases",
    )
    monkeypatch.setattr(training_worker_module, "_deny_network", lambda: None)
    output = tmp_path / "training"

    class FakeTraining:
        success = True
        train_seconds = 1.25
        device = "cuda:0"

        def to_dict(self):
            return {"success": True, "train_seconds": self.train_seconds}

    def fake_train(**_kwargs):
        output.mkdir(mode=0o700)
        (output / "training_summary.json").write_text(
            json.dumps({"success": True}), encoding="utf-8"
        )
        return FakeTraining()

    monkeypatch.setattr(
        training_worker_module,
        "train_candidate_in_process",
        fake_train,
    )
    seeds = TrainingSeedBundle.from_run_seed(1)
    candidate = ROOT / "common" / "initial_candidate.py"
    job = {
        "mode": "train",
        "candidate_path": str(candidate),
        "candidate_source_hash": sha256_file(candidate),
        "output_dir": str(output),
        "profile_name": SMOKE_TRAIN_CUDA_V1.name,
        "profile_version": SMOKE_TRAIN_CUDA_V1.version,
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "seed_bundle": seeds.__dict__,
        "seed_bundle_hash": seeds.bundle_hash,
        "task_adapter_version": DEFAULT_TASK.version,
        "task_adapter_hash": DEFAULT_TASK.config_hash,
        "requested_device": "cuda",
        "allow_cpu_for_tests": False,
        "resume": None,
        "evaluation_plan": None,
        "evaluation_context": None,
        "eligibility_threshold": 0.99,
    }

    response = training_worker_module.run_job(job)

    assert response["kind"] == "training_result"
    resource = json.loads(
        (output / "accelerator_resource_usage.json").read_text(encoding="utf-8")
    )
    assert resource["schema_version"] == "2.0"
    assert resource["accelerator_key"].startswith("cuda:")
    assert resource["resource_ledger"]["spec"]["backend"] == "cuda"
    assert not list((tmp_path / "leases").glob("*.lock"))


def test_direct_training_clis_accept_cuda_profiles_and_device() -> None:
    train_args = train_candidate.build_parser().parse_args(
        [
            "--candidate",
            "common/initial_candidate.py",
            "--profile",
            "smoke_train_cuda_v1",
            "--device",
            "cuda",
            "--seed",
            "1",
            "--output-dir",
            "/scratch/example",
        ]
    )
    assert train_args.profile == "smoke_train_cuda_v1"
    assert train_args.device == "cuda"

    retrain_args = retrain_candidate.build_parser().parse_args(
        [
            "--candidate",
            "common/initial_candidate.py",
            "--profile",
            "full_train_cuda_a40_v1",
            "--device",
            "cuda",
            "--seeds",
            "1,2",
            "--output-dir",
            "/scratch/example",
        ]
    )
    assert retrain_args.profile == "full_train_cuda_a40_v1"
    assert retrain_args.device == "cuda"


def test_environment_checker_does_not_inventory_gpu_without_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(check_environment, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(
        check_environment,
        "_nvidia_smi",
        lambda: pytest.fail("nvidia-smi must not run outside an allocation"),
    )
    monkeypatch.setattr(check_environment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(check_environment.torch.cuda, "device_count", lambda: 1)
    report = check_environment.environment_report(
        requested_device="cuda",
        profile_name="smoke_train_cuda_v1",
    )
    assert report["training"]["device_status"] == "cuda_unallocated_no_fallback"
    assert not report["cuda"]["allocated_gpu_validation_performed"]


def test_environment_checker_validates_one_allocated_a40(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(check_environment, "_git_commit", lambda: "a" * 40)
    monkeypatch.setattr(check_environment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(check_environment.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        check_environment.torch.cuda,
        "get_device_properties",
        lambda _index: SimpleNamespace(
            name="NVIDIA A40",
            uuid="GPU-mocked",
            total_memory=48 * 1024**3,
        ),
    )
    monkeypatch.setattr(
        check_environment.torch.cuda,
        "get_device_capability",
        lambda _index: (8, 6),
    )
    monkeypatch.setattr(
        check_environment,
        "_nvidia_smi",
        lambda: (
            [
                {
                    "index": "0",
                    "name": "NVIDIA A40",
                    "uuid": "GPU-mocked",
                    "memory_total_mib": "46068",
                    "driver": "mocked",
                }
            ],
            None,
        ),
    )
    report = check_environment.environment_report(
        requested_device="cuda",
        profile_name="smoke_train_cuda_v1",
    )
    assert report["training"]["device_status"] == "cuda_ready"
    assert report["cuda"]["selected_device"]["name"] == "NVIDIA A40"
    assert check_environment._readiness_errors(report, require_a40=True) == []


def test_environment_checker_rejects_masked_multi_gpu_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(check_environment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(check_environment.torch.cuda, "device_count", lambda: 1)

    report = check_environment.environment_report(
        requested_device="cuda",
        profile_name="smoke_train_cuda_v1",
    )

    assert report["training"]["device_status"] == "cuda_unallocated_no_fallback"
    assert report["scheduler_allocation"]["exactly_one_allocated_gpu"] is False
