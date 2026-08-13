import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import common.device as device_module
import common.trainer as trainer_module
import pytest
import torch
from common.device import (
    AcceleratorFingerprint,
    DeviceUnavailableError,
    resolve_training_device,
)
from common.training_config import (
    FULL_TRAIN_V1,
    SMOKE_TRAIN_CUDA_V2,
    SMOKE_TRAIN_V1,
    TrainingSeedBundle,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scientific_mps_fails_instead_of_falling_back(monkeypatch):
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(DeviceUnavailableError, match="no CPU fallback"):
        resolve_training_device(
            FULL_TRAIN_V1, "mps", allow_cpu_for_tests=False
        )


def test_cpu_requires_explicit_engineering_flag():
    with pytest.raises(DeviceUnavailableError, match="allow-cpu-for-tests"):
        resolve_training_device(
            SMOKE_TRAIN_V1, "cpu", allow_cpu_for_tests=False
        )
    selected = resolve_training_device(
        SMOKE_TRAIN_V1, "cpu", allow_cpu_for_tests=True
    )
    assert selected.device.type == "cpu"
    assert not selected.hardware_matched


def test_scientific_profile_rejects_cpu_even_with_test_flag():
    with pytest.raises(DeviceUnavailableError, match="requires MPS"):
        resolve_training_device(
            FULL_TRAIN_V1, "cpu", allow_cpu_for_tests=True
        )


def test_synchronized_clock_synchronizes_before_read(monkeypatch):
    events = []
    monkeypatch.setattr(
        device_module, "synchronize", lambda _device: events.append("sync")
    )
    monkeypatch.setattr(
        device_module.time,
        "perf_counter",
        lambda: events.append("clock") or 123.0,
    )
    assert device_module.synchronized_time(torch.device("mps")) == 123.0
    assert events == ["sync", "clock"]


def _mock_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "T4")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (7, 5))


def test_cuda_selection_is_fail_closed_and_records_fingerprint(monkeypatch):
    _mock_cuda(monkeypatch)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    selected = resolve_training_device(
        SMOKE_TRAIN_CUDA_V2,
        "cuda",
        allow_cpu_for_tests=False,
    )
    assert selected.device == torch.device("cuda", 0)
    assert selected.hardware_matched
    assert selected.fingerprint.gpu_name == "T4"
    assert selected.fingerprint.compute_capability == "7.5"
    assert selected.fingerprint.accelerator_kind == "cuda"


def test_cuda_driver_falls_back_to_bounded_nvidia_smi(monkeypatch):
    monkeypatch.setattr(torch.cuda, "driver_version", None, raising=False)
    monkeypatch.setattr(
        device_module.shutil,
        "which",
        lambda _name: "/usr/bin/nvidia-smi",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="550.54.15\n")

    monkeypatch.setattr(device_module.subprocess, "run", fake_run)
    assert device_module._cuda_driver_version() == "550.54.15"
    assert calls[0][0][0] == "/usr/bin/nvidia-smi"
    assert calls[0][0][1:] == [
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ]
    assert calls[0][1]["stdin"] is device_module.subprocess.DEVNULL
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["text"] is True
    assert calls[0][1]["timeout"] == 2
    assert calls[0][1]["check"] is False


def test_cuda_fingerprint_exact_parser_requires_driver_when_attesting():
    fingerprint = AcceleratorFingerprint(
        requested_device="cuda",
        selected_device="cuda:0",
        accelerator_kind="cuda",
        gpu_name="NVIDIA T4",
        gpu_count=1,
        compute_capability="7.5",
        cuda_runtime="12.8",
        cuda_driver="550.54.15",
        torch_version="2.7.1",
        host_platform="Linux-test",
    )
    parsed = AcceleratorFingerprint.from_dict(fingerprint.to_dict())
    assert parsed.validate_cuda(
        exact_gpu_count=1,
        require_driver=True,
    ) == fingerprint

    missing_driver = fingerprint.to_dict()
    missing_driver["cuda_driver"] = None
    with pytest.raises(ValueError, match="driver is missing"):
        AcceleratorFingerprint.from_dict(missing_driver).validate_cuda(
            exact_gpu_count=1,
            require_driver=True,
        )

    extra_field = {**fingerprint.to_dict(), "unexpected": "value"}
    with pytest.raises(ValueError, match="invalid exact schema"):
        AcceleratorFingerprint.from_dict(extra_field)


def test_cuda_unavailable_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.version, "cuda", "12.6")
    with pytest.raises(DeviceUnavailableError, match="no CPU fallback"):
        resolve_training_device(
            SMOKE_TRAIN_CUDA_V2,
            "cuda",
            allow_cpu_for_tests=False,
        )


def test_cuda_requires_bound_cublas_workspace(monkeypatch):
    _mock_cuda(monkeypatch)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(DeviceUnavailableError, match="CUBLAS_WORKSPACE_CONFIG"):
        resolve_training_device(
            SMOKE_TRAIN_CUDA_V2,
            "cuda",
            allow_cpu_for_tests=False,
        )


def test_cleanup_failure_is_not_attested_as_success(monkeypatch, tmp_path):
    profile = replace(
        SMOKE_TRAIN_V1,
        name="cleanup_failure_smoke",
        max_steps=1,
        global_batch_size=2,
        validation_interval=1,
        validation_examples=2,
        checkpoint_interval=1,
    )

    def fail_cleanup(_device):
        raise RuntimeError("sensitive driver detail")

    monkeypatch.setattr(trainer_module, "cleanup_accelerator", fail_cleanup)
    output = tmp_path / "training"
    result = trainer_module.train_candidate_in_process(
        candidate_path=ROOT / "common" / "initial_candidate.py",
        output_dir=output,
        profile=profile,
        seeds=TrainingSeedBundle.from_run_seed(31),
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )
    assert result.success is False
    assert result.cleanup_completed is False
    assert result.failure_stage == "accelerator_cleanup_failure"
    assert "sensitive driver detail" not in result.error
    cleanup = json.loads((output / "cleanup_failure.json").read_text())
    assert cleanup["error_type"] == "RuntimeError"
    assert "sensitive driver detail" not in json.dumps(cleanup)


def test_v2_training_failure_artifacts_suppress_executor_paths(
    monkeypatch,
    tmp_path,
):
    def fail_device_selection(*_args, **_kwargs):
        raise RuntimeError(
            "/mnt/discovery/runs/private /opt/architecture_discovery"
        )

    monkeypatch.setattr(
        trainer_module,
        "resolve_training_device",
        fail_device_selection,
    )
    output = tmp_path / "training"
    result = trainer_module.train_candidate_in_process(
        candidate_path=ROOT / "common" / "initial_candidate.ir.json",
        output_dir=output,
        profile=SMOKE_TRAIN_CUDA_V2,
        seeds=TrainingSeedBundle.from_run_seed(37),
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )

    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (output / "training_summary.json").read_text(encoding="utf-8")
    )
    serialized = json.dumps({"failure": failure, "summary": summary})
    assert not result.success
    assert failure["message"] == "training failed; details suppressed"
    assert "traceback" not in failure
    assert "/mnt/" not in serialized
    assert "/opt/" not in serialized
