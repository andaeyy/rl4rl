from types import SimpleNamespace

import pytest
import torch

import common.device as device_module
from common.device import DeviceUnavailableError
from common.training_config import SMOKE_TRAIN_CUDA_V1
from common.cuda_contract import slurm_tres_has_exact_count


def _allocated_cuda_environment(monkeypatch) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_JOB_GPUS", "GPU-a40")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a40")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def test_slurm_tres_count_matching_is_token_exact():
    one = "cpu=8,gres/gpu=1,gres/gpu:a40=1,mem=32G"
    ten = "cpu=8,gres/gpu=10,gres/gpu:a40=10,mem=32G"

    assert slurm_tres_has_exact_count(one, "gres/gpu", 1)
    assert slurm_tres_has_exact_count(one, "gres/gpu:a40", 1)
    assert not slurm_tres_has_exact_count(ten, "gres/gpu", 1)
    assert not slurm_tres_has_exact_count(ten, "gres/gpu:a40", 1)


def _mock_one_cuda_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _device: "NVIDIA A40")


def test_cuda_resolves_only_one_scheduler_allocated_gpu(monkeypatch):
    _allocated_cuda_environment(monkeypatch)
    _mock_one_cuda_device(monkeypatch)

    selection = device_module.resolve_training_device(
        SMOKE_TRAIN_CUDA_V1,
        "cuda",
        allow_cpu_for_tests=False,
    )

    assert str(selection.device) == "cuda:0"
    assert selection.hardware_matched
    assert not selection.fallback_requested


def test_cuda_requires_scheduler_evidence_before_querying_driver(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    queried = []
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: queried.append("cuda") or True,
    )

    with pytest.raises(DeviceUnavailableError, match="SLURM_JOB_ID"):
        device_module.resolve_training_device(
            SMOKE_TRAIN_CUDA_V1,
            "cuda",
            allow_cpu_for_tests=False,
        )

    assert queried == []


@pytest.mark.parametrize(
    ("visible", "device_count"),
    [("", 1), ("-1", 0), ("0,1", 2), ("0", 2)],
)
def test_cuda_rejects_missing_or_nonexclusive_gpu(
    monkeypatch,
    visible,
    device_count,
):
    _allocated_cuda_environment(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)

    with pytest.raises(DeviceUnavailableError, match="no CPU fallback"):
        device_module.resolve_training_device(
            SMOKE_TRAIN_CUDA_V1,
            "cuda",
            allow_cpu_for_tests=False,
        )


def test_cuda_profile_forbids_explicit_cpu_test_fallback():
    with pytest.raises(DeviceUnavailableError, match="CPU fallback is forbidden"):
        device_module.resolve_training_device(
            SMOKE_TRAIN_CUDA_V1,
            "cpu",
            allow_cpu_for_tests=True,
        )


@pytest.mark.parametrize("observed_name", ["NVIDIA RTX A6000", "NVIDIA RTX A4000"])
def test_cuda_profile_rejects_a_non_a40_allocation(monkeypatch, observed_name):
    _allocated_cuda_environment(monkeypatch)
    _mock_one_cuda_device(monkeypatch)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_name",
        lambda _device: observed_name,
    )

    with pytest.raises(DeviceUnavailableError, match="requires an NVIDIA A40"):
        device_module.resolve_training_device(
            SMOKE_TRAIN_CUDA_V1,
            "cuda",
            allow_cpu_for_tests=False,
        )


def test_cuda_rejects_two_allocated_gpus_even_if_one_is_visible(monkeypatch):
    _allocated_cuda_environment(monkeypatch)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0,1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    with pytest.raises(DeviceUnavailableError, match="exactly one allocated GPU"):
        device_module.resolve_training_device(
            SMOKE_TRAIN_CUDA_V1,
            "cuda",
            allow_cpu_for_tests=False,
        )


def test_cuda_requires_frozen_cublas_workspace_configuration(monkeypatch):
    _allocated_cuda_environment(monkeypatch)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    with pytest.raises(DeviceUnavailableError, match="CUBLAS_WORKSPACE_CONFIG"):
        device_module.resolve_training_device(
            SMOKE_TRAIN_CUDA_V1,
            "cuda",
            allow_cpu_for_tests=False,
        )


def test_cuda_synchronization_and_memory_telemetry(monkeypatch):
    calls = []
    device = torch.device("cuda:0")
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda observed: calls.append(("synchronize", str(observed))),
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 101)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 202)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 303)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 404)

    memory = device_module.accelerator_memory(device)

    assert calls == [("synchronize", "cuda:0")]
    assert memory == {
        "allocated_bytes": 101,
        "reserved_bytes": 202,
        "peak_allocated_bytes": 303,
        "peak_reserved_bytes": 404,
        "backend_specific": {},
    }


def test_cuda_runtime_metadata_records_gpu_identity_and_versions(monkeypatch):
    _allocated_cuda_environment(monkeypatch)
    properties = SimpleNamespace(
        name="NVIDIA A40",
        uuid="GPU-1234",
        total_memory=48_000_000_000,
        major=8,
        minor=6,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: properties,
    )
    monkeypatch.setattr(torch.cuda, "driver_version", lambda: 55054, raising=False)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 90501)
    monkeypatch.setattr(torch.version, "cuda", "12.6")

    metadata = device_module.accelerator_runtime_metadata(
        torch.device("cuda:0")
    )

    assert metadata["schema_version"] == "accelerator_telemetry_v1"
    assert metadata["backend"] == "cuda"
    assert metadata["identity"] == {
        "name": "NVIDIA A40",
        "uuid": "GPU-1234",
        "total_memory_bytes": 48_000_000_000,
        "compute_capability": [8, 6],
    }
    assert metadata["runtime"]["cuda_runtime_version"] == "12.6"
    assert metadata["runtime"]["cuda_driver_version"] == "55054"
    assert metadata["runtime"]["cudnn_version"] == 90501
    assert metadata["allocation"]["job_id"] == "12345"


def test_cuda_driver_version_falls_back_to_credential_free_nvml(monkeypatch):
    class FakeFunction:
        def __init__(self, implementation):
            self.implementation = implementation

        def __call__(self, *args):
            return self.implementation(*args)

    initialized = []

    def get_version(buffer, _size):
        buffer.value = b"550.54.15"
        return 0

    fake_nvml = SimpleNamespace(
        nvmlInit_v2=FakeFunction(lambda: initialized.append("init") or 0),
        nvmlSystemGetDriverVersion=FakeFunction(get_version),
        nvmlShutdown=FakeFunction(lambda: initialized.append("shutdown") or 0),
    )
    monkeypatch.setattr(torch.cuda, "driver_version", None, raising=False)
    monkeypatch.setattr(
        torch._C,
        "_cuda_getDriverVersion",
        None,
        raising=False,
    )
    monkeypatch.setattr(device_module.ctypes, "CDLL", lambda _name: fake_nvml)

    assert device_module._cuda_driver_version() == "550.54.15"
    assert initialized == ["init", "shutdown"]
