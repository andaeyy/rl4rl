import torch
import pytest

import common.device as device_module
from common.device import DeviceUnavailableError, resolve_training_device
from common.training_config import FULL_TRAIN_V1, SMOKE_TRAIN_V1


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
