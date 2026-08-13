from __future__ import annotations

import json
from types import SimpleNamespace

import common.device as device_module
import pytest
import torch
from common.device import AcceleratorFingerprint
from common.runtime_context import ExecutionContextV1


def _context() -> ExecutionContextV1:
    return ExecutionContextV1(
        execution_backend="modal",
        run_id="cuda-environment-test",
        app_name="rl4rl-architecture-discovery",
        function_name="cuda_environment",
        modal_app_id="ap-test",
        modal_function_id="fu-test",
        modal_call_id="fc-test",
        modal_image_id="im-test",
        image_source_sha256="a" * 64,
        artifact_uri=(
            "volume://rl4rl-architecture-artifacts/runs/cuda-environment-test"
        ),
    )


def _mock_cuda(monkeypatch, *, driver: str | None) -> list[object]:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "NVIDIA T4")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda _index: (7, 5),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(total_memory=16_000_000_000),
    )
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    driver_probe_calls: list[object] = []

    def driver_probe() -> str | None:
        driver_probe_calls.append(object())
        return driver

    monkeypatch.setattr(device_module, "_cuda_driver_version", driver_probe)
    return driver_probe_calls


def test_cuda_environment_writes_shared_validated_fingerprint(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    driver_probe_calls = _mock_cuda(monkeypatch, driver="550.54.15")
    monkeypatch.setattr(
        modal_app, "_git_runtime_version", lambda: "git version 2.47.3"
    )

    result = modal_app._cuda_environment_action(tmp_path, _context())

    payload = json.loads(
        (tmp_path / "cuda_environment.json").read_text(encoding="utf-8")
    )
    fingerprint = AcceleratorFingerprint.from_dict(
        payload["accelerator_fingerprint"]
    ).validate_cuda(exact_gpu_count=1, require_driver=True)
    assert len(driver_probe_calls) == 1
    assert fingerprint.selected_device == "cuda:0"
    assert fingerprint.gpu_name == "NVIDIA T4"
    assert fingerprint.compute_capability == "7.5"
    assert fingerprint.cuda_runtime == "12.8"
    assert fingerprint.cuda_driver == "550.54.15"
    assert payload["cuda_device_count"] == fingerprint.gpu_count
    assert payload["cuda_device_name"] == fingerprint.gpu_name
    assert payload["cuda_compute_capability"] == [7, 5]
    assert payload["cuda_runtime"] == fingerprint.cuda_runtime
    assert payload["cuda_driver"] == fingerprint.cuda_driver
    assert payload["torch"] == fingerprint.torch_version
    assert payload["platform"] == fingerprint.host_platform
    assert payload["git_version"] == "git version 2.47.3"
    assert result == {
        "mode": "cuda_environment",
        "observed_gpu": "NVIDIA T4",
    }


def test_cuda_environment_fails_closed_without_driver_evidence(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    driver_probe_calls = _mock_cuda(monkeypatch, driver=None)

    with pytest.raises(modal_app.RemoteActionError, match="driver is missing"):
        modal_app._cuda_environment_action(tmp_path, _context())

    assert len(driver_probe_calls) == 1
    assert not (tmp_path / "cuda_environment.json").exists()


def test_cuda_environment_fails_closed_without_git_runtime(
    tmp_path, monkeypatch
) -> None:
    import modal_app

    _mock_cuda(monkeypatch, driver="550.54.15")

    def missing_git() -> str:
        raise modal_app.RemoteActionError(
            "Modal image is missing its required Git runtime"
        )

    monkeypatch.setattr(modal_app, "_git_runtime_version", missing_git)

    with pytest.raises(modal_app.RemoteActionError, match="required Git runtime"):
        modal_app._cuda_environment_action(tmp_path, _context())

    assert not (tmp_path / "cuda_environment.json").exists()


def test_git_runtime_probe_is_bounded_and_credential_free(monkeypatch) -> None:
    import modal_app

    calls: list[tuple[object, object]] = []

    def git_probe(command, **options):
        calls.append((command, options))
        return SimpleNamespace(stdout="git version 2.47.3\n")

    monkeypatch.setattr(modal_app.subprocess, "run", git_probe)

    assert modal_app._git_runtime_version() == "git version 2.47.3"
    assert calls == [
        (
            ["/usr/bin/git", "--version"],
            {
                "env": {"PATH": "/usr/bin:/bin"},
                "stdin": modal_app.subprocess.DEVNULL,
                "capture_output": True,
                "text": True,
                "check": True,
                "timeout": 2,
            },
        )
    ]
