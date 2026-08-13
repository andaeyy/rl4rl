import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from common.process_control import OUTER_PROCESS_DEADLINE_ENV
from common.runtime_context import ExecutionContextV1
from common.training_client import (
    WorkerError,
    _bounded_worker_timeout_seconds,
    _outer_process_group_is_contained,
    _validated_execution_context,
    build_worker_environment,
    run_worker_job,
)
from common.training_config import SMOKE_TRAIN_CUDA_V2, TrainingSeedBundle

ROOT = Path(__file__).resolve().parents[1]


def test_worker_environment_is_allowlisted_and_excludes_credentials():
    secrets = {
        "DISCOVERY_API_KEY": "discovery-secret",
        "OPENAI_API_KEY": "openai-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "GOOGLE_API_KEY": "google-secret",
        "GEMINI_API_KEY": "gemini-secret",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "GITHUB_TOKEN": "github-secret",
        "HF_TOKEN": "hf-secret",
        "SUPER_SECRET_TOKEN": "other-secret",
        "DISCOVERY_SHADOW_SEED": "sealed-seed",
        "PYTHONPATH": "/untrusted",
        "PYTHONSTARTUP": "/untrusted/start.py",
        "DISCOVERY_EXECUTION_CONTEXT_JSON": '{"credential":"never-forward"}',
        OUTER_PROCESS_DEADLINE_ENV: "12345.0",
        "LANG": "en_US.UTF-8",
    }
    environment = build_worker_environment(
        requested_device="cpu",
        allow_cpu_for_tests=True,
        model_seed=1,
        parent_environment=secrets,
    )
    assert environment["LANG"] == "en_US.UTF-8"
    assert environment["DISCOVERY_TRAIN_DEVICE"] == "cpu"
    assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    for key, value in secrets.items():
        if key == "LANG":
            continue
        assert key not in environment
        assert value not in environment.values()


def test_training_path_has_no_provider_sdk_or_paid_call():
    for name in (
        "trainer.py",
        "training_client.py",
        "training_worker.py",
        "training_data.py",
    ):
        source = (ROOT / "common" / name).read_text().lower()
        assert "from openai" not in source
        assert "import openai" not in source
        assert "chat.completions" not in source


def test_nested_worker_timeout_preserves_outer_cleanup_guard():
    assert _bounded_worker_timeout_seconds(
        60,
        parent_environment={},
        monotonic_now=100.0,
    ) == 360.0
    assert _bounded_worker_timeout_seconds(
        60,
        parent_environment={OUTER_PROCESS_DEADLINE_ENV: "150.0"},
        monotonic_now=100.0,
    ) == 40.0
    with pytest.raises(WorkerError, match="insufficient worker cleanup guard"):
        _bounded_worker_timeout_seconds(
            60,
            parent_environment={OUTER_PROCESS_DEADLINE_ENV: "105.0"},
            monotonic_now=100.0,
        )
    with pytest.raises(WorkerError, match="deadline is invalid"):
        _bounded_worker_timeout_seconds(
            60,
            parent_environment={OUTER_PROCESS_DEADLINE_ENV: "nan"},
            monotonic_now=100.0,
        )


def test_spoofed_outer_deadline_does_not_disable_local_process_group(
    monkeypatch,
) -> None:
    environment = {OUTER_PROCESS_DEADLINE_ENV: "12345.0"}
    monkeypatch.setattr("common.training_client.os.getpid", lambda: 101)
    monkeypatch.setattr("common.training_client.os.getpgrp", lambda: 100)
    assert not _outer_process_group_is_contained(
        parent_environment=environment
    )

    monkeypatch.setattr("common.training_client.os.getpgrp", lambda: 101)
    assert _outer_process_group_is_contained(parent_environment=environment)


def test_cuda_worker_inherits_one_visible_device_and_determinism_only():
    environment = build_worker_environment(
        requested_device="cuda",
        allow_cpu_for_tests=False,
        model_seed=7,
        cublas_workspace_config=":4096:8",
        parent_environment={
            "CUDA_VISIBLE_DEVICES": "GPU-one",
            "NVIDIA_VISIBLE_DEVICES": "GPU-one",
            "MODAL_TOKEN_SECRET": "never-forward",
            "DISCOVERY_API_KEY": "never-forward",
        },
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-one"
    assert environment["NVIDIA_VISIBLE_DEVICES"] == "GPU-one"
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "MODAL_TOKEN_SECRET" not in environment
    assert "DISCOVERY_API_KEY" not in environment


def test_cuda_worker_rejects_multiple_visible_devices():
    with pytest.raises(WorkerError, match="exactly one"):
        build_worker_environment(
            requested_device="cuda",
            allow_cpu_for_tests=False,
            model_seed=7,
            cublas_workspace_config=":4096:8",
            parent_environment={"CUDA_VISIBLE_DEVICES": "0,1"},
        )


def test_execution_context_is_validated_as_data_but_not_inherited():
    context = ExecutionContextV1(
        execution_backend="modal",
        run_id="modal-run-1",
        app_name="rl4rl-architecture-discovery",
        function_name="candidate_smoke",
        modal_app_id=None,
        modal_function_id=None,
        modal_call_id="fc-call1",
        modal_image_id=None,
        image_source_sha256="a" * 64,
        artifact_uri="volume://rl4rl-architecture-artifacts/runs/modal-run-1",
    )
    encoded = json.dumps(context.to_dict())
    assert _validated_execution_context(
        None,
        parent_environment={"DISCOVERY_EXECUTION_CONTEXT_JSON": encoded},
    ) == context

    child = build_worker_environment(
        requested_device="cuda",
        allow_cpu_for_tests=False,
        model_seed=7,
        cublas_workspace_config=":4096:8",
        parent_environment={"DISCOVERY_EXECUTION_CONTEXT_JSON": encoded},
    )
    assert "DISCOVERY_EXECUTION_CONTEXT_JSON" not in child


def test_execution_context_rejects_extra_fields_and_oversize_values():
    with pytest.raises(WorkerError, match="invalid execution context"):
        _validated_execution_context(
            None,
            parent_environment={
                "DISCOVERY_EXECUTION_CONTEXT_JSON": json.dumps(
                    {"schema_name": "ExecutionContext", "api_key": "forbidden"}
                )
            },
        )
    with pytest.raises(WorkerError, match="16 KiB"):
        _validated_execution_context(
            None,
            parent_environment={"DISCOVERY_EXECUTION_CONTEXT_JSON": "x" * 16_385},
        )


def test_training_timeout_kills_grandchildren_before_returning(
    tmp_path, monkeypatch
) -> None:
    """A candidate-worker descendant cannot mutate outputs after timeout."""

    from common import training_client

    marker = tmp_path / "forbidden-late-write"
    ready = tmp_path / "grandchild-ready"
    grandchild = (
        "import signal,sys,time;"
        "from pathlib import Path;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "Path(sys.argv[1]).write_text('ready');"
        "time.sleep(0.75);"
        "Path(sys.argv[2]).write_text('late')"
    )
    bootstrap = tmp_path / "malicious_bootstrap.py"
    bootstrap.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "import subprocess",
                "import sys",
                "import time",
                f"grandchild = {grandchild!r}",
                f"ready = {str(ready)!r}",
                f"marker = {str(marker)!r}",
                "subprocess.Popen([sys.executable, '-c', grandchild, ready, marker])",
                "deadline = time.monotonic() + 2",
                "while not Path(ready).exists():",
                "    assert time.monotonic() < deadline",
                "    time.sleep(0.01)",
                "time.sleep(5)",
            )
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    profile = replace(SMOKE_TRAIN_CUDA_V2, maximum_wall_seconds=-299.8)
    monkeypatch.setattr(training_client, "BOOTSTRAP", bootstrap)
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin:/bin"))

    started = time.monotonic()
    with pytest.raises(WorkerError, match="hard timeout"):
        run_worker_job(
            mode="train",
            candidate_path=candidate,
            output_dir=tmp_path / "training-output",
            profile=profile,
            seeds=TrainingSeedBundle.from_run_seed(1),
            requested_device="cpu",
            allow_cpu_for_tests=True,
        )
    elapsed = time.monotonic() - started

    assert ready.is_file(), "the adversarial grandchild never started"
    assert elapsed < 2.5
    time.sleep(0.8)
    assert not marker.exists()


def test_training_worker_capture_failure_closes_expected_process_group(
    tmp_path,
    monkeypatch,
) -> None:
    from common import training_client

    class FakeProcess:
        pid = 161803

    process = FakeProcess()
    cleanup_calls = []
    monkeypatch.setattr(
        training_client,
        "_outer_process_group_is_contained",
        lambda: False,
    )
    monkeypatch.setattr(
        training_client.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def failed_capture(_process):
        raise OSError("synthetic worker PGID capture failure")

    monkeypatch.setattr(
        training_client,
        "capture_isolated_process_group",
        failed_capture,
    )
    monkeypatch.setattr(
        training_client,
        "terminate_process_group",
        lambda child, **kwargs: cleanup_calls.append(
            (child, kwargs["process_group_id"])
        ),
    )

    with pytest.raises(OSError, match="synthetic worker PGID capture failure"):
        run_worker_job(
            mode="train",
            candidate_path=ROOT / "common" / "initial_candidate.ir.json",
            output_dir=tmp_path / "training-output",
            profile=SMOKE_TRAIN_CUDA_V2,
            seeds=TrainingSeedBundle.from_run_seed(3),
            requested_device="cpu",
            allow_cpu_for_tests=True,
        )

    assert cleanup_calls == [(process, process.pid)]


def test_successful_training_worker_cannot_leave_background_grandchild(
    tmp_path, monkeypatch
) -> None:
    from common import training_client

    marker = tmp_path / "forbidden-background-write"
    ready = tmp_path / "background-ready"
    grandchild = (
        "import signal,sys,time;"
        "from pathlib import Path;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "Path(sys.argv[1]).write_text('ready');"
        "time.sleep(0.75);"
        "Path(sys.argv[2]).write_text('late')"
    )
    bootstrap = tmp_path / "successful_bootstrap.py"
    bootstrap.write_text(
        "\n".join(
            (
                "from pathlib import Path",
                "import json",
                "import subprocess",
                "import sys",
                "import time",
                f"grandchild = {grandchild!r}",
                f"ready = {str(ready)!r}",
                f"marker = {str(marker)!r}",
                "subprocess.Popen([sys.executable, '-c', grandchild, ready, marker])",
                "deadline = time.monotonic() + 2",
                "while not Path(ready).exists():",
                "    assert time.monotonic() < deadline",
                "    time.sleep(0.01)",
                "Path(sys.argv[2]).write_text(json.dumps({",
                "    'kind': 'worker_failure', 'error': 'expected-test-failure'",
                "}))",
            )
        ),
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(training_client, "BOOTSTRAP", bootstrap)

    response = run_worker_job(
        mode="train",
        candidate_path=candidate,
        output_dir=tmp_path / "training-output",
        profile=SMOKE_TRAIN_CUDA_V2,
        seeds=TrainingSeedBundle.from_run_seed(2),
        requested_device="cpu",
        allow_cpu_for_tests=True,
    )

    assert response["kind"] == "worker_failure"
    assert ready.is_file()
    time.sleep(0.8)
    assert not marker.exists()
