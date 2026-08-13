from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import modal_image_build
import pytest


def test_installer_runs_pinned_commands_under_one_deadline(monkeypatch) -> None:
    timeline = iter((100.0, 101.0, 102.0, 103.0, 104.0, 105.0))
    monkeypatch.setattr(modal_image_build.time, "monotonic", lambda: next(timeline))
    monkeypatch.setenv("PATH", "/usr/local/bin:/usr/bin:/bin")
    monkeypatch.setenv("MODAL_IDENTITY_TOKEN", "modal-secret")
    monkeypatch.setenv("DISCOVERY_API_KEY", "provider-secret")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))

    monkeypatch.setattr(modal_image_build, "_run_bounded_command", fake_run)

    modal_image_build.install_image_dependencies(
        project_root="/opt/architecture_discovery",
        thread_limit=2,
        uv_version="0.12.0",
        timeout_seconds=600,
    )

    assert calls[0][0] == ["/usr/bin/apt-get", "update"]
    assert calls[1][0] == [
        "/usr/bin/apt-get",
        "install",
        "--yes",
        "--no-install-recommends",
        "git",
    ]
    assert calls[2][0] == ["/usr/bin/git", "--version"]
    assert calls[3][0] == [
        sys.executable,
        "-m",
        "pip",
        "install",
        "uv==0.12.0",
    ]
    assert calls[4][0] == [
        str(modal_image_build.Path(sys.executable).with_name("uv")),
        "sync",
        "--frozen",
        "--no-dev",
        "--group",
        "modal",
        "--no-install-project",
    ]
    assert all(call[1]["cwd"] is None for call in calls[:4])
    assert calls[4][1]["cwd"] == "/opt/architecture_discovery"
    assert calls[0][1]["timeout_seconds"] == 584.0
    assert calls[1][1]["timeout_seconds"] == 583.0
    assert calls[2][1]["timeout_seconds"] == 582.0
    assert calls[3][1]["timeout_seconds"] == 581.0
    assert calls[4][1]["timeout_seconds"] == 580.0
    for _command, options in calls:
        environment = options["environment"]
        assert isinstance(environment, dict)
        assert environment["PIP_NO_INPUT"] == "1"
        assert environment["PYTHONNOUSERSITE"] == "1"
        assert environment["DEBIAN_FRONTEND"] == "noninteractive"
        assert environment["OMP_NUM_THREADS"] == "2"
        assert environment["OPENBLAS_NUM_THREADS"] == "2"
        assert environment["MKL_NUM_THREADS"] == "2"
        assert environment["RAYON_NUM_THREADS"] == "2"
        assert environment["UV_CONCURRENT_BUILDS"] == "2"
        assert environment["UV_CONCURRENT_DOWNLOADS"] == "2"
        assert environment["UV_CONCURRENT_INSTALLS"] == "2"
        assert environment["MAKEFLAGS"] == "-j2"
        assert environment["MAX_JOBS"] == "2"
        assert "MODAL_IDENTITY_TOKEN" not in environment
        assert "DISCOVERY_API_KEY" not in environment


def test_installer_propagates_subprocess_failure(monkeypatch) -> None:
    monkeypatch.setattr(modal_image_build.time, "monotonic", lambda: 1.0)

    def fail(command, **_kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(modal_image_build, "_run_bounded_command", fail)
    with pytest.raises(subprocess.CalledProcessError):
        modal_image_build.install_image_dependencies(
            project_root="/opt/architecture_discovery",
            thread_limit=2,
            uv_version="0.12.0",
            timeout_seconds=600,
        )


def test_installer_fails_before_starting_after_shared_deadline(monkeypatch) -> None:
    timeline = iter((100.0, 686.0))
    monkeypatch.setattr(modal_image_build.time, "monotonic", lambda: next(timeline))
    called = False

    def forbidden_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(modal_image_build, "_run_bounded_command", forbidden_run)
    with pytest.raises(TimeoutError, match="dependency installation timed out"):
        modal_image_build.install_image_dependencies(
            project_root="/opt/architecture_discovery",
            thread_limit=2,
            uv_version="0.12.0",
            timeout_seconds=600,
        )
    assert called is False


@pytest.mark.parametrize("timeout_seconds", (True, 15, 0, -1, 600.0))
def test_installer_rejects_invalid_timeout(timeout_seconds) -> None:
    with pytest.raises(ValueError, match="integer greater than 15"):
        modal_image_build.install_image_dependencies(
            project_root="/opt/architecture_discovery",
            thread_limit=2,
            uv_version="0.12.0",
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize("thread_limit", (True, 1, 3, 2.0))
def test_installer_rejects_non_frozen_thread_limit(thread_limit) -> None:
    with pytest.raises(ValueError, match="thread limit must remain two"):
        modal_image_build.install_image_dependencies(
            project_root="/opt/architecture_discovery",
            thread_limit=thread_limit,
            uv_version="0.12.0",
            timeout_seconds=600,
        )


def _write_descendant_programs(
    tmp_path: Path,
    *,
    parent_sleeps: bool,
) -> tuple[Path, Path]:
    grandchild = tmp_path / "grandchild.py"
    grandchild.write_text(
        "\n".join(
            (
                "import signal",
                "import sys",
                "import time",
                "from pathlib import Path",
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
                "Path(sys.argv[1]).write_text('ready', encoding='utf-8')",
                "time.sleep(0.8)",
                "Path(sys.argv[2]).write_text('late', encoding='utf-8')",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    statements = [
        "import subprocess",
        "import sys",
        "import time",
        "from pathlib import Path",
        ("subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2], sys.argv[3]])"),
        "deadline = time.monotonic() + 2.0",
        "while not Path(sys.argv[2]).is_file():",
        "    if time.monotonic() >= deadline:",
        "        raise RuntimeError('grandchild did not start')",
        "    time.sleep(0.01)",
    ]
    if parent_sleeps:
        statements.append("time.sleep(10.0)")
    parent.write_text("\n".join(statements) + "\n", encoding="utf-8")
    return parent, grandchild


def test_bounded_command_closes_descendants_after_timeout(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    marker = tmp_path / "late-marker"
    parent, grandchild = _write_descendant_programs(
        tmp_path,
        parent_sleeps=True,
    )

    with pytest.raises(subprocess.TimeoutExpired):
        modal_image_build._run_bounded_command(
            [
                sys.executable,
                str(parent),
                str(grandchild),
                str(ready),
                str(marker),
            ],
            cwd=None,
            environment=modal_image_build._build_subprocess_environment(),
            timeout_seconds=0.5,
        )

    assert ready.is_file()
    time.sleep(0.9)
    assert not marker.exists()


def test_bounded_command_closes_descendants_after_success(tmp_path: Path) -> None:
    ready = tmp_path / "ready"
    marker = tmp_path / "late-marker"
    parent, grandchild = _write_descendant_programs(
        tmp_path,
        parent_sleeps=False,
    )

    modal_image_build._run_bounded_command(
        [
            sys.executable,
            str(parent),
            str(grandchild),
            str(ready),
            str(marker),
        ],
        cwd=None,
        environment=modal_image_build._build_subprocess_environment(),
        timeout_seconds=2.0,
    )

    assert ready.is_file()
    time.sleep(0.9)
    assert not marker.exists()


def test_bounded_command_capture_failure_still_closes_expected_group(
    monkeypatch,
) -> None:
    process = SimpleNamespace(pid=314159)
    cleanup_calls = []
    monkeypatch.setattr(
        modal_image_build.subprocess,
        "Popen",
        lambda *_args, **_kwargs: process,
    )

    def failed_capture(_process):
        raise OSError("synthetic build PGID capture failure")

    monkeypatch.setattr(modal_image_build, "_capture_process_group", failed_capture)
    monkeypatch.setattr(
        modal_image_build,
        "_close_process_group",
        lambda child, **kwargs: cleanup_calls.append(
            (child, kwargs["process_group_id"])
        ),
    )

    with pytest.raises(OSError, match="synthetic build PGID capture failure"):
        modal_image_build._run_bounded_command(
            ["synthetic-build-command"],
            cwd=None,
            environment={},
            timeout_seconds=1.0,
        )

    assert cleanup_calls == [(process, process.pid)]
