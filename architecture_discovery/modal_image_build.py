"""Minimal, source-staged entrypoint for the bounded Modal image build.

The module intentionally imports only the standard library.  ``modal_app``
copies these exact, manifest-verified bytes into the base image before asking
Modal to import and run :func:`install_image_dependencies`.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

_BUILD_ENVIRONMENT_ALLOWLIST = (
    "LANG",
    "LC_ALL",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SSL_CERT_FILE",
    "TMPDIR",
)
_TERMINATION_GRACE_SECONDS = 0.25
_REAP_TIMEOUT_SECONDS = 1.0
_FROZEN_BUILD_THREAD_LIMIT = 2


class BuildProcessGroupClosureError(RuntimeError):
    """Raised when an image-build child group cannot be closed."""


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen[Any],
    *,
    process_group_id: int,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))


def _capture_process_group(process: subprocess.Popen[Any]) -> int:
    process_group_id = process.pid
    try:
        observed_process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        return process_group_id
    if observed_process_group_id != process_group_id:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=_REAP_TIMEOUT_SECONDS)
        raise RuntimeError(
            "image-build child was not its isolated process-group leader"
        )
    return process_group_id


def _close_process_group(
    process: subprocess.Popen[Any],
    *,
    process_group_id: int,
) -> None:
    if process_group_id != process.pid:
        raise ValueError("captured image-build process group is invalid")
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return

    group_closed = _wait_for_process_group_exit(
        process,
        process_group_id=process_group_id,
        timeout_seconds=_TERMINATION_GRACE_SECONDS,
    )
    if not group_closed:
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
        group_closed = _wait_for_process_group_exit(
            process,
            process_group_id=process_group_id,
            timeout_seconds=_REAP_TIMEOUT_SECONDS,
        )
    try:
        process.wait(timeout=_REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_REAP_TIMEOUT_SECONDS)
    if not group_closed:
        raise BuildProcessGroupClosureError(
            "image-build child group remained after its SIGKILL deadline"
        )


def _run_bounded_command(
    command: list[str],
    *,
    cwd: str | None,
    environment: dict[str, str],
    timeout_seconds: float,
) -> None:
    """Run one build command and close its entire isolated process group."""

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    # ``start_new_session=True`` fixes the expected PGID to the child PID.
    # Preserve it before observational capture so capture failure still
    # reaches the group-closure path below.
    process_group_id = process.pid
    pending_error: BaseException | None = None
    returncode: int | None = None
    try:
        process_group_id = _capture_process_group(process)
        returncode = process.wait(timeout=timeout_seconds)
    except BaseException as error:
        pending_error = error
    try:
        _close_process_group(
            process,
            process_group_id=process_group_id,
        )
    except BaseException as cleanup_error:
        if pending_error is not None:
            raise cleanup_error from pending_error
        raise
    if pending_error is not None:
        raise pending_error
    if returncode is None:  # pragma: no cover - exhaustive state guard
        raise RuntimeError("image-build child completed without a return code")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)


def _build_subprocess_environment(
    thread_limit: int = _FROZEN_BUILD_THREAD_LIMIT,
) -> dict[str, str]:
    if type(thread_limit) is not int or thread_limit != _FROZEN_BUILD_THREAD_LIMIT:
        raise ValueError("image build subprocess thread limit must remain two")
    environment = {
        name: os.environ[name]
        for name in _BUILD_ENVIRONMENT_ALLOWLIST
        if name in os.environ
    }
    environment.update(
        {
            "BLIS_NUM_THREADS": str(thread_limit),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(thread_limit),
            "DEBIAN_FRONTEND": "noninteractive",
            "MAKEFLAGS": f"-j{thread_limit}",
            "MAX_JOBS": str(thread_limit),
            "MKL_NUM_THREADS": str(thread_limit),
            "NUMEXPR_NUM_THREADS": str(thread_limit),
            "OMP_NUM_THREADS": str(thread_limit),
            "OPENBLAS_NUM_THREADS": str(thread_limit),
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "RAYON_NUM_THREADS": str(thread_limit),
            "TOKENIZERS_PARALLELISM": "false",
            "UV_CONCURRENT_BUILDS": str(thread_limit),
            "UV_CONCURRENT_DOWNLOADS": str(thread_limit),
            "UV_CONCURRENT_INSTALLS": str(thread_limit),
            "UV_NO_PROGRESS": "1",
            "VECLIB_MAXIMUM_THREADS": str(thread_limit),
        }
    )
    return environment


def install_image_dependencies(
    *,
    project_root: str,
    thread_limit: int,
    uv_version: str,
    timeout_seconds: int,
) -> None:
    """Install one frozen environment under a shared hard deadline."""

    if type(timeout_seconds) is not int or timeout_seconds <= 15:
        raise ValueError("image build timeout must be an integer greater than 15")
    deadline = time.monotonic() + timeout_seconds - 15
    environment = _build_subprocess_environment(thread_limit)

    def run(command: list[str], *, cwd: str | None = None) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Modal image dependency installation timed out")
        _run_bounded_command(
            command,
            cwd=cwd,
            environment=environment,
            timeout_seconds=remaining,
        )

    # Greedy Autoresearch records its accepted lineage in a real Git repository.
    # Install and verify that runtime binary inside this same bounded build
    # Function instead of introducing an implicit, separately resourced image
    # build step.
    run(["/usr/bin/apt-get", "update"])
    run(
        [
            "/usr/bin/apt-get",
            "install",
            "--yes",
            "--no-install-recommends",
            "git",
        ]
    )
    run(["/usr/bin/git", "--version"])
    run([sys.executable, "-m", "pip", "install", f"uv=={uv_version}"])
    run(
        [
            str(Path(sys.executable).with_name("uv")),
            "sync",
            "--frozen",
            "--no-dev",
            "--group",
            "modal",
            "--no-install-project",
        ],
        cwd=project_root,
    )
