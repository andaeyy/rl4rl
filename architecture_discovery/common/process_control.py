"""Bounded POSIX process-group cleanup for trusted subprocess launchers."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from contextlib import suppress
from typing import Any

TERMINATION_GRACE_SECONDS = 0.25
REAP_TIMEOUT_SECONDS = 1.0
OUTER_PROCESS_DEADLINE_ENV = "DISCOVERY_OUTER_PROCESS_DEADLINE_MONOTONIC"


class ProcessGroupClosureError(RuntimeError):
    """Raised when a signaled isolated process group does not disappear."""


def terminate_process(
    process: subprocess.Popen[Any],
    *,
    termination_grace_seconds: float = TERMINATION_GRACE_SECONDS,
    reap_timeout_seconds: float = REAP_TIMEOUT_SECONDS,
) -> None:
    """Terminate, then kill and reap, one direct contained subprocess."""

    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=termination_grace_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=reap_timeout_seconds)


def capture_isolated_process_group(process: subprocess.Popen[Any]) -> int:
    """Capture the PGID established by ``start_new_session=True`` at launch."""

    process_group_id = process.pid
    try:
        observed_process_group_id = os.getpgid(process.pid)
    except ProcessLookupError:
        # A very short-lived leader may exit before this check. Popen's
        # start_new_session contract still fixes its PGID to its PID, and a
        # surviving descendant remains addressable by that captured value.
        return process_group_id
    if observed_process_group_id != process_group_id:
        process.kill()
        process.wait(timeout=REAP_TIMEOUT_SECONDS)
        raise RuntimeError(
            "subprocess was not the leader of its isolated process group"
        )
    return process_group_id


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a process group still has members without signaling it."""

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
    """Reap the direct child and wait a bounded interval for its group to empty."""

    deadline = time.monotonic() + timeout_seconds
    while True:
        # poll() reaps the direct child when it has exited. Grandchildren are
        # reaped by the container's init process after the group-wide signal.
        process.poll()
        if not _process_group_exists(process_group_id):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.01, remaining))


def terminate_process_group(
    process: subprocess.Popen[Any],
    *,
    process_group_id: int,
    termination_grace_seconds: float = TERMINATION_GRACE_SECONDS,
    reap_timeout_seconds: float = REAP_TIMEOUT_SECONDS,
) -> None:
    """Terminate, then kill and reap, one isolated subprocess process group.

    ``process_group_id`` must come from :func:`capture_isolated_process_group`
    immediately after launching with ``start_new_session=True``. Retaining that
    value lets cleanup reach descendants even after the leader has exited.
    """

    if termination_grace_seconds < 0 or reap_timeout_seconds <= 0:
        raise ValueError(
            "process-group cleanup deadlines must be bounded and positive"
        )
    if process_group_id != process.pid:
        raise ValueError("captured process group does not belong to subprocess")

    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        process.poll()
        return

    group_closed = _wait_for_process_group_exit(
        process,
        process_group_id=process_group_id,
        timeout_seconds=termination_grace_seconds,
    )
    if not group_closed:
        with suppress(ProcessLookupError):
            os.killpg(process_group_id, signal.SIGKILL)
        group_closed = _wait_for_process_group_exit(
            process,
            process_group_id=process_group_id,
            timeout_seconds=reap_timeout_seconds,
        )

    # A group can briefly contain an orphaned zombie after the direct child is
    # gone. Always finish reaping the direct child handle before returning.
    try:
        process.wait(timeout=reap_timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=reap_timeout_seconds)
    if not group_closed:
        raise ProcessGroupClosureError(
            "isolated subprocess group remained after SIGKILL deadline"
        )
