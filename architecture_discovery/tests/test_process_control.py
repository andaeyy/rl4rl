from __future__ import annotations

import signal
from unittest.mock import Mock, call

import pytest
from common import process_control


def test_process_group_cleanup_fails_closed_after_sigkill_deadline(
    monkeypatch,
) -> None:
    process = Mock(pid=41_001)
    process.wait.return_value = -signal.SIGKILL
    kill_group = Mock()
    monkeypatch.setattr(process_control.os, "killpg", kill_group)
    monkeypatch.setattr(
        process_control,
        "_wait_for_process_group_exit",
        Mock(side_effect=(False, False)),
    )

    with pytest.raises(
        process_control.ProcessGroupClosureError,
        match="remained after SIGKILL",
    ):
        process_control.terminate_process_group(
            process,
            process_group_id=process.pid,
        )

    assert kill_group.call_args_list == [
        call(process.pid, signal.SIGTERM),
        call(process.pid, signal.SIGKILL),
    ]
    process.wait.assert_called_once_with(
        timeout=process_control.REAP_TIMEOUT_SECONDS
    )
