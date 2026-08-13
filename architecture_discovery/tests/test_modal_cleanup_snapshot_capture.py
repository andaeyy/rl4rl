from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import common.modal_action_lock as modal_lock
import pytest
from modal_boundary import APP_NAME, VOLUME_NAME, ModalLiveCohortIdentity
from scripts import capture_modal_cleanup_snapshots as capture

_ORIGINAL_MODAL_INSTALLATION_BINDER = capture._bind_modal_installation
_CONFIG_TOKEN_ID = "ak-capture-secret-id-sentinel"
_CONFIG_TOKEN_SECRET = "as-capture-secret-value-sentinel"


@pytest.fixture
def identity() -> ModalLiveCohortIdentity:
    return ModalLiveCohortIdentity(
        source_tree_sha256="1" * 64,
        image_source_sha256="2" * 64,
        cohort_id="modal-cleanup-cohort",
    )


def _rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "app_list": [
            {
                "app_id": "ap-test123",
                "description": APP_NAME,
                "state": "stopped",
                "tasks": "0",
                "created_at": "2026-08-10 00:00:00+00:00",
                "stopped_at": "2026-08-10 00:30:00+00:00",
            }
        ],
        "container_list": [],
        "endpoint_list": [],
        "volume_list": [
            {
                "name": VOLUME_NAME,
                "created_at": "2026-08-10 00:00:00+00:00",
                "created_by": None,
            }
        ],
        "run_directory_list": [
            {
                "filename": "/runs/modal-cleanup-cohort",
                "type": "dir",
                "created_modified": "2026-08-10 00:30 UTC",
                "size": "0 B",
            }
        ],
        "billing_report": [
            {
                "object_id": "ap-test123",
                "description": APP_NAME,
                "environment": "main",
                "interval_start": "2026-08-10T00:00:00+00:00",
                "resource": "T4 GPU",
                "cost": "0E-8",
            }
        ],
    }


class FakeBoundedRunner:
    def __init__(
        self,
        rows: dict[str, list[dict[str, Any]]] | None = None,
        *,
        fail_at: int | None = None,
    ) -> None:
        self.rows = _rows() if rows is None else rows
        self.fail_at = fail_at
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        environment: dict[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> subprocess.CompletedProcess[bytes]:
        call_index = len(self.calls)
        frozen_environment = dict(environment)
        config_path = frozen_environment.get("MODAL_CONFIG_PATH")
        config_identity = (
            os.fstat(int(Path(config_path).name)).st_ino
            if isinstance(config_path, str) and config_path.startswith("/dev/fd/")
            else None
        )
        self.calls.append(
            {
                "command": tuple(command),
                "cwd": cwd,
                "environment": frozen_environment,
                "modal_config_inode": config_identity,
                "timeout_seconds": timeout_seconds,
                "max_output_bytes": max_output_bytes,
            }
        )
        if self.fail_at == call_index:
            raise subprocess.TimeoutExpired(list(command), timeout_seconds)
        name = capture.SNAPSHOT_NAMES[call_index]
        output = json.dumps(self.rows[name], separators=(",", ":")).encode()
        return subprocess.CompletedProcess(list(command), 0, output, b"")


@pytest.fixture(autouse=True)
def _pin_fake_modal_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "pinned-modal-1.5.3"
    executable.write_bytes(b"#!/bin/sh\nexit 97\n")
    executable.chmod(0o700)

    def bind() -> capture._ModalExecutableBinding:
        return capture._open_modal_executable_binding(executable)

    monkeypatch.setattr(capture, "_bind_modal_installation", bind)
    canonical_home = tmp_path / "passwd-home"
    canonical_home.mkdir()
    config = canonical_home / ".modal.toml"
    config.write_text(
        "[scalingintelligence]\n"
        f'token_id = "{_CONFIG_TOKEN_ID}"\n'
        f'token_secret = "{_CONFIG_TOKEN_SECRET}"\n',
        encoding="utf-8",
    )
    config.chmod(0o600)
    monkeypatch.setattr(
        capture.pwd,
        "getpwuid",
        lambda _uid: SimpleNamespace(pw_dir=str(canonical_home)),
    )


def _capture(
    root: Path,
    identity: ModalLiveCohortIdentity,
    runner: Any,
    **overrides: Any,
) -> capture.ModalCleanupSnapshotCaptureResult:
    arguments = {
        "project_root": root,
        "identity": identity,
        "capture_id": "capture-001",
        "billing_window_start_utc": "2026-08-10T00:00:00Z",
        "billing_window_end_utc": "2026-08-10T01:00:00Z",
        "now_factory": lambda: datetime(2026, 8, 10, 2, tzinfo=UTC),
        "monotonic_factory": lambda: 0.0,
        "environment": {
            "HOME": "/safe/home",
            "PATH": "/safe/bin",
            "TZ": "Asia/Shanghai",
            "MODAL_PROFILE": "wrong-profile",
            "MODAL_TOKEN_ID": "should-not-cross",
            "MODAL_ENVIRONMENT": "wrong-environment",
            "OPENAI_API_KEY": "provider-key",
            "DISCOVERY_API_KEY": "provider-key",
            "UNRELATED_SECRET": "not-needed",
        },
        "_bounded_runner": runner,
    }
    arguments.update(overrides)
    return capture.capture_modal_cleanup_snapshots(**arguments)


def _capture_root(root: Path, identity: ModalLiveCohortIdentity) -> Path:
    return root / capture.modal_cleanup_snapshot_capture_directory(
        identity, "capture-001"
    )


def _modal_config_path(root: Path) -> Path:
    return root / "passwd-home/.modal.toml"


def test_capture_uses_exact_read_only_command_order_and_frozen_profile(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    runner = FakeBoundedRunner()
    result = _capture(tmp_path, identity, runner)

    expected = capture.build_modal_cleanup_snapshot_commands(
        modal_executable=Path(runner.calls[0]["command"][0]),
        billing_window_start_utc="2026-08-10T00:00:00Z",
        billing_window_end_utc="2026-08-10T01:00:00Z",
    )
    assert tuple(call["command"] for call in runner.calls) == expected
    assert len(runner.calls) == 6
    for call in runner.calls:
        assert call["timeout_seconds"] == capture.COMMAND_TIMEOUT_SECONDS
        assert call["max_output_bytes"] <= capture.MAX_COMMAND_OUTPUT_BYTES
        environment = call["environment"]
        assert environment["MODAL_PROFILE"] == "scalingintelligence"
        assert environment["MODAL_ENVIRONMENT"] == "main"
        assert environment["TZ"] == "UTC"
        config_path = environment["MODAL_CONFIG_PATH"]
        assert config_path.startswith("/dev/fd/")
        assert call["modal_config_inode"] == (
            tmp_path / "passwd-home/.modal.toml"
        ).stat().st_ino
        assert environment["HOME"] == "/safe/home"
        assert "MODAL_TOKEN_ID" not in environment
        assert "OPENAI_API_KEY" not in environment
        assert "DISCOVERY_API_KEY" not in environment
        assert "UNRELATED_SECRET" not in environment
    flattened = {part for command in expected for part in command[1:]}
    assert not flattened.intersection(
        {"activate", "deploy", "run", "serve", "shell", "stop"}
    )

    manifest = result.manifest
    assert manifest["schema_name"] == "ModalCleanupSnapshotCaptureManifest"
    assert manifest["schema_version"] == "1.0"
    assert manifest["source_tree_sha256"] == identity.source_tree_sha256
    assert manifest["image_source_sha256"] == identity.image_source_sha256
    assert manifest["cohort_id"] == identity.cohort_id
    assert manifest["capture_id"] == "capture-001"
    assert manifest["command_retry_count"] == 0
    assert tuple(manifest["snapshots"]) == tuple(sorted(capture.SNAPSHOT_NAMES))
    # Durable JSON uses deterministic sorted keys on disk; each record still
    # binds the frozen command roster by its semantic key.
    for name, command in zip(capture.SNAPSHOT_NAMES, expected, strict=True):
        record = manifest["snapshots"][name]
        path = tmp_path / record["path"]
        raw = path.read_bytes()
        assert record["argv"] == list(command)
        assert record["sha256"] == hashlib.sha256(raw).hexdigest()
        assert record["size_bytes"] == len(raw)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    billing = json.loads(
        (tmp_path / manifest["snapshots"]["billing_report"]["path"]).read_text()
    )
    assert billing[0]["cost"] == "0E-8"

    manifest_path = tmp_path / result.manifest_path
    assert result.manifest_path.endswith(
        "/resource_cleanup/snapshot_captures/capture-001/capture_manifest.v1.0.json"
    )
    assert (
        result.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600

    mutable_copy = result.manifest
    mutable_copy["cohort_id"] = "mutated"
    assert result.manifest["cohort_id"] == identity.cohort_id


def test_default_runner_passes_only_held_modal_and_config_descriptors(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations: list[dict[str, Any]] = []
    rows = _rows()

    def bounded(command: Sequence[str], **kwargs: Any) -> Any:
        environment = kwargs["environment"]
        config_descriptor = int(Path(environment["MODAL_CONFIG_PATH"]).name)
        python_path = Path(command[0])
        executable_descriptor = int(Path(command[1]).name)
        assert kwargs["pass_fds"] == (
            executable_descriptor,
            config_descriptor,
        )
        assert python_path.name == capture.PRIVATE_PYTHON_EXECUTION_FILENAME
        assert python_path.is_file()
        assert stat.S_IMODE(python_path.stat().st_mode) == 0o500
        os.fstat(executable_descriptor)
        os.fstat(config_descriptor)
        name = capture.SNAPSHOT_NAMES[len(invocations)]
        invocations.append(
            {
                "command": tuple(command),
                "python_path": python_path,
                "config_descriptor": config_descriptor,
                "executable_descriptor": executable_descriptor,
            }
        )
        return subprocess.CompletedProcess(
            list(command),
            0,
            json.dumps(rows[name], separators=(",", ":")).encode(),
            b"",
        )

    monkeypatch.setattr(capture, "_run_bounded_cli", bounded)
    _capture(tmp_path, identity, None)
    assert len(invocations) == len(capture.SNAPSHOT_NAMES)
    assert all(
        item["config_descriptor"] != item["executable_descriptor"]
        for item in invocations
    )
    assert all(not item["python_path"].exists() for item in invocations)
    with pytest.raises(OSError):
        os.fstat(invocations[0]["config_descriptor"])
    with pytest.raises(OSError):
        os.fstat(invocations[0]["executable_descriptor"])


def test_default_runner_executes_descriptor_bound_console_script_via_python_copy(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "modal-python-console-script"
    rows_json = json.dumps(_rows(), separators=(",", ":"))
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"rows = json.loads({rows_json!r})\n"
        "with open(os.environ['MODAL_CONFIG_PATH'], encoding='utf-8') as f:\n"
        "    if '[scalingintelligence]' not in f.read(): raise SystemExit(65)\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['app', 'list']: name = 'app_list'\n"
        "elif args[:2] == ['container', 'list']: name = 'container_list'\n"
        "elif args[:2] == ['endpoint', 'list']: name = 'endpoint_list'\n"
        "elif args[:2] == ['volume', 'list']: name = 'volume_list'\n"
        "elif args[:2] == ['volume', 'ls']: name = 'run_directory_list'\n"
        "elif args[:2] == ['billing', 'report']: name = 'billing_report'\n"
        "else: raise SystemExit(64)\n"
        "print(json.dumps(rows[name], separators=(',', ':')))\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    monkeypatch.setattr(
        capture,
        "_bind_modal_installation",
        lambda: capture._open_modal_executable_binding(executable),
    )

    result = _capture(tmp_path, identity, None)

    assert result.manifest["capture_id"] == "capture-001"
    assert not (
        _capture_root(tmp_path, identity)
        / capture.PRIVATE_PYTHON_EXECUTION_FILENAME
    ).exists()


def test_config_and_token_bytes_never_enter_manifest_or_output(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _capture(tmp_path, identity, FakeBoundedRunner())
    captured = capsys.readouterr()
    manifest_path = tmp_path / result.manifest_path
    persisted = [manifest_path.read_bytes()]
    persisted.extend(
        (tmp_path / record["path"]).read_bytes()
        for record in result.manifest["snapshots"].values()
    )
    forbidden = (
        _CONFIG_TOKEN_ID.encode(),
        _CONFIG_TOKEN_SECRET.encode(),
        b"token_id",
        b"token_secret",
        b"MODAL_CONFIG_PATH",
    )
    assert all(secret not in raw for secret in forbidden for raw in persisted)
    output = (captured.out + captured.err).encode()
    assert all(secret not in output for secret in forbidden)


def test_command_roster_has_exact_environment_positions() -> None:
    commands = capture.build_modal_cleanup_snapshot_commands(
        modal_executable=Path("/opt/pinned-modal-1.5.3"),
        billing_window_start_utc="2026-08-10T00:00:00Z",
        billing_window_end_utc="2026-08-10T01:00:00Z",
    )
    assert commands[:5] == (
        ("/opt/pinned-modal-1.5.3", "app", "list", "--env", "main", "--json"),
        (
            "/opt/pinned-modal-1.5.3",
            "container",
            "list",
            "--env",
            "main",
            "--json",
        ),
        (
            "/opt/pinned-modal-1.5.3",
            "endpoint",
            "list",
            "--env",
            "main",
            "--json",
        ),
        (
            "/opt/pinned-modal-1.5.3",
            "volume",
            "list",
            "--env",
            "main",
            "--json",
        ),
        (
            "/opt/pinned-modal-1.5.3",
            "volume",
            "ls",
            "--env",
            "main",
            "--json",
            VOLUME_NAME,
            "/runs",
        ),
    )
    assert commands[5] == (
        "/opt/pinned-modal-1.5.3",
        "billing",
        "report",
        "--start",
        "2026-08-10T00:00:00Z",
        "--end",
        "2026-08-10T01:00:00Z",
        "--resolution",
        "h",
        "--tz",
        "UTC",
        "--show-resources",
        "--json",
    )


@pytest.mark.parametrize(
    "timestamp",
    ("2026-08-10 08:30 CST", "2026-08-10 08:30 +08:00"),
)
def test_run_directory_snapshot_rejects_non_utc_localized_timestamp(
    timestamp: str,
) -> None:
    rows = _rows()["run_directory_list"]
    rows[0]["created_modified"] = timestamp

    with pytest.raises(ValueError, match="timestamp is not Modal CLI output"):
        capture._validate_snapshot_rows(
            json.dumps(rows).encode(),
            snapshot_name="run_directory_list",
            billing_start=datetime(2026, 8, 10, 0, tzinfo=UTC),
            billing_end=datetime(2026, 8, 10, 1, tzinfo=UTC),
        )


def test_installation_validator_pins_modal_1_5_3_beside_python() -> None:
    binding = _ORIGINAL_MODAL_INSTALLATION_BINDER()
    try:
        assert binding.canonical_path == Path(sys.executable).with_name("modal")
        assert binding.execution_path == Path(f"/dev/fd/{binding.descriptor}")
        capture._require_modal_executable_binding(binding)
    finally:
        binding.close()


def test_failed_command_preserves_partial_leaves_without_completion_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    runner = FakeBoundedRunner(fail_at=2)
    with pytest.raises(subprocess.TimeoutExpired):
        _capture(tmp_path, identity, runner)
    root = _capture_root(tmp_path, identity)
    assert (root / "app_list.json").is_file()
    assert (root / "container_list.json").is_file()
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()
    assert len(runner.calls) == 3


def test_preexisting_capture_id_blocks_before_any_command(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    root = _capture_root(tmp_path, identity)
    root.mkdir(parents=True)
    existing = root / capture.CAPTURE_MANIFEST_FILENAME
    existing.write_text("do not overwrite", encoding="utf-8")
    runner = FakeBoundedRunner()
    with pytest.raises(FileExistsError, match="already exists"):
        _capture(tmp_path, identity, runner)
    assert runner.calls == []
    assert existing.read_text(encoding="utf-8") == "do not overwrite"


def test_held_modal_action_lock_blocks_capture_before_any_command(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
    runner = FakeBoundedRunner()
    try:
        with pytest.raises(
            modal_lock.ModalActionLockContentionError,
            match="holds the lock",
        ):
            _capture(tmp_path, identity, runner)
    finally:
        modal_lock.release_modal_action_lock(descriptor)
    assert runner.calls == []


def test_capture_lock_blocks_interleaved_launcher_before_any_launcher_work(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    launcher_work: list[str] = []

    class InterleavingRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            if not self.calls:
                with pytest.raises(
                    modal_lock.ModalActionLockContentionError,
                    match="holds the lock",
                ):
                    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
                    launcher_work.append("acquired")
                    modal_lock.release_modal_action_lock(descriptor)
            return super().__call__(command, **kwargs)

    runner = InterleavingRunner()
    _capture(tmp_path, identity, runner)
    assert len(runner.calls) == len(capture.SNAPSHOT_NAMES)
    assert launcher_work == []

    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
    modal_lock.release_modal_action_lock(descriptor)


def test_lock_identity_swap_after_first_read_aborts_remaining_capture(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    readiness = tmp_path / modal_lock.MODAL_ACTION_LOCK_PATH.parent
    displaced = tmp_path / "outputs/readiness-displaced-during-capture"

    class LockSwappingRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            completed = super().__call__(command, **kwargs)
            if len(self.calls) == 1:
                readiness.rename(displaced)
                readiness.mkdir(mode=0o700)
                replacement = readiness / modal_lock.MODAL_ACTION_LOCK_PATH.name
                replacement.touch(mode=0o600)
            return completed

    runner = LockSwappingRunner()
    with pytest.raises(ValueError, match="lock ancestor was replaced"):
        _capture(tmp_path, identity, runner)
    assert len(runner.calls) == 1
    assert not (_capture_root(tmp_path, identity) / "app_list.json").exists()
    displaced_capture = displaced / _capture_root(tmp_path, identity).relative_to(
        tmp_path / "outputs/readiness"
    )
    assert not (displaced_capture / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_capture_holds_lock_through_terminal_failure_cleanup(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_remove = capture._remove_path_if_same
    cleanup_observed = False

    def fail_terminal_verification(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected terminal verification failure")

    def locked_remove(*args: Any, **kwargs: Any) -> None:
        nonlocal cleanup_observed
        with pytest.raises(
            modal_lock.ModalActionLockContentionError,
            match="holds the lock",
        ):
            modal_lock.acquire_modal_action_lock(tmp_path)
        cleanup_observed = True
        original_remove(*args, **kwargs)

    monkeypatch.setattr(
        capture,
        "_verify_terminal_capture_files",
        fail_terminal_verification,
    )
    monkeypatch.setattr(capture, "_remove_path_if_same", locked_remove)

    runner = FakeBoundedRunner()
    with pytest.raises(RuntimeError, match="terminal verification failure"):
        _capture(tmp_path, identity, runner)
    assert len(runner.calls) == len(capture.SNAPSHOT_NAMES)
    assert cleanup_observed is True
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()

    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
    modal_lock.release_modal_action_lock(descriptor)


@pytest.mark.parametrize("swap_after", range(len(capture.SNAPSHOT_NAMES)))
def test_same_byte_config_path_swap_after_every_command_is_detected(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    swap_after: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _modal_config_path(tmp_path)
    original_config = config.read_bytes()
    original_publish = capture._publish_json_in_reserved_capture
    swap_after_leaf = f"{capture.SNAPSHOT_NAMES[swap_after]}.json"

    def swapping_publish(*args: Any, **kwargs: Any) -> Any:
        published = original_publish(*args, **kwargs)
        if kwargs.get("name") == swap_after_leaf:
            replacement = config.with_name(".modal.toml.replacement")
            replacement.write_bytes(original_config)
            replacement.chmod(0o600)
            os.replace(replacement, config)
        return published

    monkeypatch.setattr(
        capture,
        "_publish_json_in_reserved_capture",
        swapping_publish,
    )
    runner = FakeBoundedRunner()
    with pytest.raises(
        ValueError,
        match="configuration (?:descriptor|path) changed",
    ):
        _capture(tmp_path, identity, runner)
    assert len(runner.calls) == swap_after + 1
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


@pytest.mark.parametrize("drift", ["mode", "link"])
def test_config_mode_or_link_drift_after_read_is_detected(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    drift: str,
) -> None:
    config = _modal_config_path(tmp_path)

    class ConfigMetadataDriftRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            completed = super().__call__(command, **kwargs)
            if len(self.calls) == 1:
                if drift == "mode":
                    config.chmod(0o640)
                else:
                    os.link(config, config.with_name(".modal.toml.extra-link"))
            return completed

    runner = ConfigMetadataDriftRunner()
    with pytest.raises(
        ValueError,
        match="configuration (?:descriptor|path) changed",
    ):
        _capture(tmp_path, identity, runner)
    assert len(runner.calls) == 1
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


def test_config_owner_drift_after_read_is_detected(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fstat = os.fstat

    class ConfigOwnerDriftRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            completed = super().__call__(command, **kwargs)
            if len(self.calls) == 1:
                config_descriptor = int(
                    Path(kwargs["environment"]["MODAL_CONFIG_PATH"]).name
                )

                def drifted_fstat(descriptor: int) -> Any:
                    metadata = real_fstat(descriptor)
                    if descriptor != config_descriptor:
                        return metadata
                    return SimpleNamespace(
                        st_dev=metadata.st_dev,
                        st_ino=metadata.st_ino,
                        st_size=metadata.st_size,
                        st_mode=metadata.st_mode,
                        st_uid=metadata.st_uid + 1,
                        st_nlink=metadata.st_nlink,
                        st_mtime_ns=metadata.st_mtime_ns,
                        st_ctime_ns=metadata.st_ctime_ns,
                    )

                monkeypatch.setattr(capture.os, "fstat", drifted_fstat)
            return completed

    runner = ConfigOwnerDriftRunner()
    with pytest.raises(ValueError, match="configuration descriptor changed"):
        _capture(tmp_path, identity, runner)
    assert len(runner.calls) == 1
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


@pytest.mark.parametrize("unsafe", ["mode", "hardlink", "symlink"])
def test_unsafe_initial_config_metadata_blocks_before_any_command(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    unsafe: str,
) -> None:
    config = _modal_config_path(tmp_path)
    if unsafe == "mode":
        config.chmod(0o640)
    elif unsafe == "hardlink":
        os.link(config, config.with_name(".modal.toml.extra-link"))
    else:
        target = config.with_name("real-modal.toml")
        config.rename(target)
        config.symlink_to(target)
    runner = FakeBoundedRunner()
    with pytest.raises(ValueError, match="configuration metadata is unsafe"):
        _capture(tmp_path, identity, runner)
    assert runner.calls == []


@pytest.mark.parametrize(
    "config_text",
    [
        "[default]\ntoken_id='x'\ntoken_secret='y'\n",
        "[scalingintelligence]\ntoken_id='x'\n",
    ],
)
def test_missing_profile_or_token_key_metadata_blocks_before_any_command(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    config_text: str,
) -> None:
    config = _modal_config_path(tmp_path)
    config.write_text(config_text, encoding="utf-8")
    config.chmod(0o600)
    runner = FakeBoundedRunner()
    with pytest.raises(ValueError, match="required profile metadata"):
        _capture(tmp_path, identity, runner)
    assert runner.calls == []


def test_concurrent_capture_directory_creator_wins_before_any_command(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _capture_root(tmp_path, identity)
    original = capture._open_directory_chain
    injected = False

    def racing_parent_open(path: Path, *, create: bool) -> int:
        nonlocal injected
        descriptor = original(path, create=create)
        if create and not injected and Path(os.path.abspath(path)) == root.parent:
            os.mkdir(root.name, 0o700, dir_fd=descriptor)
            injected = True
        return descriptor

    monkeypatch.setattr(capture, "_open_directory_chain", racing_parent_open)
    runner = FakeBoundedRunner()
    with pytest.raises(FileExistsError, match="already exists"):
        _capture(tmp_path, identity, runner)
    assert runner.calls == []
    assert root.is_dir()
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_preexisting_symlink_or_hardlink_leaf_is_never_overwritten(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    kind: str,
) -> None:
    root = _capture_root(tmp_path, identity)
    root.mkdir(parents=True)
    target = tmp_path / "outside.json"
    target.write_text("outside", encoding="utf-8")
    leaf = root / "app_list.json"
    if kind == "symlink":
        leaf.symlink_to(target)
    else:
        os.link(target, leaf)
    runner = FakeBoundedRunner()
    with pytest.raises(FileExistsError):
        _capture(tmp_path, identity, runner)
    assert runner.calls == []
    assert target.read_text(encoding="utf-8") == "outside"


def test_symlink_ancestor_blocks_publication_and_no_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    cleanup_root = (
        tmp_path
        / capture.modal_cleanup_snapshot_capture_directory(
            identity, "capture-001"
        ).parent
    )
    cleanup_root.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    cleanup_root.symlink_to(outside, target_is_directory=True)
    runner = FakeBoundedRunner()
    with pytest.raises((NotADirectoryError, ValueError)):
        _capture(tmp_path, identity, runner)
    assert runner.calls == []
    assert not (outside / "capture-001" / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_ancestor_swap_after_first_leaf_blocks_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = capture._publish_json_in_reserved_capture
    root = _capture_root(tmp_path, identity)
    calls = 0

    def swapping_create(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        published = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            moved = root.with_name("capture-001-moved")
            root.rename(moved)
            root.mkdir()
        return published

    monkeypatch.setattr(capture, "_publish_json_in_reserved_capture", swapping_create)
    with pytest.raises((FileNotFoundError, ValueError)):
        _capture(tmp_path, identity, FakeBoundedRunner())
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_leaf_swap_before_manifest_is_detected(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = capture._publish_terminal_manifest
    root = _capture_root(tmp_path, identity)

    def swapping_create(*args: Any, **kwargs: Any) -> Any:
        first = root / "app_list.json"
        first.unlink()
        first.write_text("[]\n", encoding="utf-8")
        first.chmod(0o600)
        return original(*args, **kwargs)

    monkeypatch.setattr(capture, "_publish_terminal_manifest", swapping_create)
    with pytest.raises(ValueError, match="metadata is unsafe|snapshot path changed"):
        _capture(tmp_path, identity, FakeBoundedRunner())
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_ancestor_swap_in_final_precheck_to_manifest_gap_is_detected(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = capture._publish_terminal_manifest
    root = _capture_root(tmp_path, identity)
    moved = root.with_name("capture-001-moved-late")

    def swapping_manifest(*args: Any, **kwargs: Any) -> Any:
        terminal = original(*args, **kwargs)
        root.rename(moved)
        root.mkdir(mode=0o700)
        return terminal

    monkeypatch.setattr(capture, "_publish_terminal_manifest", swapping_manifest)
    with pytest.raises(ValueError, match="canonical cleanup capture directory changed"):
        _capture(tmp_path, identity, FakeBoundedRunner())
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()
    assert not (moved / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_late_hardlink_blocks_completion_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = capture._publish_terminal_manifest
    root = _capture_root(tmp_path, identity)

    def linking_create(*args: Any, **kwargs: Any) -> Any:
        os.link(root / "app_list.json", root / "extra-hardlink.json")
        return original(*args, **kwargs)

    monkeypatch.setattr(capture, "_publish_terminal_manifest", linking_create)
    with pytest.raises(ValueError, match="metadata is unsafe"):
        _capture(tmp_path, identity, FakeBoundedRunner())
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


@pytest.mark.parametrize("platform, succeeds", [("darwin", True), ("linux", False)])
def test_published_leaf_ctime_drift_is_allowed_only_for_macos_provenance(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    succeeds: bool,
) -> None:
    original = capture._publish_json_in_reserved_capture

    def drifted_ctime(*args: Any, **kwargs: Any) -> Any:
        published = original(*args, **kwargs)
        published.ctime_ns -= 1
        return published

    monkeypatch.setattr(capture.sys, "platform", platform)
    monkeypatch.setattr(
        capture,
        "_publish_json_in_reserved_capture",
        drifted_ctime,
    )
    if succeeds:
        result = _capture(tmp_path, identity, FakeBoundedRunner())
        assert result.manifest["schema_name"] == (
            capture.CAPTURE_MANIFEST_SCHEMA_NAME
        )
    else:
        with pytest.raises(ValueError, match="metadata is unsafe"):
            _capture(tmp_path, identity, FakeBoundedRunner())


def test_final_clock_callback_leaf_swap_blocks_completion_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = capture._publish_terminal_manifest
    root = _capture_root(tmp_path, identity)
    terminal_published = False
    swapped = False

    def publishing_manifest(*args: Any, **kwargs: Any) -> Any:
        nonlocal terminal_published
        terminal = original_publish(*args, **kwargs)
        terminal_published = True
        return terminal

    def swapping_clock() -> float:
        nonlocal swapped
        if terminal_published and not swapped:
            leaf = root / "app_list.json"
            leaf.unlink()
            leaf.write_text("[]\n", encoding="utf-8")
            leaf.chmod(0o600)
            swapped = True
        return 0.0

    monkeypatch.setattr(
        capture,
        "_publish_terminal_manifest",
        publishing_manifest,
    )
    with pytest.raises(ValueError, match="metadata is unsafe|snapshot path changed"):
        _capture(
            tmp_path,
            identity,
            FakeBoundedRunner(),
            monotonic_factory=swapping_clock,
        )
    assert swapped is True
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_duplicate_json_key_and_wrong_schema_block_leaf_and_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    duplicate = (
        b'[{"app_id":"ap-a","app_id":"ap-b","description":"x",'
        b'"state":"stopped","tasks":"0","created_at":'
        b'"2026-08-10 00:00:00+00:00","stopped_at":null}]'
    )

    class DuplicateRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            super().__call__(command, **kwargs)
            return subprocess.CompletedProcess(list(command), 0, duplicate, b"")

    with pytest.raises(ValueError, match="duplicate key"):
        _capture(tmp_path, identity, DuplicateRunner())
    assert _capture_root(tmp_path, identity).is_dir()
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()

    rows = _rows()
    rows["app_list"][0]["unexpected"] = "field"
    second_root = tmp_path / "wrong-schema"
    second_root.mkdir()
    with pytest.raises(ValueError, match="differs from.*schema"):
        _capture(second_root, identity, FakeBoundedRunner(rows))
    assert _capture_root(second_root, identity).is_dir()
    assert not (
        _capture_root(second_root, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


def test_output_caps_are_enforced_even_if_injected_runner_misbehaves(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capture, "MAX_COMMAND_OUTPUT_BYTES", 32)

    class OversizeRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            self.calls.append({"command": tuple(command), **kwargs})
            return subprocess.CompletedProcess(list(command), 0, b"x" * 33, b"")

    with pytest.raises(capture.ModalCleanupSnapshotOutputLimitError):
        _capture(tmp_path, identity, OversizeRunner())
    assert _capture_root(tmp_path, identity).is_dir()
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


def test_restrictive_umask_still_publishes_exact_0600_files(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    descriptor = modal_lock.acquire_modal_action_lock(tmp_path)
    modal_lock.release_modal_action_lock(descriptor)
    previous_umask = os.umask(0o777)
    try:
        result = _capture(tmp_path, identity, FakeBoundedRunner())
    finally:
        os.umask(previous_umask)
    manifest = result.manifest
    root = _capture_root(tmp_path, identity)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    for record in manifest["snapshots"].values():
        assert stat.S_IMODE((tmp_path / record["path"]).stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / result.manifest_path).stat().st_mode) == 0o600


def test_outer_deadline_is_checked_even_if_runner_ignores_timeout(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    expired = False

    def monotonic() -> float:
        return capture.OUTER_TIMEOUT_SECONDS + 1 if expired else 0.0

    class ExpiringRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            nonlocal expired
            result = super().__call__(command, **kwargs)
            expired = True
            return result

    with pytest.raises(subprocess.TimeoutExpired):
        _capture(
            tmp_path,
            identity,
            ExpiringRunner(),
            monotonic_factory=monotonic,
        )
    assert _capture_root(tmp_path, identity).is_dir()
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


def test_outer_deadline_expiring_after_billing_leaf_blocks_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = capture._publish_json_in_reserved_capture
    expired = False

    def publishing(*args: Any, **kwargs: Any) -> Any:
        nonlocal expired
        result = original(*args, **kwargs)
        if kwargs.get("name") == "billing_report.json":
            expired = True
        return result

    monkeypatch.setattr(capture, "_publish_json_in_reserved_capture", publishing)

    def monotonic() -> float:
        return capture.OUTER_TIMEOUT_SECONDS + 1 if expired else 0.0

    with pytest.raises(subprocess.TimeoutExpired):
        _capture(
            tmp_path,
            identity,
            FakeBoundedRunner(),
            monotonic_factory=monotonic,
        )
    root = _capture_root(tmp_path, identity)
    assert (root / "billing_report.json").is_file()
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_deadline_advance_during_final_binding_blocks_completion_manifest(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_publish = capture._publish_terminal_manifest
    original_binding = capture._require_capture_directory_binding
    root = _capture_root(tmp_path, identity)
    terminal_published = False
    expired = False

    def publishing_manifest(*args: Any, **kwargs: Any) -> Any:
        nonlocal terminal_published
        terminal = original_publish(*args, **kwargs)
        terminal_published = True
        return terminal

    def expiring_binding(*args: Any, **kwargs: Any) -> Any:
        nonlocal expired
        result = original_binding(*args, **kwargs)
        if terminal_published and not expired:
            expired = True
        return result

    def monotonic() -> float:
        return capture.OUTER_TIMEOUT_SECONDS + 1 if expired else 0.0

    monkeypatch.setattr(
        capture,
        "_publish_terminal_manifest",
        publishing_manifest,
    )
    monkeypatch.setattr(
        capture,
        "_require_capture_directory_binding",
        expiring_binding,
    )
    with pytest.raises(subprocess.TimeoutExpired):
        _capture(
            tmp_path,
            identity,
            FakeBoundedRunner(),
            monotonic_factory=monotonic,
        )
    assert expired is True
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


def test_incomplete_billing_window_is_rejected_before_commands(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    runner = FakeBoundedRunner()
    with pytest.raises(ValueError, match="incomplete UTC hour"):
        _capture(
            tmp_path,
            identity,
            runner,
            billing_window_end_utc="2026-08-10T03:00:00Z",
        )
    assert runner.calls == []


@pytest.mark.parametrize("cost", ["-0", "+0", "NaN", "Infinity"])
def test_billing_rejects_signed_or_nonfinite_cost(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    cost: str,
) -> None:
    rows = _rows()
    rows["billing_report"][0]["cost"] = cost
    with pytest.raises(ValueError, match="decimal|finite|unsigned"):
        _capture(tmp_path, identity, FakeBoundedRunner(rows))
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


def test_migration_billing_row_outside_main_is_rejected(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    rows = _rows()
    rows["billing_report"][0]["environment"] = "dev"
    with pytest.raises(ValueError, match="outside environment main"):
        _capture(tmp_path, identity, FakeBoundedRunner(rows))
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


@pytest.mark.parametrize(
    "interval",
    ["2026-08-10T00:00:00", "2026-08-10T01:00:00+01:00"],
)
def test_billing_rejects_naive_or_non_utc_interval(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    interval: str,
) -> None:
    rows = _rows()
    rows["billing_report"][0]["interval_start"] = interval
    with pytest.raises(ValueError, match="UTC offset"):
        _capture(tmp_path, identity, FakeBoundedRunner(rows))
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


def test_billing_rejects_tags_without_tag_names_argv(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    rows = _rows()
    rows["billing_report"][0]["tags"] = '{"team":"rl4rl"}'
    with pytest.raises(ValueError, match="differs from.*schema"):
        _capture(tmp_path, identity, FakeBoundedRunner(rows))
    assert not (
        _capture_root(tmp_path, identity) / capture.CAPTURE_MANIFEST_FILENAME
    ).exists()


def test_atomic_console_script_replacement_cannot_execute(
    tmp_path: Path,
    identity: ModalLiveCohortIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "race-modal"
    original_bytes = b"#!/bin/sh\nexit 97\n"
    malicious_bytes = b"#!/bin/sh\ntouch should-never-run\n"
    executable.write_bytes(original_bytes)
    executable.chmod(0o700)

    def bind() -> capture._ModalExecutableBinding:
        return capture._open_modal_executable_binding(executable)

    monkeypatch.setattr(capture, "_bind_modal_installation", bind)

    class ReplacingRunner(FakeBoundedRunner):
        def __call__(self, command: Sequence[str], **kwargs: Any) -> Any:
            descriptor = int(Path(command[0]).name)
            os.lseek(descriptor, 0, os.SEEK_SET)
            assert os.read(descriptor, len(original_bytes)) == original_bytes
            replacement = executable.with_suffix(".new")
            replacement.write_bytes(malicious_bytes)
            replacement.chmod(0o700)
            os.replace(replacement, executable)
            return super().__call__(command, **kwargs)

    runner = ReplacingRunner()
    with pytest.raises(ValueError, match="executable.*changed"):
        _capture(tmp_path, identity, runner)
    assert len(runner.calls) == 1
    assert not (tmp_path / "should-never-run").exists()
    root = _capture_root(tmp_path, identity)
    assert root.is_dir()
    assert not (root / "app_list.json").exists()
    assert not (root / capture.CAPTURE_MANIFEST_FILENAME).exists()


class _PipeProcess:
    def __init__(self, stdout: bytes, stderr: bytes, returncode: int = 0) -> None:
        stdout_reader, stdout_writer = os.pipe()
        stderr_reader, stderr_writer = os.pipe()
        os.write(stdout_writer, stdout)
        os.write(stderr_writer, stderr)
        os.close(stdout_writer)
        os.close(stderr_writer)
        self.stdout = os.fdopen(stdout_reader, "rb", buffering=0)
        self.stderr = os.fdopen(stderr_reader, "rb", buffering=0)
        self.pid = 424242
        self._returncode = returncode

    def poll(self) -> int:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode

    def kill(self) -> None:
        self._returncode = -9


class _HangingPipeProcess:
    def __init__(self) -> None:
        stdout_reader, self.stdout_writer = os.pipe()
        stderr_reader, self.stderr_writer = os.pipe()
        self.stdout = os.fdopen(stdout_reader, "rb", buffering=0)
        self.stderr = os.fdopen(stderr_reader, "rb", buffering=0)
        self.pid = 424243
        self._returncode: int | None = None

    def poll(self) -> int | None:
        return self._returncode

    def wait(self, timeout: float | None = None) -> int:
        if self._returncode is None:
            raise subprocess.TimeoutExpired(["fake"], timeout)
        return self._returncode

    def kill(self) -> None:
        self.close_group()

    def close_group(self) -> None:
        for descriptor in (self.stdout_writer, self.stderr_writer):
            with suppress(OSError):
                os.close(descriptor)
        self._returncode = -15


def test_low_level_runner_closes_fake_process_group_and_caps_streams() -> None:
    invocations: list[dict[str, Any]] = []
    process = _PipeProcess(b"[]", b"")

    def popen(command: Sequence[str], **kwargs: Any) -> _PipeProcess:
        invocations.append({"command": list(command), **kwargs})
        return process

    terminated: list[int] = []
    completed = capture._run_bounded_cli(
        ["/opt/pinned-modal-1.5.3", "app", "list", "--json"],
        cwd=Path("/tmp"),
        environment={"MODAL_PROFILE": "scalingintelligence"},
        timeout_seconds=1,
        max_output_bytes=10,
        popen_factory=popen,
        process_group_capture=lambda child: child.pid,
        process_group_terminator=lambda child, *, process_group_id: terminated.append(
            process_group_id
        ),
    )
    assert completed.stdout == b"[]"
    assert terminated == [424242]
    assert invocations[0]["start_new_session"] is True
    assert invocations[0]["stdout"] is subprocess.PIPE
    assert invocations[0]["stderr"] is subprocess.PIPE

    oversized = _PipeProcess(b"123456", b"")
    with pytest.raises(capture.ModalCleanupSnapshotOutputLimitError):
        capture._run_bounded_cli(
            ["/opt/pinned-modal-1.5.3", "app", "list", "--json"],
            cwd=Path("/tmp"),
            environment={},
            timeout_seconds=1,
            max_output_bytes=5,
            popen_factory=lambda *args, **kwargs: oversized,
            process_group_capture=lambda child: child.pid,
            process_group_terminator=lambda child, *, process_group_id: None,
        )


def test_process_group_cleanup_failure_is_not_masked() -> None:
    process = _PipeProcess(b"[]", b"")

    def fail_cleanup(child: Any, *, process_group_id: int) -> None:
        raise RuntimeError("process group remained")

    with pytest.raises(RuntimeError, match="process group remained"):
        capture._run_bounded_cli(
            ["/opt/pinned-modal-1.5.3", "app", "list", "--json"],
            cwd=Path("/tmp"),
            environment={},
            timeout_seconds=1,
            max_output_bytes=10,
            popen_factory=lambda *args, **kwargs: process,
            process_group_capture=lambda child: child.pid,
            process_group_terminator=fail_cleanup,
        )


def test_low_level_timeout_always_closes_fake_process_group() -> None:
    process = _HangingPipeProcess()
    monotonic_values = iter((0.0, 2.0))
    terminated: list[int] = []

    def terminate(child: _HangingPipeProcess, *, process_group_id: int) -> None:
        terminated.append(process_group_id)
        child.close_group()

    with pytest.raises(subprocess.TimeoutExpired):
        capture._run_bounded_cli(
            ["/opt/pinned-modal-1.5.3", "app", "list", "--json"],
            cwd=Path("/tmp"),
            environment={},
            timeout_seconds=1,
            max_output_bytes=10,
            popen_factory=lambda *args, **kwargs: process,
            process_group_capture=lambda child: child.pid,
            process_group_terminator=terminate,
            monotonic_factory=lambda: next(monotonic_values),
        )
    assert terminated == [424243]
