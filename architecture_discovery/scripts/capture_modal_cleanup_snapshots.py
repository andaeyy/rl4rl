#!/usr/bin/env python3
"""Capture one immutable, provider-free Modal cleanup snapshot cohort.

The helper issues exactly six read-only Modal 1.5.3 CLI requests, once each and
in a frozen order.  It never activates a profile, starts or stops an App,
deploys code, invokes a Function, or contacts a model provider.  A capture is
complete only when its create-only manifest exists and binds all six durable
JSON leaves by their exact published byte digests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pwd
import re
import selectors
import stat
import subprocess
import sys
import sysconfig
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.modal_action_lock import (  # noqa: E402
    acquire_modal_action_lock,
    assert_modal_action_lock_identity,
    release_modal_action_lock,
)
from common.process_control import (  # noqa: E402
    capture_isolated_process_group,
    terminate_process_group,
)
from modal_boundary import (  # noqa: E402
    APP_NAME,
    MAX_MODAL_BILLING_WINDOW,
    MODAL_VERSION,
    VOLUME_NAME,
    ModalLiveCohortIdentity,
    modal_live_cohort_root,
    validate_run_id,
)
from study.serialization import json_value  # noqa: E402

MODAL_PROFILE = "scalingintelligence"
MODAL_ENVIRONMENT = "main"
CAPTURE_MANIFEST_SCHEMA_NAME = "ModalCleanupSnapshotCaptureManifest"
CAPTURE_MANIFEST_SCHEMA_VERSION = "1.0"
CAPTURE_MANIFEST_FILENAME = "capture_manifest.v1.0.json"
SNAPSHOT_NAMES = (
    "app_list",
    "container_list",
    "endpoint_list",
    "volume_list",
    "run_directory_list",
    "billing_report",
)

COMMAND_TIMEOUT_SECONDS = 45.0
OUTER_TIMEOUT_SECONDS = 300.0
MAX_COMMAND_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_CAPTURE_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_SNAPSHOT_ROWS = 100_000
MAX_MODAL_CONSOLE_SCRIPT_BYTES = 1024 * 1024
MAX_PYTHON_EXECUTABLE_BYTES = 64 * 1024 * 1024
MAX_MODAL_CONFIG_BYTES = 1024 * 1024
PRIVATE_PYTHON_EXECUTION_FILENAME = ".python-execution"
_REQUIRED_MODAL_CONFIG_KEYS = frozenset({"token_id", "token_secret"})

_RAW_SNAPSHOT_FIELDS = {
    "app_list": frozenset(
        {
            "app_id",
            "description",
            "state",
            "tasks",
            "created_at",
            "stopped_at",
        }
    ),
    "container_list": frozenset({"container_id", "app_id", "app_name", "start_time"}),
    "endpoint_list": frozenset(
        {"name", "endpoint_id", "status", "created_at", "created_by"}
    ),
    "volume_list": frozenset({"name", "created_at", "created_by"}),
    "run_directory_list": frozenset({"filename", "type", "created_modified", "size"}),
    "billing_report": frozenset(
        {
            "object_id",
            "description",
            "environment",
            "interval_start",
            "resource",
            "cost",
        }
    ),
}
_IDENTIFIER_FIELDS = {
    "app_list": "app_id",
    "container_list": "container_id",
    "endpoint_list": "endpoint_id",
    "volume_list": "name",
    "run_directory_list": "filename",
}
_TIMESTAMP_PATTERN = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2} UTC\Z"
)
_SIZE_PATTERN = re.compile(
    r"\A(?:0|[1-9][0-9]*) B|"
    r"(?:0|[1-9][0-9]*)\.[0-9] (?:KiB|MiB|GiB|TiB|PiB|EiB|ZiB)\Z"
)
_DECIMAL_PATTERN = re.compile(
    r"\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:E(?:\+?[0-9]+|-[0-9]+))?\Z"
)
_SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "CURL_CA_BUNDLE",
        "HOME",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LOGNAME",
        "NO_PROXY",
        "PATH",
        "REQUESTS_CA_BUNDLE",
        "SHELL",
        "SSL_CERT_DIR",
        "SSL_CERT_FILE",
        "TMPDIR",
        "USER",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


class ModalCleanupSnapshotCaptureError(RuntimeError):
    """Raised when a snapshot attempt cannot be frozen as complete."""


class ModalCleanupSnapshotOutputLimitError(ModalCleanupSnapshotCaptureError):
    """Raised before unbounded Modal CLI output can accumulate."""


@dataclass(frozen=True, slots=True)
class ModalCleanupSnapshotCaptureResult:
    """Immutable selector for one fully published cleanup snapshot capture."""

    manifest_path: str
    manifest_sha256: str
    _manifest_bytes: bytes

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a fresh strict copy so callers cannot mutate claimed evidence."""

        payload = _strict_json_loads(self._manifest_bytes, field="capture manifest")
        if not isinstance(payload, dict):  # pragma: no cover - construction guard
            raise TypeError("published capture manifest is not an object")
        return payload


@dataclass(slots=True)
class _ModalExecutableBinding:
    label: str
    canonical_path: Path
    descriptor: int
    device: int
    inode: int
    size_bytes: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    sha256: str
    maximum_bytes: int
    require_current_uid: bool
    require_stable_ctime: bool

    @property
    def execution_path(self) -> Path:
        return Path(f"/dev/fd/{self.descriptor}")

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _PrivatePythonExecutionCopy:
    """A byte-exact Python copy inside the held private capture directory."""

    binding: _ModalExecutableBinding
    reservation: _CaptureDirectoryReservation
    device: int
    inode: int
    removed: bool = False

    @property
    def canonical_path(self) -> Path:
        return self.binding.canonical_path

    def require_current(self) -> None:
        if self.removed:
            raise ValueError("private Python execution copy was removed")
        _require_capture_directory_binding(self.reservation)
        _require_modal_executable_binding(self.binding)
        current = os.stat(
            PRIVATE_PYTHON_EXECUTION_FILENAME,
            dir_fd=self.reservation.descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != (self.device, self.inode)
        ):
            raise ValueError("private Python execution copy path changed")

    def close_and_remove(self) -> None:
        if self.removed:
            return
        directory = os.fstat(self.reservation.descriptor)
        if (
            not stat.S_ISDIR(directory.st_mode)
            or directory.st_uid != os.getuid()
            or stat.S_IMODE(directory.st_mode) != 0o700
        ):
            raise ValueError("held cleanup capture directory changed")
        _require_modal_executable_descriptor_binding(self.binding)
        self.binding.close()
        current = os.stat(
            PRIVATE_PYTHON_EXECUTION_FILENAME,
            dir_fd=self.reservation.descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (current.st_dev, current.st_ino) != (self.device, self.inode)
        ):
            raise ValueError("private Python execution copy changed before removal")
        os.unlink(
            PRIVATE_PYTHON_EXECUTION_FILENAME,
            dir_fd=self.reservation.descriptor,
        )
        os.fsync(self.reservation.descriptor)
        self.removed = True


@dataclass(slots=True)
class _ModalConfigBinding:
    canonical_path: Path
    descriptor: int
    device: int
    inode: int
    size_bytes: int
    mode: int
    owner_uid: int
    mtime_ns: int
    ctime_ns: int

    @property
    def execution_path(self) -> str:
        if self.descriptor < 0:
            raise ValueError("canonical Modal configuration descriptor is closed")
        return f"/dev/fd/{self.descriptor}"

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


@dataclass(slots=True)
class _CaptureDirectoryReservation:
    canonical_path: Path
    parent_descriptor: int
    descriptor: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1
        if self.parent_descriptor >= 0:
            os.close(self.parent_descriptor)
            self.parent_descriptor = -1


@dataclass(slots=True)
class _PublishedJSONFile:
    name: str
    descriptor: int
    raw: bytes
    sha256: str
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1


def modal_cleanup_snapshot_capture_directory(
    identity: ModalLiveCohortIdentity,
    capture_id: str,
) -> PurePosixPath:
    """Return the canonical attempt-versioned cleanup capture directory."""

    if not isinstance(identity, ModalLiveCohortIdentity):
        raise TypeError("identity must be a ModalLiveCohortIdentity")
    validated_capture_id = validate_run_id(capture_id)
    return (
        modal_live_cohort_root(identity)
        / "resource_cleanup"
        / "snapshot_captures"
        / validated_capture_id
    )


def modal_cleanup_snapshot_capture_manifest_path(
    identity: ModalLiveCohortIdentity,
    capture_id: str,
) -> PurePosixPath:
    return (
        modal_cleanup_snapshot_capture_directory(identity, capture_id)
        / CAPTURE_MANIFEST_FILENAME
    )


def _utc_z(value: datetime, *, field: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    if value.utcoffset() != UTC.utcoffset(value):
        value = value.astimezone(UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_hour(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be an explicit UTC Z timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an explicit UTC Z timestamp") from error
    if (
        parsed.utcoffset() != UTC.utcoffset(parsed)
        or parsed.minute
        or parsed.second
        or parsed.microsecond
    ):
        raise ValueError(f"{field} must be aligned to a complete UTC hour")
    return parsed


def _validate_completed_billing_window(
    start_text: str,
    end_text: str,
    *,
    captured_at: datetime,
) -> tuple[datetime, datetime]:
    start = _parse_utc_hour(start_text, field="billing_window_start_utc")
    end = _parse_utc_hour(end_text, field="billing_window_end_utc")
    captured_utc = captured_at.astimezone(UTC)
    completed_through = captured_utc.replace(minute=0, second=0, microsecond=0)
    if start >= end:
        raise ValueError("billing window start must precede its exclusive end")
    if end > completed_through:
        raise ValueError("billing window includes an incomplete UTC hour")
    if end - start > MAX_MODAL_BILLING_WINDOW:
        raise ValueError("billing window exceeds the frozen 31-day limit")
    return start, end


def _read_descriptor_bytes(
    descriptor: int,
    *,
    maximum_bytes: int,
    label: str = "bound executable",
) -> bytes:
    metadata = os.fstat(descriptor)
    if metadata.st_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) <= maximum_bytes:
        try:
            chunk = os.read(
                descriptor, min(64 * 1024, maximum_bytes + 1 - len(payload))
            )
        except InterruptedError:
            continue
        if not chunk:
            break
        payload.extend(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    if len(payload) > maximum_bytes or len(payload) != metadata.st_size:
        raise ValueError(f"{label} changed while it was read")
    return bytes(payload)


def _open_nofollow_directory(path: Path) -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory = getattr(os, "O_DIRECTORY", None)
    if nofollow is None or directory is None:
        raise ValueError("platform cannot enforce no-follow Modal config reads")
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | nofollow | directory
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            before = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError(
                    "canonical Modal configuration path has an unsafe ancestor"
                )
            child = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(child)
                raise ValueError(
                    "canonical Modal configuration ancestor changed"
                )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _canonical_passwd_home() -> Path:
    record = pwd.getpwuid(os.getuid())
    raw_home = getattr(record, "pw_dir", None)
    if (
        not isinstance(raw_home, str)
        or not raw_home
        or not Path(raw_home).is_absolute()
    ):
        raise ValueError("current account has no canonical absolute home directory")
    try:
        canonical = Path(raw_home).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("current account home directory cannot be resolved") from error
    descriptor = _open_nofollow_directory(canonical)
    os.close(descriptor)
    return canonical


def _modal_config_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _validate_modal_config_metadata(binding: _ModalConfigBinding) -> None:
    """Validate only table/key metadata without retaining token values."""

    metadata = os.fstat(binding.descriptor)
    if metadata.st_size > MAX_MODAL_CONFIG_BYTES:
        raise ValueError("canonical Modal configuration exceeds its byte limit")
    current_profile: str | None = None
    target_profile_count = 0
    required_keys: set[str] = set()
    prefix = bytearray()
    assignment = False
    comment = False
    value_started = False
    opening_quote: int | None = None
    opening_quote_count = 0

    def finish_metadata_line() -> None:
        nonlocal current_profile, target_profile_count
        selected = bytes(prefix).strip()
        prefix.clear()
        if not selected:
            return
        match = re.fullmatch(rb"\[\s*([A-Za-z0-9_-]+)\s*\]", selected)
        if match is None:
            raise ValueError("canonical Modal configuration metadata is invalid")
        current_profile = match.group(1).decode("ascii")
        if current_profile == MODAL_PROFILE:
            target_profile_count += 1
            if target_profile_count != 1:
                raise ValueError(
                    "canonical Modal configuration profile metadata is duplicated"
                )

    def finish_assignment_prefix() -> None:
        selected = bytes(prefix).strip()
        prefix.clear()
        if re.fullmatch(rb"[A-Za-z0-9_-]+", selected) is None:
            raise ValueError("canonical Modal configuration metadata is invalid")
        key = selected.decode("ascii")
        if current_profile == MODAL_PROFILE and key in _REQUIRED_MODAL_CONFIG_KEYS:
            if key in required_keys:
                raise ValueError(
                    "canonical Modal configuration key metadata is duplicated"
                )
            required_keys.add(key)

    for offset in range(metadata.st_size):
        octet = os.pread(binding.descriptor, 1, offset)
        if len(octet) != 1:
            raise ValueError(
                "canonical Modal configuration changed during metadata validation"
            )
        value = octet[0]
        if assignment:
            if value == 0x0A:
                if not value_started:
                    raise ValueError(
                        "canonical Modal configuration metadata is invalid"
                    )
                assignment = False
                value_started = False
                opening_quote = None
                opening_quote_count = 0
                continue
            if not value_started and value in {0x09, 0x0D, 0x20}:
                continue
            if not value_started:
                value_started = True
                if value in {ord("["), ord("{")}:
                    raise ValueError(
                        "canonical Modal configuration uses unsupported "
                        "multiline metadata"
                    )
                if value in {ord("'"), ord('"')}:
                    opening_quote = value
                    opening_quote_count = 1
                continue
            if opening_quote is not None and opening_quote_count < 3:
                if value == opening_quote:
                    opening_quote_count += 1
                    if opening_quote_count == 3:
                        raise ValueError(
                            "canonical Modal configuration uses unsupported "
                            "multiline metadata"
                        )
                else:
                    opening_quote = None
            continue
        if comment:
            if value == 0x0A:
                finish_metadata_line()
                comment = False
            continue
        if value == ord("#"):
            comment = True
        elif value == ord("="):
            finish_assignment_prefix()
            assignment = True
        elif value == 0x0A:
            finish_metadata_line()
        else:
            if len(prefix) >= 1024:
                raise ValueError("canonical Modal configuration metadata is too long")
            prefix.append(value)
    if assignment:
        if not value_started:
            raise ValueError("canonical Modal configuration metadata is invalid")
    else:
        finish_metadata_line()
    if target_profile_count != 1 or required_keys != _REQUIRED_MODAL_CONFIG_KEYS:
        raise ValueError(
            "canonical Modal configuration lacks the required profile metadata"
        )


def _open_modal_config_binding() -> _ModalConfigBinding:
    absolute = _canonical_passwd_home() / ".modal.toml"
    parent_descriptor = _open_nofollow_directory(absolute.parent)
    descriptor: int | None = None
    try:
        before = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > MAX_MODAL_CONFIG_BYTES
        ):
            raise ValueError("canonical Modal configuration metadata is unsafe")
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            _modal_config_identity(opened) != _modal_config_identity(before)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            raise ValueError(
                "canonical Modal configuration changed while it was opened"
            )
        binding = _ModalConfigBinding(
            canonical_path=absolute,
            descriptor=descriptor,
            device=opened.st_dev,
            inode=opened.st_ino,
            size_bytes=opened.st_size,
            mode=stat.S_IMODE(opened.st_mode),
            owner_uid=opened.st_uid,
            mtime_ns=opened.st_mtime_ns,
            ctime_ns=opened.st_ctime_ns,
        )
        _validate_modal_config_metadata(binding)
        _require_modal_config_binding(binding)
        descriptor = None
        return binding
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _require_modal_config_binding(binding: _ModalConfigBinding) -> None:
    if binding.descriptor < 0:
        raise ValueError("canonical Modal configuration descriptor is closed")
    opened = os.fstat(binding.descriptor)
    expected = (
        binding.device,
        binding.inode,
        binding.size_bytes,
        binding.mode,
        binding.owner_uid,
        1,
        binding.mtime_ns,
        binding.ctime_ns,
    )
    if (
        _modal_config_identity(opened) != expected
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.getuid()
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) != 0o600
    ):
        raise ValueError("canonical Modal configuration descriptor changed")
    parent_descriptor = _open_nofollow_directory(binding.canonical_path.parent)
    try:
        current = os.stat(
            binding.canonical_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise ValueError("canonical Modal configuration path was removed") from None
    finally:
        os.close(parent_descriptor)
    if (
        _modal_config_identity(current) != expected
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or current.st_uid != os.getuid()
        or current.st_nlink != 1
        or stat.S_IMODE(current.st_mode) != 0o600
    ):
        raise ValueError("canonical Modal configuration path changed")


def _open_executable_binding(
    executable: Path,
    *,
    label: str,
    maximum_bytes: int,
    require_current_uid: bool,
    require_stable_ctime: bool = True,
) -> _ModalExecutableBinding:
    absolute = Path(os.path.abspath(os.fspath(executable)))
    try:
        before = os.lstat(absolute)
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} is missing: {absolute}") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    if (
        before.st_nlink != 1
        or (require_current_uid and before.st_uid != os.getuid())
        or stat.S_IMODE(before.st_mode) & 0o022
        or not before.st_mode & stat.S_IXUSR
        or before.st_size > maximum_bytes
    ):
        raise ValueError(f"{label} metadata is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_nlink != 1
            or (require_current_uid and opened.st_uid != os.getuid())
            or not opened.st_mode & stat.S_IXUSR
        ):
            raise ValueError(f"{label} changed while it was opened")
        raw = _read_descriptor_bytes(descriptor, maximum_bytes=maximum_bytes)
        after = os.fstat(descriptor)
        before_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            stat.S_IMODE(opened.st_mode),
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            stat.S_IMODE(after.st_mode),
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ValueError(f"{label} changed while it was hashed")
        return _ModalExecutableBinding(
            label=label,
            canonical_path=absolute,
            descriptor=descriptor,
            device=after.st_dev,
            inode=after.st_ino,
            size_bytes=after.st_size,
            mode=stat.S_IMODE(after.st_mode),
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
            sha256=hashlib.sha256(raw).hexdigest(),
            maximum_bytes=maximum_bytes,
            require_current_uid=require_current_uid,
            require_stable_ctime=require_stable_ctime,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _open_modal_executable_binding(modal_executable: Path) -> _ModalExecutableBinding:
    return _open_executable_binding(
        modal_executable,
        label="pinned Modal executable",
        maximum_bytes=MAX_MODAL_CONSOLE_SCRIPT_BYTES,
        require_current_uid=True,
    )


def _open_python_executable_binding() -> _ModalExecutableBinding:
    try:
        resolved = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("active Python executable cannot be resolved") from error
    if Path(sys.prefix).resolve(strict=True) == Path(sys.base_prefix).resolve(strict=True):
        raise ValueError("cleanup capture requires the active project virtual environment")
    return _open_executable_binding(
        resolved,
        label="resolved Python executable",
        maximum_bytes=MAX_PYTHON_EXECUTABLE_BYTES,
        require_current_uid=False,
    )


def _canonical_venv_site_packages() -> Path:
    try:
        venv_root = Path(sys.prefix).resolve(strict=True)
        purelib = Path(sysconfig.get_path("purelib")).resolve(strict=True)
        purelib.relative_to(venv_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "virtual-environment site-packages cannot be resolved"
        ) from error
    descriptor = _open_nofollow_directory(purelib)
    os.close(descriptor)
    return purelib


def _require_modal_executable_descriptor_binding(
    binding: _ModalExecutableBinding,
) -> None:
    if binding.descriptor < 0:
        raise ValueError(f"{binding.label} descriptor is closed")
    opened = os.fstat(binding.descriptor)
    expected = (
        binding.device,
        binding.inode,
        binding.size_bytes,
        binding.mode,
        binding.mtime_ns,
        binding.ctime_ns,
    )
    observed = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        stat.S_IMODE(opened.st_mode),
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    if (
        expected[:-1] != observed[:-1]
        or (binding.require_stable_ctime and expected[-1] != observed[-1])
        or opened.st_nlink != 1
        or (binding.require_current_uid and opened.st_uid != os.getuid())
        or stat.S_IMODE(opened.st_mode) & 0o022
        or not opened.st_mode & stat.S_IXUSR
        or hashlib.sha256(
            _read_descriptor_bytes(
                binding.descriptor,
                maximum_bytes=binding.maximum_bytes,
            )
        ).hexdigest()
        != binding.sha256
    ):
        raise ValueError(f"{binding.label} descriptor changed")


def _require_modal_executable_binding(binding: _ModalExecutableBinding) -> None:
    _require_modal_executable_descriptor_binding(binding)
    expected = (
        binding.device,
        binding.inode,
        binding.size_bytes,
        binding.mode,
        binding.mtime_ns,
        binding.ctime_ns,
    )
    try:
        canonical = os.lstat(binding.canonical_path)
    except FileNotFoundError:
        raise ValueError(f"{binding.label} path was removed") from None
    canonical_identity = (
        canonical.st_dev,
        canonical.st_ino,
        canonical.st_size,
        stat.S_IMODE(canonical.st_mode),
        canonical.st_mtime_ns,
        canonical.st_ctime_ns,
    )
    if (
        stat.S_ISLNK(canonical.st_mode)
        or canonical_identity[:-1] != expected[:-1]
        or (
            binding.require_stable_ctime
            and canonical_identity[-1] != expected[-1]
        )
        or canonical.st_nlink != 1
        or (binding.require_current_uid and canonical.st_uid != os.getuid())
        or stat.S_IMODE(canonical.st_mode) & 0o022
        or not canonical.st_mode & stat.S_IXUSR
    ):
        raise ValueError(f"{binding.label} path changed")


def _bind_modal_installation() -> _ModalExecutableBinding:
    expected = Path(sys.executable).with_name("modal")
    try:
        installed_version = version("modal")
    except PackageNotFoundError as error:
        raise ValueError("the pinned Modal package is not installed") from error
    if installed_version != MODAL_VERSION or MODAL_VERSION != "1.5.3":
        raise ValueError("cleanup capture requires exactly Modal 1.5.3")
    binding = _open_modal_executable_binding(expected)
    try:
        _require_modal_executable_binding(binding)
    except BaseException:
        binding.close()
        raise
    return binding


def _build_child_environment(
    environment: Mapping[str, str] | None,
) -> dict[str, str]:
    ambient = os.environ if environment is None else environment
    child: dict[str, str] = {}
    for key in _SAFE_ENVIRONMENT_KEYS:
        value = ambient.get(key)
        if isinstance(value, str):
            child[key] = value
    # No ambient Modal selection or token override may cross this boundary.
    child["MODAL_PROFILE"] = MODAL_PROFILE
    child["MODAL_ENVIRONMENT"] = MODAL_ENVIRONMENT
    child["TZ"] = "UTC"
    return child


def build_modal_cleanup_snapshot_commands(
    *,
    modal_executable: Path,
    billing_window_start_utc: str,
    billing_window_end_utc: str,
) -> tuple[tuple[str, ...], ...]:
    """Build the frozen, ordered, read-only Modal 1.5.3 command roster."""

    executable = str(modal_executable)
    commands = (
        (executable, "app", "list", "--env", MODAL_ENVIRONMENT, "--json"),
        (
            executable,
            "container",
            "list",
            "--env",
            MODAL_ENVIRONMENT,
            "--json",
        ),
        (
            executable,
            "endpoint",
            "list",
            "--env",
            MODAL_ENVIRONMENT,
            "--json",
        ),
        (
            executable,
            "volume",
            "list",
            "--env",
            MODAL_ENVIRONMENT,
            "--json",
        ),
        (
            executable,
            "volume",
            "ls",
            "--env",
            MODAL_ENVIRONMENT,
            "--json",
            VOLUME_NAME,
            "/runs",
        ),
        (
            executable,
            "billing",
            "report",
            "--start",
            billing_window_start_utc,
            "--end",
            billing_window_end_utc,
            "--resolution",
            "h",
            "--tz",
            "UTC",
            "--show-resources",
            "--json",
        ),
    )
    if len(commands) != len(SNAPSHOT_NAMES) or len(set(commands)) != len(commands):
        raise AssertionError("Modal cleanup command roster is incomplete or duplicated")
    forbidden = {"activate", "deploy", "run", "serve", "shell", "stop"}
    if any(forbidden.intersection(command[1:]) for command in commands):
        raise AssertionError("Modal cleanup command roster contains a write action")
    return commands


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise ValueError(f"JSON contains a non-finite constant: {value}")


def _strict_json_loads(payload: bytes, *, field: str) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} must be UTF-8 JSON") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_non_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} must be one complete JSON value") from error


def _require_text(value: object, *, field: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "text" if allow_empty else "nonempty text"
        raise ValueError(f"{field} must be {qualifier}")
    if len(value.encode("utf-8")) > 4096:
        raise ValueError(f"{field} exceeds its text limit")
    return value


def _parse_raw_timestamp(
    value: object,
    *,
    field: str,
    allow_naive: bool,
) -> datetime:
    text = _require_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        if not allow_naive:
            raise ValueError(f"{field} must include a timezone")
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_billing_interval_utc(value: object, *, field: str) -> datetime:
    text = _require_text(value, field=field)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include the CLI's UTC offset")
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must use the CLI's UTC offset")
    return parsed.astimezone(UTC)


def _validate_decimal_text(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    if _DECIMAL_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be an unsigned Modal decimal string")
    try:
        parsed = Decimal(text)
    except InvalidOperation as error:  # pragma: no cover - regex is stricter
        raise ValueError(f"{field} must be a decimal string") from error
    if not parsed.is_finite() or parsed.is_signed() or parsed < 0:
        raise ValueError(f"{field} must be finite, unsigned, and non-negative")
    return text


def _validate_common_text_row(
    row: Mapping[str, Any],
    *,
    snapshot_name: str,
    index: int,
) -> None:
    for key, value in row.items():
        if snapshot_name == "app_list" and key == "stopped_at" and value is None:
            continue
        if snapshot_name == "volume_list" and key == "created_by" and value is None:
            continue
        _require_text(
            value,
            field=f"{snapshot_name}[{index}].{key}",
            allow_empty=(
                snapshot_name == "billing_report"
                and key in {"description", "environment", "tags"}
            ),
        )


def _validate_snapshot_rows(
    raw: bytes,
    *,
    snapshot_name: str,
    billing_start: datetime,
    billing_end: datetime,
) -> list[dict[str, Any]]:
    payload = _strict_json_loads(raw, field=snapshot_name)
    if not isinstance(payload, list):
        raise TypeError(f"{snapshot_name} must be a Modal CLI JSON row list")
    if len(payload) > MAX_SNAPSHOT_ROWS:
        raise ValueError(f"{snapshot_name} exceeds its row-count limit")
    expected = _RAW_SNAPSHOT_FIELDS[snapshot_name]
    allowed = expected
    rows: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    canonical_rows: set[str] = set()
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise TypeError(f"{snapshot_name}[{index}] must be an object")
        if frozenset(row) not in {expected, allowed}:
            raise ValueError(
                f"{snapshot_name}[{index}] differs from the Modal 1.5.3 JSON schema"
            )
        _validate_common_text_row(row, snapshot_name=snapshot_name, index=index)

        identifier_field = _IDENTIFIER_FIELDS.get(snapshot_name)
        if identifier_field is not None:
            identifier = _require_text(
                row[identifier_field],
                field=f"{snapshot_name}[{index}].{identifier_field}",
            )
            if identifier in identifiers:
                raise ValueError(
                    f"{snapshot_name} contains duplicate {identifier_field} values"
                )
            identifiers.add(identifier)

        if snapshot_name == "app_list":
            _parse_raw_timestamp(
                row["created_at"],
                field=f"app_list[{index}].created_at",
                allow_naive=False,
            )
            if row["stopped_at"] is not None:
                _parse_raw_timestamp(
                    row["stopped_at"],
                    field=f"app_list[{index}].stopped_at",
                    allow_naive=False,
                )
            tasks = _require_text(row["tasks"], field=f"app_list[{index}].tasks")
            if not tasks.isdigit():
                raise ValueError(f"app_list[{index}].tasks must be a decimal integer")
        elif snapshot_name == "container_list":
            if row["start_time"] != "Pending":
                _parse_raw_timestamp(
                    row["start_time"],
                    field=f"container_list[{index}].start_time",
                    allow_naive=False,
                )
        elif snapshot_name in {"endpoint_list", "volume_list"}:
            _parse_raw_timestamp(
                row["created_at"],
                field=f"{snapshot_name}[{index}].created_at",
                allow_naive=False,
            )
        elif snapshot_name == "run_directory_list":
            if row["type"] not in {"dir", "fifo", "file", "link", "socket"}:
                raise ValueError(
                    "run_directory_list contains an unsupported entry type"
                )
            timestamp = _require_text(
                row["created_modified"],
                field=f"run_directory_list[{index}].created_modified",
            )
            if _TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
                raise ValueError("run_directory_list timestamp is not Modal CLI output")
            try:
                datetime.strptime(timestamp[:16], "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
            except ValueError as error:
                raise ValueError("run_directory_list timestamp is invalid") from error
            size = _require_text(row["size"], field=f"run_directory_list[{index}].size")
            if _SIZE_PATTERN.fullmatch(size) is None:
                raise ValueError("run_directory_list size is not Modal CLI output")
            normalized = row["filename"].removeprefix("/").removesuffix("/")
            parts = PurePosixPath(normalized).parts
            if len(parts) != 2 or parts[0] != "runs":
                raise ValueError("run_directory_list path is outside /runs")
            validate_run_id(parts[1])
        elif snapshot_name == "billing_report":
            interval = _parse_billing_interval_utc(
                row["interval_start"],
                field=f"billing_report[{index}].interval_start",
            )
            if interval.minute or interval.second or interval.microsecond:
                raise ValueError("billing_report interval is not hourly aligned")
            if not billing_start <= interval < billing_end:
                raise ValueError("billing_report row lies outside the requested window")
            _validate_decimal_text(row["cost"], field=f"billing_report[{index}].cost")
            if (
                row["description"] == APP_NAME
                and row["environment"] != MODAL_ENVIRONMENT
            ):
                raise ValueError("migration billing row is outside environment main")

        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        if canonical in canonical_rows:
            raise ValueError(f"{snapshot_name} contains a duplicate row")
        canonical_rows.add(canonical)
        rows.append(row)
    return rows


def _encoded_json(value: Any) -> bytes:
    return (
        json.dumps(
            json_value(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_directory_chain(path: Path, *, create: bool) -> int:
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(absolute.anchor, _directory_flags())
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError("cleanup capture path contains an unsafe component")
            created = False
            if create:
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    created = True
                    os.fsync(descriptor)
            try:
                before = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"cleanup capture parent is missing: {absolute}"
                ) from None
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError("cleanup capture path may not traverse a symlink")
            if created:
                os.chmod(
                    component,
                    0o755,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            opened_descriptor = os.open(
                component, _directory_flags(), dir_fd=descriptor
            )
            opened = os.fstat(opened_descriptor)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                os.close(opened_descriptor)
                raise ValueError("cleanup capture ancestor changed while opening")
            os.close(descriptor)
            descriptor = opened_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _reserve_capture_directory(path: Path) -> _CaptureDirectoryReservation:
    """Atomically reserve a fresh capture ID before any Modal CLI request."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_descriptor = _open_directory_chain(absolute.parent, create=True)
    capture_descriptor = -1
    try:
        try:
            os.mkdir(absolute.name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            raise FileExistsError(
                f"cleanup snapshot capture ID already exists: {absolute.name}"
            ) from None
        os.fsync(parent_descriptor)
        os.chmod(
            absolute.name,
            0o700,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        before = os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError("reserved cleanup capture is not a directory")
        capture_descriptor = os.open(
            absolute.name, _directory_flags(), dir_fd=parent_descriptor
        )
        opened = os.fstat(capture_descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("reserved cleanup capture changed while opening")
        os.fchmod(capture_descriptor, 0o700)
        os.fsync(capture_descriptor)
        os.fsync(parent_descriptor)
        reservation = _CaptureDirectoryReservation(
            canonical_path=absolute,
            parent_descriptor=parent_descriptor,
            descriptor=capture_descriptor,
        )
        _require_capture_directory_binding(reservation)
        return reservation
    except BaseException:
        if capture_descriptor >= 0:
            os.close(capture_descriptor)
        os.close(parent_descriptor)
        raise


def _require_capture_directory_binding(
    reservation: _CaptureDirectoryReservation,
) -> None:
    if reservation.descriptor < 0:
        raise ValueError("cleanup capture directory descriptor is closed")
    expected = os.fstat(reservation.descriptor)
    reopened = _open_directory_chain(reservation.canonical_path, create=False)
    try:
        observed = os.fstat(reopened)
        if (expected.st_dev, expected.st_ino) != (
            observed.st_dev,
            observed.st_ino,
        ):
            raise ValueError("canonical cleanup capture directory changed")
    finally:
        os.close(reopened)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        try:
            written = os.write(descriptor, remaining)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("cleanup snapshot write made no progress")
        remaining = remaining[written:]


def _materialize_python_execution_copy(
    source: _ModalExecutableBinding,
    reservation: _CaptureDirectoryReservation,
) -> _PrivatePythonExecutionCopy:
    """Create one create-only executable copy of the held Python bytes."""

    _require_modal_executable_binding(source)
    _require_capture_directory_binding(reservation)
    destination_descriptor: int | None = None
    destination_identity: tuple[int, int] | None = None
    binding: _ModalExecutableBinding | None = None
    digest = hashlib.sha256()
    destination = reservation.canonical_path / PRIVATE_PYTHON_EXECUTION_FILENAME
    try:
        destination_descriptor = os.open(
            PRIVATE_PYTHON_EXECUTION_FILENAME,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o500,
            dir_fd=reservation.descriptor,
        )
        created = os.fstat(destination_descriptor)
        destination_identity = (created.st_dev, created.st_ino)
        offset = 0
        while offset < source.size_bytes:
            try:
                chunk = os.pread(
                    source.descriptor,
                    min(64 * 1024, source.size_bytes - offset),
                    offset,
                )
            except InterruptedError:
                continue
            if not chunk:
                raise ValueError("resolved Python executable changed during copy")
            digest.update(chunk)
            _write_all(destination_descriptor, chunk)
            offset += len(chunk)
        if os.pread(source.descriptor, 1, source.size_bytes):
            raise ValueError("resolved Python executable changed during copy")
        os.fchmod(destination_descriptor, 0o500)
        os.fsync(destination_descriptor)
        copied = os.fstat(destination_descriptor)
        if (
            not stat.S_ISREG(copied.st_mode)
            or copied.st_uid != os.getuid()
            or copied.st_nlink != 1
            or stat.S_IMODE(copied.st_mode) != 0o500
            or copied.st_size != source.size_bytes
            or digest.hexdigest() != source.sha256
        ):
            raise ValueError("private Python executable copy differs from its source")
        os.close(destination_descriptor)
        destination_descriptor = None
        os.fsync(reservation.descriptor)
        _require_modal_executable_binding(source)
        binding = _open_executable_binding(
            destination,
            label="private Python execution copy",
            maximum_bytes=MAX_PYTHON_EXECUTABLE_BYTES,
            require_current_uid=True,
            require_stable_ctime=False,
        )
        if binding.sha256 != source.sha256:
            raise ValueError("private Python executable copy digest changed")
        result = _PrivatePythonExecutionCopy(
            binding=binding,
            reservation=reservation,
            device=binding.device,
            inode=binding.inode,
        )
        binding = None
        result.require_current()
        return result
    except BaseException:
        if destination_descriptor is not None:
            os.close(destination_descriptor)
        if binding is not None:
            binding.close()
        if destination_identity is not None:
            current = os.stat(
                PRIVATE_PYTHON_EXECUTION_FILENAME,
                dir_fd=reservation.descriptor,
                follow_symlinks=False,
            )
            if (
                stat.S_ISLNK(current.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (current.st_dev, current.st_ino) != destination_identity
            ):
                raise RuntimeError(
                    "partial private Python execution copy was replaced"
                ) from None
            os.unlink(
                PRIVATE_PYTHON_EXECUTION_FILENAME,
                dir_fd=reservation.descriptor,
            )
            os.fsync(reservation.descriptor)
        raise


def _read_bound_json_file(
    published: _PublishedJSONFile,
    *,
    maximum_bytes: int,
) -> bytes:
    metadata = os.fstat(published.descriptor)
    if metadata.st_size > maximum_bytes:
        raise ValueError("published snapshot file exceeds its byte limit")
    os.lseek(published.descriptor, 0, os.SEEK_SET)
    payload = bytearray()
    while len(payload) <= maximum_bytes:
        try:
            chunk = os.read(
                published.descriptor,
                min(64 * 1024, maximum_bytes + 1 - len(payload)),
            )
        except InterruptedError:
            continue
        if not chunk:
            break
        payload.extend(chunk)
    os.lseek(published.descriptor, 0, os.SEEK_SET)
    if len(payload) > maximum_bytes or len(payload) != metadata.st_size:
        raise ValueError("published snapshot changed while it was read")
    return bytes(payload)


def _verify_published_json_file(
    reservation: _CaptureDirectoryReservation,
    published: _PublishedJSONFile,
    *,
    maximum_bytes: int,
    require_capture_binding: bool = True,
) -> bytes:
    if require_capture_binding:
        _require_capture_directory_binding(reservation)
    metadata = os.fstat(published.descriptor)
    ctime_is_safe = metadata.st_ctime_ns == published.ctime_ns or (
        sys.platform == "darwin" and metadata.st_ctime_ns > published.ctime_ns
    )
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or (metadata.st_dev, metadata.st_ino) != (published.device, published.inode)
        or metadata.st_size != published.size_bytes
        or metadata.st_mtime_ns != published.mtime_ns
        or not ctime_is_safe
    ):
        raise ValueError("published snapshot file metadata is unsafe")
    try:
        path_metadata = os.stat(
            published.name,
            dir_fd=reservation.descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise ValueError("published snapshot path was removed") from None
    if (
        stat.S_ISLNK(path_metadata.st_mode)
        or (path_metadata.st_dev, path_metadata.st_ino)
        != (published.device, published.inode)
        or stat.S_IMODE(path_metadata.st_mode) != 0o600
        or path_metadata.st_nlink != 1
        or path_metadata.st_size != published.size_bytes
        or path_metadata.st_mtime_ns != published.mtime_ns
        or not (
            path_metadata.st_ctime_ns == published.ctime_ns
            or (
                sys.platform == "darwin"
                and path_metadata.st_ctime_ns > published.ctime_ns
            )
        )
    ):
        raise ValueError("published snapshot path changed")
    raw = _read_bound_json_file(published, maximum_bytes=maximum_bytes)
    if (
        raw != published.raw
        or len(raw) != metadata.st_size
        or hashlib.sha256(raw).hexdigest() != published.sha256
    ):
        raise ValueError("published snapshot bytes changed")
    return raw


def _verify_terminal_capture_files(
    reservation: _CaptureDirectoryReservation,
    published: Mapping[str, _PublishedJSONFile],
    terminal_manifest: _PublishedJSONFile,
    *,
    billing_start: datetime,
    billing_end: datetime,
) -> None:
    """Perform the final callback-free canonical path and byte verification."""

    _require_capture_directory_binding(reservation)
    for snapshot_name in SNAPSHOT_NAMES:
        observed = _verify_published_json_file(
            reservation,
            published[snapshot_name],
            maximum_bytes=MAX_COMMAND_OUTPUT_BYTES,
            require_capture_binding=False,
        )
        _validate_snapshot_rows(
            observed,
            snapshot_name=snapshot_name,
            billing_start=billing_start,
            billing_end=billing_end,
        )
    _verify_published_json_file(
        reservation,
        terminal_manifest,
        maximum_bytes=MAX_COMMAND_OUTPUT_BYTES,
        require_capture_binding=False,
    )


def _publish_json_in_reserved_capture(
    reservation: _CaptureDirectoryReservation,
    *,
    name: str,
    payload: Any,
    maximum_bytes: int = MAX_COMMAND_OUTPUT_BYTES,
) -> _PublishedJSONFile:
    if not name or "/" in name or name in {".", ".."}:
        raise ValueError("cleanup snapshot leaf name is unsafe")
    _require_capture_directory_binding(reservation)
    raw = _encoded_json(payload)
    if len(raw) > maximum_bytes:
        raise ValueError("validated cleanup snapshot exceeds its byte limit")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=reservation.descriptor)
    try:
        # Explicit chmod is required because a restrictive umask can turn the
        # O_CREAT mode into 0400/0200.  Frozen evidence is exactly 0600.
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        os.fsync(reservation.descriptor)
        metadata = os.fstat(descriptor)
        published = _PublishedJSONFile(
            name=name,
            descriptor=descriptor,
            raw=raw,
            sha256=hashlib.sha256(raw).hexdigest(),
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size_bytes=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )
        _verify_published_json_file(reservation, published, maximum_bytes=maximum_bytes)
        return published
    except BaseException:
        os.close(descriptor)
        raise


def _remove_path_if_same(
    reservation: _CaptureDirectoryReservation,
    *,
    name: str,
    published: _PublishedJSONFile,
) -> None:
    try:
        metadata = os.stat(name, dir_fd=reservation.descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (metadata.st_dev, metadata.st_ino) != (
        published.device,
        published.inode,
    ):
        return
    os.unlink(name, dir_fd=reservation.descriptor)
    os.fsync(reservation.descriptor)


def _publish_terminal_manifest(
    reservation: _CaptureDirectoryReservation,
    payload: Mapping[str, Any],
) -> _PublishedJSONFile:
    staging_name = f".{CAPTURE_MANIFEST_FILENAME}.partial"
    staging = _publish_json_in_reserved_capture(
        reservation,
        name=staging_name,
        payload=payload,
        maximum_bytes=MAX_COMMAND_OUTPUT_BYTES,
    )
    terminal_linked = False
    try:
        os.link(
            staging_name,
            CAPTURE_MANIFEST_FILENAME,
            src_dir_fd=reservation.descriptor,
            dst_dir_fd=reservation.descriptor,
            follow_symlinks=False,
        )
        terminal_linked = True
        os.fsync(reservation.descriptor)
        os.unlink(staging_name, dir_fd=reservation.descriptor)
        os.fsync(reservation.descriptor)
        metadata = os.fstat(staging.descriptor)
        terminal = _PublishedJSONFile(
            name=CAPTURE_MANIFEST_FILENAME,
            descriptor=staging.descriptor,
            raw=staging.raw,
            sha256=staging.sha256,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size_bytes=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )
        staging.descriptor = -1
        _verify_published_json_file(
            reservation, terminal, maximum_bytes=MAX_COMMAND_OUTPUT_BYTES
        )
        return terminal
    except BaseException:
        if terminal_linked:
            _remove_path_if_same(
                reservation,
                name=CAPTURE_MANIFEST_FILENAME,
                published=staging,
            )
        raise
    finally:
        staging.close()


def _run_bounded_cli(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    max_output_bytes: int,
    pass_fds: tuple[int, ...] = (),
    popen_factory: Callable[..., Any] = subprocess.Popen,
    process_group_capture: Callable[[Any], int] = capture_isolated_process_group,
    process_group_terminator: Callable[..., None] = terminate_process_group,
    monotonic_factory: Callable[[], float] = time.monotonic,
) -> subprocess.CompletedProcess[bytes]:
    """Run one CLI request with pre-accumulation limits and group closure."""

    if (
        not command
        or not all(isinstance(item, str) and item for item in command)
        or timeout_seconds <= 0
        or max_output_bytes <= 0
    ):
        raise ValueError("bounded Modal command and limits must be positive")
    process = popen_factory(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        pass_fds=pass_fds,
    )
    process_group_id = process.pid
    group_closed = False
    pending_error: BaseException | None = None
    selector: selectors.BaseSelector | None = None
    stdout = bytearray()
    stderr = bytearray()
    returncode: int | None = None
    try:
        process_group_id = process_group_capture(process)
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Modal CLI pipes were not created")
        selector = selectors.DefaultSelector()
        streams = {
            process.stdout.fileno(): stdout,
            process.stderr.fileno(): stderr,
        }
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ)
        deadline = monotonic_factory() + timeout_seconds
        while selector.get_map() or process.poll() is None:
            observed = process.poll()
            if observed is not None and returncode is None:
                returncode = int(observed)
                process_group_terminator(process, process_group_id=process_group_id)
                group_closed = True
            remaining = deadline - monotonic_factory()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(list(command), timeout_seconds)
            for key, _events in selector.select(min(0.05, remaining)):
                descriptor = int(key.fd)
                target = streams[descriptor]
                remaining_budget = max_output_bytes - len(stdout) - len(stderr)
                try:
                    chunk = os.read(descriptor, min(64 * 1024, remaining_budget + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                if len(chunk) > remaining_budget:
                    raise ModalCleanupSnapshotOutputLimitError(
                        "Modal CLI stdout/stderr exceeded the shared byte limit"
                    )
                target.extend(chunk)
        if returncode is None:
            returncode = int(process.wait(timeout=0))
    except BaseException as error:  # noqa: BLE001 - always close process group
        pending_error = error
    finally:
        if not group_closed:
            try:
                process_group_terminator(process, process_group_id=process_group_id)
                group_closed = True
            except BaseException as cleanup_error:  # noqa: BLE001
                pending_error = cleanup_error
        if selector is not None:
            selector.close()
        if getattr(process, "stdout", None) is not None:
            process.stdout.close()
        if getattr(process, "stderr", None) is not None:
            process.stderr.close()
    if pending_error is not None:
        raise pending_error
    if not group_closed or returncode is None:  # pragma: no cover - state guard
        raise RuntimeError("Modal CLI process group did not close")
    completed = subprocess.CompletedProcess(
        list(command), returncode, bytes(stdout), bytes(stderr)
    )
    if returncode != 0:
        raise subprocess.CalledProcessError(
            returncode,
            list(command),
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _remaining_outer_seconds(
    *,
    outer_started: float,
    monotonic_factory: Callable[[], float],
    command: Sequence[str] | None = None,
) -> float:
    remaining = OUTER_TIMEOUT_SECONDS - (monotonic_factory() - outer_started)
    if remaining <= 0:
        raise subprocess.TimeoutExpired(
            list(command)
            if command is not None
            else ["modal-cleanup-snapshot-capture"],
            OUTER_TIMEOUT_SECONDS,
        )
    return remaining


def capture_modal_cleanup_snapshots(
    *,
    project_root: str | Path,
    identity: ModalLiveCohortIdentity,
    capture_id: str,
    billing_window_start_utc: str,
    billing_window_end_utc: str,
    now_factory: Callable[[], datetime] = lambda: datetime.now(UTC),
    monotonic_factory: Callable[[], float] = time.monotonic,
    environment: Mapping[str, str] | None = None,
    _bounded_runner: Callable[..., subprocess.CompletedProcess[bytes]] | None = None,
    _popen_factory: Callable[..., Any] = subprocess.Popen,
    _process_group_capture: Callable[[Any], int] = capture_isolated_process_group,
    _process_group_terminator: Callable[..., None] = terminate_process_group,
) -> ModalCleanupSnapshotCaptureResult:
    """Capture and freeze one complete read-only Modal cleanup observation."""

    if not isinstance(identity, ModalLiveCohortIdentity):
        raise TypeError("identity must be a ModalLiveCohortIdentity")
    validated_capture_id = validate_run_id(capture_id)
    root = Path(os.path.abspath(os.fspath(project_root)))
    outer_started = monotonic_factory()
    started_at = now_factory()
    started_text = _utc_z(started_at, field="started_at_utc")
    billing_start, billing_end = _validate_completed_billing_window(
        billing_window_start_utc,
        billing_window_end_utc,
        captured_at=started_at,
    )
    _remaining_outer_seconds(
        outer_started=outer_started, monotonic_factory=monotonic_factory
    )
    logical_capture_root = modal_cleanup_snapshot_capture_directory(
        identity, validated_capture_id
    )
    capture_root = root / logical_capture_root
    child_environment = _build_child_environment(environment)
    total_output_bytes = 0
    snapshots: dict[str, dict[str, Any]] = {}
    published: dict[str, _PublishedJSONFile] = {}
    executable_binding: _ModalExecutableBinding | None = None
    python_binding: _ModalExecutableBinding | None = None
    python_execution: _PrivatePythonExecutionCopy | None = None
    config_binding: _ModalConfigBinding | None = None
    reservation: _CaptureDirectoryReservation | None = None
    terminal_manifest: _PublishedJSONFile | None = None
    completed_successfully = False

    def default_runner(
        command: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> subprocess.CompletedProcess[bytes]:
        if (
            executable_binding is None
            or python_binding is None
            or python_execution is None
            or config_binding is None
        ):
            raise RuntimeError("cleanup capture launch bindings are incomplete")
        if not command or Path(command[0]) != executable_binding.execution_path:
            raise ValueError("cleanup capture command is not descriptor-bound")
        _require_modal_executable_binding(python_binding)
        python_execution.require_current()
        execution_command = [
            os.fspath(python_execution.canonical_path),
            *command,
        ]
        os.lseek(config_binding.descriptor, 0, os.SEEK_SET)
        try:
            completed = _run_bounded_cli(
                execution_command,
                cwd=cwd,
                environment=environment,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
                pass_fds=(
                    executable_binding.descriptor,
                    config_binding.descriptor,
                ),
                popen_factory=_popen_factory,
                process_group_capture=_process_group_capture,
                process_group_terminator=_process_group_terminator,
                monotonic_factory=monotonic_factory,
            )
        finally:
            os.lseek(config_binding.descriptor, 0, os.SEEK_SET)
        _require_modal_executable_binding(python_binding)
        python_execution.require_current()
        return subprocess.CompletedProcess(
            list(command),
            completed.returncode,
            completed.stdout,
            completed.stderr,
        )

    lock_descriptor = acquire_modal_action_lock(root)
    try:
        assert_modal_action_lock_identity(lock_descriptor)
        executable_binding = _bind_modal_installation()
        python_binding = _open_python_executable_binding()
        config_binding = _open_modal_config_binding()
        child_environment["MODAL_CONFIG_PATH"] = config_binding.execution_path
        child_environment["PYTHONPATH"] = os.pathsep.join(
            (str(root), str(_canonical_venv_site_packages()))
        )
        _require_modal_config_binding(config_binding)
        _remaining_outer_seconds(
            outer_started=outer_started, monotonic_factory=monotonic_factory
        )
        # The fresh directory is an O_EXCL-equivalent mkdir reservation and is
        # durably present before the first networked read-only CLI command.
        assert_modal_action_lock_identity(lock_descriptor)
        reservation = _reserve_capture_directory(capture_root)
        python_execution = _materialize_python_execution_copy(
            python_binding,
            reservation,
        )
        _remaining_outer_seconds(
            outer_started=outer_started, monotonic_factory=monotonic_factory
        )
        commands = build_modal_cleanup_snapshot_commands(
            modal_executable=executable_binding.execution_path,
            billing_window_start_utc=billing_window_start_utc,
            billing_window_end_utc=billing_window_end_utc,
        )
        runner = default_runner if _bounded_runner is None else _bounded_runner
        for snapshot_name, command in zip(SNAPSHOT_NAMES, commands, strict=True):
            remaining_time = _remaining_outer_seconds(
                outer_started=outer_started,
                monotonic_factory=monotonic_factory,
                command=command,
            )
            remaining_bytes = MAX_CAPTURE_OUTPUT_BYTES - total_output_bytes
            if remaining_bytes <= 0:
                raise ModalCleanupSnapshotOutputLimitError(
                    "cleanup capture exceeded its aggregate output limit"
                )
            assert_modal_action_lock_identity(lock_descriptor)
            _require_modal_executable_binding(executable_binding)
            _require_modal_executable_binding(python_binding)
            python_execution.require_current()
            _require_modal_config_binding(config_binding)
            remaining_time = _remaining_outer_seconds(
                outer_started=outer_started,
                monotonic_factory=monotonic_factory,
                command=command,
            )
            completed = runner(
                command,
                cwd=root,
                environment=child_environment,
                timeout_seconds=min(COMMAND_TIMEOUT_SECONDS, remaining_time),
                max_output_bytes=min(MAX_COMMAND_OUTPUT_BYTES, remaining_bytes),
            )
            assert_modal_action_lock_identity(lock_descriptor)
            _require_modal_executable_binding(executable_binding)
            _require_modal_executable_binding(python_binding)
            python_execution.require_current()
            _require_modal_config_binding(config_binding)
            _remaining_outer_seconds(
                outer_started=outer_started,
                monotonic_factory=monotonic_factory,
                command=command,
            )
            if not isinstance(completed, subprocess.CompletedProcess):
                raise TypeError("bounded Modal runner returned the wrong result type")
            if completed.args != list(command) and completed.args != tuple(command):
                raise ValueError(
                    "bounded Modal runner returned a result for another command"
                )
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    list(command),
                    output=completed.stdout,
                    stderr=completed.stderr,
                )
            if not isinstance(completed.stdout, bytes) or not isinstance(
                completed.stderr, bytes
            ):
                raise TypeError("bounded Modal runner must return byte output")
            observed_bytes = len(completed.stdout) + len(completed.stderr)
            if (
                observed_bytes > MAX_COMMAND_OUTPUT_BYTES
                or observed_bytes > remaining_bytes
            ):
                raise ModalCleanupSnapshotOutputLimitError(
                    "bounded Modal runner violated its output limit"
                )
            total_output_bytes += observed_bytes
            rows = _validate_snapshot_rows(
                completed.stdout,
                snapshot_name=snapshot_name,
                billing_start=billing_start,
                billing_end=billing_end,
            )
            _remaining_outer_seconds(
                outer_started=outer_started,
                monotonic_factory=monotonic_factory,
                command=command,
            )
            leaf_name = f"{snapshot_name}.json"
            leaf_logical_path = logical_capture_root / leaf_name
            frozen = _publish_json_in_reserved_capture(
                reservation, name=leaf_name, payload=rows
            )
            assert_modal_action_lock_identity(lock_descriptor)
            published[snapshot_name] = frozen
            _remaining_outer_seconds(
                outer_started=outer_started,
                monotonic_factory=monotonic_factory,
                command=command,
            )
            captured_at_text = _utc_z(
                now_factory(), field=f"{snapshot_name}.captured_at_utc"
            )
            snapshots[snapshot_name] = {
                "path": leaf_logical_path.as_posix(),
                "sha256": frozen.sha256,
                "size_bytes": len(frozen.raw),
                "argv": list(command),
                "captured_at_utc": captured_at_text,
            }

        _require_modal_executable_binding(python_binding)
        python_execution.close_and_remove()
        python_execution = None

        # Bind every still-open leaf descriptor and its canonical path directly
        # before staging the terminal manifest.
        for snapshot_name in SNAPSHOT_NAMES:
            assert_modal_action_lock_identity(lock_descriptor)
            _remaining_outer_seconds(
                outer_started=outer_started, monotonic_factory=monotonic_factory
            )
            observed = _verify_published_json_file(
                reservation,
                published[snapshot_name],
                maximum_bytes=MAX_COMMAND_OUTPUT_BYTES,
            )
            _validate_snapshot_rows(
                observed,
                snapshot_name=snapshot_name,
                billing_start=billing_start,
                billing_end=billing_end,
            )
        _require_modal_executable_binding(executable_binding)
        _require_modal_config_binding(config_binding)
        _require_capture_directory_binding(reservation)
        assert_modal_action_lock_identity(lock_descriptor)
        _remaining_outer_seconds(
            outer_started=outer_started, monotonic_factory=monotonic_factory
        )

        finished_at = now_factory()
        if finished_at.astimezone(UTC) < started_at.astimezone(UTC):
            raise ValueError("cleanup capture finish timestamp precedes its start")
        finished_text = _utc_z(finished_at, field="finished_at_utc")
        manifest = {
            "schema_name": CAPTURE_MANIFEST_SCHEMA_NAME,
            "schema_version": CAPTURE_MANIFEST_SCHEMA_VERSION,
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "capture_id": validated_capture_id,
            "modal_profile": MODAL_PROFILE,
            "modal_environment": MODAL_ENVIRONMENT,
            "modal_cli_version": MODAL_VERSION,
            "billing_window_start_utc": billing_window_start_utc,
            "billing_window_end_utc": billing_window_end_utc,
            "started_at_utc": started_text,
            "finished_at_utc": finished_text,
            "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
            "outer_timeout_seconds": OUTER_TIMEOUT_SECONDS,
            "command_retry_count": 0,
            "snapshots": snapshots,
        }
        manifest_logical_path = modal_cleanup_snapshot_capture_manifest_path(
            identity, validated_capture_id
        )
        _remaining_outer_seconds(
            outer_started=outer_started, monotonic_factory=monotonic_factory
        )
        assert_modal_action_lock_identity(lock_descriptor)
        _require_modal_config_binding(config_binding)
        terminal_manifest = _publish_terminal_manifest(reservation, manifest)
        assert_modal_action_lock_identity(lock_descriptor)
        _require_modal_config_binding(config_binding)
        parsed_manifest = _strict_json_loads(
            terminal_manifest.raw,
            field="capture manifest",
        )
        if parsed_manifest != manifest:
            raise ValueError("published cleanup capture manifest changed")
        result = ModalCleanupSnapshotCaptureResult(
            manifest_path=manifest_logical_path.as_posix(),
            manifest_sha256=terminal_manifest.sha256,
            _manifest_bytes=terminal_manifest.raw,
        )

        # Complete every injected observation and binding check before the
        # final callback-free descriptor/path verification.  The deadline read
        # catches time consumed by the bindings; any mutation performed by that
        # final injected clock callback is caught by the verification below.
        _require_modal_executable_binding(executable_binding)
        _require_modal_config_binding(config_binding)
        _require_capture_directory_binding(reservation)
        _remaining_outer_seconds(
            outer_started=outer_started,
            monotonic_factory=monotonic_factory,
        )
        _verify_terminal_capture_files(
            reservation,
            published,
            terminal_manifest,
            billing_start=billing_start,
            billing_end=billing_end,
        )
        assert_modal_action_lock_identity(lock_descriptor)
        _require_modal_config_binding(config_binding)
        completed_successfully = True
        return result
    finally:
        try:
            try:
                assert_modal_action_lock_identity(lock_descriptor)
                if config_binding is not None:
                    _require_modal_config_binding(config_binding)
            finally:
                if (
                    terminal_manifest is not None
                    and reservation is not None
                    and not completed_successfully
                ):
                    _remove_path_if_same(
                        reservation,
                        name=CAPTURE_MANIFEST_FILENAME,
                        published=terminal_manifest,
                    )
                if terminal_manifest is not None:
                    terminal_manifest.close()
                for frozen in published.values():
                    frozen.close()
                try:
                    if python_execution is not None:
                        python_execution.close_and_remove()
                finally:
                    if reservation is not None:
                        reservation.close()
                    if executable_binding is not None:
                        executable_binding.close()
                    if python_binding is not None:
                        python_binding.close()
                    if config_binding is not None:
                        config_binding.close()
            assert_modal_action_lock_identity(lock_descriptor)
        finally:
            release_modal_action_lock(lock_descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--image-source-sha256", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--billing-window-start-utc", required=True)
    parser.add_argument("--billing-window-end-utc", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=arguments.source_tree_sha256,
        image_source_sha256=arguments.image_source_sha256,
        cohort_id=arguments.cohort_id,
    )
    result = capture_modal_cleanup_snapshots(
        project_root=arguments.project_root,
        identity=identity,
        capture_id=arguments.capture_id,
        billing_window_start_utc=arguments.billing_window_start_utc,
        billing_window_end_utc=arguments.billing_window_end_utc,
    )
    print(
        json.dumps(
            {
                "manifest_path": result.manifest_path,
                "manifest_sha256": result.manifest_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
