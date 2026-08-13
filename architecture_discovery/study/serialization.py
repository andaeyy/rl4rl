"""Canonical JSON, stable hashes, and fail-closed atomic state writes."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any


def require_bool(value: object, field_name: str) -> bool:
    """Return an exact JSON boolean and reject truthiness-based coercion."""

    if type(value) is not bool:
        raise ValueError(f"{field_name} must be boolean")
    return value


def require_int(value: object, field_name: str) -> int:
    """Return an exact integer; booleans and numeric strings are not integers."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def require_str(value: object, field_name: str) -> str:
    """Return an exact string without silently stringifying another JSON type."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def json_value(value: Any) -> Any:
    """Return a JSON-compatible value with stable tuple, enum, and path handling."""

    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(json_value(item) for item in value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, value: Any, *, length: int = 20) -> str:
    return f"{prefix}-{content_hash(value)[:length]}"


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically replace one JSON record and fsync both file and parent directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                json_value(value),
                handle,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _open_exclusive_json_parent(
    destination: Path,
    *,
    create: bool,
) -> tuple[int, Path]:
    """Open a path's parent without following any directory symlink."""

    absolute = Path(os.path.abspath(os.fspath(destination)))
    if not absolute.name or absolute.name in {".", ".."}:
        raise ValueError("exclusive JSON destination must name a file")
    components = absolute.parts
    if not absolute.is_absolute() or not components:
        raise ValueError("exclusive JSON destination could not be anchored")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in components[1:-1]:
            if component in {"", ".", ".."}:
                raise ValueError(
                    "exclusive JSON parent contains an unsafe path component"
                )
            if create:
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    os.fsync(descriptor)
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"exclusive JSON parent does not exist: {absolute.parent}"
                ) from None
            if stat.S_ISLNK(before.st_mode):
                raise ValueError("exclusive JSON parent may not contain a symlink")
            if not stat.S_ISDIR(before.st_mode):
                raise NotADirectoryError(
                    f"exclusive JSON parent component is not a directory: {component}"
                )
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ValueError(
                    "exclusive JSON parent changed while it was opened"
                ) from error
            opened = os.fstat(next_descriptor)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(next_descriptor)
                raise ValueError(
                    "exclusive JSON parent changed while it was opened"
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _require_same_exclusive_json_parent(
    destination: Path,
    expected_descriptor: int,
) -> None:
    reopened_descriptor, _ = _open_exclusive_json_parent(
        destination,
        create=False,
    )
    try:
        expected = os.fstat(expected_descriptor)
        reopened = os.fstat(reopened_descriptor)
        if (expected.st_dev, expected.st_ino) != (
            reopened.st_dev,
            reopened.st_ino,
        ):
            raise ValueError(
                "exclusive JSON parent changed during create-only publication"
            )
    finally:
        os.close(reopened_descriptor)


def create_json_exclusive(path: str | Path, value: Any) -> None:
    """Durably create one frozen JSON file without following path symlinks."""

    destination = Path(path)
    encoded = (
        json.dumps(
            json_value(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    parent_descriptor, absolute = _open_exclusive_json_parent(
        destination,
        create=True,
    )
    published = False
    try:
        _require_same_exclusive_json_parent(absolute, parent_descriptor)
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        write_flags |= getattr(os, "O_CLOEXEC", 0)
        write_flags |= getattr(os, "O_NOFOLLOW", 0)
        published_descriptor = os.open(
            absolute.name,
            write_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        published = True
        try:
            remaining = memoryview(encoded)
            while remaining:
                try:
                    written = os.write(published_descriptor, remaining)
                except InterruptedError:
                    continue
                if written <= 0:
                    raise OSError("exclusive JSON write made no progress")
                remaining = remaining[written:]
            os.fsync(published_descriptor)
            published_stat = os.fstat(published_descriptor)
            if (
                not stat.S_ISREG(published_stat.st_mode)
                or published_stat.st_nlink != 1
                or published_stat.st_uid != os.getuid()
                or published_stat.st_size != len(encoded)
                or stat.S_IMODE(published_stat.st_mode) & 0o077
            ):
                raise ValueError("published exclusive JSON file is unsafe")
        except BaseException:
            # Never remove or replace a partially written frozen record.  A
            # durable invalid file quarantines the run and blocks silent reuse.
            with suppress(OSError):
                os.fsync(published_descriptor)
            raise
        finally:
            os.close(published_descriptor)
        os.fsync(parent_descriptor)

        read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        read_flags |= getattr(os, "O_NOFOLLOW", 0)
        reopened_descriptor = os.open(
            absolute.name,
            read_flags,
            dir_fd=parent_descriptor,
        )
        try:
            reopened_stat = os.fstat(reopened_descriptor)
            if (
                not stat.S_ISREG(reopened_stat.st_mode)
                or reopened_stat.st_nlink != 1
                or reopened_stat.st_uid != os.getuid()
                or reopened_stat.st_size != len(encoded)
                or stat.S_IMODE(reopened_stat.st_mode) & 0o077
            ):
                raise ValueError("published exclusive JSON file is unsafe")
            with os.fdopen(reopened_descriptor, "rb", closefd=False) as handle:
                reopened = handle.read(len(encoded) + 1)
        finally:
            os.close(reopened_descriptor)
        if reopened != encoded:
            raise ValueError("published exclusive JSON bytes differ")
        _require_same_exclusive_json_parent(absolute, parent_descriptor)
    finally:
        if published:
            with suppress(OSError):
                os.fsync(parent_descriptor)
        os.close(parent_descriptor)


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload
