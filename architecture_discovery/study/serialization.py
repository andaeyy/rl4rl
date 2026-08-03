"""Canonical JSON, stable hashes, and fail-closed atomic state writes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
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


def create_json_exclusive(path: str | Path, value: Any) -> None:
    """Create a frozen JSON file exactly once, leaving an existing file untouched."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        payload = json.dumps(
            json_value(value),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial frozen plan is safer left visible than silently replaced. Loading it
        # will fail closed and require deliberate operator recovery.
        raise


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload
