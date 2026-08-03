"""Canonical serialization helpers for post-search scientific records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence


def require_bool(value: object, field_name: str) -> bool:
    """Reject JSON type confusion instead of applying Python truthiness."""

    if type(value) is not bool:
        raise ValueError(f"{field_name} must be boolean")
    return value


def require_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def json_value(value: Any) -> Any:
    """Return a strictly JSON-compatible value with deterministic mappings."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific records cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_value(item) for item in value]
    if hasattr(value, "to_dict"):
        return json_value(value.to_dict())
    raise TypeError(f"unsupported scientific-record value {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_sha256(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")


def require_identifier(value: str, field_name: str) -> None:
    if not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed identifier")
    if any(character in value for character in ("/", "\\", "\x00")):
        raise ValueError(f"{field_name} must not contain path separators")


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically replace ``path`` with canonical JSON and a trailing newline."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(value))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
