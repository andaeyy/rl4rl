"""Run-scoped executable-architecture identity registry.

Each architecture hash is claimed by atomically creating a marker file.  The
filesystem operation is process-safe, which lets the native OpenEvolve workers
share the same registry without trusting candidate-provided identifiers or
metadata.
"""

from __future__ import annotations

import os
from pathlib import Path


_SHA256_HEX_LENGTH = 64


def validate_architecture_hash(value: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError("architecture hash must be a lowercase SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError("architecture hash must be a lowercase SHA-256 digest")
    return value


class ArchitectureHashRegistry:
    """Atomically claim normalized executable identities within one run."""

    def __init__(self, directory: str | Path) -> None:
        raw = Path(directory).expanduser()
        if raw.is_symlink():
            raise ValueError(f"architecture registry may not be a symlink: {raw}")
        raw.mkdir(parents=True, exist_ok=True)
        self.directory = raw.resolve()

    def claim(self, architecture_hash: str) -> bool:
        """Return true exactly once for a valid hash in this registry."""

        digest = validate_architecture_hash(architecture_hash)
        marker = self.directory / digest
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(marker, flags, 0o600)
        except FileExistsError:
            return False
        try:
            os.write(descriptor, (digest + "\n").encode("ascii"))
        finally:
            os.close(descriptor)
        return True

    def contains(self, architecture_hash: str) -> bool:
        digest = validate_architecture_hash(architecture_hash)
        return (self.directory / digest).is_file()
