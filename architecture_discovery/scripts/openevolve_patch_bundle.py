#!/usr/bin/env python3
"""Validate the frozen OpenEvolve commit and reviewed patch provenance.

The vendored repository must be at the exact reviewed commit, both historical
patch files must match their frozen digests, the three integrated implementation
files must match their reviewed hashes, and the worktree must be pristine.  No
command in this module contacts a network service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).resolve().parents[1]
OPENEVOLVE_VENDOR_RELATIVE_PATH = "vendor/openevolve"
OPENEVOLVE_BASE_COMMIT = "5ecb48b5ca453d3f2b9c316a4ffe45d45725bf0c"
OPENEVOLVE_PATCH_RELATIVE_PATH = (
    "vendor_patches/openevolve_process_isolation.patch"
)
OPENEVOLVE_PATCH_SHA256 = (
    "f39fa2a2ed50b7d22a28a5c5ce5547838f66b8445b2f17ad24f899a4e92560a8"
)
OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH = (
    "vendor_patches/openevolve_provider_attempt_ledger.patch"
)
OPENEVOLVE_PROVIDER_PATCH_SHA256 = (
    "b0f731fa87fda394188dadc8fdab9c687b22c0a635eebef06d04cc4740017dc6"
)
OPENEVOLVE_BASE_FILE_SHA256: Mapping[str, str] = {
    "openevolve/llm/openai.py": (
        "c01228a2a47b7d22096206e9f9da999dfb031fed728568583da6fbc667fec1d1"
    ),
    "openevolve/process_parallel.py": (
        "d740466efc43d6f70f341bcf6d33701188edf215e2347f66febaa7d45ad10971"
    ),
    "tests/test_process_parallel.py": (
        "65f7402c8caec92254eec677bccd709569e221dc43f4e81f48648e98137b3afe"
    ),
}
OPENEVOLVE_ISOLATION_PATCHED_FILE_SHA256: Mapping[str, str] = {
    "openevolve/process_parallel.py": (
        "d740466efc43d6f70f341bcf6d33701188edf215e2347f66febaa7d45ad10971"
    ),
    "tests/test_process_parallel.py": (
        "65f7402c8caec92254eec677bccd709569e221dc43f4e81f48648e98137b3afe"
    ),
}
OPENEVOLVE_PROVIDER_PATCHED_FILE_SHA256: Mapping[str, str] = {
    "openevolve/llm/openai.py": (
        "c01228a2a47b7d22096206e9f9da999dfb031fed728568583da6fbc667fec1d1"
    ),
}
OPENEVOLVE_PATCHED_FILE_SHA256: Mapping[str, str] = {
    **OPENEVOLVE_PROVIDER_PATCHED_FILE_SHA256,
    **OPENEVOLVE_ISOLATION_PATCHED_FILE_SHA256,
}
_GIT_TIMEOUT_SECONDS = 15


class OpenEvolvePatchBundleError(ValueError):
    """The frozen patch bundle or vendored source is not in an exact state."""


@dataclass(frozen=True)
class OpenEvolvePatchBundleStatus:
    base_commit: str
    patch_relative_path: str
    patch_sha256: str
    provider_patch_relative_path: str
    provider_patch_sha256: str
    patched_file_sha256: Mapping[str, str]
    applied: bool

    SCHEMA_NAME: ClassVar[str] = "OpenEvolvePatchBundleStatus"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "base_commit": self.base_commit,
            "patch_relative_path": self.patch_relative_path,
            "patch_sha256": self.patch_sha256,
            "provider_patch_relative_path": self.provider_patch_relative_path,
            "provider_patch_sha256": self.provider_patch_sha256,
            "patched_file_sha256": dict(sorted(self.patched_file_sha256.items())),
            "applied": self.applied,
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise OpenEvolvePatchBundleError(
            f"patch-bundle input is not a regular file: {path.name}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contained_path(project_root: Path, relative_path: str, label: str) -> Path:
    """Resolve a fixed project-relative path without traversing symlinks."""

    current = project_root
    for component in Path(relative_path).parts:
        current /= component
        if current.is_symlink():
            raise OpenEvolvePatchBundleError(
                f"{label} may not traverse symbolic links"
            )
    return current


def _git(vendor_root: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=vendor_root,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise OpenEvolvePatchBundleError(
            "unable to inspect the vendored OpenEvolve repository"
        ) from error
    if completed.returncode != 0:
        raise OpenEvolvePatchBundleError(
            "vendored OpenEvolve Git verification failed for "
            f"{arguments[0] if arguments else 'unknown command'}"
        )
    return completed.stdout


def _resolved_inputs(
    project_root: str | Path,
) -> tuple[Path, Path, tuple[Path, Path]]:
    raw_project = Path(project_root)
    if raw_project.is_symlink():
        raise OpenEvolvePatchBundleError("project root may not be a symlink")
    project = raw_project.resolve()
    if not project.is_dir():
        raise OpenEvolvePatchBundleError("project root is missing")
    raw_vendor = _contained_path(
        project,
        OPENEVOLVE_VENDOR_RELATIVE_PATH,
        "OpenEvolve vendor root",
    )
    vendor = raw_vendor.resolve()
    if not vendor.is_dir():
        raise OpenEvolvePatchBundleError("OpenEvolve vendor root is missing")
    patch = _contained_path(
        project,
        OPENEVOLVE_PATCH_RELATIVE_PATH,
        "reviewed OpenEvolve patch file",
    )
    if not patch.is_file():
        raise OpenEvolvePatchBundleError("reviewed OpenEvolve patch file is missing")
    provider_patch = _contained_path(
        project,
        OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH,
        "reviewed OpenEvolve provider patch file",
    )
    if not provider_patch.is_file():
        raise OpenEvolvePatchBundleError(
            "reviewed OpenEvolve provider patch file is missing"
        )
    return project, vendor, (patch, provider_patch)


def _verify_repository_identity(vendor_root: Path) -> None:
    top_level_raw = _git(vendor_root, "rev-parse", "--show-toplevel")
    try:
        top_level = Path(top_level_raw.decode("utf-8").strip()).resolve()
        head = _git(vendor_root, "rev-parse", "--verify", "HEAD").decode(
            "ascii"
        ).strip()
    except (UnicodeDecodeError, ValueError) as error:
        raise OpenEvolvePatchBundleError(
            "vendored OpenEvolve Git identity is malformed"
        ) from error
    if top_level != vendor_root:
        raise OpenEvolvePatchBundleError(
            "vendored OpenEvolve path is not its Git worktree root"
        )
    if head != OPENEVOLVE_BASE_COMMIT:
        raise OpenEvolvePatchBundleError(
            "vendored OpenEvolve HEAD differs from the frozen base commit"
        )
    for relative_path, expected_hash in OPENEVOLVE_BASE_FILE_SHA256.items():
        blob = _git(
            vendor_root,
            "show",
            f"{OPENEVOLVE_BASE_COMMIT}:{relative_path}",
        )
        if _sha256_bytes(blob) != expected_hash:
            raise OpenEvolvePatchBundleError(
                "frozen OpenEvolve base blob differs from the reviewed identity"
            )


def _worktree_status(vendor_root: Path) -> frozenset[bytes]:
    payload = _git(
        vendor_root,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )
    records = tuple(record for record in payload.split(b"\0") if record)
    if len(records) != len(set(records)):
        raise OpenEvolvePatchBundleError(
            "vendored OpenEvolve status contains duplicate records"
        )
    return frozenset(records)


def _worktree_hashes(vendor_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative_path in OPENEVOLVE_PATCHED_FILE_SHA256:
        path = vendor_root.joinpath(*relative_path.split("/"))
        observed[relative_path] = _sha256_file(path)
    return observed


def _state(vendor_root: Path) -> str:
    observed = _worktree_hashes(vendor_root)
    status = _worktree_status(vendor_root)
    if observed != dict(OPENEVOLVE_PATCHED_FILE_SHA256):
        raise OpenEvolvePatchBundleError(
            "OpenEvolve files differ from the exact reviewed integrated state"
        )
    if status:
        raise OpenEvolvePatchBundleError(
            "integrated OpenEvolve files coexist with unexpected worktree changes"
        )
    return "integrated"


def ensure_openevolve_patch_bundle(
    project_root: str | Path = ROOT,
    *,
    apply: bool = False,
) -> OpenEvolvePatchBundleStatus:
    """Validate the integrated commit and frozen patch provenance.

    ``apply`` is retained as a backwards-compatible no-op for bootstrap
    callers that predate integration of the reviewed patches into the commit.
    """

    _project, vendor, patches = _resolved_inputs(project_root)
    expected_patches = (
        (patches[0], OPENEVOLVE_PATCH_SHA256),
        (patches[1], OPENEVOLVE_PROVIDER_PATCH_SHA256),
    )
    for patch, expected_sha256 in expected_patches:
        if _sha256_file(patch) != expected_sha256:
            raise OpenEvolvePatchBundleError(
                "reviewed OpenEvolve patch SHA-256 differs"
            )
    _verify_repository_identity(vendor)
    _state(vendor)
    return OpenEvolvePatchBundleStatus(
        base_commit=OPENEVOLVE_BASE_COMMIT,
        patch_relative_path=OPENEVOLVE_PATCH_RELATIVE_PATH,
        patch_sha256=OPENEVOLVE_PATCH_SHA256,
        provider_patch_relative_path=OPENEVOLVE_PROVIDER_PATCH_RELATIVE_PATH,
        provider_patch_sha256=OPENEVOLVE_PROVIDER_PATCH_SHA256,
        patched_file_sha256=OPENEVOLVE_PATCHED_FILE_SHA256,
        applied=True,
    )


def validate_applied_patch_bundle(
    project_root: str | Path = ROOT,
) -> OpenEvolvePatchBundleStatus:
    """Require the exact reviewed applied state without changing any file."""

    return ensure_openevolve_patch_bundle(project_root, apply=False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the frozen integrated OpenEvolve commit"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="compatibility no-op; validate the integrated reviewed state",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        status = ensure_openevolve_patch_bundle(apply=arguments.apply)
    except OpenEvolvePatchBundleError as error:
        parser.error(str(error))
    print(json.dumps(status.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
