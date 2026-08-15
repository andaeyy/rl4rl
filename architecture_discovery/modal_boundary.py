"""Pure, cost-free contracts for the optional Modal execution boundary.

This module never imports Modal and never performs network or provider calls.
It defines the image source allowlist, bounded function specifications, safe
Volume paths, artifact transfer checks, and synchronous orchestration helpers
used by :mod:`modal_app`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Protocol

from common.runtime_context import ExecutionContextV1

APP_NAME = "rl4rl-architecture-discovery"
VOLUME_NAME = "rl4rl-architecture-artifacts"
VOLUME_MOUNT_PATH = PurePosixPath("/mnt/discovery")
VOLUME_RUNS_PATH = VOLUME_MOUNT_PATH / "runs"
REMOTE_PROJECT_ROOT = PurePosixPath("/opt/architecture_discovery")
PROVIDER_SECRET_NAME = "rl4rl-discovery-provider"
PYTHON_VERSION = "3.12"
UV_VERSION = "0.12.0"
MODAL_VERSION = "1.5.3"
MODAL_ENVIRONMENT_NAME = "main"
MAX_MODAL_BILLING_WINDOW = timedelta(days=31)
IMAGE_RECIPE_VERSION = "modal-cuda-image-v1"
MODAL_DOWNLOAD_OUTPUT_ROOT = "outputs/development/modal_downloads"
IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS = 600
IMAGE_BUILD_CPU_REQUEST_CORES = 2.0
IMAGE_BUILD_MEMORY_REQUEST_MIB = 8192
IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT = 2
FUNCTION_CPU_REQUEST_CORES = 2.0
FUNCTION_CPU_SOFT_LIMIT_CORES = 2.0
FUNCTION_MEMORY_REQUEST_MIB = 8192
FUNCTION_MEMORY_LIMIT_MIB = 8192
FUNCTION_TIMEOUT_SECONDS = 300
CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS = 240
PROVIDER_REQUEST_TIMEOUT_SECONDS = 180
PROVIDER_ATTEMPT_FINALIZATION_RESERVE_SECONDS = 60
MAX_CONTAINERS = 1
MIN_CONTAINERS = 0
FUNCTION_RETRIES = 0
GPU_TYPE = "T4"
CANARY_ORDER = (
    "greedy_autoresearch",
    "semantic_autoresearch",
    "openevolve_generic",
    "openevolve_semantic",
)
MODAL_ACTIONS = frozenset(
    {
        "canaries",
        "canary",
        "candidate-smoke",
        "checkpoint-resume",
        "cuda-environment",
        "exploratory_c0c3_pilot",
        "download",
        "offline-smoke",
        "verify",
    }
)
MODAL_ACTION_ATTEMPT_ID_ENV = "RL4RL_MODAL_ACTION_ATTEMPT_ID"
MODAL_ACTION_INTENT_SHA256_ENV = "RL4RL_MODAL_ACTION_INTENT_SHA256"
MODAL_LIVE_COHORT_ROOT = PurePosixPath(
    "outputs/readiness/modal_only_final/modal_live_cohorts"
)
MODAL_LAUNCH_REJECTION_ROOT = PurePosixPath(
    "outputs/readiness/modal_only_final/modal_launch_rejections"
)
MODAL_GLOBAL_LAUNCH_REJECTION_SEAL_PATH = MODAL_LAUNCH_REJECTION_ROOT / "seal.v1.json"
MODAL_REMOTE_RUN_RESERVATION_ROOT = PurePosixPath(
    "outputs/readiness/modal_only_final/modal_remote_run_reservations"
)
MODAL_LOCAL_CONTAINMENT_ROOT = PurePosixPath(
    "outputs/readiness/.modal_local_containment"
)
MODAL_LOCAL_HOST_ANCHOR_PATH = MODAL_LOCAL_CONTAINMENT_ROOT / "host_anchor.json"
MODAL_LOCAL_PROCESS_START_ROOT = MODAL_LOCAL_CONTAINMENT_ROOT / "process_starts"
PROVIDER_CANARY_AGGREGATE_OUTCOME_SCHEMA_NAME = "ProviderCanaryAggregateOutcomeReceipt"
PROVIDER_CANARY_AGGREGATE_OUTCOME_SCHEMA_VERSION = "1.1"


_RUN_ID = re.compile(r"\A[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_ACTION_ATTEMPT_ID = re.compile(r"\A[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
ARTIFACT_MANIFEST_FILENAMES = (
    "artifact_manifest.json",
    "artifact_manifest.checkpoint.json",
)
_SAFE_COMPONENT = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
_SAFE_PATH_COMPONENT = re.compile(r"\A[a-zA-Z0-9_.-]{1,128}\Z")
_ALLOWED_SOURCE_SUFFIXES = frozenset(
    {
        ".json",
        ".lock",
        ".md",
        ".patch",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
MAX_IMAGE_SOURCE_FILE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_SOURCE_TOTAL_BYTES = 32 * 1024 * 1024
MAX_ARTIFACT_DOWNLOAD_FILE_BYTES = 64 * 1024 * 1024
MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ARTIFACT_MANIFEST_BYTES = 2 * 1024 * 1024
_SENSITIVE_SOURCE_STEMS = frozenset(
    {
        "api-key",
        "api_key",
        "apikey",
        "credential",
        "credentials",
        "private-key",
        "private_key",
        "secret",
        "secrets",
        "token",
        "tokens",
    }
)
_HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(
        rb"(?<![A-Za-z0-9_-])(?:gh[pousr]_[A-Za-z0-9]{20,}"
        rb"|github_pat_[A-Za-z0-9_]{20,})"
    ),
    re.compile(rb"(?<![A-Za-z0-9_-])hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"(?<![A-Za-z0-9_-])a[ks]-[A-Za-z0-9_-]{20,}"),
    re.compile(
        rb"(?<![A-Za-z0-9_-])(?:tinker|tml)[_-][A-Za-z0-9_-]{20,}",
        re.IGNORECASE,
    ),
)
_ALLOWED_ROOT_FILES = (
    "experiment_manifest.yaml",
    "modal_action_journal.py",
    "modal_app.py",
    "modal_boundary.py",
    "modal_image_build.py",
    "exploratory_pilot.py",
    "pyproject.toml",
    "scientific_decisions.yaml",
    "uv.lock",
    "vendor/openevolve/README.md",
    "vendor/openevolve/pyproject.toml",
)
IMAGE_SOURCE_DIRECTORIES = (
    "agents",
    "analysis",
    "architecture_ir",
    "artifacts",
    "audits",
    "baselines",
    "common",
    "configs",
    "containment",
    "evaluation",
    "mechanism",
    "novelty",
    "reconstruction",
    "replication",
    "reporting",
    "research_ledger",
    "review",
    "sealed_eval",
    "scripts",
    "study",
    "vendor/openevolve/openevolve",
    "vendor_patches",
)
_CODE_ONLY_SOURCE_DIRECTORIES = frozenset({"sealed_eval"})
_FORBIDDEN_COMPONENTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        ".venv-tinker",
        "__pycache__",
        "checkpoints",
        "custody",
        "logs",
        "outputs",
        "private_eval",
        "tests",
    }
)
_TRANSIENT_ARTIFACT_NAMES = frozenset(
    {
        "artifact_manifest.json",
        "artifact_manifest.checkpoint.json",
        ".candidate_training.lock",
        ".study_accelerator.lock",
    }
)


class ModalBoundaryError(RuntimeError):
    pass


class ArtifactIntegrityError(ModalBoundaryError):
    pass


@dataclass(frozen=True, slots=True)
class ModalLiveCohortIdentity:
    """Immutable source, image, and operator cohort identity for live evidence."""

    source_tree_sha256: str
    image_source_sha256: str
    cohort_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_tree_sha256, str)
            or _SHA256.fullmatch(self.source_tree_sha256) is None
        ):
            raise ValueError("source-tree digest must be a lowercase SHA-256")
        if (
            not isinstance(self.image_source_sha256, str)
            or _SHA256.fullmatch(self.image_source_sha256) is None
        ):
            raise ValueError("image-source digest must be a lowercase SHA-256")
        validate_run_id(self.cohort_id)


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    gpu: str | None
    provider_secret: bool
    cpu_request_cores: float = FUNCTION_CPU_REQUEST_CORES
    cpu_soft_limit_cores: float = FUNCTION_CPU_SOFT_LIMIT_CORES
    memory_request_mib: int = FUNCTION_MEMORY_REQUEST_MIB
    memory_limit_mib: int = FUNCTION_MEMORY_LIMIT_MIB
    region: str | None = None
    timeout_seconds: int = FUNCTION_TIMEOUT_SECONDS
    max_containers: int = MAX_CONTAINERS
    min_containers: int = MIN_CONTAINERS
    retries: int = FUNCTION_RETRIES
    volume_mount_path: str = str(VOLUME_MOUNT_PATH)

    def __post_init__(self) -> None:
        if _SAFE_COMPONENT.fullmatch(self.name) is None:
            raise ValueError("function name is unsafe")
        if self.gpu not in {None, GPU_TYPE}:
            raise ValueError(f"only {GPU_TYPE!r} or no GPU is permitted")
        if (
            self.cpu_request_cores != FUNCTION_CPU_REQUEST_CORES
            or self.cpu_soft_limit_cores != FUNCTION_CPU_SOFT_LIMIT_CORES
            or self.memory_request_mib != FUNCTION_MEMORY_REQUEST_MIB
            or self.memory_limit_mib != FUNCTION_MEMORY_LIMIT_MIB
        ):
            raise ValueError(
                "Modal CPU request/soft throttle and memory request/hard limit "
                "are frozen"
            )
        if (
            self.cpu_request_cores != self.cpu_soft_limit_cores
            or self.memory_request_mib != self.memory_limit_mib
        ):
            raise ValueError(
                "Modal Function CPU soft and memory hard limits must equal requests"
            )
        if self.region is not None:
            raise ValueError("Modal Functions must retain base-rate region selection")
        if self.timeout_seconds != 300:
            raise ValueError("Modal functions must retain the five-minute timeout")
        if (self.max_containers, self.min_containers, self.retries) != (1, 0, 0):
            raise ValueError("Modal concurrency and retry ceilings are frozen")
        if self.volume_mount_path != str(VOLUME_MOUNT_PATH):
            raise ValueError("Modal functions must use the single documented mount")


FUNCTION_SPECS: Mapping[str, FunctionSpec] = {
    "offline_smoke": FunctionSpec("offline_smoke", None, False),
    "cuda_environment": FunctionSpec("cuda_environment", GPU_TYPE, False),
    "candidate_smoke": FunctionSpec("candidate_smoke", GPU_TYPE, False),
    "checkpoint_resume": FunctionSpec("checkpoint_resume", GPU_TYPE, False),
    "canary_greedy_autoresearch": FunctionSpec(
        "canary_greedy_autoresearch", GPU_TYPE, True
    ),
    "canary_semantic_autoresearch": FunctionSpec(
        "canary_semantic_autoresearch", GPU_TYPE, True
    ),
    "canary_openevolve_generic": FunctionSpec(
        "canary_openevolve_generic", GPU_TYPE, True
    ),
    "canary_openevolve_semantic": FunctionSpec(
        "canary_openevolve_semantic", GPU_TYPE, True
    ),
    "artifact_verify": FunctionSpec("artifact_verify", None, False),
}

# Kept outside ``FUNCTION_SPECS`` for backward-compatible plan/receipt schemas:
# the historical provider-name roster is intentionally immutable.  The
# exploratory lane has its own explicit provider-bearing function contract.
EXPLORATORY_FUNCTION_SPEC = FunctionSpec(
    "exploratory_c0c3_pilot", GPU_TYPE, True
)


def function_spec(name: str) -> FunctionSpec:
    if name == EXPLORATORY_FUNCTION_SPEC.name:
        return EXPLORATORY_FUNCTION_SPEC
    return FUNCTION_SPECS[name]


@dataclass(frozen=True)
class SourceFileV1:
    relative_path: str
    sha256: str
    size_bytes: int

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"relative_path", "sha256", "size_bytes"}
    )

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path)
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("source-file digest must be a lowercase SHA-256")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("source-file size must be a non-negative integer")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class ImageSourceManifestV1:
    dependency_lock_sha256: str
    files: tuple[SourceFileV1, ...]
    python_version: str = PYTHON_VERSION
    uv_version: str = UV_VERSION
    modal_version: str = MODAL_VERSION
    recipe_version: str = IMAGE_RECIPE_VERSION

    SCHEMA_NAME: ClassVar[str] = "ModalImageSourceManifest"
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.dependency_lock_sha256) is None:
            raise ValueError("dependency lock digest must be a lowercase SHA-256")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("image source paths must be sorted and unique")
        if self.python_version != PYTHON_VERSION:
            raise ValueError("image Python version differs from the frozen recipe")
        if self.uv_version != UV_VERSION or self.modal_version != MODAL_VERSION:
            raise ValueError("image tool versions differ from the frozen recipe")
        if self.recipe_version != IMAGE_RECIPE_VERSION:
            raise ValueError("unsupported image recipe version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "recipe_version": self.recipe_version,
            "python_version": self.python_version,
            "uv_version": self.uv_version,
            "modal_version": self.modal_version,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "files": [item.to_dict() for item in self.files],
        }

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ImageSourceSnapshotFile:
    """One immutable in-memory image-source file captured from a stable FD."""

    relative_path: str
    payload: bytes

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path)
        if type(self.payload) is not bytes:
            raise TypeError("image-source snapshot payload must be exact bytes")

    @property
    def source_file(self) -> SourceFileV1:
        return SourceFileV1(
            relative_path=self.relative_path,
            sha256=hashlib.sha256(self.payload).hexdigest(),
            size_bytes=len(self.payload),
        )


@dataclass(frozen=True)
class ImageSourceSnapshot:
    """Exact bytes used for secret scanning, hashing, and Modal staging."""

    manifest: ImageSourceManifestV1
    files: tuple[ImageSourceSnapshotFile, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ImageSourceManifestV1):
            raise TypeError("image-source snapshot manifest has the wrong type")
        if not all(isinstance(item, ImageSourceSnapshotFile) for item in self.files):
            raise TypeError("image-source snapshot files have the wrong type")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("image-source snapshot paths must be sorted and unique")
        total_bytes = 0
        for item in self.files:
            relative = safe_relative_path(item.relative_path)
            if len(item.payload) > MAX_IMAGE_SOURCE_FILE_BYTES:
                raise ValueError("image-source snapshot file exceeds its byte cap")
            total_bytes += len(item.payload)
            _validate_image_source_name(relative)
            _validate_image_source_payload(relative, item.payload)
        if total_bytes > MAX_IMAGE_SOURCE_TOTAL_BYTES:
            raise ValueError("image-source snapshot exceeds its total byte cap")
        source_files = tuple(item.source_file for item in self.files)
        if source_files != self.manifest.files:
            raise ValueError("image-source snapshot bytes differ from its manifest")
        dependency_locks = tuple(
            item for item in source_files if item.relative_path == "uv.lock"
        )
        if len(dependency_locks) != 1:
            raise ValueError("image-source snapshot requires exactly one uv.lock")
        if self.manifest.dependency_lock_sha256 != dependency_locks[0].sha256:
            raise ValueError("image-source snapshot lock digest differs from uv.lock")


@dataclass(frozen=True)
class ArtifactFileV1:
    relative_path: str
    sha256: str
    size_bytes: int

    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"relative_path", "sha256", "size_bytes"}
    )

    def __post_init__(self) -> None:
        safe_relative_path(self.relative_path)
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("artifact digest must be a lowercase SHA-256")
        if (
            not isinstance(self.size_bytes, int)
            or isinstance(self.size_bytes, bool)
            or self.size_bytes < 0
        ):
            raise ValueError("artifact size must be a non-negative integer")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactFileV1:
        if not isinstance(payload, Mapping) or set(payload) != cls.FIELDS:
            raise ValueError("artifact entry has unexpected or missing fields")
        relative_path = payload["relative_path"]
        sha256 = payload["sha256"]
        size_bytes = payload["size_bytes"]
        if not isinstance(relative_path, str) or not isinstance(sha256, str):
            raise ValueError("artifact path and digest must be text")
        return cls(
            relative_path=relative_path,
            sha256=sha256,
            size_bytes=size_bytes,
        )


@dataclass(frozen=True)
class ArtifactManifestV1:
    run_id: str
    created_at_utc: str
    image_source_sha256: str
    files: tuple[ArtifactFileV1, ...]

    SCHEMA_NAME: ClassVar[str] = "ModalRunArtifactManifest"
    SCHEMA_VERSION: ClassVar[str] = "1.0"
    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_name",
            "schema_version",
            "run_id",
            "created_at_utc",
            "image_source_sha256",
            "files",
        }
    )

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        try:
            created_at = datetime.fromisoformat(
                self.created_at_utc.replace("Z", "+00:00")
            )
        except (TypeError, ValueError) as error:
            raise ValueError("created_at_utc must be an ISO-8601 timestamp") from error
        if created_at.tzinfo is None:
            raise ValueError("created_at_utc must include a timezone")
        if _SHA256.fullmatch(self.image_source_sha256) is None:
            raise ValueError("image source digest must be a lowercase SHA-256")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be sorted and unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "image_source_sha256": self.image_source_sha256,
            "files": [item.to_dict() for item in self.files],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactManifestV1:
        if not isinstance(payload, Mapping) or set(payload) != cls.FIELDS:
            raise ValueError("artifact manifest has unexpected or missing fields")
        if payload["schema_name"] != cls.SCHEMA_NAME:
            raise ValueError("expected ModalRunArtifactManifest schema")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported artifact manifest version")
        raw_files = payload["files"]
        if not isinstance(raw_files, list):
            raise ValueError("artifact manifest files must be a list")
        run_id = payload["run_id"]
        created_at_utc = payload["created_at_utc"]
        image_source_sha256 = payload["image_source_sha256"]
        if not all(
            isinstance(value, str)
            for value in (run_id, created_at_utc, image_source_sha256)
        ):
            raise ValueError("artifact manifest identity fields must be text")
        return cls(
            run_id=run_id,
            created_at_utc=created_at_utc,
            image_source_sha256=image_source_sha256,
            files=tuple(ArtifactFileV1.from_dict(item) for item in raw_files),
        )

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_modal_action_identity(
    *,
    action: object,
    run_id: object,
    source_run_id: object = None,
    verifier_run_id: object = None,
    harness: object = None,
) -> tuple[str, str, str | None, str | None, str | None]:
    """Validate the action-specific run identity shared by producer and reader."""

    if not isinstance(action, str) or action not in MODAL_ACTIONS:
        raise ValueError("Modal CLI action is unsupported")
    selected_run_id = validate_run_id(run_id)
    selected_source = (
        validate_run_id(source_run_id) if source_run_id is not None else None
    )
    selected_verifier = (
        validate_run_id(verifier_run_id) if verifier_run_id is not None else None
    )
    if harness is not None and (
        not isinstance(harness, str) or harness not in CANARY_ORDER
    ):
        raise ValueError("Modal CLI harness is unsupported")
    selected_harness = harness

    if action == "checkpoint-resume":
        if (
            selected_source is None
            or selected_verifier is not None
            or selected_harness is not None
        ):
            raise ValueError("checkpoint-resume Modal CLI identity is incomplete")
        if selected_source == selected_run_id:
            raise ValueError(
                "checkpoint-resume source and destination run IDs must differ"
            )
    elif action in {"download", "verify"}:
        if (
            selected_verifier is None
            or selected_source is not None
            or selected_harness is not None
        ):
            raise ValueError("verifier Modal CLI identity is incomplete")
        if selected_verifier == selected_run_id:
            raise ValueError("verifier source and destination run IDs must differ")
    elif action == "canary":
        if (
            selected_harness is None
            or selected_source is not None
            or selected_verifier is not None
        ):
            raise ValueError("single-canary Modal CLI identity is incomplete")
        suffix = canary_run_suffix(selected_harness)
        if selected_run_id != (
            f"{selected_run_id.removesuffix('-' + suffix)}-{suffix}"
        ):
            raise ValueError(
                "single-canary run ID lacks its exact harness-specific suffix"
            )
    elif any(
        value is not None
        for value in (selected_source, selected_verifier, selected_harness)
    ):
        raise ValueError("Modal CLI action contains unrelated identity fields")
    return (
        action,
        selected_run_id,
        selected_source,
        selected_verifier,
        selected_harness,
    )


def build_modal_cli_command(
    *,
    python_executable: str | Path,
    project_root: str | Path,
    action: str,
    run_id: str,
    source_run_id: str | None = None,
    verifier_run_id: str | None = None,
    harness: str | None = None,
    source_tree_sha256: str,
    cohort_id: str,
    image_source_sha256: str,
    provider_approved: bool,
) -> tuple[str, ...]:
    """Reconstruct the one canonical paid ``modal run`` argv.

    This pure helper is shared by the launcher producer and the independent
    receipt consumer.  Download output is intentionally fixed because it is
    part of the readiness artifact namespace and otherwise cannot be recovered
    from the sanitized journal schema.
    """

    (
        action,
        selected_run_id,
        selected_source,
        selected_verifier,
        harness,
    ) = validate_modal_action_identity(
        action=action,
        run_id=run_id,
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        harness=harness,
    )
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256,
        image_source_sha256=image_source_sha256,
        cohort_id=cohort_id,
    )
    if action == "cuda-environment" and identity.cohort_id != selected_run_id:
        raise ValueError("first CUDA environment run ID must equal the cohort ID")
    if type(provider_approved) is not bool:
        raise TypeError("Modal CLI provider approval must be boolean")
    if provider_approved is not (
        action in {"canary", "canaries", "exploratory_c0c3_pilot"}
    ):
        raise ValueError("Modal CLI provider approval differs from its action")

    root = Path(os.path.abspath(os.fspath(project_root)))
    executable = Path(os.path.abspath(os.fspath(python_executable))).with_name("modal")
    command = [
        str(executable),
        "run",
        "--env",
        MODAL_ENVIRONMENT_NAME,
        str(root / "modal_app.py"),
        "--action",
        action,
        "--run-id",
        selected_run_id,
        "--source-tree-sha256",
        identity.source_tree_sha256,
        "--cohort-id",
        identity.cohort_id,
    ]
    if selected_source is not None:
        command.extend(("--source-run-id", selected_source))
    if selected_verifier is not None:
        command.extend(("--verifier-run-id", selected_verifier))
    if harness is not None:
        command.extend(("--harness", harness))
    if action == "download":
        command.extend(("--local-output", MODAL_DOWNLOAD_OUTPUT_ROOT))
    command.extend(
        (
            "--expected-image-source-sha256",
            image_source_sha256,
            "--approved",
        )
    )
    if provider_approved:
        command.append("--provider-approved")
    return tuple(command)


def modal_cli_command_sha256(**kwargs: Any) -> str:
    command = build_modal_cli_command(**kwargs)
    encoded = json.dumps(
        list(command),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_artifact_manifest_filename(filename: str) -> str:
    if filename not in ARTIFACT_MANIFEST_FILENAMES:
        raise ValueError("artifact manifest filename is unsafe")
    return filename


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ArtifactIntegrityError(
                f"artifact manifest contains duplicate JSON key: {key}"
            )
        payload[key] = value
    return payload


def parse_artifact_manifest_bytes(raw_bytes: bytes) -> ArtifactManifestV1:
    """Parse one size-bounded manifest without accepting duplicate JSON keys."""

    if type(raw_bytes) is not bytes:
        raise TypeError("raw artifact manifest must be bytes")
    if len(raw_bytes) > MAX_ARTIFACT_MANIFEST_BYTES:
        raise ArtifactIntegrityError("artifact manifest exceeds the 2 MiB limit")
    try:
        source = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactIntegrityError("artifact manifest is not UTF-8") from error
    try:
        payload = json.loads(source, object_pairs_hook=_reject_duplicate_json_keys)
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError("artifact manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ArtifactIntegrityError("artifact manifest is not an object")
    try:
        return ArtifactManifestV1.from_dict(payload)
    except (TypeError, ValueError) as error:
        raise ArtifactIntegrityError(str(error)) from error


@dataclass(frozen=True)
class RawArtifactManifestV1:
    """The exact selected manifest bytes and their parsed canonical meaning."""

    filename: str
    raw_bytes: bytes
    manifest: ArtifactManifestV1

    def __post_init__(self) -> None:
        validate_artifact_manifest_filename(self.filename)
        if type(self.raw_bytes) is not bytes:
            raise TypeError("raw artifact manifest must be bytes")
        parsed = parse_artifact_manifest_bytes(self.raw_bytes)
        if parsed != self.manifest:
            raise ArtifactIntegrityError(
                "raw artifact manifest differs from its parsed manifest"
            )

    @classmethod
    def from_bytes(
        cls,
        *,
        filename: str,
        raw_bytes: bytes,
    ) -> RawArtifactManifestV1:
        return cls(
            filename=filename,
            raw_bytes=raw_bytes,
            manifest=parse_artifact_manifest_bytes(raw_bytes),
        )

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    @property
    def raw_size_bytes(self) -> int:
        return len(self.raw_bytes)


@dataclass(frozen=True)
class ArtifactVerificationV1:
    """Exact remote proof for one raw source-manifest verification."""

    source_run_id: str
    verifier_run_id: str
    manifest_filename: str
    raw_manifest_sha256: str
    raw_manifest_size_bytes: int
    canonical_manifest_sha256: str
    file_count: int
    verifier_execution_context: ExecutionContextV1
    verified: bool = True

    SCHEMA_NAME: ClassVar[str] = "ModalArtifactVerificationResult"
    SCHEMA_VERSION: ClassVar[str] = "1.0"
    FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "schema_name",
            "schema_version",
            "source_run_id",
            "verifier_run_id",
            "manifest_filename",
            "raw_manifest_sha256",
            "raw_manifest_size_bytes",
            "canonical_manifest_sha256",
            "file_count",
            "verifier_execution_context",
            "verified",
        }
    )

    def __post_init__(self) -> None:
        validate_run_id(self.source_run_id)
        validate_run_id(self.verifier_run_id)
        if self.source_run_id == self.verifier_run_id:
            raise ValueError("verifier run ID must differ from the source run ID")
        validate_artifact_manifest_filename(self.manifest_filename)
        for digest in (
            self.raw_manifest_sha256,
            self.canonical_manifest_sha256,
        ):
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError("verification digests must be lowercase SHA-256")
        for value, field in (
            (self.raw_manifest_size_bytes, "raw_manifest_size_bytes"),
            (self.file_count, "file_count"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
        if self.raw_manifest_size_bytes > MAX_ARTIFACT_MANIFEST_BYTES:
            raise ValueError("raw manifest size exceeds the 2 MiB limit")
        if type(self.verified) is not bool or self.verified is not True:
            raise ValueError("artifact verification must be exactly true")
        context = self.verifier_execution_context
        if (
            not isinstance(context, ExecutionContextV1)
            or context.execution_backend != "modal"
            or context.run_id != self.verifier_run_id
            or context.app_name != APP_NAME
            or context.function_name != "artifact_verify"
            or context.artifact_uri != volume_artifact_uri(self.source_run_id)
            or context.modal_app_id is None
            or context.modal_function_id is None
            or context.modal_call_id is None
            or context.modal_image_id is None
            or context.image_source_sha256 is None
        ):
            raise ValueError("verifier execution context is not source-run bound")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "source_run_id": self.source_run_id,
            "verifier_run_id": self.verifier_run_id,
            "manifest_filename": self.manifest_filename,
            "raw_manifest_sha256": self.raw_manifest_sha256,
            "raw_manifest_size_bytes": self.raw_manifest_size_bytes,
            "canonical_manifest_sha256": self.canonical_manifest_sha256,
            "file_count": self.file_count,
            "verifier_execution_context": self.verifier_execution_context.to_dict(),
            "verified": self.verified,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactVerificationV1:
        if not isinstance(payload, Mapping) or set(payload) != cls.FIELDS:
            raise ValueError("artifact verification has unexpected or missing fields")
        if payload["schema_name"] != cls.SCHEMA_NAME:
            raise ValueError("expected ModalArtifactVerificationResult schema")
        if payload["schema_version"] != cls.SCHEMA_VERSION:
            raise ValueError("unsupported artifact verification version")
        text_fields = (
            "source_run_id",
            "verifier_run_id",
            "manifest_filename",
            "raw_manifest_sha256",
            "canonical_manifest_sha256",
        )
        if any(not isinstance(payload[field], str) for field in text_fields):
            raise ValueError("artifact verification identity fields must be text")
        raw_context = payload["verifier_execution_context"]
        if not isinstance(raw_context, Mapping):
            raise ValueError("verifier execution context must be an object")
        return cls(
            source_run_id=payload["source_run_id"],
            verifier_run_id=payload["verifier_run_id"],
            manifest_filename=payload["manifest_filename"],
            raw_manifest_sha256=payload["raw_manifest_sha256"],
            raw_manifest_size_bytes=payload["raw_manifest_size_bytes"],
            canonical_manifest_sha256=payload["canonical_manifest_sha256"],
            file_count=payload["file_count"],
            verifier_execution_context=ExecutionContextV1.from_dict(raw_context),
            verified=payload["verified"],
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be 1-63 lowercase letters, digits, or hyphens")
    return run_id


def _validate_modal_action_attempt_id(attempt_id: str) -> str:
    if (
        not isinstance(attempt_id, str)
        or _ACTION_ATTEMPT_ID.fullmatch(attempt_id) is None
    ):
        raise ValueError("Modal action attempt ID must be 32 lowercase hex digits")
    return attempt_id


def modal_live_cohort_root(identity: ModalLiveCohortIdentity) -> PurePosixPath:
    if not isinstance(identity, ModalLiveCohortIdentity):
        raise TypeError("Modal live cohort identity has the wrong type")
    return (
        MODAL_LIVE_COHORT_ROOT
        / identity.source_tree_sha256
        / identity.image_source_sha256
        / identity.cohort_id
    )


def modal_action_attempt_directory(
    identity: ModalLiveCohortIdentity,
) -> PurePosixPath:
    return modal_live_cohort_root(identity) / "action_attempts"


def modal_migration_lineage_path(
    identity: ModalLiveCohortIdentity,
) -> PurePosixPath:
    """Return the immutable terminal seal for one finalized live cohort."""

    return modal_live_cohort_root(identity) / "migration_lineage.v1.1.json"


def modal_action_intent_receipt_path(
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> PurePosixPath:
    return modal_action_attempt_directory(identity) / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.intent.json"
    )


def modal_action_terminal_receipt_path(
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> PurePosixPath:
    return modal_action_attempt_directory(identity) / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.json"
    )


def modal_action_recovery_directory(
    identity: ModalLiveCohortIdentity,
) -> PurePosixPath:
    """Return the immutable recovery journal for one live cohort."""

    return modal_live_cohort_root(identity) / "action_recoveries"


def modal_action_recovery_intent_path(
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> PurePosixPath:
    return modal_action_recovery_directory(identity) / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.intent.v1.0.json"
    )


def modal_action_host_containment_path(
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> PurePosixPath:
    return modal_action_recovery_directory(identity) / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.host-containment.v1.0.json"
    )


def modal_action_recovery_resolution_path(
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> PurePosixPath:
    return modal_action_recovery_directory(identity) / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.resolution.v1.0.json"
    )


def modal_launch_rejection_receipt_path(attempt_id: str) -> PurePosixPath:
    return MODAL_LAUNCH_REJECTION_ROOT / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.json"
    )


def modal_global_launch_rejection_seal_path() -> PurePosixPath:
    """Return the create-only closure seal for the global rejection roster."""

    return MODAL_GLOBAL_LAUNCH_REJECTION_SEAL_PATH


def modal_remote_run_reservation_path(run_id: str) -> PurePosixPath:
    """Return the global create-only reservation for one Volume run leaf."""

    return MODAL_REMOTE_RUN_RESERVATION_ROOT / f"{validate_run_id(run_id)}.json"


def modal_local_host_anchor_path() -> PurePosixPath:
    """Return the private local anchor used by crash-containment evidence."""

    return MODAL_LOCAL_HOST_ANCHOR_PATH


def modal_local_process_start_receipt_path(attempt_id: str) -> PurePosixPath:
    """Return the globally unique local process-start marker for one attempt."""

    return MODAL_LOCAL_PROCESS_START_ROOT / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.json"
    )


def _modal_artifact_verification_root(
    identity: ModalLiveCohortIdentity,
    source_run_id: str,
    verifier_run_id: str,
    attempt_id: str,
) -> PurePosixPath:
    source = validate_run_id(source_run_id)
    verifier = validate_run_id(verifier_run_id)
    if source == verifier:
        raise ValueError("artifact verifier run ID must differ from its source")
    return (
        modal_live_cohort_root(identity)
        / "artifact_verifications"
        / source
        / verifier
        / _validate_modal_action_attempt_id(attempt_id)
    )


def modal_remote_verification_receipt_path(
    identity: ModalLiveCohortIdentity,
    source_run_id: str,
    verifier_run_id: str,
    attempt_id: str,
) -> PurePosixPath:
    return (
        _modal_artifact_verification_root(
            identity,
            source_run_id,
            verifier_run_id,
            attempt_id,
        )
        / "remote_verification.json"
    )


def modal_artifact_verifier_capture_parent_path(
    identity: ModalLiveCohortIdentity,
    source_run_id: str,
    verifier_run_id: str,
    attempt_id: str,
) -> PurePosixPath:
    return (
        _modal_artifact_verification_root(
            identity,
            source_run_id,
            verifier_run_id,
            attempt_id,
        )
        / "volume_capture"
    )


def modal_artifact_verifier_capture_directory_path(
    identity: ModalLiveCohortIdentity,
    source_run_id: str,
    verifier_run_id: str,
    attempt_id: str,
) -> PurePosixPath:
    return modal_artifact_verifier_capture_parent_path(
        identity,
        source_run_id,
        verifier_run_id,
        attempt_id,
    ) / validate_run_id(verifier_run_id)


def canary_run_suffix(harness: str) -> str:
    if harness not in CANARY_ORDER:
        raise ValueError("canary harness is not in the frozen order")
    return harness.replace("_autoresearch", "-ar").replace("_", "-")


def provider_canary_aggregate_outcome_receipt_path(
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> PurePosixPath:
    return modal_action_attempt_directory(identity) / (
        f"{_validate_modal_action_attempt_id(attempt_id)}.aggregate.json"
    )


def validate_provider_canary_aggregate_outcome_receipt(
    payload: Mapping[str, Any],
    *,
    expected_attempt_id: str | None = None,
    expected_run_id_prefix: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_image_source_sha256: str | None = None,
    expected_cohort_id: str | None = None,
) -> dict[str, Any]:
    fields = {
        "schema_name",
        "schema_version",
        "attempt_id",
        "run_id_prefix",
        "source_tree_sha256",
        "image_source_sha256",
        "cohort_id",
        "harness_order",
        "outcomes",
        "all_succeeded",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("provider canary aggregate outcome has the wrong schema")
    if (
        payload["schema_name"] != PROVIDER_CANARY_AGGREGATE_OUTCOME_SCHEMA_NAME
        or payload["schema_version"] != PROVIDER_CANARY_AGGREGATE_OUTCOME_SCHEMA_VERSION
    ):
        raise ValueError("provider canary aggregate outcome identity changed")
    attempt_id = payload["attempt_id"]
    _validate_modal_action_attempt_id(attempt_id)
    run_id_prefix = validate_run_id(payload["run_id_prefix"])
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=payload["source_tree_sha256"],
        image_source_sha256=payload["image_source_sha256"],
        cohort_id=payload["cohort_id"],
    )
    if payload["harness_order"] != list(CANARY_ORDER):
        raise ValueError("provider canary aggregate outcome order changed")
    outcomes = payload["outcomes"]
    if not isinstance(outcomes, list) or len(outcomes) != len(CANARY_ORDER):
        raise ValueError("provider canary aggregate outcomes are incomplete")
    outcome_fields = {"harness", "run_id", "status", "error_type"}
    for harness, outcome in zip(CANARY_ORDER, outcomes, strict=True):
        if (
            not isinstance(outcome, Mapping)
            or set(outcome) != outcome_fields
            or outcome["harness"] != harness
            or outcome["run_id"] != f"{run_id_prefix}-{canary_run_suffix(harness)}"
            or outcome["status"] not in {"success", "failed"}
        ):
            raise ValueError("provider canary aggregate outcome record is invalid")
        if outcome["status"] == "success":
            if outcome["error_type"] is not None:
                raise ValueError("successful aggregate outcome contains an error")
        elif (
            not isinstance(outcome["error_type"], str)
            or not outcome["error_type"].isidentifier()
            or len(outcome["error_type"]) > 128
        ):
            raise ValueError("failed aggregate outcome has an unsafe error class")
    all_succeeded = payload["all_succeeded"]
    observed_all_succeeded = all(outcome["status"] == "success" for outcome in outcomes)
    if type(all_succeeded) is not bool or all_succeeded is not observed_all_succeeded:
        raise ValueError("provider canary aggregate outcome status is inconsistent")
    for expected, observed, label in (
        (expected_attempt_id, attempt_id, "attempt ID"),
        (expected_run_id_prefix, run_id_prefix, "run prefix"),
        (
            expected_source_tree_sha256,
            identity.source_tree_sha256,
            "source-tree SHA-256",
        ),
        (
            expected_image_source_sha256,
            identity.image_source_sha256,
            "image source SHA-256",
        ),
        (expected_cohort_id, identity.cohort_id, "cohort ID"),
    ):
        if expected is not None and observed != expected:
            raise ValueError(f"provider canary aggregate outcome {label} differs")
    return dict(payload)


def build_provider_canary_aggregate_outcome_receipt(
    aggregate_result: Mapping[str, Any],
    *,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> dict[str, Any]:
    if not isinstance(identity, ModalLiveCohortIdentity):
        raise TypeError("provider aggregate requires a Modal live cohort identity")
    aggregate_fields = {
        "schema_name",
        "schema_version",
        "run_id_prefix",
        "harness_order",
        "outcomes",
        "all_succeeded",
    }
    if not isinstance(aggregate_result, Mapping) or set(aggregate_result) != (
        aggregate_fields
    ):
        raise ValueError("provider canary aggregate result has the wrong schema")
    if (
        aggregate_result["schema_name"] != "ModalProviderCanaryAggregateResult"
        or aggregate_result["schema_version"] != "1.0"
        or aggregate_result["harness_order"] != list(CANARY_ORDER)
    ):
        raise ValueError("provider canary aggregate result identity changed")
    run_id_prefix = validate_run_id(aggregate_result["run_id_prefix"])
    raw_outcomes = aggregate_result["outcomes"]
    if not isinstance(raw_outcomes, list) or len(raw_outcomes) != len(CANARY_ORDER):
        raise ValueError("provider canary aggregate result outcomes are invalid")
    outcomes: list[dict[str, Any]] = []
    expected_fields = {"harness", "run_id", "status", "result", "error_type"}
    for harness, outcome in zip(CANARY_ORDER, raw_outcomes, strict=True):
        if (
            not isinstance(outcome, Mapping)
            or set(outcome) != expected_fields
            or outcome["harness"] != harness
            or outcome["run_id"] != f"{run_id_prefix}-{canary_run_suffix(harness)}"
            or outcome["status"] not in {"success", "failed"}
            or (outcome["status"] == "failed") != (outcome["result"] is None)
        ):
            raise ValueError("provider canary aggregate result outcome changed")
        if outcome["status"] == "success":
            if outcome["error_type"] is not None:
                raise ValueError("successful canary result contains an error")
        elif (
            not isinstance(outcome["error_type"], str)
            or not outcome["error_type"].isidentifier()
            or len(outcome["error_type"]) > 128
        ):
            raise ValueError("failed canary result has an unsafe error class")
        outcomes.append(
            {
                "harness": outcome["harness"],
                "run_id": outcome["run_id"],
                "status": outcome["status"],
                "error_type": outcome["error_type"],
            }
        )
    observed_all_succeeded = all(
        outcome["status"] == "success" for outcome in raw_outcomes
    )
    if (
        type(aggregate_result["all_succeeded"]) is not bool
        or aggregate_result["all_succeeded"] is not observed_all_succeeded
    ):
        raise ValueError("provider canary aggregate result status is inconsistent")
    payload = {
        "schema_name": PROVIDER_CANARY_AGGREGATE_OUTCOME_SCHEMA_NAME,
        "schema_version": PROVIDER_CANARY_AGGREGATE_OUTCOME_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "run_id_prefix": run_id_prefix,
        "source_tree_sha256": identity.source_tree_sha256,
        "image_source_sha256": identity.image_source_sha256,
        "cohort_id": identity.cohort_id,
        "harness_order": list(CANARY_ORDER),
        "outcomes": outcomes,
        "all_succeeded": aggregate_result["all_succeeded"],
    }
    return validate_provider_canary_aggregate_outcome_receipt(
        payload,
        expected_attempt_id=attempt_id,
        expected_run_id_prefix=run_id_prefix,
        expected_source_tree_sha256=identity.source_tree_sha256,
        expected_image_source_sha256=identity.image_source_sha256,
        expected_cohort_id=identity.cohort_id,
    )


def new_run_id(prefix: str = "modal") -> str:
    if _RUN_ID.fullmatch(prefix) is None or len(prefix) > 32:
        raise ValueError("run ID prefix is unsafe or too long")
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return validate_run_id(f"{prefix}-{timestamp}-{secrets.token_hex(4)}")


def safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("relative path is empty or contains forbidden characters")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("path must be normalized, relative, and non-traversing")
    if not all(_SAFE_PATH_COMPONENT.fullmatch(part) for part in path.parts):
        raise ValueError("relative path contains an unsafe component")
    return path


def volume_run_path(run_id: str) -> PurePosixPath:
    return VOLUME_RUNS_PATH / validate_run_id(run_id)


def volume_object_path(run_id: str, relative_path: str) -> str:
    relative = safe_relative_path(relative_path)
    return str(PurePosixPath("/runs") / validate_run_id(run_id) / relative)


def volume_artifact_uri(run_id: str) -> str:
    return f"volume://{VOLUME_NAME}/runs/{validate_run_id(run_id)}"


def _open_directory_no_follow(
    path: str | Path,
    *,
    label: str,
    dir_fd: int | None = None,
) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise ModalBoundaryError(
            "platform cannot enforce no-follow Volume directory creation"
        )
    flags = os.O_RDONLY | no_follow | directory_only | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, dir_fd=dir_fd)
    except OSError as error:
        raise ModalBoundaryError(f"{label} is unsafe") from error
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ModalBoundaryError(f"{label} is not a directory")
    return descriptor


def create_fresh_run_directory(
    mount_root: str | Path,
    run_id: str,
    *,
    allow_mount_root_symlink: bool = False,
) -> Path:
    validated = validate_run_id(run_id)
    if not isinstance(allow_mount_root_symlink, bool):
        raise TypeError("allow_mount_root_symlink must be a boolean")
    raw_root = Path(mount_root)
    if not raw_root.is_absolute():
        raise ModalBoundaryError("Volume mount root must be absolute")
    if ".." in raw_root.parts:
        raise ModalBoundaryError("Volume mount root may not contain traversal")
    if raw_root.is_symlink() and not allow_mount_root_symlink:
        raise ModalBoundaryError(
            "Volume mount root may not be a symlink without explicit trust"
        )
    try:
        # With the explicit opt-in, the mount point is the one trusted alias at
        # this boundary. Resolve it once and use only the pinned canonical
        # target below. Symlinks underneath that target remain forbidden.
        root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ModalBoundaryError(
            "Volume mount root is missing or cannot be resolved safely"
        ) from error
    if not root.is_dir():
        raise ModalBoundaryError("Volume mount root must resolve to a directory")
    root_descriptor = _open_directory_no_follow(
        root,
        label="resolved Volume mount root",
    )
    try:
        runs = root / "runs"
        if runs.is_symlink():
            raise ModalBoundaryError("Volume runs directory may not be a symlink")
        try:
            os.mkdir("runs", mode=0o700, dir_fd=root_descriptor)
        except FileExistsError:
            pass
        except OSError as error:
            raise ModalBoundaryError("Volume runs directory is unsafe") from error
        runs_descriptor = _open_directory_no_follow(
            "runs",
            label="Volume runs directory",
            dir_fd=root_descriptor,
        )
        try:
            if (
                runs.is_symlink()
                or not runs.is_dir()
                or runs.resolve(strict=True).parent != root
            ):
                raise ModalBoundaryError("Volume runs directory escaped its mount root")
            destination = runs / validated
            if destination.exists() or destination.is_symlink():
                raise ModalBoundaryError(
                    f"run directory already exists; refusing overwrite: {validated}"
                )
            try:
                os.mkdir(validated, mode=0o700, dir_fd=runs_descriptor)
            except FileExistsError as error:
                raise ModalBoundaryError(
                    f"run directory already exists; refusing overwrite: {validated}"
                ) from error
            except OSError as error:
                raise ModalBoundaryError(
                    "run directory could not be created safely"
                ) from error
            destination_descriptor = _open_directory_no_follow(
                validated,
                label="new Volume run directory",
                dir_fd=runs_descriptor,
            )
            os.close(destination_descriptor)
            if (
                destination.is_symlink()
                or not destination.is_dir()
                or destination.resolve(strict=True).parent != runs
            ):
                raise ModalBoundaryError(
                    "run directory escaped the Volume runs directory"
                )
            canonical_destination = destination.resolve(strict=True)
        finally:
            os.close(runs_descriptor)
    finally:
        os.close(root_descriptor)
    return canonical_destination


def resolve_existing_volume_run_directory(
    mount_root: str | Path,
    run_id: str,
    *,
    allow_mount_root_symlink: bool = False,
) -> Path:
    """Pin an existing run below one explicitly trusted Volume mount alias.

    Modal exposes a mounted Volume through a symlink in its container runtime.
    Only that caller-selected mount alias may be followed.  The resolved mount
    target, ``runs`` directory, and run directory are then opened with
    ``O_NOFOLLOW`` and identity-checked so no symlink below the trusted anchor
    can redirect an artifact read.
    """

    validated = validate_run_id(run_id)
    if not isinstance(allow_mount_root_symlink, bool):
        raise TypeError("allow_mount_root_symlink must be a boolean")
    raw_root = Path(mount_root)
    if not raw_root.is_absolute():
        raise ModalBoundaryError("Volume mount root must be absolute")
    if ".." in raw_root.parts:
        raise ModalBoundaryError("Volume mount root may not contain traversal")
    if raw_root.is_symlink() and not allow_mount_root_symlink:
        raise ModalBoundaryError(
            "Volume mount root may not be a symlink without explicit trust"
        )
    try:
        root = raw_root.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ModalBoundaryError(
            "Volume mount root is missing or cannot be resolved safely"
        ) from error
    if not root.is_dir():
        raise ModalBoundaryError("Volume mount root must resolve to a directory")

    root_descriptor = _open_directory_no_follow(
        root,
        label="resolved Volume mount root",
    )
    try:
        runs = root / "runs"
        if runs.is_symlink():
            raise ModalBoundaryError("Volume runs directory may not be a symlink")
        runs_descriptor = _open_directory_no_follow(
            "runs",
            label="Volume runs directory",
            dir_fd=root_descriptor,
        )
        try:
            run_metadata = os.stat(
                validated,
                dir_fd=runs_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(run_metadata.st_mode):
                raise ModalBoundaryError("Volume run directory may not be a symlink")
            if not stat.S_ISDIR(run_metadata.st_mode):
                raise ModalBoundaryError("Volume run path is not a directory")
            run_descriptor = _open_directory_no_follow(
                validated,
                label="Volume run directory",
                dir_fd=runs_descriptor,
            )
            try:
                opened = os.fstat(run_descriptor)
                if _artifact_object_identity(opened) != _artifact_object_identity(
                    run_metadata
                ):
                    raise ModalBoundaryError(
                        "Volume run directory changed while opening"
                    )
                destination = runs / validated
                if (
                    destination.is_symlink()
                    or destination.resolve(strict=True).parent != runs
                ):
                    raise ModalBoundaryError(
                        "Volume run directory escaped its mount root"
                    )
                canonical_destination = destination.resolve(strict=True)
                reopened = os.stat(
                    validated,
                    dir_fd=runs_descriptor,
                    follow_symlinks=False,
                )
                if _artifact_object_identity(reopened) != _artifact_object_identity(
                    opened
                ):
                    raise ModalBoundaryError(
                        "Volume run directory changed while resolving"
                    )
            finally:
                os.close(run_descriptor)
        except FileNotFoundError as error:
            raise ModalBoundaryError("Volume run directory is missing") from error
        finally:
            os.close(runs_descriptor)
    finally:
        os.close(root_descriptor)
    return canonical_destination


def _path_is_forbidden(relative: PurePosixPath) -> bool:
    for raw_part in relative.parts:
        part = raw_part.lower()
        if part in _FORBIDDEN_COMPONENTS:
            return True
        if part.startswith(".env") or part.startswith(".venv"):
            return True
        if "custody" in part or part.startswith("reviewer_roster"):
            return True
    return False


def _include_source_file(root: Path, path: Path) -> bool:
    try:
        relative = PurePosixPath(path.relative_to(root).as_posix())
    except ValueError:
        return False
    if _path_is_forbidden(relative) or path.name.startswith("test_"):
        return False
    if (
        relative.parts
        and relative.parts[0] in _CODE_ONLY_SOURCE_DIRECTORIES
        and path.suffix.lower() != ".py"
    ):
        return False
    if path.suffix.lower() not in _ALLOWED_SOURCE_SUFFIXES:
        return False
    return path.is_file() and not path.is_symlink()


def _validate_image_source_name(relative: PurePosixPath) -> None:
    stem = relative.stem.lower().replace("-", "_")
    stem_parts = frozenset(part for part in stem.split("_") if part)
    if stem in _SENSITIVE_SOURCE_STEMS or stem_parts & _SENSITIVE_SOURCE_STEMS:
        raise ModalBoundaryError(
            f"image source has a credential-like filename: {relative.as_posix()}"
        )


def _validate_image_source_payload(relative: PurePosixPath, payload: bytes) -> None:
    if any(pattern.search(payload) for pattern in _HIGH_CONFIDENCE_SECRET_PATTERNS):
        raise ModalBoundaryError(
            "image source contains high-confidence credential material: "
            f"{relative.as_posix()}"
        )


def _image_source_file_is_included(relative: PurePosixPath) -> bool:
    if _path_is_forbidden(relative) or relative.name.startswith("test_"):
        return False
    if (
        relative.parts
        and relative.parts[0] in _CODE_ONLY_SOURCE_DIRECTORIES
        and relative.suffix.lower() != ".py"
    ):
        return False
    return relative.suffix.lower() in _ALLOWED_SOURCE_SUFFIXES


def _read_image_source_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    relative: PurePosixPath,
) -> ImageSourceSnapshotFile:
    """Read, scan, size, and hash one descriptor-bound source file once."""

    _validate_image_source_name(relative)
    descriptor, before = _open_artifact_regular_at(
        parent_descriptor,
        name,
        relative_path=relative.as_posix(),
    )
    try:
        if before.st_size > MAX_IMAGE_SOURCE_FILE_BYTES:
            raise ModalBoundaryError(
                f"image source exceeds the per-file byte cap: {relative.as_posix()}"
            )
        payload = bytearray()
        while len(payload) <= MAX_IMAGE_SOURCE_FILE_BYTES:
            try:
                chunk = os.read(
                    descriptor,
                    min(
                        1024 * 1024,
                        MAX_IMAGE_SOURCE_FILE_BYTES + 1 - len(payload),
                    ),
                )
            except InterruptedError:
                continue
            except OSError as error:
                raise ModalBoundaryError(
                    f"image source read failed: {relative.as_posix()}"
                ) from error
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > MAX_IMAGE_SOURCE_FILE_BYTES:
            raise ModalBoundaryError(
                f"image source exceeds the per-file byte cap: {relative.as_posix()}"
            )
        after = os.fstat(descriptor)
        if (
            _artifact_metadata_identity(after) != _artifact_metadata_identity(before)
            or len(payload) != after.st_size
        ):
            raise ModalBoundaryError(
                f"image source changed while reading: {relative.as_posix()}"
            )
        _require_same_artifact_regular_at(
            parent_descriptor,
            name,
            relative_path=relative.as_posix(),
            expected=after,
        )
        exact = bytes(payload)
        _validate_image_source_payload(relative, exact)
        return ImageSourceSnapshotFile(relative.as_posix(), exact)
    finally:
        os.close(descriptor)


def _scan_image_source_directory(
    directory_descriptor: int,
    *,
    relative_parent: PurePosixPath,
) -> tuple[list[ImageSourceSnapshotFile], os.stat_result]:
    before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ModalBoundaryError("image source traversal escaped its directory")
    try:
        names = sorted(os.listdir(directory_descriptor))
    except OSError as error:
        raise ModalBoundaryError("image source directory listing failed") from error
    if len(names) != len(set(names)):
        raise ModalBoundaryError("image source directory listing is inconsistent")
    files: list[ImageSourceSnapshotFile] = []
    for name in names:
        relative = relative_parent / name
        safe_relative_path(relative.as_posix())
        if _path_is_forbidden(relative):
            continue
        try:
            metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ModalBoundaryError(
                f"image source changed during traversal: {relative.as_posix()}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ModalBoundaryError(
                f"image source path is a symlink: {relative.as_posix()}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            child_descriptor, _opened = _open_artifact_directory_at(
                directory_descriptor,
                name,
                relative_path=relative.as_posix(),
            )
            try:
                child_files, child_after = _scan_image_source_directory(
                    child_descriptor,
                    relative_parent=relative,
                )
            finally:
                os.close(child_descriptor)
            _require_same_artifact_directory_at(
                directory_descriptor,
                name,
                relative_path=relative.as_posix(),
                expected=child_after,
            )
            files.extend(child_files)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ModalBoundaryError(
                "image source path is not a regular file or directory: "
                f"{relative.as_posix()}"
            )
        if not _image_source_file_is_included(relative):
            continue
        files.append(
            _read_image_source_regular_at(
                directory_descriptor,
                name,
                relative=relative,
            )
        )
    try:
        names_after = sorted(os.listdir(directory_descriptor))
    except OSError as error:
        raise ModalBoundaryError("image source directory re-listing failed") from error
    after = os.fstat(directory_descriptor)
    if names_after != names or _artifact_metadata_identity(
        after
    ) != _artifact_metadata_identity(before):
        raise ModalBoundaryError("image source directory changed during traversal")
    return files, after


def _open_image_source_directory_chain(
    root_descriptor: int,
    relative: PurePosixPath,
) -> tuple[
    int,
    list[tuple[int, str, str, os.stat_result]],
    list[int],
]:
    descriptors = [os.dup(root_descriptor)]
    bindings: list[tuple[int, str, str, os.stat_result]] = []
    current = descriptors[0]
    traversed = PurePosixPath()
    for component in relative.parts:
        traversed /= component
        child, opened = _open_artifact_directory_at(
            current,
            component,
            relative_path=traversed.as_posix(),
        )
        bindings.append((current, component, traversed.as_posix(), opened))
        descriptors.append(child)
        current = child
    return current, bindings, descriptors


def _revalidate_image_source_directory_chain(
    bindings: list[tuple[int, str, str, os.stat_result]],
) -> None:
    for parent, name, relative, expected in reversed(bindings):
        _require_same_artifact_directory_at(
            parent,
            name,
            relative_path=relative,
            expected=expected,
        )


def _snapshot_image_source_file(
    root_descriptor: int,
    relative: PurePosixPath,
) -> ImageSourceSnapshotFile:
    parent_relative = relative.parent
    current, bindings, descriptors = _open_image_source_directory_chain(
        root_descriptor,
        parent_relative if parent_relative.as_posix() != "." else PurePosixPath(),
    )
    try:
        item = _read_image_source_regular_at(
            current,
            relative.name,
            relative=relative,
        )
        _revalidate_image_source_directory_chain(bindings)
        return item
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _snapshot_image_source_tree(
    root_descriptor: int,
    relative: PurePosixPath,
) -> list[ImageSourceSnapshotFile]:
    current, bindings, descriptors = _open_image_source_directory_chain(
        root_descriptor,
        relative,
    )
    try:
        files, after = _scan_image_source_directory(
            current,
            relative_parent=relative,
        )
        if _artifact_metadata_identity(after) != _artifact_metadata_identity(
            bindings[-1][3]
        ):
            raise ModalBoundaryError(
                f"image source directory changed: {relative.as_posix()}"
            )
        _revalidate_image_source_directory_chain(bindings)
        return files
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def build_image_source_snapshot(
    project_root: str | Path,
) -> ImageSourceSnapshot:
    """Capture the allowlisted tree once through stable no-follow descriptors."""

    raw_root = Path(project_root)
    if raw_root.is_symlink():
        raise ModalBoundaryError(
            f"architecture_discovery root may not be a symlink: {raw_root}"
        )
    root, root_descriptor, opened = _open_anchored_artifact_directory(
        raw_root,
        label="image source root",
    )
    selected: dict[str, ImageSourceSnapshotFile] = {}
    try:
        for filename in _ALLOWED_ROOT_FILES:
            relative = safe_relative_path(filename)
            item = _snapshot_image_source_file(root_descriptor, relative)
            if item.relative_path in selected:
                raise ModalBoundaryError("image source allowlist contains a duplicate")
            selected[item.relative_path] = item
        for directory_name in IMAGE_SOURCE_DIRECTORIES:
            relative = safe_relative_path(directory_name)
            for item in _snapshot_image_source_tree(root_descriptor, relative):
                if item.relative_path in selected:
                    raise ModalBoundaryError(
                        "image source directories overlap or contain duplicates"
                    )
                selected[item.relative_path] = item
        after = os.fstat(root_descriptor)
        _require_same_anchored_artifact_directory(
            root,
            expected=opened,
            label="image source root",
        )
        if _artifact_metadata_identity(after) != _artifact_metadata_identity(opened):
            raise ModalBoundaryError("image source root changed during snapshot")
    finally:
        os.close(root_descriptor)
    files = tuple(selected[path] for path in sorted(selected))
    total_bytes = sum(len(item.payload) for item in files)
    if total_bytes > MAX_IMAGE_SOURCE_TOTAL_BYTES:
        raise ModalBoundaryError("image source exceeds the total byte cap")
    source_files = tuple(item.source_file for item in files)
    lock_entries = tuple(
        item for item in source_files if item.relative_path == "uv.lock"
    )
    if len(lock_entries) != 1:
        raise ModalBoundaryError(
            "image source manifest must contain exactly one dependency lock"
        )
    manifest = ImageSourceManifestV1(
        dependency_lock_sha256=lock_entries[0].sha256,
        files=source_files,
    )
    return ImageSourceSnapshot(manifest=manifest, files=files)


def selected_image_source_paths(project_root: str | Path) -> tuple[Path, ...]:
    snapshot = build_image_source_snapshot(project_root)
    root = Path(os.path.abspath(os.fspath(project_root)))
    return tuple(
        root.joinpath(*PurePosixPath(item.relative_path).parts)
        for item in snapshot.files
    )


def build_image_source_manifest(
    project_root: str | Path,
) -> ImageSourceManifestV1:
    return build_image_source_snapshot(project_root).manifest


def stage_image_source(
    project_root: str | Path,
    destination_root: str | Path,
    manifest: ImageSourceManifestV1,
    *,
    snapshot: ImageSourceSnapshot | None = None,
) -> Path:
    """Create one immutable-by-convention, manifest-exact upload snapshot.

    Modal resolves local paths after an Image recipe is declared. Copying every
    allowlisted file into a verified temporary tree prevents later edits to the
    checkout from changing the bytes uploaded for this recipe.
    """

    if snapshot is not None and not isinstance(snapshot, ImageSourceSnapshot):
        raise TypeError("image source snapshot has the wrong type")
    captured = snapshot or build_image_source_snapshot(project_root)
    if captured.manifest != manifest:
        raise ModalBoundaryError("image source changed after approval planning")
    destination = Path(destination_root)
    if destination.exists() or destination.is_symlink():
        raise ModalBoundaryError("image staging destination must not exist")
    raw_parent = destination.parent
    if raw_parent.is_symlink():
        raise ModalBoundaryError("image staging parent is invalid")
    parent = raw_parent.resolve()
    if not parent.is_dir():
        raise ModalBoundaryError("image staging parent is invalid")
    temporary = Path(tempfile.mkdtemp(prefix=".rl4rl-image-stage-", dir=parent))
    published = False
    try:
        for item in captured.files:
            relative = safe_relative_path(item.relative_path)
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(target, flags, 0o600)
            try:
                remaining = memoryview(item.payload)
                while remaining:
                    try:
                        written = os.write(descriptor, remaining)
                    except InterruptedError:
                        continue
                    if written <= 0:
                        raise OSError("image-source staging write made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                    or metadata.st_size != len(item.payload)
                ):
                    raise ModalBoundaryError(
                        f"staged image source is unsafe: {item.relative_path}"
                    )
                os.fchmod(descriptor, 0o444)
            finally:
                os.close(descriptor)
        os.replace(temporary, destination)
        published = True
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        shutil.rmtree(destination if published else temporary, ignore_errors=True)
        raise
    return destination


def image_source_groups(
    project_root: str | Path,
    manifest: ImageSourceManifestV1,
) -> tuple[tuple[Path, PurePosixPath], ...]:
    """Return coarse local/remote copy roots whose contents are manifest-bound."""

    root = Path(project_root).resolve()
    manifest_paths = {item.relative_path for item in manifest.files}
    actual = {
        item.relative_to(root).as_posix() for item in selected_image_source_paths(root)
    }
    if actual != manifest_paths:
        raise ModalBoundaryError("image sources changed after manifest construction")
    groups = tuple(
        (root / name, REMOTE_PROJECT_ROOT / name) for name in IMAGE_SOURCE_DIRECTORIES
    )
    return groups


def _artifact_metadata_identity(metadata: os.stat_result) -> tuple[int, ...]:
    """Return metadata that must remain stable across one artifact read."""

    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _artifact_object_identity(metadata: os.stat_result) -> tuple[int, int]:
    return metadata.st_dev, metadata.st_ino


def _open_anchored_artifact_directory(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, int, os.stat_result]:
    """Open every directory component with ``O_NOFOLLOW`` from the anchor."""

    raw = Path(path)
    if ".." in raw.parts:
        raise ArtifactIntegrityError(f"{label} may not contain traversal")
    absolute = Path(os.path.abspath(os.fspath(raw)))
    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise ArtifactIntegrityError(
            "platform cannot enforce no-follow artifact traversal"
        )
    flags = os.O_RDONLY | no_follow | directory_only
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as error:
        raise ArtifactIntegrityError(f"{label} anchor is unsafe") from error
    try:
        opened = os.fstat(descriptor)
        for component in absolute.parts[1:]:
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise ArtifactIntegrityError(f"{label} is missing or unsafe") from error
            if stat.S_ISLNK(before.st_mode):
                raise ArtifactIntegrityError(f"{label} may not traverse symlinks")
            if not stat.S_ISDIR(before.st_mode):
                raise ArtifactIntegrityError(f"{label} is not a directory")
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except OSError as error:
                raise ArtifactIntegrityError(
                    f"{label} changed while opening"
                ) from error
            try:
                opened = os.fstat(next_descriptor)
            except OSError as error:
                os.close(next_descriptor)
                raise ArtifactIntegrityError(
                    f"{label} changed while opening"
                ) from error
            if not stat.S_ISDIR(opened.st_mode) or _artifact_object_identity(
                opened
            ) != _artifact_object_identity(before):
                os.close(next_descriptor)
                raise ArtifactIntegrityError(f"{label} changed while opening")
            os.close(descriptor)
            descriptor = next_descriptor
        return absolute, descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _require_same_anchored_artifact_directory(
    path: Path,
    *,
    expected: os.stat_result,
    label: str,
) -> None:
    reopened_descriptor, reopened = _open_anchored_artifact_directory(
        path,
        label=label,
    )[1:]
    try:
        if _artifact_object_identity(reopened) != _artifact_object_identity(expected):
            raise ArtifactIntegrityError(f"{label} path identity changed")
    finally:
        os.close(reopened_descriptor)


def _open_artifact_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    relative_path: str,
) -> tuple[int, os.stat_result]:
    """Open one descendant regular file and bind its pre-open identity."""

    try:
        before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ArtifactIntegrityError(
            f"artifact path disappeared or became unsafe: {relative_path}"
        ) from error
    if stat.S_ISLNK(before.st_mode):
        raise ArtifactIntegrityError(f"artifact path is a symlink: {relative_path}")
    if not stat.S_ISREG(before.st_mode):
        raise ArtifactIntegrityError(
            f"artifact path is not a regular file: {relative_path}"
        )
    if before.st_nlink != 1:
        raise ArtifactIntegrityError(
            f"artifact file link count is not one: {relative_path}"
        )
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ArtifactIntegrityError(
            "platform cannot enforce no-follow artifact file reads"
        )
    flags = os.O_RDONLY | no_follow | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ArtifactIntegrityError(
            f"artifact file changed while opening: {relative_path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise ArtifactIntegrityError(
            f"artifact file changed while opening: {relative_path}"
        ) from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or _artifact_metadata_identity(opened) != _artifact_metadata_identity(before)
    ):
        os.close(descriptor)
        raise ArtifactIntegrityError(
            f"artifact file changed while opening: {relative_path}"
        )
    return descriptor, opened


def _require_same_artifact_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    relative_path: str,
    expected: os.stat_result,
) -> None:
    descriptor, reopened = _open_artifact_regular_at(
        parent_descriptor,
        name,
        relative_path=relative_path,
    )
    try:
        if _artifact_metadata_identity(reopened) != _artifact_metadata_identity(
            expected
        ):
            raise ArtifactIntegrityError(
                f"artifact file path identity changed: {relative_path}"
            )
    finally:
        os.close(descriptor)


def _hash_artifact_regular_at(
    parent_descriptor: int,
    name: str,
    *,
    relative_path: str,
    maximum_bytes: int,
) -> ArtifactFileV1:
    """Hash and size one stable file descriptor within a fixed ceiling."""

    descriptor, before = _open_artifact_regular_at(
        parent_descriptor,
        name,
        relative_path=relative_path,
    )
    try:
        if before.st_size > maximum_bytes:
            raise ArtifactIntegrityError(
                f"artifact exceeds the per-file byte cap: {relative_path}"
            )
        digest = hashlib.sha256()
        observed_size = 0
        while observed_size <= maximum_bytes:
            try:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, maximum_bytes + 1 - observed_size),
                )
            except InterruptedError:
                continue
            except OSError as error:
                raise ArtifactIntegrityError(
                    f"artifact file read failed: {relative_path}"
                ) from error
            if not chunk:
                break
            observed_size += len(chunk)
            digest.update(chunk)
        if observed_size > maximum_bytes:
            raise ArtifactIntegrityError(
                f"artifact exceeds the per-file byte cap: {relative_path}"
            )
        after = os.fstat(descriptor)
        if (
            _artifact_metadata_identity(after) != _artifact_metadata_identity(before)
            or observed_size != after.st_size
        ):
            raise ArtifactIntegrityError(
                f"artifact file changed while hashing: {relative_path}"
            )
        _require_same_artifact_regular_at(
            parent_descriptor,
            name,
            relative_path=relative_path,
            expected=after,
        )
        return ArtifactFileV1(
            relative_path=relative_path,
            sha256=digest.hexdigest(),
            size_bytes=observed_size,
        )
    finally:
        os.close(descriptor)


def _read_artifact_regular_bytes_at(
    parent_descriptor: int,
    name: str,
    *,
    relative_path: str,
    maximum_bytes: int,
) -> bytes:
    """Read exact bytes from one stable, path-revalidated regular file."""

    descriptor, before = _open_artifact_regular_at(
        parent_descriptor,
        name,
        relative_path=relative_path,
    )
    try:
        if before.st_size > maximum_bytes:
            raise ArtifactIntegrityError(
                f"artifact exceeds its byte cap: {relative_path}"
            )
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            try:
                chunk = os.read(
                    descriptor,
                    min(1024 * 1024, maximum_bytes + 1 - len(payload)),
                )
            except InterruptedError:
                continue
            except OSError as error:
                raise ArtifactIntegrityError(
                    f"artifact file read failed: {relative_path}"
                ) from error
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > maximum_bytes:
            raise ArtifactIntegrityError(
                f"artifact exceeds its byte cap: {relative_path}"
            )
        after = os.fstat(descriptor)
        if (
            _artifact_metadata_identity(after) != _artifact_metadata_identity(before)
            or len(payload) != after.st_size
        ):
            raise ArtifactIntegrityError(
                f"artifact file changed while reading: {relative_path}"
            )
        _require_same_artifact_regular_at(
            parent_descriptor,
            name,
            relative_path=relative_path,
            expected=after,
        )
        return bytes(payload)
    finally:
        os.close(descriptor)


def _open_artifact_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    relative_path: str,
) -> tuple[int, os.stat_result]:
    try:
        before = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as error:
        raise ArtifactIntegrityError(
            f"artifact directory disappeared or became unsafe: {relative_path}"
        ) from error
    if stat.S_ISLNK(before.st_mode):
        raise ArtifactIntegrityError(f"artifact path is a symlink: {relative_path}")
    if not stat.S_ISDIR(before.st_mode):
        raise ArtifactIntegrityError(
            f"artifact path is not a directory: {relative_path}"
        )
    try:
        descriptor = _open_directory_no_follow(
            name,
            label=f"artifact directory {relative_path}",
            dir_fd=parent_descriptor,
        )
    except ModalBoundaryError as error:
        raise ArtifactIntegrityError(
            f"artifact directory changed or became unsafe: {relative_path}"
        ) from error
    try:
        opened = os.fstat(descriptor)
    except OSError as error:
        os.close(descriptor)
        raise ArtifactIntegrityError(
            f"artifact directory changed while opening: {relative_path}"
        ) from error
    if _artifact_metadata_identity(opened) != _artifact_metadata_identity(before):
        os.close(descriptor)
        raise ArtifactIntegrityError(
            f"artifact directory changed while opening: {relative_path}"
        )
    return descriptor, opened


def _require_same_artifact_directory_at(
    parent_descriptor: int,
    name: str,
    *,
    relative_path: str,
    expected: os.stat_result,
) -> None:
    descriptor, reopened = _open_artifact_directory_at(
        parent_descriptor,
        name,
        relative_path=relative_path,
    )
    try:
        if _artifact_metadata_identity(reopened) != _artifact_metadata_identity(
            expected
        ):
            raise ArtifactIntegrityError(
                f"artifact directory path identity changed: {relative_path}"
            )
    finally:
        os.close(descriptor)


def _scan_artifact_directory(
    directory_descriptor: int,
    *,
    relative_parent: PurePosixPath | None,
) -> tuple[list[ArtifactFileV1], os.stat_result]:
    before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ArtifactIntegrityError("artifact traversal escaped its directory")
    try:
        names = sorted(os.listdir(directory_descriptor))
    except OSError as error:
        raise ArtifactIntegrityError("artifact directory listing failed") from error
    if len(names) != len(set(names)):
        raise ArtifactIntegrityError("artifact directory listing is inconsistent")
    files: list[ArtifactFileV1] = []
    for name in names:
        relative = (
            PurePosixPath(name) if relative_parent is None else relative_parent / name
        )
        try:
            safe_relative_path(relative.as_posix())
        except ValueError as error:
            raise ArtifactIntegrityError(
                f"artifact path is unsafe: {relative.as_posix()}"
            ) from error
        try:
            metadata = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise ArtifactIntegrityError(
                f"artifact path changed during traversal: {relative.as_posix()}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactIntegrityError(
                f"artifact path is a symlink: {relative.as_posix()}"
            )
        if stat.S_ISDIR(metadata.st_mode):
            child_descriptor, _opened = _open_artifact_directory_at(
                directory_descriptor,
                name,
                relative_path=relative.as_posix(),
            )
            try:
                child_files, child_after = _scan_artifact_directory(
                    child_descriptor,
                    relative_parent=relative,
                )
            finally:
                os.close(child_descriptor)
            _require_same_artifact_directory_at(
                directory_descriptor,
                name,
                relative_path=relative.as_posix(),
                expected=child_after,
            )
            files.extend(child_files)
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactIntegrityError(
                f"artifact path is not a regular file or directory: "
                f"{relative.as_posix()}"
            )
        maximum_bytes = (
            MAX_ARTIFACT_MANIFEST_BYTES
            if name in ARTIFACT_MANIFEST_FILENAMES
            else MAX_ARTIFACT_DOWNLOAD_FILE_BYTES
        )
        item = _hash_artifact_regular_at(
            directory_descriptor,
            name,
            relative_path=relative.as_posix(),
            maximum_bytes=maximum_bytes,
        )
        if name not in _TRANSIENT_ARTIFACT_NAMES:
            files.append(item)
    try:
        names_after = sorted(os.listdir(directory_descriptor))
    except OSError as error:
        raise ArtifactIntegrityError("artifact directory re-listing failed") from error
    after = os.fstat(directory_descriptor)
    if names_after != names or _artifact_metadata_identity(
        after
    ) != _artifact_metadata_identity(before):
        raise ArtifactIntegrityError("artifact directory changed during traversal")
    return files, after


def _artifact_files(run_directory: str | Path) -> tuple[ArtifactFileV1, ...]:
    root, root_descriptor, opened = _open_anchored_artifact_directory(
        run_directory,
        label="run artifact directory",
    )
    try:
        files, after = _scan_artifact_directory(
            root_descriptor,
            relative_parent=None,
        )
        _require_same_anchored_artifact_directory(
            root,
            expected=opened,
            label="run artifact directory",
        )
        if _artifact_metadata_identity(after) != _artifact_metadata_identity(opened):
            raise ArtifactIntegrityError(
                "run artifact directory changed during traversal"
            )
    finally:
        os.close(root_descriptor)
    ordered = tuple(sorted(files, key=lambda item: item.relative_path))
    total_bytes = 0
    for item in ordered:
        total_bytes += item.size_bytes
        if total_bytes > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES:
            raise ArtifactIntegrityError(
                "artifact manifest exceeds the aggregate download byte cap"
            )
    return ordered


def build_artifact_manifest(
    run_directory: str | Path,
    *,
    run_id: str,
    image_source_sha256: str,
) -> ArtifactManifestV1:
    validated_run_id = validate_run_id(run_id)
    root = Path(os.path.abspath(os.fspath(run_directory)))
    if root.name != validated_run_id:
        raise ArtifactIntegrityError("artifact directory name differs from run ID")
    files = _artifact_files(root)
    return ArtifactManifestV1(
        run_id=validated_run_id,
        created_at_utc=datetime.now(UTC).isoformat(),
        image_source_sha256=image_source_sha256,
        files=files,
    )


def write_artifact_manifest(
    run_directory: str | Path,
    manifest: ArtifactManifestV1,
    *,
    filename: str = "artifact_manifest.json",
) -> Path:
    validate_artifact_manifest_filename(filename)
    encoded = (json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if len(encoded) > MAX_ARTIFACT_MANIFEST_BYTES:
        raise ArtifactIntegrityError("artifact manifest exceeds the 2 MiB limit")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise ArtifactIntegrityError(
            "platform cannot enforce exclusive artifact manifest publication"
        )
    root, root_descriptor, opened_root = _open_anchored_artifact_directory(
        run_directory,
        label="run artifact directory",
    )
    if root.name != manifest.run_id:
        os.close(root_descriptor)
        raise ArtifactIntegrityError("artifact directory name differs from run ID")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow
    flags |= getattr(os, "O_CLOEXEC", 0)
    manifest_descriptor: int | None = None
    created = False
    try:
        try:
            manifest_descriptor = os.open(
                filename,
                flags,
                0o600,
                dir_fd=root_descriptor,
            )
            created = True
        except FileExistsError as error:
            raise ArtifactIntegrityError("artifact manifest already exists") from error
        except OSError as error:
            raise ArtifactIntegrityError(
                "artifact manifest destination is unsafe"
            ) from error
        remaining = memoryview(encoded)
        while remaining:
            try:
                written = os.write(manifest_descriptor, remaining)
            except InterruptedError:
                continue
            except OSError as error:
                raise ArtifactIntegrityError(
                    "artifact manifest write failed; partial create is quarantined"
                ) from error
            if written <= 0 or written > len(remaining):
                raise ArtifactIntegrityError(
                    "artifact manifest write made no progress; "
                    "partial create is quarantined"
                )
            remaining = remaining[written:]
        os.fsync(manifest_descriptor)
        published = os.fstat(manifest_descriptor)
        if (
            not stat.S_ISREG(published.st_mode)
            or published.st_nlink != 1
            or published.st_size != len(encoded)
        ):
            raise ArtifactIntegrityError(
                "artifact manifest publication identity is invalid; "
                "created leaf is quarantined"
            )
        _require_same_artifact_regular_at(
            root_descriptor,
            filename,
            relative_path=filename,
            expected=published,
        )
        os.fsync(root_descriptor)
        _require_same_anchored_artifact_directory(
            root,
            expected=opened_root,
            label="run artifact directory",
        )
    except BaseException:
        if created and manifest_descriptor is not None:
            with suppress(OSError):
                os.fsync(manifest_descriptor)
            with suppress(OSError):
                os.fsync(root_descriptor)
        raise
    finally:
        if manifest_descriptor is not None:
            os.close(manifest_descriptor)
        os.close(root_descriptor)
    return root / filename


def load_raw_artifact_manifest(path: str | Path) -> RawArtifactManifestV1:
    manifest_path = Path(path)
    validate_artifact_manifest_filename(manifest_path.name)
    parent, parent_descriptor, opened_parent = _open_anchored_artifact_directory(
        manifest_path.parent,
        label="artifact manifest parent",
    )
    try:
        raw_bytes = _read_artifact_regular_bytes_at(
            parent_descriptor,
            manifest_path.name,
            relative_path=manifest_path.name,
            maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
        )
        _require_same_anchored_artifact_directory(
            parent,
            expected=opened_parent,
            label="artifact manifest parent",
        )
    finally:
        os.close(parent_descriptor)
    return RawArtifactManifestV1.from_bytes(
        filename=manifest_path.name,
        raw_bytes=raw_bytes,
    )


def load_artifact_manifest(path: str | Path) -> ArtifactManifestV1:
    return load_raw_artifact_manifest(path).manifest


def verify_artifact_manifest(
    run_directory: str | Path,
    manifest: ArtifactManifestV1,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(run_directory)))
    if root.name != manifest.run_id:
        raise ArtifactIntegrityError("artifact directory name differs from run ID")
    actual_files = _artifact_files(root)
    actual_paths = {item.relative_path: item for item in actual_files}
    expected_paths = {item.relative_path for item in manifest.files}
    if set(actual_paths) != expected_paths:
        raise ArtifactIntegrityError("artifact file set differs from its manifest")
    for item in manifest.files:
        actual = actual_paths[item.relative_path]
        if actual.size_bytes != item.size_bytes:
            raise ArtifactIntegrityError(
                f"artifact size mismatch: {item.relative_path}"
            )
        if actual.sha256 != item.sha256:
            raise ArtifactIntegrityError(
                f"artifact digest mismatch: {item.relative_path}"
            )
    return {
        "run_id": manifest.run_id,
        "file_count": len(manifest.files),
        "manifest_sha256": manifest.manifest_sha256,
        "verified": True,
    }


def _safe_local_destination(root: Path, relative_path: str) -> Path:
    relative = safe_relative_path(relative_path)
    destination = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ArtifactIntegrityError("download parent may not be a symlink")
    return destination


def validate_artifact_download_bounds(manifest: ArtifactManifestV1) -> int:
    """Return declared bytes after enforcing the fixed download ceilings."""

    if not isinstance(manifest, ArtifactManifestV1):
        raise TypeError("download manifest must be a ModalRunArtifactManifest")
    total_bytes = 0
    for item in manifest.files:
        if item.size_bytes > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
            raise ArtifactIntegrityError(
                f"artifact exceeds the per-file download byte cap: {item.relative_path}"
            )
        total_bytes += item.size_bytes
        if total_bytes > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES:
            raise ArtifactIntegrityError(
                "artifact manifest exceeds the aggregate download byte cap"
            )
    return total_bytes


def download_artifacts(
    raw_manifest: RawArtifactManifestV1,
    *,
    local_root: str | Path,
    reader: Callable[[str], bytes | Iterable[bytes]],
) -> Path:
    """Materialize artifacts plus the exact remotely verified manifest bytes."""

    if not isinstance(raw_manifest, RawArtifactManifestV1):
        raise TypeError("download requires an exact raw artifact manifest")
    manifest = raw_manifest.manifest

    if any(
        PurePosixPath(item.relative_path).name in _TRANSIENT_ARTIFACT_NAMES
        for item in manifest.files
    ):
        raise ArtifactIntegrityError(
            "artifact manifest may not list its reserved verifier metadata path"
        )
    declared_total_bytes = validate_artifact_download_bounds(manifest)
    raw_parent = Path(local_root)
    if raw_parent.is_symlink():
        raise ArtifactIntegrityError("local download root may not be a symlink")
    parent = raw_parent.resolve()
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    final_destination = parent / manifest.run_id
    if final_destination.exists() or final_destination.is_symlink():
        raise ArtifactIntegrityError("local run destination already exists")
    staging_parent = Path(
        tempfile.mkdtemp(prefix=f".{manifest.run_id}.download-", dir=parent)
    )
    staging_parent.chmod(0o700)
    destination_root = staging_parent / manifest.run_id
    destination_root.mkdir(mode=0o700)
    downloaded_total_bytes = 0
    published = False
    try:
        for item in manifest.files:
            destination = _safe_local_destination(destination_root, item.relative_path)
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            chunks = reader(volume_object_path(manifest.run_id, item.relative_path))
            stream = (chunks,) if isinstance(chunks, bytes) else chunks
            digest = hashlib.sha256()
            size = 0
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as handle:
                os.fchmod(handle.fileno(), 0o600)
                for chunk in stream:
                    if not isinstance(chunk, bytes):
                        raise ArtifactIntegrityError("Volume reader yielded non-bytes")
                    next_size = size + len(chunk)
                    next_total_bytes = downloaded_total_bytes + len(chunk)
                    if next_size > item.size_bytes:
                        raise ArtifactIntegrityError(
                            "Volume reader exceeded the manifest-declared artifact size"
                        )
                    if next_size > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
                        raise ArtifactIntegrityError(
                            "download exceeded the per-file artifact byte cap"
                        )
                    if next_total_bytes > declared_total_bytes:
                        raise ArtifactIntegrityError(
                            "download exceeded the manifest-declared aggregate size"
                        )
                    if next_total_bytes > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES:
                        raise ArtifactIntegrityError(
                            "download exceeded the aggregate artifact byte cap"
                        )
                    handle.write(chunk)
                    digest.update(chunk)
                    size = next_size
                    downloaded_total_bytes = next_total_bytes
                handle.flush()
                os.fsync(handle.fileno())
            if size != item.size_bytes or digest.hexdigest() != item.sha256:
                raise ArtifactIntegrityError(
                    f"downloaded artifact failed verification: {item.relative_path}"
                )
        verify_artifact_manifest(destination_root, manifest)
        persisted_manifest = destination_root / raw_manifest.filename
        manifest_descriptor = os.open(
            persisted_manifest,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(manifest_descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(raw_manifest.raw_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = load_raw_artifact_manifest(persisted_manifest)
        if (
            persisted.raw_bytes != raw_manifest.raw_bytes
            or persisted.raw_sha256 != raw_manifest.raw_sha256
            or persisted.raw_size_bytes != raw_manifest.raw_size_bytes
            or persisted.manifest != manifest
        ):
            raise ArtifactIntegrityError(
                "persisted raw manifest differs from the remotely verified bytes"
            )
        if final_destination.exists() or final_destination.is_symlink():
            raise ArtifactIntegrityError("local run destination already exists")
        os.rename(destination_root, final_destination)
        published = True
        parent_descriptor = os.open(
            parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        if not published:
            shutil.rmtree(staging_parent, ignore_errors=True)
        raise
    staging_parent.rmdir()
    return final_destination


class RemoteFunction(Protocol):
    def remote(self, **kwargs: Any) -> Any: ...


def invoke_synchronously(function: RemoteFunction, **kwargs: Any) -> Any:
    remote = getattr(function, "remote", None)
    if not callable(remote):
        raise TypeError("remote function does not expose synchronous .remote()")
    return remote(**kwargs)


def run_canaries_synchronously(
    functions: Mapping[str, RemoteFunction],
    *,
    run_id_prefix: str,
) -> dict[str, Any]:
    """Attempt each one-opportunity canary once, in frozen order.

    Ordinary remote failures are reduced to their exception class so later
    harnesses still receive their one approved attempt without retaining
    provider-bearing exception text. Process-control exceptions such as
    ``KeyboardInterrupt`` remain uncaught.
    """

    validate_run_id(run_id_prefix)
    if set(functions) != set(CANARY_ORDER):
        raise ValueError("canary function mapping must cover exactly four harnesses")
    outcomes: list[dict[str, Any]] = []
    for harness in CANARY_ORDER:
        suffix = canary_run_suffix(harness)
        run_id = validate_run_id(f"{run_id_prefix}-{suffix}")
        try:
            result = invoke_synchronously(
                functions[harness],
                run_id=run_id,
                opportunities=1,
            )
        except Exception as error:
            outcomes.append(
                {
                    "harness": harness,
                    "run_id": run_id,
                    "status": "failed",
                    "result": None,
                    "error_type": type(error).__name__,
                }
            )
        else:
            outcomes.append(
                {
                    "harness": harness,
                    "run_id": run_id,
                    "status": "success",
                    "result": result,
                    "error_type": None,
                }
            )
    all_succeeded = all(item["status"] == "success" for item in outcomes)
    return {
        "schema_name": "ModalProviderCanaryAggregateResult",
        "schema_version": "1.0",
        "run_id_prefix": run_id_prefix,
        "harness_order": list(CANARY_ORDER),
        "outcomes": outcomes,
        "all_succeeded": all_succeeded,
    }
