"""Optional Modal app for bounded Architecture Discovery validation.

Importing this module constructs lazy Modal objects only when the optional SDK
is installed. It does not build an image, hydrate a Secret or Volume, start a
container, make a provider request, or invoke a remote function.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from pathlib import Path, PurePosixPath
from typing import Any

from common.gpt56_sol import OFFICIAL_OPENAI_API_BASE, TARGET_MODEL
from common.process_control import (
    OUTER_PROCESS_DEADLINE_ENV,
    ProcessGroupClosureError,
    capture_isolated_process_group,
    terminate_process_group,
)
from common.provider_attempts import (
    PROVIDER_ATTEMPT_LEDGER_FILENAME,
    load_provider_attempt_ledger,
)
from common.runtime_context import ExecutionContextV1
from modal_boundary import (
    APP_NAME,
    ARTIFACT_MANIFEST_FILENAMES,
    CANARY_ORDER,
    CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS,
    FUNCTION_SPECS,
    IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
    IMAGE_BUILD_CPU_REQUEST_CORES,
    IMAGE_BUILD_MEMORY_REQUEST_MIB,
    IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
    IMAGE_RECIPE_VERSION,
    MAX_ARTIFACT_DOWNLOAD_FILE_BYTES,
    MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MODAL_ACTION_ATTEMPT_ID_ENV,
    MODAL_ACTION_INTENT_SHA256_ENV,
    MODAL_VERSION,
    PROVIDER_SECRET_NAME,
    PYTHON_VERSION,
    REMOTE_PROJECT_ROOT,
    UV_VERSION,
    VOLUME_MOUNT_PATH,
    VOLUME_NAME,
    ArtifactFileV1,
    ArtifactManifestV1,
    ArtifactVerificationV1,
    ModalBoundaryError,
    ModalLiveCohortIdentity,
    RawArtifactManifestV1,
    build_artifact_manifest,
    build_image_source_snapshot,
    build_provider_canary_aggregate_outcome_receipt,
    canary_run_suffix,
    create_fresh_run_directory,
    download_artifacts,
    invoke_synchronously,
    load_artifact_manifest,
    load_raw_artifact_manifest,
    modal_action_intent_receipt_path,
    modal_artifact_verifier_capture_directory_path,
    modal_artifact_verifier_capture_parent_path,
    modal_remote_verification_receipt_path,
    new_run_id,
    provider_canary_aggregate_outcome_receipt_path,
    resolve_existing_volume_run_directory,
    run_canaries_synchronously,
    safe_relative_path,
    stage_image_source,
    validate_artifact_download_bounds,
    validate_provider_canary_aggregate_outcome_receipt,
    validate_run_id,
    verify_artifact_manifest,
    volume_artifact_uri,
    write_artifact_manifest,
)
from modal_image_build import install_image_dependencies
from scripts.launch_modal import (
    MODAL_LAUNCH_NONCE_ENV,
    local_launch_authorized,
    validate_local_action_intent_for_entrypoint,
)
from study.serialization import create_json_exclusive

try:
    import modal
except ModuleNotFoundError as error:  # Core/offline imports remain usable.
    if error.name != "modal":
        raise
    modal = None


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODAL_COMMAND_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
CUDA_SMOKE_PROFILE = "smoke_train_cuda_v2"
RESUME_ACTION_DEADLINE_SECONDS = 240
RESUME_PROBE_TIMEOUT_SECONDS = 20
RESUME_TRAIN_TIMEOUT_SECONDS = 180
RESUME_PROGRESSION_TIMEOUT_SECONDS = 20
RESUME_INTERNAL_RESERVE_SECONDS = 10
PROVIDER_ENVIRONMENT_KEYS = (
    "DISCOVERY_API_KEY",
    "DISCOVERY_API_BASE",
    "DISCOVERY_MODEL",
)
PROVIDER_FREE_NETWORK_PROBE_FUNCTIONS = frozenset(
    {
        "offline_smoke",
        "cuda_environment",
        "candidate_smoke",
        "checkpoint_resume",
    }
)
# Cloudflare's public anycast HTTPS address is intentionally routable.  A
# documentation-only TEST-NET address could time out even when egress worked
# and would therefore be incapable of distinguishing Modal's network block.
NETWORK_DENIAL_PROBE_IP = "1.1.1.1"
NETWORK_DENIAL_PROBE_PORT = 443
NETWORK_DENIAL_PROBE_TIMEOUT_SECONDS = 1.0
_STATIC_SUBPROCESS_PATH = (
    f"{REMOTE_PROJECT_ROOT}/.venv/bin:/usr/local/bin:/usr/bin:/bin"
)
IMAGE_SOURCE_SNAPSHOT = build_image_source_snapshot(PROJECT_ROOT)
IMAGE_SOURCE_MANIFEST = IMAGE_SOURCE_SNAPSHOT.manifest
IMAGE_SOURCE_SHA256 = IMAGE_SOURCE_MANIFEST.manifest_sha256
IMAGE_SOURCE_IDENTITY_ENV = "RL4RL_IMAGE_SOURCE_SHA256"
_IMAGE_SOURCE_STAGING: tempfile.TemporaryDirectory[str] | None = None
_DEPENDENCY_FILES = (
    "modal_image_build.py",
    "pyproject.toml",
    "uv.lock",
    "vendor/openevolve/README.md",
    "vendor/openevolve/pyproject.toml",
)
_DEPENDENCY_SOURCE_DIRECTORY = "vendor/openevolve/openevolve"

if modal is not None and modal.__version__ != MODAL_VERSION:
    raise RuntimeError(
        f"modal_app requires Modal {MODAL_VERSION}; "
        "install the pinned optional SDK version"
    )

_MODAL_OBJECTS_ENABLED = modal is not None and (
    not modal.is_local()
    or local_launch_authorized(
        os.environ,
        image_source_sha256=IMAGE_SOURCE_SHA256,
        modal_command_python_executable=LOCAL_MODAL_COMMAND_PYTHON,
    )
)


class RemoteActionError(RuntimeError):
    pass


class ProviderFreeNetworkExposureError(RemoteActionError):
    """Raised when a block-network function unexpectedly reaches the Internet."""


def _dependency_image_source_paths() -> frozenset[str]:
    manifest_paths = frozenset(
        item.relative_path for item in IMAGE_SOURCE_MANIFEST.files
    )
    required_files = frozenset(_DEPENDENCY_FILES)
    missing = sorted(required_files - manifest_paths)
    if missing:
        raise RuntimeError(
            "dependency image source is absent from the verified manifest: "
            + ", ".join(missing)
        )
    vendor_prefix = _DEPENDENCY_SOURCE_DIRECTORY.rstrip("/") + "/"
    vendor_paths = frozenset(
        path for path in manifest_paths if path.startswith(vendor_prefix)
    )
    if not vendor_paths:
        raise RuntimeError(
            "dependency image source contains no vendored OpenEvolve files"
        )
    return required_files | vendor_paths


def _source_subset_ignore(local_root: Path, allowed: frozenset[str]):
    if not allowed:
        raise ValueError("image source subset must not be empty")

    def ignore(candidate: Path) -> bool:
        raw = Path(candidate)
        try:
            relative = raw.relative_to(local_root) if raw.is_absolute() else raw
        except ValueError:
            return True
        if relative.as_posix() in {"", "."}:
            return False
        logical = PurePosixPath(relative.as_posix()).as_posix()
        if logical in allowed:
            return False
        prefix = logical.rstrip("/") + "/"
        return not any(path.startswith(prefix) for path in allowed)

    return ignore


def _build_image():
    global _IMAGE_SOURCE_STAGING

    if _IMAGE_SOURCE_STAGING is not None:
        raise RuntimeError("Modal image source may be staged only once")
    staging = tempfile.TemporaryDirectory(prefix="rl4rl-modal-image-source-")
    staged_root = stage_image_source(
        PROJECT_ROOT,
        Path(staging.name) / "source",
        IMAGE_SOURCE_MANIFEST,
        snapshot=IMAGE_SOURCE_SNAPSHOT,
    )
    _IMAGE_SOURCE_STAGING = staging
    image = modal.Image.debian_slim(python_version=PYTHON_VERSION)
    image = image.add_local_dir(
        staged_root,
        str(REMOTE_PROJECT_ROOT),
        copy=True,
        ignore=_source_subset_ignore(
            staged_root,
            _dependency_image_source_paths(),
        ),
    )
    image = image.run_function(
        install_image_dependencies,
        cpu=IMAGE_BUILD_CPU_REQUEST_CORES,
        memory=IMAGE_BUILD_MEMORY_REQUEST_MIB,
        timeout=IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
        env={"PYTHONPATH": str(REMOTE_PROJECT_ROOT)},
        kwargs={
            "project_root": str(REMOTE_PROJECT_ROOT),
            "thread_limit": IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
            "uv_version": UV_VERSION,
            "timeout_seconds": IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
        },
        include_source=False,
    )
    image = image.add_local_dir(
        staged_root,
        str(REMOTE_PROJECT_ROOT),
        copy=True,
    )
    return image.env(
        {
            "PATH": (f"{REMOTE_PROJECT_ROOT}/.venv/bin:/usr/local/bin:/usr/bin:/bin"),
            "PYTHONPATH": str(REMOTE_PROJECT_ROOT),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            IMAGE_SOURCE_IDENTITY_ENV: IMAGE_SOURCE_SHA256,
        }
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _exclusive_json_bytes(payload: dict[str, Any]) -> bytes:
    """Mirror the frozen create-only JSON encoding for post-write hashing."""

    return (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = _json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_exclusive_bytes(parent: Path, filename: str, payload: bytes) -> None:
    """Create one durable final leaf without a check-then-replace window."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise RemoteActionError("platform cannot enforce exclusive no-follow writes")
    parent_flags = (
        os.O_RDONLY
        | no_follow
        | directory_only
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_descriptor = os.open(parent, parent_flags)
    except OSError as error:
        raise RemoteActionError("run JSON parent directory is unsafe") from error
    descriptor: int | None = None
    created = False
    try:
        leaf_flags = (
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | no_follow
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                filename,
                leaf_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
        except FileExistsError as error:
            raise RemoteActionError("run JSON artifact already exists") from error
        except OSError as error:
            raise RemoteActionError("run JSON artifact is unsafe") from error
        created = True
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(descriptor, payload[offset:])
            except InterruptedError:
                continue
            if not isinstance(written, int) or written <= 0:
                raise RemoteActionError("run JSON publication made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(parent_descriptor)
    except BaseException:
        if descriptor is not None:
            with suppress(OSError):
                os.close(descriptor)
            descriptor = None
        if created:
            try:
                os.unlink(filename, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            except OSError as cleanup_error:
                raise RemoteActionError(
                    "partial run JSON could not be quarantined safely"
                ) from cleanup_error
        raise
    finally:
        os.close(parent_descriptor)


def _execution_context(
    function_name: str,
    run_id: str,
    *,
    artifact_run_id: str | None = None,
) -> ExecutionContextV1:
    call_id = modal.current_function_call_id()
    if call_id is None:
        raise RemoteActionError("Modal did not expose the current function call ID")
    function_id = _current_modal_function_id(function_name)
    app_id = app.app_id if app is not None else None
    image_id = os.environ.get("MODAL_IMAGE_ID") or _hydrated_object_id(IMAGE)
    return ExecutionContextV1(
        execution_backend="modal",
        run_id=run_id,
        app_name=APP_NAME,
        function_name=function_name,
        modal_app_id=app_id,
        modal_function_id=function_id,
        modal_call_id=call_id,
        modal_image_id=image_id,
        image_source_sha256=IMAGE_SOURCE_SHA256,
        artifact_uri=volume_artifact_uri(artifact_run_id or run_id),
    )


def _provider_free_network_denial_probe(
    context: ExecutionContextV1,
    *,
    connector: Callable[[tuple[str, int], float], Any] = socket.create_connection,
) -> dict[str, Any]:
    """Require one bounded numeric-IP connection attempt to be denied."""

    if (
        context.execution_backend != "modal"
        or context.function_name not in PROVIDER_FREE_NETWORK_PROBE_FUNCTIONS
    ):
        raise ValueError("network denial probe context is not provider-free")
    try:
        connection = connector(
            (NETWORK_DENIAL_PROBE_IP, NETWORK_DENIAL_PROBE_PORT),
            NETWORK_DENIAL_PROBE_TIMEOUT_SECONDS,
        )
    except OSError as error:
        permission_denied = isinstance(error, PermissionError) or error.errno in {
            errno.EACCES,
            errno.EPERM,
        }
        network_unreachable = error.errno == errno.ENETUNREACH
        if not permission_denied and not network_unreachable:
            raise ProviderFreeNetworkExposureError(
                "provider-free network policy denial could not be proven"
            ) from error
        return {
            "schema_name": "ProviderFreeNetworkDenialProbe",
            "schema_version": "1.0",
            "attempted_endpoint": {
                "ip": NETWORK_DENIAL_PROBE_IP,
                "port": NETWORK_DENIAL_PROBE_PORT,
            },
            "timeout_seconds": NETWORK_DENIAL_PROBE_TIMEOUT_SECONDS,
            "denied": True,
            "exception_type": (
                "PermissionError"
                if permission_denied
                else "NetworkUnreachableError"
            ),
            "execution_context": context.to_dict(),
        }
    close = getattr(connection, "close", None)
    if callable(close):
        close()
    raise ProviderFreeNetworkExposureError(
        "provider-free function unexpectedly established an egress connection"
    )


def _current_modal_function_id(function_name: str) -> str | None:
    """Return the executing bound Function ID under the pinned Modal runtime.

    Provider canaries are Secret-bound Function variants whose executing ID can
    differ from the registered base Function object. Modal 1.5.3 retains the
    exact container Function ID in its task-lifecycle context. The registered
    object ID remains a fail-closed compatibility fallback for mocked/local
    contexts where no container lifecycle exists.
    """

    try:
        from modal._runtime.task_lifecycle_manager import _TaskLifecycleManager
    except (ImportError, ModuleNotFoundError):
        manager = None
    else:
        manager = _TaskLifecycleManager._singleton
    current_id = getattr(manager, "function_id", None)
    if isinstance(current_id, str) and current_id:
        return current_id
    return _hydrated_object_id(globals().get(function_name))


def _hydrated_object_id(value: object) -> str | None:
    if value is None:
        return None
    hydrated = getattr(value, "is_hydrated", False)
    if callable(hydrated):
        hydrated = hydrated()
    if hydrated is not True:
        return None
    try:
        object_id = getattr(value, "object_id", None)
    except (RuntimeError, ValueError):
        return None
    return object_id if isinstance(object_id, str) and object_id else None


def _remote_environment(
    context: ExecutionContextV1,
    *,
    provider: bool,
    requested_device: str,
) -> dict[str, str]:
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("Modal subprocess device must be 'cpu' or 'cuda'")
    # Never inherit operational variables from the Secret-bearing Function
    # container. Modal 1.5.3 can assert that required Secret keys exist, but its
    # metadata does not expose the complete key set. Constructing these values
    # here prevents an unexpected Secret key such as PATH, PYTHONPATH, TMPDIR,
    # SSL_CERT_FILE, or a CUDA/NVIDIA variable from silently changing the
    # controller subprocess. The container already has exactly one assigned GPU;
    # CUDA device zero is therefore the deterministic child-visible selection.
    environment = {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": _STATIC_SUBPROCESS_PATH,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(REMOTE_PROJECT_ROOT),
        "PYTHONUNBUFFERED": "1",
        "TMPDIR": "/tmp",
    }
    if provider:
        missing = [
            name for name in PROVIDER_ENVIRONMENT_KEYS if not os.environ.get(name)
        ]
        if missing:
            raise RemoteActionError(
                "provider Secret is missing required keys: " + ", ".join(missing)
            )
        if (
            os.environ["DISCOVERY_API_BASE"] != OFFICIAL_OPENAI_API_BASE
            or os.environ["DISCOVERY_MODEL"] != TARGET_MODEL
        ):
            raise RemoteActionError(
                "provider Secret does not match the pinned Modal provider contract"
            )
        environment.update(
            {name: os.environ[name] for name in PROVIDER_ENVIRONMENT_KEYS}
        )
    environment.update(
        {
            "DISCOVERY_EXECUTION_CONTEXT_JSON": json.dumps(
                context.to_dict(), sort_keys=True, separators=(",", ":")
            ),
            "DISCOVERY_TRAIN_DEVICE": requested_device,
            "PYTHONHASHSEED": "0",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        }
    )
    if requested_device == "cuda":
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return environment


def _provider_credential_bytes() -> bytes:
    raw_key = os.environ.get("DISCOVERY_API_KEY")
    if not raw_key:
        raise RemoteActionError("provider credential artifact scan unavailable")
    return raw_key.encode("utf-8")


def _run_command(
    command: list[str],
    *,
    context: ExecutionContextV1,
    provider: bool,
    requested_device: str,
    timeout_seconds: int = CONTROLLER_SUBPROCESS_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if timeout_seconds <= 0 or timeout_seconds >= 300:
        raise ValueError("subprocess timeout must remain inside the function timeout")
    with tempfile.TemporaryDirectory(prefix="modal-command-") as temporary:
        temporary_path = Path(temporary)
        stdout_path = temporary_path / "stdout"
        stderr_path = temporary_path / "stderr"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            environment = _remote_environment(
                context,
                provider=provider,
                requested_device=requested_device,
            )
            environment[OUTER_PROCESS_DEADLINE_ENV] = format(
                time.monotonic() + timeout_seconds,
                ".9f",
            )
            process = subprocess.Popen(
                command,
                cwd=REMOTE_PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                # This makes the child the leader of a fresh process group, so
                # timeout cleanup also reaches any subprocesses it launches.
                start_new_session=True,
            )
            # The new session fixes the expected PGID to the child PID. Keep
            # that value before observing it so a capture failure still flows
            # through group cleanup instead of stranding descendants.
            process_group_id = process.pid
            try:
                process_group_id = capture_isolated_process_group(process)
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as error:
                terminate_process_group(
                    process,
                    process_group_id=process_group_id,
                )
                # A trusted controller can itself launch a worker in a nested
                # session. Its PGID is intentionally outside the controller's
                # group, so an outer timeout cannot attest full-tree closure.
                # Fail without artifact finalization or a Volume commit.
                raise ProcessGroupClosureError(
                    "outer subprocess timeout left nested-process closure unverified"
                ) from error
            except BaseException:
                terminate_process_group(
                    process,
                    process_group_id=process_group_id,
                )
                raise
            terminate_process_group(
                process,
                process_group_id=process_group_id,
            )
        stdout_bytes = stdout_path.read_bytes()
        stderr_bytes = stderr_path.read_bytes()
        if provider:
            credential = _provider_credential_bytes()
            if credential in stdout_bytes or credential in stderr_bytes:
                # Never derive or retain a digest from credential-bearing bytes.
                raise RemoteActionError(
                    "provider credential material reached subprocess streams"
                )
        result = {
            "returncode": returncode,
            "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
            "stdout_size_bytes": len(stdout_bytes),
            "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
            "stderr_size_bytes": len(stderr_bytes),
        }
        if returncode != 0:
            raise RemoteActionError(
                "remote action failed; raw subprocess output was not retained "
                f"(exit={returncode}, stderr_sha256="
                f"{result['stderr_sha256']})"
            )
        return result


def _retained_artifact_bytes(run_directory: Path) -> int:
    """Return retained non-manifest bytes after enforcing both storage caps."""

    try:
        root_metadata = run_directory.lstat()
    except OSError as error:
        raise RemoteActionError(
            "run artifact directory is missing or unsafe"
        ) from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RemoteActionError("run artifact directory is missing or unsafe")
    total = 0
    for path in sorted(run_directory.rglob("*")):
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RemoteActionError(
                "retained run artifact changed during byte accounting"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RemoteActionError("retained run artifacts may not use symlinks")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise RemoteActionError("retained run contains a non-regular artifact")
        if path.name in ARTIFACT_MANIFEST_FILENAMES:
            if metadata.st_size > MAX_ARTIFACT_MANIFEST_BYTES:
                raise RemoteActionError(
                    "retained artifact manifest exceeds its byte cap"
                )
            continue
        if metadata.st_size > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
            raise RemoteActionError("retained artifact exceeds the per-file byte cap")
        total += metadata.st_size
        if total > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES:
            raise RemoteActionError("retained artifacts exceed the aggregate byte cap")
    return total


def _ensure_retained_artifact_budget(
    run_directory: Path,
    *,
    future_payloads: tuple[bytes, ...] = (),
    additional_bytes: int = 0,
) -> int:
    """Reserve exact future bytes before publishing anything to the Volume."""

    if type(additional_bytes) is not int or additional_bytes < 0:
        raise TypeError("additional artifact bytes must be a nonnegative integer")
    future_total = 0
    for payload in future_payloads:
        if type(payload) is not bytes:
            raise TypeError("reserved artifact payloads must be exact bytes")
        if len(payload) > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
            raise RemoteActionError("reserved artifact exceeds the per-file byte cap")
        future_total += len(payload)
    total = _retained_artifact_bytes(run_directory) + future_total + additional_bytes
    if total > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES:
        raise RemoteActionError("retained artifacts exceed the aggregate byte cap")
    return total


def _validate_prospective_artifact_manifest(
    run_directory: Path,
    snapshots: tuple[tuple[PurePosixPath, bytes], ...],
    reserved_artifacts: tuple[tuple[str, bytes], ...],
) -> None:
    """Prove the eventual manifest can fit before any staged file is published."""

    entries: dict[str, int] = {}
    for path in sorted(run_directory.rglob("*")):
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RemoteActionError("retained run contains an unsafe manifest entry")
        if path.name in ARTIFACT_MANIFEST_FILENAMES:
            raise RemoteActionError("staged publication cannot follow a final manifest")
        relative = safe_relative_path(path.relative_to(run_directory).as_posix())
        entries[relative.as_posix()] = metadata.st_size

    for relative, payload in snapshots:
        name = relative.as_posix()
        if name in entries:
            raise RemoteActionError("prospective artifact manifest path collides")
        entries[name] = len(payload)

    for raw_name, payload in reserved_artifacts:
        relative = safe_relative_path(raw_name)
        name = relative.as_posix()
        if name in entries:
            raise RemoteActionError("reserved artifact manifest path collides")
        if type(payload) is not bytes:
            raise TypeError("reserved artifact payloads must be exact bytes")
        entries[name] = len(payload)

    # SHA-256 strings have a fixed serialized width. The timestamp below is the
    # longest form emitted by datetime.now(UTC).isoformat(), so this synthetic
    # manifest is an upper bound for the final manifest's exact JSON bytes.
    manifest = ArtifactManifestV1(
        run_id=run_directory.name,
        created_at_utc="9999-12-31T23:59:59.999999+00:00",
        image_source_sha256=IMAGE_SOURCE_SHA256,
        files=tuple(
            ArtifactFileV1(
                relative_path=name,
                sha256="0" * 64,
                size_bytes=size,
            )
            for name, size in sorted(entries.items())
        ),
    )
    if len(_json_bytes(manifest.to_dict())) > MAX_ARTIFACT_MANIFEST_BYTES:
        raise RemoteActionError("prospective artifact manifest exceeds its byte cap")


def _write_bounded_run_json(
    run_directory: Path,
    filename: str,
    payload: dict[str, Any],
) -> bytes:
    relative = safe_relative_path(filename)
    if len(relative.parts) != 1:
        raise ValueError("run JSON filename must be one safe path component")
    destination = run_directory / relative.name
    if destination.exists() or destination.is_symlink():
        raise RemoteActionError("run JSON artifact already exists")
    encoded = _json_bytes(payload)
    _ensure_retained_artifact_budget(
        run_directory,
        future_payloads=(encoded,),
    )
    _write_exclusive_bytes(run_directory, relative.name, encoded)
    return encoded


def _validate_manifest_payload(
    manifest: ArtifactManifestV1,
    *,
    filename: str,
) -> None:
    raw = RawArtifactManifestV1.from_bytes(
        filename=filename,
        raw_bytes=_json_bytes(manifest.to_dict()),
    )
    if raw.manifest != manifest or raw.raw_size_bytes > MAX_ARTIFACT_MANIFEST_BYTES:
        raise RemoteActionError("artifact manifest exceeds its exact byte contract")


def _prepare_run(
    function_name: str,
    run_id: str,
    *,
    artifact_run_id: str | None = None,
) -> tuple[Path, ExecutionContextV1]:
    validate_run_id(run_id)
    if os.environ.get(IMAGE_SOURCE_IDENTITY_ENV) != IMAGE_SOURCE_SHA256:
        raise RemoteActionError(
            "baked image source identity differs from the executing source tree"
        )
    # A restarted input must reload the last committed Volume state before it
    # attempts the create-once lease. Context construction is kept before the
    # directory mutation so a context failure cannot consume a run ID.
    ARTIFACT_VOLUME.reload()
    if artifact_run_id is None:
        context = _execution_context(function_name, run_id)
    else:
        context = _execution_context(
            function_name,
            run_id,
            artifact_run_id=artifact_run_id,
        )
    run_directory = _create_volume_run_directory(run_id)
    _write_bounded_run_json(
        run_directory,
        "execution_context.json",
        context.to_dict(),
    )
    _write_bounded_run_json(
        run_directory,
        "image_source_manifest.json",
        IMAGE_SOURCE_MANIFEST.to_dict(),
    )
    # This commit is the durable pre-action lease. No action probe, subprocess,
    # provider request, training, staged publication, or artifact-verifier
    # source read may start before it.
    ARTIFACT_VOLUME.commit()
    return run_directory, context


def _create_volume_run_directory(run_id: str) -> Path:
    """Sole trusted opt-in for Modal's symlinked Volume mount alias."""

    return create_fresh_run_directory(
        VOLUME_MOUNT_PATH,
        run_id,
        allow_mount_root_symlink=True,
    )


def _finalize_run(
    run_directory: Path,
    *,
    run_id: str,
    result: dict[str, Any],
    manifest_filename: str = "artifact_manifest.json",
    result_filename: str = "remote_action_result.json",
) -> dict[str, Any]:
    if result_filename not in {
        "remote_action_result.json",
        "resume_action_result.json",
    }:
        raise ValueError("remote action result filename is unsafe")
    _write_bounded_run_json(run_directory, result_filename, result)
    manifest = build_artifact_manifest(
        run_directory,
        run_id=run_id,
        image_source_sha256=IMAGE_SOURCE_SHA256,
    )
    validate_artifact_download_bounds(manifest)
    _validate_manifest_payload(manifest, filename=manifest_filename)
    write_artifact_manifest(
        run_directory,
        manifest,
        filename=manifest_filename,
    )
    ARTIFACT_VOLUME.commit()
    return {
        **result,
        "run_id": run_id,
        "artifact_uri": volume_artifact_uri(run_id),
        "artifact_manifest_sha256": manifest.manifest_sha256,
        "image_source_sha256": IMAGE_SOURCE_SHA256,
    }


def _record_failure(
    run_directory: Path,
    error: BaseException,
) -> None:
    _write_bounded_run_json(
        run_directory,
        "remote_action_failure.json",
        {
            "error_type": type(error).__name__,
            # Raw exception text can contain credentials, absolute image paths,
            # or mounted Volume paths. Persist only a path-free classification.
            "message": "remote action failed; details suppressed",
        },
    )


def _file_contains_bytes(path: Path, needle: bytes) -> bool:
    """Search a regular file without retaining or deriving the secret value."""

    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            combined = overlap + chunk
            if needle in combined:
                return True
            overlap_size = len(needle) - 1
            overlap = combined[-overlap_size:] if overlap_size else b""
    return False


def _purge_provider_credential_files(run_directory: Path) -> None:
    """Delete all generated files containing the active provider credential."""

    key_bytes = _provider_credential_bytes()
    offending: list[Path] = []
    for path in sorted(run_directory.rglob("*")):
        if path.is_symlink():
            if key_bytes in path.relative_to(run_directory).as_posix().encode("utf-8"):
                offending.append(path)
            continue
        if not path.is_file():
            continue
        relative_bytes = path.relative_to(run_directory).as_posix().encode("utf-8")
        if key_bytes in relative_bytes or _file_contains_bytes(path, key_bytes):
            offending.append(path)
    for path in offending:
        path.unlink()
    if offending:
        # Deliberately do not include file names, the key, a digest, or content.
        raise RemoteActionError("provider credential material reached run artifacts")


def _reject_provider_credential_payload(payload: dict[str, Any]) -> None:
    """Reject a return payload containing the provider key before it is written."""

    key_bytes = _provider_credential_bytes()
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if key_bytes in encoded:
        raise RemoteActionError("provider credential material reached result payload")


def _publish_artifact_snapshots(
    snapshots: tuple[tuple[PurePosixPath, bytes], ...],
    run_directory: Path,
) -> None:
    """Exclusively publish one immutable off-Volume byte snapshot."""

    created_files: list[Path] = []
    created_directories: list[Path] = []
    try:
        for relative, payload in snapshots:
            current = run_directory
            for part in relative.parts[:-1]:
                current = current / part
                try:
                    metadata = current.lstat()
                except FileNotFoundError:
                    try:
                        current.mkdir(mode=0o700)
                    except FileExistsError:
                        metadata = current.lstat()
                    else:
                        created_directories.append(current)
                        metadata = current.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise RemoteActionError(
                        "staged artifact parent collides with run provenance"
                    )

            destination = run_directory.joinpath(*relative.parts)
            try:
                with destination.open("xb") as writer:
                    created_files.append(destination)
                    remaining = memoryview(payload)
                    while remaining:
                        written = writer.write(remaining)
                        if not isinstance(written, int) or written <= 0:
                            raise RemoteActionError(
                                "staged artifact publication made no progress"
                            )
                        remaining = remaining[written:]
                    writer.flush()
                    os.fsync(writer.fileno())
            except FileExistsError as error:
                raise RemoteActionError(
                    "staged artifact collides with run provenance"
                ) from error
    except BaseException:
        cleanup_failed = False
        for path in reversed(created_files):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                cleanup_failed = True
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except FileNotFoundError:
                continue
            except OSError:
                cleanup_failed = True
        if cleanup_failed:
            raise RemoteActionError(
                "staged artifact publication cleanup failed"
            ) from None
        raise


def _artifact_destination(
    run_directory: Path,
    relative: PurePosixPath,
) -> Path:
    """Preflight one publication path without creating anything on the Volume."""

    try:
        root_metadata = run_directory.lstat()
    except OSError as error:
        raise RemoteActionError(
            "run artifact directory is missing or unsafe"
        ) from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RemoteActionError("run artifact directory is missing or unsafe")
    current = run_directory
    for part in relative.parts[:-1]:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RemoteActionError(
                "staged artifact parent collides with run provenance"
            )
    destination = run_directory.joinpath(*relative.parts)
    try:
        destination.lstat()
    except FileNotFoundError:
        return destination
    raise RemoteActionError("staged artifact collides with run provenance")


def _snapshot_staged_artifacts(
    staging_directory: Path,
    run_directory: Path,
    *,
    scan_provider_credential: bool,
    reserved_artifacts: tuple[tuple[str, bytes], ...] = (),
) -> tuple[tuple[PurePosixPath, bytes], ...]:
    """Capture every stable byte and reserve the complete retained byte budget."""

    try:
        root_metadata = staging_directory.lstat()
    except OSError as error:
        raise RemoteActionError("staging directory is missing or unsafe") from error
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise RemoteActionError("staging directory is missing or unsafe")

    retained_and_reserved = _ensure_retained_artifact_budget(
        run_directory,
        future_payloads=tuple(payload for _, payload in reserved_artifacts),
    )

    preflight_files: list[tuple[Path, PurePosixPath, os.stat_result]] = []
    declared_total_bytes = 0
    for path in sorted(staging_directory.rglob("*")):
        try:
            metadata = path.lstat()
        except OSError as error:
            raise RemoteActionError(
                "staging artifact changed during snapshot"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RemoteActionError("staging artifacts may not use symlinks")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RemoteActionError("staging contains a non-regular artifact")
        try:
            relative = safe_relative_path(
                path.relative_to(staging_directory).as_posix()
            )
        except ValueError as error:
            raise RemoteActionError("staging artifact path is unsafe") from error
        _artifact_destination(run_directory, relative)
        if metadata.st_size > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
            raise RemoteActionError("staged artifact exceeds the per-file byte cap")
        declared_total_bytes += metadata.st_size
        if (
            retained_and_reserved + declared_total_bytes
            > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES
        ):
            raise RemoteActionError("staged artifacts exceed the aggregate byte cap")
        preflight_files.append((path, relative, metadata))

    credential = _provider_credential_bytes() if scan_provider_credential else None
    snapshots: list[tuple[PurePosixPath, bytes]] = []
    total_bytes = 0
    for path, relative, metadata in preflight_files:
        if credential is not None and credential in relative.as_posix().encode("utf-8"):
            raise RemoteActionError(
                "provider credential material reached run artifacts"
            )
        chunks: list[bytes] = []
        size = 0
        try:
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                opened_metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened_metadata.st_mode)
                    or opened_metadata.st_nlink != 1
                ):
                    raise RemoteActionError("staging contains a non-regular artifact")
                if (
                    opened_metadata.st_dev != metadata.st_dev
                    or opened_metadata.st_ino != metadata.st_ino
                    or opened_metadata.st_size != metadata.st_size
                ):
                    raise RemoteActionError(
                        "staging artifact changed during snapshot"
                    )
                with os.fdopen(descriptor, "rb", closefd=False) as reader:
                    while chunk := reader.read(1024 * 1024):
                        next_size = size + len(chunk)
                        next_total = retained_and_reserved + total_bytes + next_size
                        if next_size > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
                            raise RemoteActionError(
                                "staged artifact exceeds the per-file byte cap"
                            )
                        if next_total > MAX_ARTIFACT_DOWNLOAD_TOTAL_BYTES:
                            raise RemoteActionError(
                                "staged artifacts exceed the aggregate byte cap"
                            )
                        chunks.append(chunk)
                        size = next_size
                closed_metadata = os.fstat(descriptor)
                if (
                    closed_metadata.st_dev != opened_metadata.st_dev
                    or closed_metadata.st_ino != opened_metadata.st_ino
                    or closed_metadata.st_size != opened_metadata.st_size
                    or closed_metadata.st_mtime_ns != opened_metadata.st_mtime_ns
                    or size != opened_metadata.st_size
                ):
                    raise RemoteActionError(
                        "staging artifact changed during snapshot"
                    )
            finally:
                os.close(descriptor)
        except OSError as error:
            raise RemoteActionError(
                "staging artifact changed during snapshot"
            ) from error
        payload = b"".join(chunks)
        if credential is not None and credential in payload:
            raise RemoteActionError(
                "provider credential material reached run artifacts"
            )
        total_bytes += len(payload)
        snapshots.append((relative, payload))
    captured = tuple(snapshots)
    _validate_prospective_artifact_manifest(
        run_directory,
        captured,
        reserved_artifacts,
    )
    return captured


def _snapshot_provider_artifacts(
    staging_directory: Path,
    run_directory: Path,
    *,
    reserved_artifacts: tuple[tuple[str, bytes], ...] = (),
) -> tuple[tuple[PurePosixPath, bytes], ...]:
    return _snapshot_staged_artifacts(
        staging_directory,
        run_directory,
        scan_provider_credential=True,
        reserved_artifacts=reserved_artifacts,
    )


def _publish_staged_artifacts(
    staging_directory: Path,
    run_directory: Path,
    *,
    scan_provider_credential: bool,
    reserved_artifacts: tuple[tuple[str, bytes], ...] = (),
) -> None:
    if scan_provider_credential:
        snapshots = _snapshot_provider_artifacts(
            staging_directory,
            run_directory,
            reserved_artifacts=reserved_artifacts,
        )
    else:
        snapshots = _snapshot_staged_artifacts(
            staging_directory,
            run_directory,
            scan_provider_credential=False,
            reserved_artifacts=reserved_artifacts,
        )
    _publish_artifact_snapshots(snapshots, run_directory)


def _publish_scanned_provider_artifacts(
    staging_directory: Path,
    run_directory: Path,
    *,
    reserved_artifacts: tuple[tuple[str, bytes], ...] = (),
) -> None:
    _publish_staged_artifacts(
        staging_directory,
        run_directory,
        scan_provider_credential=True,
        reserved_artifacts=reserved_artifacts,
    )


def _publish_failed_provider_attempt_ledger(
    staging_directory: Path,
    run_directory: Path,
    *,
    harness: str,
    context: ExecutionContextV1,
    reserved_artifacts: tuple[tuple[str, bytes], ...] = (),
) -> None:
    """Persist only a validated attempt ledger from a failed provider action.

    All other partial controller artifacts remain quarantined off-Volume.  The
    ledger is first copied through an ``O_NOFOLLOW`` byte snapshot into a new
    private directory, strictly parsed there, and then passed through the same
    credential scanner and create-only publisher used by successful actions.
    """

    source = (
        staging_directory / "controller" / PROVIDER_ATTEMPT_LEDGER_FILENAME
    )
    ledger_state = "missing"
    try:
        metadata = source.lstat()
    except FileNotFoundError:
        payload = b""
    except OSError as error:
        raise RemoteActionError(
            "provider failure attempt ledger is unavailable"
        ) from error
    else:
        ledger_state = "present"
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or metadata.st_size > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES
        ):
            raise RemoteActionError("provider failure attempt ledger is unsafe")
        try:
            descriptor = os.open(
                source,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_dev != metadata.st_dev
                    or opened.st_ino != metadata.st_ino
                    or opened.st_size != metadata.st_size
                ):
                    raise RemoteActionError(
                        "provider failure attempt ledger changed during snapshot"
                    )
                chunks: list[bytes] = []
                size = 0
                while chunk := os.read(descriptor, 64 * 1024):
                    size += len(chunk)
                    if size > MAX_ARTIFACT_DOWNLOAD_FILE_BYTES:
                        raise RemoteActionError(
                            "provider failure attempt ledger exceeds its size cap"
                        )
                    chunks.append(chunk)
                if size != opened.st_size:
                    raise RemoteActionError(
                        "provider failure attempt ledger changed during snapshot"
                    )
            finally:
                os.close(descriptor)
        except OSError as error:
            raise RemoteActionError(
                "provider failure attempt ledger changed during snapshot"
            ) from error
        payload = b"".join(chunks)
    with tempfile.TemporaryDirectory(
        prefix="rl4rl-provider-failure-ledger-",
        ignore_cleanup_errors=True,
    ) as temporary:
        snapshot_root = Path(temporary)
        snapshot_controller = snapshot_root / "controller"
        snapshot_controller.mkdir(mode=0o700)
        snapshot_path = snapshot_controller / PROVIDER_ATTEMPT_LEDGER_FILENAME
        with snapshot_path.open("xb") as writer:
            writer.write(payload)
            writer.flush()
            os.fsync(writer.fileno())
        try:
            records = load_provider_attempt_ledger(snapshot_path)
        except (OSError, TypeError, ValueError) as error:
            raise RemoteActionError(
                "provider failure attempt ledger is invalid"
            ) from error
        if len(records) > 1:
            raise RemoteActionError(
                "one-opportunity provider failure ledger contains multiple attempts"
            )
        for record in records:
            if (
                record.harness != harness
                or record.action != "one_opportunity_engineering_canary"
                or record.execution_backend != "modal"
                or record.action_run_id != context.run_id
                or record.modal_call_id != context.modal_call_id
                or record.api_endpoint != OFFICIAL_OPENAI_API_BASE
                or record.model != TARGET_MODEL
                or record.status not in {"success", "error"}
            ):
                raise RemoteActionError(
                    "provider failure attempt ledger differs from its canary context"
                )
        # A terminal record is appended only after the provider operation
        # returns or raises. An empty initialized ledger therefore cannot prove
        # that transport never started: abrupt child death can leave the request
        # in flight without a terminal append. Publish an explicit upper bound
        # of one so billing and recovery logic can never count this as zero.
        if not records:
            _atomic_json(
                snapshot_controller / "provider_request_start_uncertain.json",
                {
                    "schema_name": "ProviderRequestStartUncertainEvidence",
                    "schema_version": "1.0",
                    "harness": harness,
                    "action": "one_opportunity_engineering_canary",
                    "execution_backend": "modal",
                    "action_run_id": context.run_id,
                    "modal_call_id": context.modal_call_id,
                    "api_endpoint": OFFICIAL_OPENAI_API_BASE,
                    "model": TARGET_MODEL,
                    "provider_attempt_count_lower_bound": 0,
                    "provider_attempt_count_upper_bound": 1,
                    "provider_request_started": "unknown",
                    "provider_attempt_ledger_state": ledger_state,
                    "billing_treatment": "reserve_one_full_approved_request",
                    "reason": "controller_terminated_without_terminal_attempt_record",
                },
            )
        _publish_scanned_provider_artifacts(
            snapshot_root,
            run_directory,
            reserved_artifacts=reserved_artifacts,
        )


def _execute_new_run(
    function_name: str,
    run_id: str,
    action,
    *,
    manifest_filename: str = "artifact_manifest.json",
    scan_provider_credential: bool | None = None,
) -> dict[str, Any]:
    required_scan = FUNCTION_SPECS[function_name].provider_secret
    if scan_provider_credential is None:
        scan_provider_credential = required_scan
    if scan_provider_credential is not required_scan:
        raise ValueError(
            "provider credential scan must match the function specification"
        )
    run_directory, context = _prepare_run(function_name, run_id)
    action_staging = tempfile.TemporaryDirectory(
        prefix=f"rl4rl-action-{run_id}-",
        ignore_cleanup_errors=True,
    )
    action_directory = Path(action_staging.name) / "run"
    action_directory.mkdir(mode=0o700)
    if action_directory.resolve().is_relative_to(Path(VOLUME_MOUNT_PATH).resolve()):
        action_staging.cleanup()
        raise RemoteActionError("action staging may not reside on the Volume")
    try:
        if function_name in PROVIDER_FREE_NETWORK_PROBE_FUNCTIONS:
            probe = _provider_free_network_denial_probe(context)
            _atomic_json(
                action_directory / "provider_free_network_denial_probe.json",
                probe,
            )
        result = action(action_directory, context)
        success_result = {"success": True, **result}
        if scan_provider_credential:
            _reject_provider_credential_payload(success_result)
            harness = function_name.removeprefix("canary_")
            if harness not in CANARY_ORDER:
                raise RemoteActionError(
                    "provider action is not a frozen engineering canary"
                )
            # Import only inside an executing provider function.  The validator
            # is provider-free and reads the private off-Volume tree before the
            # credential scanner snapshots any bytes for publication.
            from scripts.validate_engineering_canaries import (
                validate_private_canary_staging,
            )

            validate_private_canary_staging(
                action_directory / "controller",
                harness=harness,
                execution_context=context,
            )
        _publish_staged_artifacts(
            action_directory,
            run_directory,
            scan_provider_credential=scan_provider_credential,
            reserved_artifacts=(
                ("remote_action_result.json", _json_bytes(success_result)),
            ),
        )
    except ProviderFreeNetworkExposureError as error:
        _record_failure(run_directory, error)
        # The pre-action lease is already durable. Do not create an acceptance
        # manifest or make another commit: successful egress quarantines this ID.
        raise
    except ProcessGroupClosureError:
        # The mutable tree is private and off-Volume. Retain only the durable
        # provenance lease; do not publish, finalize, or make another commit.
        raise
    except BaseException:
        error = sys.exception()
        if error is None:  # pragma: no cover - guaranteed inside except
            raise RuntimeError(
                "remote action failed without an active exception"
            ) from None
        if scan_provider_credential:
            try:
                _purge_provider_credential_files(action_directory)
            except RemoteActionError as scan_error:
                error = scan_error
            try:
                failure_result = {
                    "success": False,
                    "error_type": type(error).__name__,
                }
                _publish_failed_provider_attempt_ledger(
                    action_directory,
                    run_directory,
                    harness=function_name.removeprefix("canary_"),
                    context=context,
                    reserved_artifacts=(
                        (
                            "remote_action_failure.json",
                            _json_bytes(
                                {
                                    "error_type": type(error).__name__,
                                    "message": (
                                        "remote action failed; details suppressed"
                                    ),
                                }
                            ),
                        ),
                        (
                            "remote_action_result.json",
                            _json_bytes(failure_result),
                        ),
                    ),
                )
            except RemoteActionError as ledger_error:
                error = ledger_error
        failure_result = {
            "success": False,
            "error_type": type(error).__name__,
        }
        _record_failure(
            run_directory,
            error,
        )
        _finalize_run(
            run_directory,
            run_id=run_id,
            result=failure_result,
            manifest_filename=manifest_filename,
        )
        raise error from None
    finally:
        action_staging.cleanup()
    return _finalize_run(
        run_directory,
        run_id=run_id,
        result=success_result,
        manifest_filename=manifest_filename,
    )


def _offline_smoke_action(run_directory: Path, context: ExecutionContextV1):
    output = run_directory / "offline_study"
    command = [
        sys.executable,
        str(REMOTE_PROJECT_ROOT / "scripts" / "study_offline_smoke.py"),
        "--output-dir",
        str(output),
        "--study-id",
        f"modal-offline-{context.run_id}",
        "--blocks",
        "1",
        "--opportunities",
        "1",
    ]
    return {
        "mode": "provider_free_offline_smoke",
        **_run_command(
            command,
            context=context,
            provider=False,
            requested_device="cpu",
        ),
    }


def _git_runtime_version() -> str:
    """Prove the immutable image contains the Greedy controller dependency."""

    try:
        git_probe = subprocess.run(
            ["/usr/bin/git", "--version"],
            env={"PATH": "/usr/bin:/bin"},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RemoteActionError(
            "Modal image is missing its required Git runtime"
        ) from error
    git_version = git_probe.stdout.strip()
    if not git_version.startswith("git version "):
        raise RemoteActionError("Modal image returned an invalid Git runtime version")
    return git_version


def _cuda_environment_action(run_directory: Path, context: ExecutionContextV1):
    import platform

    import torch
    from common.device import AcceleratorFingerprint, accelerator_fingerprint

    if not torch.cuda.is_available():
        raise RemoteActionError("T4 function started without an available CUDA device")
    count = int(torch.cuda.device_count())
    if count != 1:
        raise RemoteActionError(f"expected exactly one visible GPU, observed {count}")
    device = torch.device("cuda:0")
    fingerprint = accelerator_fingerprint(device, requested_device="cuda")
    try:
        fingerprint = AcceleratorFingerprint.from_dict(
            fingerprint.to_dict()
        ).validate_cuda(exact_gpu_count=1, require_driver=True)
    except ValueError as error:
        raise RemoteActionError(
            f"invalid CUDA accelerator fingerprint: {error}"
        ) from None
    if fingerprint.selected_device != str(device):
        raise RemoteActionError(
            "CUDA accelerator fingerprint selected an unexpected device"
        )
    capability = fingerprint.compute_capability
    if capability is None:  # validate_cuda makes this unreachable.
        raise RemoteActionError("CUDA compute capability is unavailable")
    capability_parts = [int(part) for part in capability.split(".")]
    properties = torch.cuda.get_device_properties(device)
    git_version = _git_runtime_version()
    report = {
        "python": platform.python_version(),
        "platform": fingerprint.host_platform,
        "torch": fingerprint.torch_version,
        "cuda_available": True,
        "cuda_device_count": fingerprint.gpu_count,
        "cuda_device_name": fingerprint.gpu_name,
        "cuda_compute_capability": capability_parts,
        "cuda_runtime": fingerprint.cuda_runtime,
        "cuda_driver": fingerprint.cuda_driver,
        "cuda_total_memory_bytes": int(properties.total_memory),
        "git_version": git_version,
        "accelerator_fingerprint": fingerprint.to_dict(),
        "execution_context": context.to_dict(),
    }
    _atomic_json(run_directory / "cuda_environment.json", report)
    return {"mode": "cuda_environment", "observed_gpu": report["cuda_device_name"]}


def _candidate_smoke_action(
    run_directory: Path,
    context: ExecutionContextV1,
    *,
    seed: int,
):
    command = [
        sys.executable,
        str(REMOTE_PROJECT_ROOT / "scripts" / "retrain_candidate.py"),
        "--candidate",
        str(REMOTE_PROJECT_ROOT / "common" / "initial_candidate.ir.json"),
        "--profile",
        CUDA_SMOKE_PROFILE,
        "--seeds",
        str(seed),
        "--device",
        "cuda",
        "--output-dir",
        str(run_directory / "candidate_smoke"),
        "--layer-a-cases",
        "24",
        "--evaluation-profile",
        "smoke_eval_v1",
    ]
    return {
        "mode": "cuda_candidate_train_and_layer_a",
        **_run_command(
            command,
            context=context,
            provider=False,
            requested_device="cuda",
        ),
    }


def _manifest_artifact_named(
    manifest: ArtifactManifestV1,
    filename: str,
):
    matches = tuple(
        item
        for item in manifest.files
        if PurePosixPath(item.relative_path).name == filename
    )
    if len(matches) != 1:
        raise RemoteActionError(
            f"source checkpoint manifest must bind exactly one {filename}"
        )
    return matches[0]


def _verify_source_run_unchanged(
    source_directory: Path,
    source_manifest_path: Path,
    source_manifest: ArtifactManifestV1,
    *,
    expected_manifest_file_sha256: str,
    expected_verification: dict[str, Any],
) -> None:
    observed_verification = verify_artifact_manifest(
        source_directory,
        source_manifest,
    )
    observed_manifest_file_sha256 = hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest()
    if (
        observed_verification != expected_verification
        or observed_manifest_file_sha256 != expected_manifest_file_sha256
        or load_artifact_manifest(source_manifest_path).to_dict()
        != source_manifest.to_dict()
    ):
        raise RemoteActionError("source candidate-smoke evidence changed")


def _read_bounded_json_object(path: Path, *, maximum_bytes: int) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RemoteActionError("source JSON artifact is missing or unsafe")
    if path.stat().st_size > maximum_bytes:
        raise RemoteActionError("source JSON artifact exceeds its size limit")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError("source JSON artifact contains a duplicate key")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise RemoteActionError("source JSON artifact is invalid") from error
    if not isinstance(payload, dict):
        raise RemoteActionError("source JSON artifact must be an object")
    return payload


def _open_nofollow_directory(path: Path) -> int:
    """Open one absolute directory by descriptors without following ancestors."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            before = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError("receipt parent contains an unsafe component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            opened = os.fstat(next_descriptor)
            if (before.st_dev, before.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                os.close(next_descriptor)
                raise ValueError("receipt parent changed while it was opened")
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _reopen_exclusive_json_receipt(
    destination: Path,
    *,
    maximum_bytes: int,
) -> tuple[dict[str, Any], str]:
    """Reopen a create-only JSON leaf through a stable no-follow parent."""

    absolute = Path(os.path.abspath(os.fspath(destination)))
    parent_descriptor = _open_nofollow_directory(absolute.parent)
    try:
        parent_identity = os.fstat(parent_descriptor)
        before = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_size > maximum_bytes
            or stat.S_IMODE(before.st_mode) & 0o077
        ):
            raise ValueError("published receipt leaf is unsafe")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            absolute.name,
            flags,
            dir_fd=parent_descriptor,
        )
        try:
            opened = os.fstat(descriptor)
            if (
                (before.st_dev, before.st_ino, before.st_size)
                != (opened.st_dev, opened.st_ino, opened.st_size)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
            ):
                raise ValueError("published receipt changed before reopen")
            chunks: list[bytes] = []
            size = 0
            while chunk := os.read(descriptor, 64 * 1024):
                size += len(chunk)
                if size > maximum_bytes:
                    raise ValueError("published receipt exceeds its size limit")
                chunks.append(chunk)
            after = os.fstat(descriptor)
            if (
                (opened.st_dev, opened.st_ino, opened.st_size)
                != (after.st_dev, after.st_ino, after.st_size)
                or size != opened.st_size
            ):
                raise ValueError("published receipt changed while reopened")
        finally:
            os.close(descriptor)
        final_leaf = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            stat.S_ISLNK(final_leaf.st_mode)
            or (before.st_dev, before.st_ino, before.st_size)
            != (final_leaf.st_dev, final_leaf.st_ino, final_leaf.st_size)
        ):
            raise ValueError("published receipt leaf changed during reopen")
        reopened_parent = _open_nofollow_directory(absolute.parent)
        try:
            observed_parent = os.fstat(reopened_parent)
            if (parent_identity.st_dev, parent_identity.st_ino) != (
                observed_parent.st_dev,
                observed_parent.st_ino,
            ):
                raise ValueError("published receipt parent changed during reopen")
        finally:
            os.close(reopened_parent)
    finally:
        os.close(parent_descriptor)

    raw = b"".join(chunks)

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError("published receipt contains a duplicate key")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("published receipt JSON is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError("published receipt JSON must be an object")
    return payload, hashlib.sha256(raw).hexdigest()


def _copy_manifest_artifact(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if source.is_symlink() or not source.is_file():
        raise RemoteActionError("manifest-bound source artifact is unsafe")
    if destination.exists() or destination.is_symlink():
        raise RemoteActionError("resume attempt artifact destination already exists")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if size != expected_size or digest.hexdigest() != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RemoteActionError("copied resume artifact failed its source manifest")


def _copy_resume_event_prefix(
    source: Path,
    destination: Path,
    *,
    partial_step: int,
    profile_max_steps: int,
    global_batch_size: int,
) -> tuple[int, str]:
    if source.is_symlink() or not source.is_file():
        raise RemoteActionError("source training event log is missing or unsafe")
    if source.stat().st_size > 64 * 1024 * 1024:
        raise RemoteActionError("source training event log exceeds 64 MiB")
    if destination.exists() or destination.is_symlink():
        raise RemoteActionError("resume attempt event log already exists")
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    retained = 0
    total = 0
    try:
        with source.open("r", encoding="utf-8") as reader, destination.open(
            "x", encoding="utf-8"
        ) as writer:
            for line_number, line in enumerate(reader, start=1):
                total = line_number
                if not line.endswith("\n") or not line.strip():
                    raise RemoteActionError(
                        "source training event log contains an incomplete record"
                    )
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RemoteActionError(
                        "source training event log contains invalid JSON"
                    ) from error
                if not isinstance(event, dict):
                    raise RemoteActionError(
                        "source training event log contains a non-object record"
                    )
                step = event.get("optimizer_step")
                examples = event.get("examples_processed")
                if (
                    not isinstance(step, int)
                    or isinstance(step, bool)
                    or step != line_number
                    or not isinstance(examples, int)
                    or isinstance(examples, bool)
                    or examples != step * global_batch_size
                ):
                    raise RemoteActionError(
                        "source training event progression is not contiguous"
                    )
                if line_number > profile_max_steps:
                    raise RemoteActionError(
                        "source training events exceed the profile maximum"
                    )
                if step <= partial_step:
                    writer.write(line)
                    retained += 1
            writer.flush()
            os.fsync(writer.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if total != profile_max_steps or retained != partial_step:
        destination.unlink(missing_ok=True)
        raise RemoteActionError(
            "source events do not prove the complete smoke trajectory"
        )
    return retained, hashlib.sha256(destination.read_bytes()).hexdigest()


def _validate_source_partial_checkpoint(
    checkpoint_path: Path,
    *,
    candidate_sha256: str,
    seed: int,
) -> dict[str, Any]:
    import torch
    from common.training_config import TrainingSeedBundle, get_training_profile

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise RemoteActionError(
            "source partial checkpoint is not weights-only loadable"
        ) from error
    if not isinstance(checkpoint, dict):
        raise RemoteActionError("source partial checkpoint must be a mapping")
    profile = get_training_profile(CUDA_SMOKE_PROFILE)
    expected_step = profile.checkpoint_interval
    if expected_step != 5:
        raise RemoteActionError("smoke profile checkpoint interval is not step 5")
    expected = {
        "checkpoint_kind": "trusted_resume_state_v2",
        "candidate_source_hash": candidate_sha256,
        "profile_hash": profile.profile_hash,
        "seed_bundle_hash": TrainingSeedBundle.from_run_seed(seed).bundle_hash,
        "global_step": expected_step,
        "next_data_step": expected_step,
        "examples_processed": expected_step * profile.global_batch_size,
    }
    if any(checkpoint.get(field) != value for field, value in expected.items()):
        raise RemoteActionError(
            "source partial checkpoint does not match the step-5 smoke identity"
        )
    if not 0 < expected_step < profile.max_steps:
        raise RemoteActionError("smoke profile lacks a nonterminal checkpoint step")
    return {
        "examples_processed": expected["examples_processed"],
        "optimizer_step": expected_step,
        "profile_hash": profile.profile_hash,
        "profile_max_steps": profile.max_steps,
        "global_batch_size": profile.global_batch_size,
    }


def _execute_staged_resume_attempt(
    *,
    run_directory: Path,
    context: ExecutionContextV1,
    source_run_id: str,
    run_id: str,
    source_manifest: ArtifactManifestV1,
    source_training_relative: PurePosixPath,
    source_partial: Path,
    source_candidate: Path,
    source_events: Path,
    partial_entry: Any,
    candidate_entry: Any,
    events_entry: Any,
    partial_identity: dict[str, Any],
    seed: int,
    deadline: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Stage and execute one fresh attempt inside its failure boundary."""

    training_directory = run_directory.joinpath(*source_training_relative.parts)
    partial_checkpoint = training_directory / "partial_resume_checkpoint.pt"
    latest_checkpoint = training_directory / "latest_resume_checkpoint.pt"
    candidate = training_directory / "candidate_graph.json"
    events = training_directory / "training_events.jsonl"
    _copy_manifest_artifact(
        source_partial,
        partial_checkpoint,
        expected_sha256=partial_entry.sha256,
        expected_size=partial_entry.size_bytes,
    )
    _copy_manifest_artifact(
        partial_checkpoint,
        latest_checkpoint,
        expected_sha256=partial_entry.sha256,
        expected_size=partial_entry.size_bytes,
    )
    partial_checkpoint.chmod(0o444)
    _copy_manifest_artifact(
        source_candidate,
        candidate,
        expected_sha256=candidate_entry.sha256,
        expected_size=candidate_entry.size_bytes,
    )
    candidate.chmod(0o444)
    retained_events, retained_event_sha256 = _copy_resume_event_prefix(
        source_events,
        events,
        partial_step=partial_identity["optimizer_step"],
        profile_max_steps=partial_identity["profile_max_steps"],
        global_batch_size=partial_identity["global_batch_size"],
    )
    source_binding = {
        "schema_name": "ResumeSourceBinding",
        "schema_version": "1.0",
        "source_run_id": source_run_id,
        "resume_attempt_run_id": run_id,
        "source_manifest_sha256": source_manifest.manifest_sha256,
        "source_image_sha256": source_manifest.image_source_sha256,
        "source_training_relative_path": source_training_relative.as_posix(),
        "partial_optimizer_step": partial_identity["optimizer_step"],
        "partial_examples_processed": partial_identity["examples_processed"],
        "retained_event_count": retained_events,
        "source_artifacts": {
            "candidate": candidate_entry.to_dict(),
            "partial_resume_checkpoint": partial_entry.to_dict(),
            "training_events": events_entry.to_dict(),
        },
        "attempt_artifacts": {
            "candidate_relative_path": candidate.relative_to(run_directory).as_posix(),
            "candidate_sha256": candidate_entry.sha256,
            "partial_checkpoint_relative_path": partial_checkpoint.relative_to(
                run_directory
            ).as_posix(),
            "partial_checkpoint_sha256": partial_entry.sha256,
            "initial_latest_checkpoint_relative_path": latest_checkpoint.relative_to(
                run_directory
            ).as_posix(),
            "initial_latest_checkpoint_sha256": partial_entry.sha256,
            "event_prefix_relative_path": events.relative_to(
                run_directory
            ).as_posix(),
            "event_prefix_sha256": retained_event_sha256,
        },
    }
    _atomic_json(run_directory / "resume_source_binding.json", source_binding)
    _atomic_json(run_directory / "resume_execution_context.json", context.to_dict())
    contract_evidence = run_directory / "resume_contract_verification.json"
    probe_command = [
        sys.executable,
        str(REMOTE_PROJECT_ROOT / "scripts" / "verify_resume_contract.py"),
        "--checkpoint",
        str(latest_checkpoint),
        "--candidate",
        str(candidate),
        "--profile",
        CUDA_SMOKE_PROFILE,
        "--seed",
        str(seed),
        "--output",
        str(contract_evidence),
    ]
    training_command = [
        sys.executable,
        str(REMOTE_PROJECT_ROOT / "scripts" / "train_candidate.py"),
        "--candidate",
        str(candidate),
        "--profile",
        CUDA_SMOKE_PROFILE,
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--output-dir",
        str(training_directory),
        "--resume",
        str(latest_checkpoint),
    ]
    progression_evidence = run_directory / "resume_progression_verification.json"
    progression_command = [
        sys.executable,
        str(REMOTE_PROJECT_ROOT / "scripts" / "verify_resume_progression.py"),
        "--artifact-root",
        str(run_directory),
        "--training-output-dir",
        str(training_directory),
        "--profile",
        CUDA_SMOKE_PROFILE,
        "--seed",
        str(seed),
        "--output",
        str(progression_evidence),
    ]
    probe_timeout = _resume_stage_timeout(
        deadline,
        maximum_seconds=RESUME_PROBE_TIMEOUT_SECONDS,
        reserve_after_seconds=(
            RESUME_TRAIN_TIMEOUT_SECONDS
            + RESUME_PROGRESSION_TIMEOUT_SECONDS
            + RESUME_INTERNAL_RESERVE_SECONDS
        ),
        stage="contract probes",
    )
    probe_result = _run_command(
        probe_command,
        context=context,
        provider=False,
        requested_device="cuda",
        timeout_seconds=probe_timeout,
    )
    training_timeout = _resume_stage_timeout(
        deadline,
        maximum_seconds=RESUME_TRAIN_TIMEOUT_SECONDS,
        reserve_after_seconds=(
            RESUME_PROGRESSION_TIMEOUT_SECONDS + RESUME_INTERNAL_RESERVE_SECONDS
        ),
        stage="training",
    )
    training_result = _run_command(
        training_command,
        context=context,
        provider=False,
        requested_device="cuda",
        timeout_seconds=training_timeout,
    )
    progression_timeout = _resume_stage_timeout(
        deadline,
        maximum_seconds=RESUME_PROGRESSION_TIMEOUT_SECONDS,
        reserve_after_seconds=RESUME_INTERNAL_RESERVE_SECONDS,
        stage="progression verification",
    )
    progression_result = _run_command(
        progression_command,
        context=context,
        provider=False,
        requested_device="cuda",
        timeout_seconds=progression_timeout,
    )
    return probe_result, training_result, progression_result


def _resume_stage_timeout(
    deadline: float,
    *,
    maximum_seconds: int,
    reserve_after_seconds: int,
    stage: str,
) -> int:
    """Allocate a stage from one resume deadline while preserving later work."""

    available = deadline - time.monotonic() - reserve_after_seconds
    timeout = min(maximum_seconds, int(available))
    if timeout <= 0:
        raise RemoteActionError(f"resume action deadline exhausted before {stage}")
    return timeout


def _resume_action(
    source_run_id: str,
    run_id: str,
    *,
    seed: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + RESUME_ACTION_DEADLINE_SECONDS
    source_run_id = validate_run_id(source_run_id)
    run_id = validate_run_id(run_id)
    if source_run_id == run_id:
        raise RemoteActionError(
            "resume attempt run ID must differ from the immutable source run ID"
        )
    ARTIFACT_VOLUME.reload()
    source_directory = resolve_existing_volume_run_directory(
        VOLUME_MOUNT_PATH,
        source_run_id,
        allow_mount_root_symlink=True,
    )
    source_manifest_path = _select_artifact_manifest(source_directory)
    if source_manifest_path.name != "artifact_manifest.checkpoint.json":
        raise RemoteActionError(
            "resume source requires exactly one checkpoint artifact manifest"
        )
    source_manifest = load_artifact_manifest(source_manifest_path)
    source_manifest_file_sha256 = hashlib.sha256(
        source_manifest_path.read_bytes()
    ).hexdigest()
    source_verification = verify_artifact_manifest(
        source_directory,
        source_manifest,
    )
    if source_manifest.run_id != source_run_id:
        raise RemoteActionError("source checkpoint manifest run identity differs")
    if source_manifest.image_source_sha256 != IMAGE_SOURCE_SHA256:
        raise RemoteActionError("source checkpoint image identity differs")
    source_result_entry = _manifest_artifact_named(
        source_manifest,
        "remote_action_result.json",
    )
    source_result = _read_bounded_json_object(
        source_directory / source_result_entry.relative_path,
        maximum_bytes=2 * 1024 * 1024,
    )
    if (
        source_result.get("success") is not True
        or source_result.get("mode") != "cuda_candidate_train_and_layer_a"
    ):
        raise RemoteActionError("source run is not a successful candidate smoke")
    partial_entry = _manifest_artifact_named(
        source_manifest,
        "partial_resume_checkpoint.pt",
    )
    candidate_entry = _manifest_artifact_named(
        source_manifest,
        "candidate_graph.json",
    )
    events_entry = _manifest_artifact_named(
        source_manifest,
        "training_events.jsonl",
    )
    source_parents = {
        PurePosixPath(entry.relative_path).parent
        for entry in (partial_entry, candidate_entry, events_entry)
    }
    if len(source_parents) != 1:
        raise RemoteActionError("source resume artifacts are not colocated")
    source_training_relative = next(iter(source_parents))
    source_partial = source_directory / partial_entry.relative_path
    source_candidate = source_directory / candidate_entry.relative_path
    source_events = source_directory / events_entry.relative_path
    partial_identity = _validate_source_partial_checkpoint(
        source_partial,
        candidate_sha256=candidate_entry.sha256,
        seed=seed,
    )
    run_directory, context = _prepare_run("checkpoint_resume", run_id)
    action_staging = tempfile.TemporaryDirectory(
        prefix=f"rl4rl-resume-{run_id}-",
        ignore_cleanup_errors=True,
    )
    action_directory = Path(action_staging.name) / "run"
    action_directory.mkdir(mode=0o700)
    if action_directory.resolve().is_relative_to(Path(VOLUME_MOUNT_PATH).resolve()):
        action_staging.cleanup()
        raise RemoteActionError("resume staging may not reside on the Volume")
    try:
        probe = _provider_free_network_denial_probe(context)
        _atomic_json(
            action_directory / "provider_free_network_denial_probe.json",
            probe,
        )
        probe_result, result, progression_result = _execute_staged_resume_attempt(
            run_directory=action_directory,
            context=context,
            source_run_id=source_run_id,
            run_id=run_id,
            source_manifest=source_manifest,
            source_training_relative=source_training_relative,
            source_partial=source_partial,
            source_candidate=source_candidate,
            source_events=source_events,
            partial_entry=partial_entry,
            candidate_entry=candidate_entry,
            events_entry=events_entry,
            partial_identity=partial_identity,
            seed=seed,
            deadline=deadline,
        )
        _verify_source_run_unchanged(
            source_directory,
            source_manifest_path,
            source_manifest,
            expected_manifest_file_sha256=source_manifest_file_sha256,
            expected_verification=source_verification,
        )
        success_result = {
            "success": True,
            "mode": "checkpoint_resume",
            "source_run_id": source_run_id,
            "source_manifest_sha256": source_manifest.manifest_sha256,
            "resume_contract_probe": probe_result,
            "resume_progression_verification": progression_result,
            **result,
        }
        _publish_staged_artifacts(
            action_directory,
            run_directory,
            scan_provider_credential=False,
            reserved_artifacts=(
                ("resume_action_result.json", _json_bytes(success_result)),
            ),
        )
    except ProviderFreeNetworkExposureError as error:
        _record_failure(run_directory, error)
        raise
    except ProcessGroupClosureError:
        _verify_source_run_unchanged(
            source_directory,
            source_manifest_path,
            source_manifest,
            expected_manifest_file_sha256=source_manifest_file_sha256,
            expected_verification=source_verification,
        )
        raise
    except BaseException:
        error = sys.exception()
        if error is None:  # pragma: no cover - guaranteed inside except
            raise RuntimeError("resume failed without an active exception") from None
        try:
            _verify_source_run_unchanged(
                source_directory,
                source_manifest_path,
                source_manifest,
                expected_manifest_file_sha256=source_manifest_file_sha256,
                expected_verification=source_verification,
            )
        except BaseException as source_error:
            error = source_error
        _record_failure(run_directory, error)
        _finalize_run(
            run_directory,
            run_id=run_id,
            result={
                "success": False,
                "mode": "checkpoint_resume",
                "error_type": type(error).__name__,
                "source_run_id": source_run_id,
            },
            result_filename="resume_action_result.json",
        )
        raise error from None
    finally:
        action_staging.cleanup()
    return _finalize_run(
        run_directory,
        run_id=run_id,
        result=success_result,
        result_filename="resume_action_result.json",
    )


_CANARY_MODULES = {
    "greedy_autoresearch": "agents.greedy_autoresearch.run",
    "semantic_autoresearch": "agents.semantic_autoresearch.run",
    "openevolve_generic": "agents.openevolve_generic.run",
    "openevolve_semantic": "agents.openevolve_semantic.run",
}

_PROVIDER_CANARY_ACTIONS = frozenset({"canary", "canaries"})


def _require_local_action_approvals(
    *,
    action: str,
    approved: bool,
    provider_approved: bool,
    harness: str,
    expected_image_source_sha256: str,
) -> None:
    """Enforce separate infrastructure and provider-spend consent locally."""

    if not approved:
        raise SystemExit(
            "remote action refused: obtain cost approval, then pass --approved"
        )
    if expected_image_source_sha256 != IMAGE_SOURCE_SHA256:
        raise SystemExit(
            "remote action refused: --expected-image-source-sha256 must equal "
            "the approved local Modal plan hash"
        )
    if action in _PROVIDER_CANARY_ACTIONS and not provider_approved:
        raise SystemExit(
            "provider action refused: obtain separate provider-cost approval, "
            "then pass --provider-approved"
        )
    if action == "canary" and harness not in CANARY_ORDER:
        raise SystemExit(
            "single canary requires --harness with one frozen ID: "
            + ", ".join(CANARY_ORDER)
        )
    if action == "canaries" and harness:
        raise SystemExit(
            "aggregate canaries use the frozen order and do not accept --harness"
        )


def _require_durable_local_action_intent(
    *,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    action: str,
    run_id: str,
    source_run_id: str | None,
    verifier_run_id: str | None,
    harness: str | None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed on any missing, swapped, corrupt, or mismatched intent."""

    selected_environment = os.environ if environment is None else environment
    try:
        return validate_local_action_intent_for_entrypoint(
            project_root=PROJECT_ROOT,
            identity=identity,
            attempt_id=attempt_id,
            expected_intent_sha256=selected_environment.get(
                MODAL_ACTION_INTENT_SHA256_ENV,
                "",
            ),
            launch_nonce=selected_environment.get(MODAL_LAUNCH_NONCE_ENV, ""),
            action=action,
            run_id=run_id,
            source_run_id=source_run_id,
            verifier_run_id=verifier_run_id,
            harness=harness,
            modal_command_python_executable=LOCAL_MODAL_COMMAND_PYTHON,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise SystemExit(
            "paid action requires its exact durable launch intent before any "
            "remote invocation"
        ) from error


def _run_single_canary_synchronously(
    functions: dict[str, Any],
    *,
    harness: str,
    run_id: str,
) -> dict[str, Any]:
    """Run exactly one frozen recovery canary with one opportunity."""

    if set(functions) != set(CANARY_ORDER):
        raise ValueError("canary function mapping must cover exactly four harnesses")
    if harness not in CANARY_ORDER:
        raise ValueError("single canary harness is not frozen")
    selected_run_id = validate_run_id(run_id)
    result = invoke_synchronously(
        functions[harness],
        run_id=selected_run_id,
        opportunities=1,
    )
    return {"harness": harness, "result": result}


def _provider_canary_aggregate_exit_code(result: object) -> int:
    if not isinstance(result, dict) or set(result) != {
        "schema_name",
        "schema_version",
        "run_id_prefix",
        "harness_order",
        "outcomes",
        "all_succeeded",
    }:
        raise ValueError("provider canary aggregate result has the wrong schema")
    if (
        result["schema_name"] != "ModalProviderCanaryAggregateResult"
        or result["schema_version"] != "1.0"
        or result["harness_order"] != list(CANARY_ORDER)
        or type(result["all_succeeded"]) is not bool
    ):
        raise ValueError("provider canary aggregate result is invalid")
    outcomes = result["outcomes"]
    run_id_prefix = result["run_id_prefix"]
    if not isinstance(run_id_prefix, str):
        raise ValueError("provider canary aggregate prefix is invalid")
    validate_run_id(run_id_prefix)
    if not isinstance(outcomes, list) or len(outcomes) != len(CANARY_ORDER):
        raise ValueError("provider canary aggregate outcomes are incomplete")
    expected_fields = {"harness", "run_id", "status", "result", "error_type"}
    for harness, outcome in zip(CANARY_ORDER, outcomes, strict=True):
        if (
            not isinstance(outcome, dict)
            or set(outcome) != expected_fields
            or outcome["harness"] != harness
            or outcome["run_id"]
            != f"{run_id_prefix}-{canary_run_suffix(harness)}"
            or outcome["status"] not in {"success", "failed"}
            or (outcome["status"] == "failed") != (outcome["result"] is None)
        ):
            raise ValueError("provider canary aggregate outcome is invalid")
        if outcome["status"] == "success":
            if outcome["error_type"] is not None:
                raise ValueError("successful canary outcome contains an error")
        elif (
            not isinstance(outcome["error_type"], str)
            or not outcome["error_type"].isidentifier()
            or len(outcome["error_type"]) > 128
        ):
            raise ValueError("failed canary outcome has an unsafe error class")
    observed_all_succeeded = all(
        outcome["status"] == "success" for outcome in outcomes
    )
    if result["all_succeeded"] is not observed_all_succeeded:
        raise ValueError("provider canary aggregate status is inconsistent")
    return 0 if observed_all_succeeded else 2


def _persist_provider_canary_aggregate_outcome(
    result: object,
    *,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
) -> Path:
    if not isinstance(result, dict):
        raise ValueError("provider canary aggregate result must be an object")
    payload = build_provider_canary_aggregate_outcome_receipt(
        result,
        identity=identity,
        attempt_id=attempt_id,
    )
    logical = provider_canary_aggregate_outcome_receipt_path(identity, attempt_id)
    raw_root = PROJECT_ROOT
    if raw_root.is_symlink():
        raise ValueError("aggregate outcome project root may not be a symlink")
    root = raw_root.resolve()
    destination = root.joinpath(*logical.parts)
    create_json_exclusive(destination, payload)
    reopened, raw_sha256 = _reopen_exclusive_json_receipt(
        destination,
        maximum_bytes=64 * 1024,
    )
    expected_raw_sha256 = hashlib.sha256(_exclusive_json_bytes(payload)).hexdigest()
    if raw_sha256 != expected_raw_sha256:
        raise ValueError("reopened provider canary aggregate bytes differ")
    validated = validate_provider_canary_aggregate_outcome_receipt(
        reopened,
        expected_attempt_id=attempt_id,
        expected_run_id_prefix=result["run_id_prefix"],
        expected_source_tree_sha256=identity.source_tree_sha256,
        expected_image_source_sha256=identity.image_source_sha256,
        expected_cohort_id=identity.cohort_id,
    )
    if validated != payload:
        raise ValueError("reopened provider canary aggregate outcome differs")
    return destination


def _canary_action(
    harness: str,
    run_directory: Path,
    context: ExecutionContextV1,
    *,
    opportunities: int,
):
    if harness not in CANARY_ORDER or opportunities != 1:
        raise ValueError("provider canaries require one frozen harness opportunity")
    command = [
        sys.executable,
        "-m",
        _CANARY_MODULES[harness],
        "--iterations",
        "1",
        "--seed",
        "1",
        "--output-dir",
        str(run_directory / "controller"),
        "--engineering-pilot",
        "--training-profile",
        CUDA_SMOKE_PROFILE,
        "--evaluation-profile",
        "smoke_eval_v1",
        "--evaluation-cases",
        "24",
        "--device",
        "cuda",
    ]
    return {
        "mode": "one_opportunity_engineering_canary",
        "harness": harness,
        **_run_command(
            command,
            context=context,
            provider=True,
            requested_device="cuda",
        ),
    }


def _load_volume_manifest(run_id: str) -> RawArtifactManifestV1:
    selected_run_id = validate_run_id(run_id)
    stored_manifests: list[tuple[str, bytes]] = []
    for filename in ARTIFACT_MANIFEST_FILENAMES:
        try:
            raw_chunks = ARTIFACT_VOLUME.read_file(
                f"/runs/{selected_run_id}/{filename}"
            )
            chunks = (raw_chunks,) if isinstance(raw_chunks, bytes) else raw_chunks
            payload_parts: list[bytes] = []
            payload_size = 0
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    raise ModalBoundaryError(
                        "Volume manifest reader yielded non-bytes"
                    )
                payload_size += len(chunk)
                if payload_size > 2 * 1024 * 1024:
                    raise ModalBoundaryError(
                        "stored artifact manifest exceeds 2 MiB"
                    )
                payload_parts.append(chunk)
        except Exception as error:
            if isinstance(error, FileNotFoundError) or type(error).__name__ == (
                "NotFoundError"
            ):
                continue
            raise
        stored_manifests.append((filename, b"".join(payload_parts)))
    if not stored_manifests:
        raise ModalBoundaryError("run has no final or checkpoint artifact manifest")
    if len(stored_manifests) != 1:
        raise ModalBoundaryError(
            "run must contain exactly one final or checkpoint artifact manifest"
        )
    filename, raw_bytes = stored_manifests[0]
    try:
        raw_manifest = RawArtifactManifestV1.from_bytes(
            filename=filename,
            raw_bytes=raw_bytes,
        )
    except (TypeError, ValueError, ModalBoundaryError) as error:
        raise ModalBoundaryError("stored artifact manifest is invalid") from error
    if raw_manifest.manifest.run_id != selected_run_id:
        raise ModalBoundaryError("stored artifact manifest run identity differs")
    return raw_manifest


def _select_artifact_manifest(run_directory: Path) -> Path:
    stored_manifests: list[Path] = []
    for filename in ARTIFACT_MANIFEST_FILENAMES:
        candidate = run_directory / filename
        try:
            metadata = candidate.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ModalBoundaryError(
                "stored artifact manifest could not be inspected"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ModalBoundaryError("stored artifact manifest may not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise ModalBoundaryError("stored artifact manifest must be a regular file")
        stored_manifests.append(candidate)
    if not stored_manifests:
        raise ModalBoundaryError("run has no final or checkpoint artifact manifest")
    if len(stored_manifests) != 1:
        raise ModalBoundaryError(
            "run must contain exactly one final or checkpoint artifact manifest"
        )
    return stored_manifests[0]


def _finalize_artifact_verifier_directory(
    verifier_directory: Path,
    *,
    verifier_run_id: str,
    filename: str,
    payload: dict[str, Any],
) -> None:
    if filename not in {
        "artifact_verification_result.json",
        "artifact_verification_failure.json",
    }:
        raise ValueError("artifact verifier receipt filename is unsafe")
    _write_bounded_run_json(verifier_directory, filename, payload)
    manifest = build_artifact_manifest(
        verifier_directory,
        run_id=verifier_run_id,
        image_source_sha256=IMAGE_SOURCE_SHA256,
    )
    validate_artifact_download_bounds(manifest)
    _validate_manifest_payload(manifest, filename="artifact_manifest.json")
    write_artifact_manifest(verifier_directory, manifest)
    ARTIFACT_VOLUME.commit()


def _artifact_verify_action(
    source_run_id: str,
    verifier_run_id: str,
) -> dict[str, Any]:
    selected_source_run_id = validate_run_id(source_run_id)
    selected_verifier_run_id = validate_run_id(verifier_run_id)
    if selected_source_run_id == selected_verifier_run_id:
        raise RemoteActionError(
            "artifact verifier run ID must differ from the source run ID"
        )
    verifier_directory, context = _prepare_run(
        "artifact_verify",
        selected_verifier_run_id,
        artifact_run_id=selected_source_run_id,
    )
    try:
        run_directory = resolve_existing_volume_run_directory(
            VOLUME_MOUNT_PATH,
            selected_source_run_id,
            allow_mount_root_symlink=True,
        )
        raw_manifest = load_raw_artifact_manifest(
            _select_artifact_manifest(run_directory)
        )
        verification = verify_artifact_manifest(
            run_directory,
            raw_manifest.manifest,
        )
        source_context = ExecutionContextV1.from_dict(
            _read_bounded_json_object(
                run_directory / "execution_context.json",
                maximum_bytes=64 * 1024,
            )
        )
        if (
            source_context.execution_backend != "modal"
            or source_context.run_id != selected_source_run_id
            or source_context.app_name != APP_NAME
            or source_context.artifact_uri
            != volume_artifact_uri(selected_source_run_id)
            or source_context.image_source_sha256
            != raw_manifest.manifest.image_source_sha256
            or source_context.modal_image_id is None
            or source_context.modal_image_id != context.modal_image_id
            or context.image_source_sha256
            != raw_manifest.manifest.image_source_sha256
        ):
            raise RemoteActionError(
                "artifact verifier differs from the source image cohort"
            )
        result = ArtifactVerificationV1(
            source_run_id=selected_source_run_id,
            verifier_run_id=selected_verifier_run_id,
            manifest_filename=raw_manifest.filename,
            raw_manifest_sha256=raw_manifest.raw_sha256,
            raw_manifest_size_bytes=raw_manifest.raw_size_bytes,
            canonical_manifest_sha256=raw_manifest.manifest.manifest_sha256,
            file_count=verification["file_count"],
            verifier_execution_context=context,
            verified=verification["verified"],
        ).to_dict()
    except BaseException as error:
        _finalize_artifact_verifier_directory(
            verifier_directory,
            verifier_run_id=selected_verifier_run_id,
            filename="artifact_verification_failure.json",
            payload={
                "schema_name": "ModalArtifactVerificationFailure",
                "schema_version": "1.0",
                "source_run_id": selected_source_run_id,
                "verifier_run_id": selected_verifier_run_id,
                "error_type": type(error).__name__,
                "message": "artifact verification failed; details suppressed",
                "verifier_execution_context": context.to_dict(),
            },
        )
        raise
    _finalize_artifact_verifier_directory(
        verifier_directory,
        verifier_run_id=selected_verifier_run_id,
        filename="artifact_verification_result.json",
        payload=result,
    )
    return result


def _validated_artifact_verification(
    *,
    source_run_id: str,
    verifier_run_id: str,
    remote_verification: dict[str, Any],
    raw_manifest: RawArtifactManifestV1,
    source_execution_context: ExecutionContextV1 | None = None,
) -> ArtifactVerificationV1:
    try:
        verification = ArtifactVerificationV1.from_dict(remote_verification)
    except (TypeError, ValueError) as error:
        raise ModalBoundaryError(
            "remote verification has an invalid exact schema"
        ) from error
    expected = {
        "source_run_id": validate_run_id(source_run_id),
        "verifier_run_id": validate_run_id(verifier_run_id),
        "manifest_filename": raw_manifest.filename,
        "raw_manifest_sha256": raw_manifest.raw_sha256,
        "raw_manifest_size_bytes": raw_manifest.raw_size_bytes,
        "canonical_manifest_sha256": raw_manifest.manifest.manifest_sha256,
        "file_count": len(raw_manifest.manifest.files),
        "verified": True,
    }
    observed = verification.to_dict()
    for field, expected_value in expected.items():
        if observed[field] != expected_value:
            raise ModalBoundaryError(
                f"remote verification {field} differs from the raw manifest"
            )
    verifier_context = verification.verifier_execution_context
    if (
        verifier_context.image_source_sha256
        != raw_manifest.manifest.image_source_sha256
    ):
        raise ModalBoundaryError(
            "remote verifier image source differs from the source manifest"
        )
    if source_execution_context is not None:
        if (
            source_execution_context.execution_backend != "modal"
            or source_execution_context.run_id != verification.source_run_id
            or source_execution_context.app_name != APP_NAME
            or source_execution_context.artifact_uri
            != volume_artifact_uri(verification.source_run_id)
            or source_execution_context.image_source_sha256
            != raw_manifest.manifest.image_source_sha256
        ):
            raise ModalBoundaryError(
                "downloaded source execution context differs from its manifest"
            )
        if (
            source_execution_context.modal_image_id is None
            or verifier_context.modal_image_id
            != source_execution_context.modal_image_id
        ):
            raise ModalBoundaryError(
                "remote verifier image ID differs from the downloaded source"
            )
    return verification


def _persist_download_verification(
    *,
    source_run_id: str,
    verifier_run_id: str,
    remote_verification: dict[str, Any],
    raw_manifest: RawArtifactManifestV1,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    project_root: Path = PROJECT_ROOT,
) -> tuple[str, Path]:
    """Save the exact remote verification already used by ``download``."""

    verification = _validated_artifact_verification(
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        remote_verification=remote_verification,
        raw_manifest=raw_manifest,
    )
    logical_path = modal_remote_verification_receipt_path(
        identity,
        verification.source_run_id,
        verification.verifier_run_id,
        attempt_id,
    )
    logical = logical_path.as_posix()
    destination = project_root / logical
    expected_payload = verification.to_dict()
    create_json_exclusive(destination, expected_payload)
    reopened, raw_sha256 = _reopen_exclusive_json_receipt(
        destination,
        maximum_bytes=64 * 1024,
    )
    expected_raw_sha256 = hashlib.sha256(
        _exclusive_json_bytes(expected_payload)
    ).hexdigest()
    if reopened != expected_payload or raw_sha256 != expected_raw_sha256:
        raise ValueError("reopened remote verification receipt differs")
    return logical, destination


def _download_after_verification_receipt(
    *,
    source_run_id: str,
    verifier_run_id: str,
    remote_verification: dict[str, Any],
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    local_root: Path,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, str]:
    """Persist verifier evidence before publishing the local source directory."""

    raw_manifest = _load_volume_manifest(source_run_id)
    _validated_artifact_verification(
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        remote_verification=remote_verification,
        raw_manifest=raw_manifest,
    )
    verification_logical, _ = _persist_download_verification(
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        remote_verification=remote_verification,
        raw_manifest=raw_manifest,
        identity=identity,
        attempt_id=attempt_id,
        project_root=project_root,
    )
    destination = download_artifacts(
        raw_manifest,
        local_root=local_root,
        reader=ARTIFACT_VOLUME.read_file,
    )
    source_execution_context = ExecutionContextV1.from_dict(
        _read_bounded_json_object(
            destination / "execution_context.json",
            maximum_bytes=64 * 1024,
        )
    )
    _validated_artifact_verification(
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        remote_verification=remote_verification,
        raw_manifest=raw_manifest,
        source_execution_context=source_execution_context,
    )
    return destination, verification_logical


def _capture_artifact_verifier_directory(
    *,
    source_run_id: str,
    verifier_run_id: str,
    expected_remote_verification: dict[str, Any] | None,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Download and independently validate one immutable verifier directory.

    This capture is deliberately completed before the larger source-run transfer.
    A later download or local-persistence failure therefore cannot erase whether
    the paid remote verifier itself succeeded or failed.  The destination is
    create-only; recovery requires a fresh verifier run ID.
    """

    selected_source_run_id = validate_run_id(source_run_id)
    selected_verifier_run_id = validate_run_id(verifier_run_id)
    if selected_source_run_id == selected_verifier_run_id:
        raise ModalBoundaryError(
            "artifact verifier run ID must differ from the source run ID"
        )
    raw_manifest = _load_volume_manifest(selected_verifier_run_id)
    if raw_manifest.filename != "artifact_manifest.json":
        raise ModalBoundaryError(
            "artifact verifier capture requires the final artifact manifest"
        )
    capture_parent_logical = modal_artifact_verifier_capture_parent_path(
        identity,
        selected_source_run_id,
        selected_verifier_run_id,
        attempt_id,
    )
    capture_parent = project_root.joinpath(*capture_parent_logical.parts)
    destination = download_artifacts(
        raw_manifest,
        local_root=capture_parent,
        reader=ARTIFACT_VOLUME.read_file,
    )
    expected_logical = modal_artifact_verifier_capture_directory_path(
        identity,
        selected_source_run_id,
        selected_verifier_run_id,
        attempt_id,
    )
    expected_destination = project_root.joinpath(*expected_logical.parts)
    if destination != expected_destination.resolve():
        raise ModalBoundaryError("artifact verifier capture destination drifted")

    # Import lazily so remote worker initialization remains independent of the
    # local readiness recorder.  The validator reopens all four downloaded files,
    # checks the exact roster, hashes, manifest, source identity, and execution
    # context, and performs no network call.
    from scripts.record_modal_readiness import validate_artifact_verifier_capture

    validated = validate_artifact_verifier_capture(
        source_run_id=selected_source_run_id,
        verifier_run_id=selected_verifier_run_id,
        identity=identity,
        attempt_id=attempt_id,
        root=project_root,
    )
    if (
        validated["source_run_id"] != selected_source_run_id
        or validated["verifier_run_id"] != selected_verifier_run_id
        or validated["validated"] is not True
    ):
        raise ModalBoundaryError("artifact verifier capture identity drifted")
    if expected_remote_verification is not None:
        if validated["remote_verifier_outcome"] != "success":
            raise ModalBoundaryError(
                "returned remote verifier result differs from its Volume capture"
            )
        captured = _read_bounded_json_object(
            destination / "artifact_verification_result.json",
            maximum_bytes=64 * 1024,
        )
        if captured != expected_remote_verification:
            raise ModalBoundaryError(
                "returned remote verifier result differs from its Volume capture"
            )
    return validated


def _function_options(name: str) -> dict[str, Any]:
    spec = FUNCTION_SPECS[name]
    options: dict[str, Any] = {
        "image": IMAGE,
        "volumes": {str(VOLUME_MOUNT_PATH): ARTIFACT_VOLUME},
        "cpu": (spec.cpu_request_cores, spec.cpu_soft_limit_cores),
        "memory": (spec.memory_request_mib, spec.memory_limit_mib),
        "min_containers": spec.min_containers,
        "max_containers": spec.max_containers,
        "retries": spec.retries,
        "timeout": spec.timeout_seconds,
        "name": spec.name,
        "include_source": False,
        "single_use_containers": True,
        "block_network": not spec.provider_secret,
    }
    if spec.gpu is not None:
        options["gpu"] = spec.gpu
    return options


def _provider_variant(function):
    """Attach the provider Secret only when an approved canary is invoked.

    Modal loads every statically registered Function dependency when starting an
    ephemeral App. Keeping the Secret out of the decorator dependency graph lets
    provider-free CUDA and checkpoint smokes run before the separately approved
    provider phase and before the named Secret has been confirmed.
    """

    if PROVIDER_SECRET is None:
        raise RemoteActionError("Modal is unavailable; provider canary cannot run")
    return function.with_options(secrets=[PROVIDER_SECRET])


if not _MODAL_OBJECTS_ENABLED:
    app = None
    IMAGE = None
    ARTIFACT_VOLUME = None
    PROVIDER_SECRET = None
else:
    IMAGE = _build_image()
    ARTIFACT_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    PROVIDER_SECRET = modal.Secret.from_name(
        PROVIDER_SECRET_NAME,
        required_keys=list(PROVIDER_ENVIRONMENT_KEYS),
    )
    app = modal.App(APP_NAME)

    @app.function(**_function_options("offline_smoke"))
    def offline_smoke(run_id: str) -> dict[str, Any]:
        return _execute_new_run("offline_smoke", run_id, _offline_smoke_action)

    @app.function(**_function_options("cuda_environment"))
    def cuda_environment(run_id: str) -> dict[str, Any]:
        return _execute_new_run("cuda_environment", run_id, _cuda_environment_action)

    @app.function(**_function_options("candidate_smoke"))
    def candidate_smoke(run_id: str, seed: int = 1) -> dict[str, Any]:
        return _execute_new_run(
            "candidate_smoke",
            run_id,
            lambda directory, context: _candidate_smoke_action(
                directory, context, seed=seed
            ),
            manifest_filename="artifact_manifest.checkpoint.json",
        )

    @app.function(**_function_options("checkpoint_resume"))
    def checkpoint_resume(
        source_run_id: str,
        run_id: str,
        seed: int = 1,
    ) -> dict[str, Any]:
        return _resume_action(source_run_id, run_id, seed=seed)

    @app.function(**_function_options("canary_greedy_autoresearch"))
    def canary_greedy_autoresearch(
        run_id: str, opportunities: int = 1
    ) -> dict[str, Any]:
        return _execute_new_run(
            "canary_greedy_autoresearch",
            run_id,
            lambda directory, context: _canary_action(
                "greedy_autoresearch",
                directory,
                context,
                opportunities=opportunities,
            ),
            scan_provider_credential=True,
        )

    @app.function(**_function_options("canary_semantic_autoresearch"))
    def canary_semantic_autoresearch(
        run_id: str, opportunities: int = 1
    ) -> dict[str, Any]:
        return _execute_new_run(
            "canary_semantic_autoresearch",
            run_id,
            lambda directory, context: _canary_action(
                "semantic_autoresearch",
                directory,
                context,
                opportunities=opportunities,
            ),
            scan_provider_credential=True,
        )

    @app.function(**_function_options("canary_openevolve_generic"))
    def canary_openevolve_generic(
        run_id: str, opportunities: int = 1
    ) -> dict[str, Any]:
        return _execute_new_run(
            "canary_openevolve_generic",
            run_id,
            lambda directory, context: _canary_action(
                "openevolve_generic",
                directory,
                context,
                opportunities=opportunities,
            ),
            scan_provider_credential=True,
        )

    @app.function(**_function_options("canary_openevolve_semantic"))
    def canary_openevolve_semantic(
        run_id: str, opportunities: int = 1
    ) -> dict[str, Any]:
        return _execute_new_run(
            "canary_openevolve_semantic",
            run_id,
            lambda directory, context: _canary_action(
                "openevolve_semantic",
                directory,
                context,
                opportunities=opportunities,
            ),
            scan_provider_credential=True,
        )

    @app.function(**_function_options("artifact_verify"))
    def artifact_verify(
        source_run_id: str,
        verifier_run_id: str,
    ) -> dict[str, Any]:
        return _artifact_verify_action(source_run_id, verifier_run_id)

    @app.local_entrypoint()
    def main(
        action: str = "plan",
        run_id: str = "",
        source_run_id: str = "",
        verifier_run_id: str = "",
        approved: bool = False,
        provider_approved: bool = False,
        harness: str = "",
        local_output: str = "",
        expected_image_source_sha256: str = "",
        source_tree_sha256: str = "",
        cohort_id: str = "",
    ) -> None:
        if action == "plan":
            print(
                json.dumps(
                    {
                        "app": APP_NAME,
                        "volume": VOLUME_NAME,
                        "mount": str(VOLUME_MOUNT_PATH),
                        "python": PYTHON_VERSION,
                        "image_recipe": IMAGE_RECIPE_VERSION,
                        "image_build": {
                            "cpu_request_cores": IMAGE_BUILD_CPU_REQUEST_CORES,
                            "cpu_soft_limit_cores": None,
                            "memory_request_mib": IMAGE_BUILD_MEMORY_REQUEST_MIB,
                            "memory_limit_mib": None,
                            "region": None,
                            "subprocess_thread_limit": (
                                IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT
                            ),
                            "resource_limits_exposed": False,
                            "platform_compute_cost_ceiling_enforced": False,
                            "timeout_seconds": (
                                IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS
                            ),
                        },
                        "image_source_sha256": IMAGE_SOURCE_SHA256,
                        "functions": {
                            name: spec.__dict__ for name, spec in FUNCTION_SPECS.items()
                        },
                        "provider_actions": sorted(_PROVIDER_CANARY_ACTIONS),
                        "provider_approval_flag": "--provider-approved",
                        "single_canary_harnesses": list(CANARY_ORDER),
                        "remote_calls_started": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        _require_local_action_approvals(
            action=action,
            approved=approved,
            provider_approved=provider_approved,
            harness=harness,
            expected_image_source_sha256=expected_image_source_sha256,
        )
        try:
            identity = ModalLiveCohortIdentity(
                source_tree_sha256=source_tree_sha256,
                image_source_sha256=expected_image_source_sha256,
                cohort_id=cohort_id,
            )
            attempt_id = os.environ.get(MODAL_ACTION_ATTEMPT_ID_ENV, "")
            modal_action_intent_receipt_path(identity, attempt_id)
        except (TypeError, ValueError) as error:
            raise SystemExit(
                "paid action requires canonical source, image, cohort, and attempt "
                "identity from launch_modal.py"
            ) from error
        selected_run_id = validate_run_id(run_id) if run_id else new_run_id("modal")
        if action == "cuda-environment" and identity.cohort_id != selected_run_id:
            raise SystemExit(
                "the first CUDA environment run ID must equal the cohort ID"
            )
        selected_source_run_id: str | None = None
        selected_verifier_run_id: str | None = None
        if action == "checkpoint-resume":
            if not source_run_id:
                raise SystemExit(
                    "checkpoint-resume requires --source-run-id for the immutable "
                    "successful candidate-smoke run"
                )
            selected_source_run_id = validate_run_id(source_run_id)
            if selected_source_run_id == selected_run_id:
                raise SystemExit(
                    "checkpoint-resume requires a fresh --run-id distinct from "
                    "--source-run-id"
                )
        elif source_run_id:
            raise SystemExit(
                "--source-run-id is accepted only with --action checkpoint-resume"
            )
        if action in {"verify", "download"}:
            if not verifier_run_id:
                raise SystemExit(
                    f"{action} requires a fresh caller-supplied --verifier-run-id"
                )
            selected_verifier_run_id = validate_run_id(verifier_run_id)
            if selected_verifier_run_id == selected_run_id:
                raise SystemExit(
                    "--verifier-run-id must differ from the source --run-id"
                )
        elif verifier_run_id:
            raise SystemExit(
                "--verifier-run-id is accepted only with --action verify or download"
            )
        _require_durable_local_action_intent(
            identity=identity,
            attempt_id=attempt_id,
            action=action,
            run_id=selected_run_id,
            source_run_id=selected_source_run_id,
            verifier_run_id=selected_verifier_run_id,
            harness=harness or None,
        )
        if action == "offline-smoke":
            result = invoke_synchronously(offline_smoke, run_id=selected_run_id)
        elif action == "cuda-environment":
            result = invoke_synchronously(cuda_environment, run_id=selected_run_id)
        elif action == "candidate-smoke":
            result = invoke_synchronously(candidate_smoke, run_id=selected_run_id)
        elif action == "checkpoint-resume":
            result = invoke_synchronously(
                checkpoint_resume,
                source_run_id=selected_source_run_id,
                run_id=selected_run_id,
            )
        elif action in _PROVIDER_CANARY_ACTIONS:
            functions = {
                "greedy_autoresearch": _provider_variant(canary_greedy_autoresearch),
                "semantic_autoresearch": _provider_variant(
                    canary_semantic_autoresearch
                ),
                "openevolve_generic": _provider_variant(canary_openevolve_generic),
                "openevolve_semantic": _provider_variant(canary_openevolve_semantic),
            }
            if action == "canaries":
                result = run_canaries_synchronously(
                    functions, run_id_prefix=selected_run_id
                )
            else:
                result = _run_single_canary_synchronously(
                    functions,
                    harness=harness,
                    run_id=selected_run_id,
                )
        elif action == "verify":
            try:
                result = invoke_synchronously(
                    artifact_verify,
                    source_run_id=selected_run_id,
                    verifier_run_id=selected_verifier_run_id,
                )
            except Exception:
                with suppress(Exception):
                    _capture_artifact_verifier_directory(
                        source_run_id=selected_run_id,
                        verifier_run_id=selected_verifier_run_id,
                        expected_remote_verification=None,
                        identity=identity,
                        attempt_id=attempt_id,
                    )
                raise
            _capture_artifact_verifier_directory(
                source_run_id=selected_run_id,
                verifier_run_id=selected_verifier_run_id,
                expected_remote_verification=result,
                identity=identity,
                attempt_id=attempt_id,
            )
        elif action == "download":
            try:
                remote_verification = invoke_synchronously(
                    artifact_verify,
                    source_run_id=selected_run_id,
                    verifier_run_id=selected_verifier_run_id,
                )
            except Exception:
                # The remote verifier commits its own success/failure directory
                # before returning or raising.  Recover that bounded evidence
                # when possible, but preserve the original invocation failure.
                with suppress(Exception):
                    _capture_artifact_verifier_directory(
                        source_run_id=selected_run_id,
                        verifier_run_id=selected_verifier_run_id,
                        expected_remote_verification=None,
                        identity=identity,
                        attempt_id=attempt_id,
                    )
                raise
            _capture_artifact_verifier_directory(
                source_run_id=selected_run_id,
                verifier_run_id=selected_verifier_run_id,
                expected_remote_verification=remote_verification,
                identity=identity,
                attempt_id=attempt_id,
            )
            destination, verification_logical = (
                _download_after_verification_receipt(
                    source_run_id=selected_run_id,
                    verifier_run_id=selected_verifier_run_id,
                    remote_verification=remote_verification,
                    identity=identity,
                    attempt_id=attempt_id,
                    local_root=(
                        Path(local_output).expanduser()
                        if local_output
                        else PROJECT_ROOT
                        / "outputs"
                        / "development"
                        / "modal_downloads"
                    ),
                )
            )
            result = {
                "downloaded_to": str(destination),
                "verified": True,
                "remote_verification": remote_verification,
                "remote_verification_path": verification_logical,
            }
        else:
            raise SystemExit(f"unknown action: {action}")
        aggregate_outcome_path: Path | None = None
        if action == "canaries":
            aggregate_outcome_path = _persist_provider_canary_aggregate_outcome(
                result,
                identity=identity,
                attempt_id=attempt_id,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        if action == "canaries":
            print(
                f"Provider canary aggregate outcome: {aggregate_outcome_path}",
                file=sys.stderr,
            )
            aggregate_exit_code = _provider_canary_aggregate_exit_code(result)
            if aggregate_exit_code:
                raise SystemExit(aggregate_exit_code)
