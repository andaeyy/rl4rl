#!/usr/bin/env python3
"""Pre-hydration approval gate for every paid Modal migration action.

This module deliberately does not import Modal.  It validates the complete
approved action and current image-source digest before starting the Modal CLI,
then supplies a process-local launch nonce that lets ``modal_app`` construct
remote objects.  Invoking ``modal run modal_app.py`` directly therefore exposes
no App or Function objects in a local process.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import hmac
import importlib.metadata
import json
import os
import pwd
import re
import secrets
import stat
import subprocess
import sys
import sysconfig
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.gpt56_sol import TARGET_MODEL  # noqa: E402
from common.modal_action_lock import (  # noqa: E402
    MODAL_ACTION_LOCK_PATH,
    ModalActionLockContentionError,
    acquire_modal_action_lock,
    assert_modal_action_lock_identity,
    release_modal_action_lock,
)
from common.process_control import (  # noqa: E402
    ProcessGroupClosureError,
    capture_isolated_process_group,
    terminate_process_group,
)
from modal_action_journal import (  # noqa: E402
    ModalActionJournalBlockedError,
    ModalActionJournalIntegrityError,
    require_modal_global_action_gate_clear,
    scan_modal_global_action_journal,
)
from modal_boundary import (  # noqa: E402
    CANARY_ORDER,
    FUNCTION_SPECS,
    FUNCTION_TIMEOUT_SECONDS,
    IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
    IMAGE_BUILD_CPU_REQUEST_CORES,
    IMAGE_BUILD_MEMORY_REQUEST_MIB,
    IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
    MODAL_ACTION_INTENT_SHA256_ENV,
    MODAL_DOWNLOAD_OUTPUT_ROOT,
    MODAL_ENVIRONMENT_NAME,
    MODAL_LOCAL_CONTAINMENT_ROOT,
    MODAL_VERSION,
    ModalLiveCohortIdentity,
    build_image_source_manifest,
    build_modal_cli_command,
    canary_run_suffix,
    canonical_sha256,
    modal_action_attempt_directory,
    modal_action_intent_receipt_path,
    modal_action_terminal_receipt_path,
    modal_launch_rejection_receipt_path,
    modal_local_host_anchor_path,
    modal_local_process_start_receipt_path,
    modal_migration_lineage_path,
    modal_remote_run_reservation_path,
    provider_canary_aggregate_outcome_receipt_path,
    safe_relative_path,
    validate_provider_canary_aggregate_outcome_receipt,
    validate_run_id,
)
from scripts import record_modal_readiness as modal_readiness  # noqa: E402
from scripts.openevolve_patch_bundle import (  # noqa: E402
    validate_applied_patch_bundle,
)
from scripts.provider_canary_plan import (  # noqa: E402
    build_provider_canary_approval_plan,
    verify_provider_canary_approval_plan,
)
from scripts.record_local_engineering_evidence import (  # noqa: E402
    LOCAL_ENGINEERING_FREEZE_GATES,
    local_engineering_freeze_predecessor_bindings,
    validate_local_engineering_freeze_receipt,
)
from study.serialization import create_json_exclusive  # noqa: E402

MODAL_LAUNCH_LOCK_PATH = MODAL_ACTION_LOCK_PATH
ModalLaunchLockContentionError = ModalActionLockContentionError
_acquire_launcher_lock = acquire_modal_action_lock
_assert_launcher_lock_identity = assert_modal_action_lock_identity
_release_launcher_lock = release_modal_action_lock
_scan_modal_global_action_journal = scan_modal_global_action_journal
_require_modal_global_action_gate_clear = require_modal_global_action_gate_clear
MODAL_LAUNCH_NONCE_ENV = "RL4RL_MODAL_LAUNCH_NONCE"
MODAL_LAUNCH_SOURCE_ENV = "RL4RL_MODAL_LAUNCH_IMAGE_SOURCE_SHA256"
MODAL_LAUNCH_SOURCE_TREE_ENV = "RL4RL_MODAL_LAUNCH_SOURCE_TREE_SHA256"
MODAL_LAUNCH_COHORT_ENV = "RL4RL_MODAL_LAUNCH_COHORT_ID"
MODAL_ACTION_ATTEMPT_ID_ENV = "RL4RL_MODAL_ACTION_ATTEMPT_ID"
MODAL_PROFILE_ENV = "MODAL_PROFILE"
MODAL_PROFILE = "scalingintelligence"
MODAL_ENVIRONMENT_ENV = "MODAL_ENVIRONMENT"
MODAL_ENVIRONMENT = MODAL_ENVIRONMENT_NAME
MODAL_CLI_ORCHESTRATION_RESERVE_SECONDS = 300
PROVIDER_PRICE_BASIS_MAX_AGE = timedelta(hours=48)
PROVIDER_PRICE_BASIS_FUTURE_SKEW = timedelta(minutes=5)
_NONCE = re.compile(r"\A[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")
_ATTEMPT_ID = re.compile(r"\A[0-9a-f]{32}\Z")
_BOOT_UUID_TEXT = re.compile(
    rb"\A[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    rb"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\Z"
)
_CANONICAL_DECIMAL = re.compile(r"\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_ACTION_INTENT_FILENAME = re.compile(r"\A([0-9a-f]{32})\.intent\.json\Z")
_MAX_LOCAL_APPROVAL_BYTES = 4 * 1024 * 1024
_MAX_MODAL_CONFIG_BYTES = 1024 * 1024
_MAX_MODAL_CONSOLE_SCRIPT_BYTES = 64 * 1024
_MAX_PYTHON_EXECUTABLE_BYTES = 64 * 1024 * 1024
_MAX_LOCAL_HOST_ANCHOR_BYTES = 4 * 1024
_MAX_LOCAL_PROCESS_START_BYTES = 64 * 1024
_MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS = 946_684_800_000_000
_MAX_BOOT_FUTURE_SKEW_MICROSECONDS = 5 * 60 * 1_000_000
_MODAL_PYTHON_RUNTIME_ROOT = Path(
    "outputs/readiness/.modal_python_runtime"
)
_LEGACY_RESERVED_REMOTE_RUN_IDS = frozenset({"modal-cuda-env-20260809-02"})
_OFFICIAL_OPENAI_PRICE_URL = re.compile(
    r"\Ahttps://(?:platform\.)?openai\.com/[^\s]*\Z"
)
_PROVIDER_ACTIONS = frozenset({"canary", "canaries"})
_VERIFIER_ACTIONS = frozenset({"download", "verify"})
_SOURCE_PRODUCING_ACTIONS = frozenset(
    {
        "canaries",
        "canary",
        "candidate-smoke",
        "checkpoint-resume",
        "cuda-environment",
        "offline-smoke",
    }
)
_CANARY_RUN_SUFFIXES = {harness: canary_run_suffix(harness) for harness in CANARY_ORDER}
_ACTIONS = frozenset(
    {
        "canaries",
        "canary",
        "candidate-smoke",
        "checkpoint-resume",
        "cuda-environment",
        "download",
        "offline-smoke",
        "verify",
    }
)
_PREDECESSOR_ARGUMENT_PAIRS = {
    "cuda": ("cuda_receipt_path", "cuda_receipt_sha256"),
    "offline_smoke": (
        "offline_smoke_receipt_path",
        "offline_smoke_receipt_sha256",
    ),
    "artifact_round_trip": (
        "artifact_round_trip_receipt_path",
        "artifact_round_trip_receipt_sha256",
    ),
    "candidate_resume_preflight": (
        "candidate_resume_preflight_receipt_path",
        "candidate_resume_preflight_receipt_sha256",
    ),
}
_ACTION_PREDECESSOR_ARGUMENTS = {
    "cuda-environment": (),
    "offline-smoke": ("cuda",),
    "candidate-smoke": ("cuda", "offline_smoke"),
    "checkpoint-resume": ("artifact_round_trip",),
    "canary": ("candidate_resume_preflight",),
    "canaries": ("candidate_resume_preflight",),
    "download": (),
    "verify": (),
}
_PAID_MODAL_BASE_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": os.defpath,
    "PYTHONNOUSERSITE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTHONUTF8": "1",
}
_PRICE_BASIS_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "model",
        "official_source_url",
        "retrieved_at_utc",
        "uncached_input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "per_request_fee_usd",
    }
)


class ModalCLITimeoutError(TimeoutError):
    """Raised after the bounded outer Modal CLI deadline is exhausted."""


class ModalAttemptReceiptError(RuntimeError):
    """Raised when an immutable local attempt receipt cannot be created."""


class ModalProcessStartReceiptError(RuntimeError):
    """Raised when a started local process cannot be durably journaled."""


@dataclass(frozen=True, slots=True)
class ModalActionAttemptReceipt:
    """Sanitized immutable record of one local paid-action launch attempt."""

    schema_name: str
    schema_version: str
    attempt_id: str
    started_at_utc: str
    finished_at_utc: str
    status: str
    failure_kind: str | None
    action: str | None
    run_id: str | None
    concrete_remote_run_ids: tuple[str, ...]
    remote_run_reservations: tuple[dict[str, str], ...]
    local_host_anchor_path: str | None
    local_host_anchor_sha256: str | None
    local_boot_started_at_unix_microseconds: int | None
    local_boot_session_sha256: str | None
    source_run_id: str | None
    verifier_run_id: str | None
    harness: str | None
    source_tree_sha256: str | None
    cohort_id: str | None
    approved_image_source_sha256: str | None
    modal_command_sha256: str | None
    launch_capability_sha256: str | None
    modal_profile: str
    modal_environment: str
    outer_cli_timeout_seconds: int | None
    modal_cost_cap_usd: str | None
    modal_resource_profile: dict[str, Any] | None
    modal_price_basis_path: str | None
    modal_price_basis_sha256: str | None
    modal_cost_estimate: dict[str, Any] | None
    modal_cost_approved: bool
    provider_cost_approved: bool
    provider_cost_cap_usd: str | None
    provider_approval_plan_path: str | None
    approval_plan_sha256: str | None
    provider_price_basis_path: str | None
    provider_price_basis_sha256: str | None
    predecessor_receipts: tuple[dict[str, str], ...]
    source_evidence_recovery: bool
    local_process_start_receipt_path: str | None
    local_process_start_receipt_sha256: str | None
    local_process_id: int | None
    local_process_group_id: int | None
    local_session_id: int | None
    modal_cli_process_started: bool
    remote_execution_state: str
    returncode: int | None
    process_group_closed: bool | None


@dataclass(frozen=True, slots=True)
class ModalActionIntent:
    """Durable pre-Popen journal entry for one approved paid action."""

    schema_name: str
    schema_version: str
    attempt_id: str
    created_at_utc: str
    action: str
    run_id: str
    concrete_remote_run_ids: tuple[str, ...]
    remote_run_reservations: tuple[dict[str, str], ...]
    local_host_anchor_path: str
    local_host_anchor_sha256: str
    local_boot_started_at_unix_microseconds: int
    local_boot_session_sha256: str
    source_run_id: str | None
    verifier_run_id: str | None
    harness: str | None
    source_tree_sha256: str
    cohort_id: str
    approved_image_source_sha256: str
    modal_command_sha256: str
    launch_capability_sha256: str
    modal_profile: str
    modal_environment: str
    outer_cli_timeout_seconds: int
    modal_cost_cap_usd: str
    modal_resource_profile: dict[str, Any]
    modal_price_basis_path: str
    modal_price_basis_sha256: str
    modal_cost_estimate: dict[str, Any]
    modal_cost_approved: bool
    provider_cost_approved: bool
    provider_cost_cap_usd: str | None
    provider_approval_plan_path: str | None
    approval_plan_sha256: str | None
    provider_price_basis_path: str | None
    provider_price_basis_sha256: str | None
    predecessor_receipts: tuple[dict[str, str], ...]
    source_evidence_recovery: bool


@dataclass(frozen=True, slots=True)
class ModalLocalProcessStartReceipt:
    """Create-only local proof captured immediately after successful Popen."""

    schema_name: str
    schema_version: str
    attempt_id: str
    created_at_utc: str
    action: str
    run_id: str
    intent_path: str
    intent_sha256: str
    source_tree_sha256: str
    image_source_sha256: str
    cohort_id: str
    modal_command_sha256: str
    launch_capability_sha256: str
    modal_cost_cap_usd: str
    provider_cost_cap_usd: str | None
    local_host_anchor_path: str
    local_host_anchor_sha256: str
    local_boot_started_at_unix_microseconds: int
    local_boot_session_sha256: str
    process_id: int
    expected_process_group_id: int
    expected_session_id: int
    process_birth_identity_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedApprovalChain:
    """Exact local approvals and predecessor evidence bound before Popen."""

    source_tree_sha256: str
    cohort_id: str
    modal_cost_cap_usd: str
    modal_resource_profile: dict[str, Any]
    modal_price_basis_path: str
    modal_price_basis_sha256: str
    modal_cost_estimate: dict[str, Any]
    provider_cost_cap_usd: str | None
    provider_approval_plan_path: str | None
    approval_plan_sha256: str | None
    provider_price_basis_path: str | None
    provider_price_basis_sha256: str | None
    predecessor_receipts: tuple[dict[str, str], ...]


@dataclass(slots=True)
class _HeldLaunchFileBinding:
    """One immutable local launch file held open across the paid CLI lifetime."""

    label: str
    canonical_path: Path
    descriptor: int
    device: int
    inode: int
    size_bytes: int
    mode: int
    owner_uid: int
    mtime_ns: int
    ctime_ns: int
    sha256: str | None
    maximum_bytes: int
    require_owner_executable: bool
    require_current_uid: bool
    required_mode: int | None
    require_stable_ctime: bool

    @property
    def execution_path(self) -> str:
        if self.descriptor < 0:
            raise ValueError(f"{self.label} descriptor is closed")
        return f"/dev/fd/{self.descriptor}"

    def close(self) -> None:
        if self.descriptor >= 0:
            os.close(self.descriptor)
            self.descriptor = -1

    def require_current(self) -> None:
        _require_held_launch_file_binding(self)


@dataclass(slots=True)
class _PrivatePythonExecutionCopy:
    """A verified per-attempt Python copy and held directories for safe removal."""

    binding: _HeldLaunchFileBinding
    runtime_directory_descriptor: int
    attempt_directory_descriptor: int
    runtime_directory_identity: tuple[int, int]
    attempt_directory_identity: tuple[int, int]
    attempt_id: str
    removed: bool = False

    @property
    def canonical_path(self) -> Path:
        return self.binding.canonical_path

    def require_current(self) -> None:
        if (
            self.runtime_directory_descriptor < 0
            or self.attempt_directory_descriptor < 0
        ):
            raise ValueError("private Python execution copy directories are closed")
        self.binding.require_current()
        runtime = os.fstat(self.runtime_directory_descriptor)
        attempt = os.fstat(self.attempt_directory_descriptor)
        if (
            (runtime.st_dev, runtime.st_ino) != self.runtime_directory_identity
            or (attempt.st_dev, attempt.st_ino) != self.attempt_directory_identity
            or not stat.S_ISDIR(runtime.st_mode)
            or not stat.S_ISDIR(attempt.st_mode)
            or runtime.st_uid != os.getuid()
            or attempt.st_uid != os.getuid()
            or stat.S_IMODE(runtime.st_mode) != 0o700
            or stat.S_IMODE(attempt.st_mode) != 0o700
        ):
            raise ValueError("private Python execution copy directory changed")
        rebound_attempt = os.stat(
            self.attempt_id,
            dir_fd=self.runtime_directory_descriptor,
            follow_symlinks=False,
        )
        rebound_leaf = os.stat(
            self.binding.canonical_path.name,
            dir_fd=self.attempt_directory_descriptor,
            follow_symlinks=False,
        )
        if (
            (rebound_attempt.st_dev, rebound_attempt.st_ino)
            != self.attempt_directory_identity
            or stat.S_ISLNK(rebound_attempt.st_mode)
            or not stat.S_ISDIR(rebound_attempt.st_mode)
            or (rebound_leaf.st_dev, rebound_leaf.st_ino)
            != (self.binding.device, self.binding.inode)
            or stat.S_ISLNK(rebound_leaf.st_mode)
            or not stat.S_ISREG(rebound_leaf.st_mode)
        ):
            raise ValueError("private Python execution copy namespace changed")

    def close_and_remove(self) -> None:
        """Remove only the verified held leaf and then its exact empty directory."""

        pending_error: BaseException | None = None
        try:
            self.require_current()
            os.unlink(
                self.binding.canonical_path.name,
                dir_fd=self.attempt_directory_descriptor,
            )
            os.fsync(self.attempt_directory_descriptor)
            if os.listdir(self.attempt_directory_descriptor):
                raise ValueError("private Python execution directory is not empty")
            rebound_attempt = os.stat(
                self.attempt_id,
                dir_fd=self.runtime_directory_descriptor,
                follow_symlinks=False,
            )
            if (
                (rebound_attempt.st_dev, rebound_attempt.st_ino)
                != self.attempt_directory_identity
                or stat.S_ISLNK(rebound_attempt.st_mode)
                or not stat.S_ISDIR(rebound_attempt.st_mode)
            ):
                raise ValueError("private Python execution directory was replaced")
            os.rmdir(
                self.attempt_id,
                dir_fd=self.runtime_directory_descriptor,
            )
            os.fsync(self.runtime_directory_descriptor)
            self.removed = True
        except BaseException as error:
            pending_error = error
        finally:
            self.binding.close()
            if self.attempt_directory_descriptor >= 0:
                os.close(self.attempt_directory_descriptor)
                self.attempt_directory_descriptor = -1
            if self.runtime_directory_descriptor >= 0:
                os.close(self.runtime_directory_descriptor)
                self.runtime_directory_descriptor = -1
        if pending_error is not None:
            raise pending_error


@dataclass(slots=True)
class _ModalLaunchBindings:
    python_executable: _HeldLaunchFileBinding
    modal_executable: _HeldLaunchFileBinding
    modal_config: _HeldLaunchFileBinding

    @property
    def pass_fds(self) -> tuple[int, int]:
        return (
            self.modal_executable.descriptor,
            self.modal_config.descriptor,
        )

    def require_current(self) -> None:
        _require_exact_modal_version()
        self.python_executable.require_current()
        _require_held_launch_file_binding(self.modal_executable)
        _require_held_launch_file_binding(self.modal_config)

    def close(self) -> None:
        self.modal_config.close()
        self.modal_executable.close()
        self.python_executable.close()


@dataclass(slots=True)
class _ModalLocalContainmentBinding:
    anchor: _HeldLaunchFileBinding
    host_anchor_path: str
    host_anchor_sha256: str
    host_anchor_id: str
    machine_binding_sha256: str
    boot_started_at_unix_microseconds: int
    boot_session_sha256: str
    machine_identity_provider: Callable[[], bytes]
    boot_session_provider: Callable[[], int]
    boot_identity_provider: Callable[[], bytes]

    def require_current(self) -> None:
        self.anchor.require_current()
        if _local_machine_binding_sha256(
            self.host_anchor_id,
            self.machine_identity_provider(),
        ) != self.machine_binding_sha256:
            raise ValueError("local host anchor belongs to another machine")
        current = _validated_boot_started_at_unix_microseconds(
            self.boot_session_provider()
        )
        if _boot_started_at_unix_second(current) != _boot_started_at_unix_second(
            self.boot_started_at_unix_microseconds
        ):
            raise ValueError("local boot-start time changed during Modal launch")
        current_session_sha256 = _local_boot_session_sha256(
            self.host_anchor_sha256,
            self.boot_identity_provider(),
        )
        if current_session_sha256 != self.boot_session_sha256:
            raise ValueError("local OS boot identity changed during Modal launch")

    def close(self) -> None:
        self.anchor.close()


@dataclass(slots=True)
class _HeldModalLocalProcessStart:
    logical_path: str
    receipt: ModalLocalProcessStartReceipt
    binding: _HeldLaunchFileBinding

    @property
    def sha256(self) -> str:
        digest = self.binding.sha256
        if digest is None:  # pragma: no cover - construction invariant
            raise AssertionError("process-start marker lacks its raw SHA-256")
        return digest

    def require_current(self) -> None:
        self.binding.require_current()

    def close(self) -> None:
        self.binding.close()


def expected_outer_cli_timeout_seconds(action: str) -> int:
    """Return the only accepted local deadline for one approved action."""

    if action not in _ACTIONS:
        raise ValueError("action is not an approved bounded migration action")
    remote_call_count = len(CANARY_ORDER) if action == "canaries" else 1
    return (
        IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS
        + remote_call_count * FUNCTION_TIMEOUT_SECONDS
        + MODAL_CLI_ORCHESTRATION_RESERVE_SECONDS
    )


def modal_resource_profile(action: str, harness: str = "") -> dict[str, Any]:
    """Return the exact request/limit disclosure for one launcher action."""

    if action not in _ACTIONS:
        raise ValueError("resource profile action is unsupported")
    if action == "canaries":
        function_names = [f"canary_{item}" for item in CANARY_ORDER]
    elif action == "canary":
        if harness not in CANARY_ORDER:
            raise ValueError("single-canary resource profile requires a harness")
        function_names = [f"canary_{harness}"]
    elif action in _VERIFIER_ACTIONS:
        function_names = ["artifact_verify"]
    else:
        function_names = [action.replace("-", "_")]
    runtime_calls: list[dict[str, Any]] = []
    for function_name in function_names:
        spec = FUNCTION_SPECS[function_name]
        runtime_calls.append(
            {
                "function_name": function_name,
                "call_count": 1,
                "cpu_request_cores": spec.cpu_request_cores,
                "cpu_soft_limit_cores": spec.cpu_soft_limit_cores,
                "memory_request_mib": spec.memory_request_mib,
                "memory_limit_mib": spec.memory_limit_mib,
                "gpu": spec.gpu,
                "region": spec.region,
                "timeout_seconds": spec.timeout_seconds,
                "max_containers": spec.max_containers,
                "min_containers": spec.min_containers,
                "retries": spec.retries,
                "provider_secret_attached": spec.provider_secret,
                "network_mode": (
                    "provider_egress_enabled"
                    if spec.provider_secret
                    else "blocked"
                ),
            }
        )
    return {
        "modal_environment": MODAL_ENVIRONMENT,
        "runtime_function_calls": runtime_calls,
        "runtime_cpu_request_equals_soft_limit": True,
        "runtime_memory_request_equals_hard_limit": True,
        "runtime_platform_compute_cost_ceiling_enforced": False,
        "runtime_functions_preemptible": True,
        "platform_preemption_restart_possible": True,
        "logical_call_count_is_not_container_attempt_ceiling": True,
        "image_build": {
            "invocation_condition": "backend_cache_miss",
            "cpu_request_cores": IMAGE_BUILD_CPU_REQUEST_CORES,
            "cpu_soft_limit_cores": None,
            "memory_request_mib": IMAGE_BUILD_MEMORY_REQUEST_MIB,
            "memory_limit_mib": None,
            "gpu": None,
            "region": None,
            "timeout_seconds": IMAGE_BUILD_COMMAND_TIMEOUT_SECONDS,
            "subprocess_thread_limit": IMAGE_BUILD_SUBPROCESS_THREAD_LIMIT,
            "resource_limits_exposed": False,
            "platform_compute_cost_ceiling_enforced": False,
            "provider_secret_attached": False,
            "network_mode": "dependency_install_egress_required",
        },
    }


def local_launch_authorized(
    environment: Mapping[str, str],
    *,
    image_source_sha256: str,
    project_root: str | Path = ROOT,
    modal_command_python_executable: str | Path | None = None,
) -> bool:
    """Require the exact live durable intent before exposing Modal objects."""

    nonce = environment.get(MODAL_LAUNCH_NONCE_ENV, "")
    approved_source = environment.get(MODAL_LAUNCH_SOURCE_ENV, "")
    source_tree_sha256 = environment.get(MODAL_LAUNCH_SOURCE_TREE_ENV, "")
    cohort_id = environment.get(MODAL_LAUNCH_COHORT_ENV, "")
    attempt_id = environment.get(MODAL_ACTION_ATTEMPT_ID_ENV, "")
    intent_sha256 = environment.get(MODAL_ACTION_INTENT_SHA256_ENV, "")
    if not (
        _NONCE.fullmatch(nonce) is not None
        and approved_source == image_source_sha256
        and _SHA256.fullmatch(source_tree_sha256) is not None
        and _ATTEMPT_ID.fullmatch(attempt_id) is not None
        and _SHA256.fullmatch(intent_sha256) is not None
        and environment.get(MODAL_PROFILE_ENV) == MODAL_PROFILE
        and environment.get(MODAL_ENVIRONMENT_ENV) == MODAL_ENVIRONMENT
    ):
        return False
    try:
        identity = ModalLiveCohortIdentity(
            source_tree_sha256=source_tree_sha256,
            image_source_sha256=approved_source,
            cohort_id=cohort_id,
        )
        root = Path(os.path.abspath(os.fspath(project_root)))
        intent_logical = modal_action_intent_receipt_path(identity, attempt_id)
        payload, _raw, observed_sha256 = _read_project_json_file(
            root,
            intent_logical.as_posix(),
            "durable_action_intent",
        )
        if observed_sha256 != intent_sha256:
            return False
        _validate_action_intent_contract(
            payload,
            attempt_id=attempt_id,
            identity=identity,
            project_root=root,
            launch_nonce=nonce,
            modal_command_python_executable=(
                sys.executable
                if modal_command_python_executable is None
                else modal_command_python_executable
            ),
        )
        _validate_global_remote_run_reservations(
            payload,
            project_root=root,
            receipt_directory=None,
            identity=identity,
        )
        _require_live_cohort_unsealed(project_root=root, identity=identity)
        terminal_logical = modal_action_terminal_receipt_path(identity, attempt_id)
        _require_path_absent_secure(
            root.joinpath(*terminal_logical.parts),
            "action terminal receipt",
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return False
    return True


def _launch_capability_sha256(nonce: str) -> str:
    """Hash the process-local launch capability without journaling the secret."""

    if not isinstance(nonce, str) or _NONCE.fullmatch(nonce) is None:
        raise ValueError("launch nonce is not a canonical capability")
    return hashlib.sha256(
        b"rl4rl-modal-launch-capability-v1\0" + nonce.encode("ascii")
    ).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate approval and source identity before starting a paid Modal run"
        )
    )
    parser.add_argument("--action", required=True)
    parser.add_argument(
        "--attempt-id",
        default="",
        help=(
            "optional reviewed 32-hex attempt ID; required when consuming a "
            "recovery-designated fresh attempt"
        ),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--verifier-run-id", default="")
    parser.add_argument("--harness", default="")
    parser.add_argument("--local-output", default="")
    parser.add_argument("--expected-image-source-sha256", required=True)
    parser.add_argument("--outer-cli-timeout-seconds", required=True, type=int)
    parser.add_argument("--modal-cost-cap-usd", required=True)
    parser.add_argument("--modal-price-basis-path", required=True)
    parser.add_argument("--modal-price-basis-sha256", required=True)
    parser.add_argument("--approved", action="store_true")
    parser.add_argument("--provider-approved", action="store_true")
    parser.add_argument("--provider-cost-cap-usd", default="")
    parser.add_argument("--provider-approval-plan-path", default="")
    parser.add_argument("--approval-plan-sha256", default="")
    parser.add_argument("--provider-price-basis-path", default="")
    parser.add_argument("--provider-price-basis-sha256", default="")
    parser.add_argument("--source-action-attempt-receipt-path", default="")
    parser.add_argument("--source-action-attempt-receipt-sha256", default="")
    parser.add_argument("--source-evidence-recovery", action="store_true")
    parser.add_argument("--cuda-receipt-path", default="")
    parser.add_argument("--cuda-receipt-sha256", default="")
    parser.add_argument("--offline-smoke-receipt-path", default="")
    parser.add_argument("--offline-smoke-receipt-sha256", default="")
    parser.add_argument("--artifact-round-trip-receipt-path", default="")
    parser.add_argument("--artifact-round-trip-receipt-sha256", default="")
    parser.add_argument("--candidate-resume-preflight-receipt-path", default="")
    parser.add_argument("--candidate-resume-preflight-receipt-sha256", default="")
    return parser


def _validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.action not in _ACTIONS:
        raise ValueError("action is not an approved bounded migration action")
    expected_timeout = expected_outer_cli_timeout_seconds(arguments.action)
    if (
        type(arguments.outer_cli_timeout_seconds) is not int
        or arguments.outer_cli_timeout_seconds != expected_timeout
    ):
        raise ValueError(
            "outer Modal CLI timeout must be exactly "
            f"{expected_timeout} seconds for action {arguments.action}"
        )
    if arguments.approved is not True:
        raise ValueError("explicit Modal cost approval is required")
    _canonical_decimal_amount(
        arguments.modal_cost_cap_usd,
        "modal_cost_cap_usd",
        require_positive=True,
    )
    safe_relative_path(arguments.modal_price_basis_path)
    if _SHA256.fullmatch(arguments.modal_price_basis_sha256) is None:
        raise ValueError("modal_price_basis_sha256 must be a lowercase SHA-256")
    validate_run_id(arguments.run_id)
    validate_run_id(arguments.cohort_id)
    if (
        arguments.action == "cuda-environment"
        and arguments.cohort_id != arguments.run_id
    ):
        raise ValueError(
            "the first CUDA environment run ID must equal --cohort-id"
        )
    required_predecessors = set(
        _ACTION_PREDECESSOR_ARGUMENTS[arguments.action]
    )
    for predecessor, (path_field, sha_field) in (
        _PREDECESSOR_ARGUMENT_PAIRS.items()
    ):
        logical_path = getattr(arguments, path_field)
        expected_sha256 = getattr(arguments, sha_field)
        if predecessor in required_predecessors:
            if not logical_path or not expected_sha256:
                raise ValueError(
                    f"{arguments.action} requires --{path_field.replace('_', '-')} "
                    f"and --{sha_field.replace('_', '-')}"
                )
            safe_relative_path(logical_path)
            if _SHA256.fullmatch(expected_sha256) is None:
                raise ValueError(
                    f"{sha_field} must be a lowercase SHA-256"
                )
        elif logical_path or expected_sha256:
            raise ValueError(
                f"{path_field} and {sha_field} are unrelated to "
                f"action {arguments.action}"
            )
    if arguments.action in _PROVIDER_ACTIONS:
        if arguments.provider_approved is not True:
            raise ValueError("separate provider cost approval is required")
    elif arguments.provider_approved:
        raise ValueError("provider approval is valid only for provider canaries")
    if arguments.action == "canary":
        if arguments.harness not in CANARY_ORDER:
            raise ValueError("single canary requires one frozen harness ID")
        expected_suffix = f"-{_CANARY_RUN_SUFFIXES[arguments.harness]}"
        if not arguments.run_id.endswith(expected_suffix):
            raise ValueError(
                "single canary run ID must end with its frozen harness suffix"
            )
    elif arguments.harness:
        raise ValueError("--harness is valid only for a single canary")
    if arguments.action == "canaries":
        for suffix in _CANARY_RUN_SUFFIXES.values():
            validate_run_id(f"{arguments.run_id}-{suffix}")
    if arguments.action == "checkpoint-resume":
        if not arguments.source_run_id:
            raise ValueError("checkpoint resume requires --source-run-id")
        validate_run_id(arguments.source_run_id)
        if arguments.source_run_id == arguments.run_id:
            raise ValueError("resume source and attempt run IDs must differ")
    elif arguments.source_run_id:
        raise ValueError("--source-run-id is valid only for checkpoint resume")
    if arguments.action in _VERIFIER_ACTIONS:
        if not arguments.verifier_run_id:
            raise ValueError(
                "verify and download require a fresh --verifier-run-id"
            )
        validate_run_id(arguments.verifier_run_id)
        if arguments.verifier_run_id == arguments.run_id:
            raise ValueError("verifier and source run IDs must differ")
        if not arguments.source_action_attempt_receipt_path:
            raise ValueError(
                "verify and download require the source action's terminal receipt"
            )
        safe_relative_path(arguments.source_action_attempt_receipt_path)
        if (
            _SHA256.fullmatch(
                arguments.source_action_attempt_receipt_sha256
            )
            is None
        ):
            raise ValueError(
                "verify and download require the source terminal receipt's "
                "lowercase raw SHA-256"
            )
    elif arguments.verifier_run_id:
        raise ValueError(
            "--verifier-run-id is valid only for verify and download"
        )
    elif arguments.source_action_attempt_receipt_path:
        raise ValueError(
            "--source-action-attempt-receipt-path is valid only for verify and "
            "download"
        )
    elif arguments.source_action_attempt_receipt_sha256:
        raise ValueError(
            "--source-action-attempt-receipt-sha256 is valid only for verify "
            "and download"
        )
    if arguments.action not in _VERIFIER_ACTIONS and arguments.source_evidence_recovery:
        raise ValueError(
            "--source-evidence-recovery is valid only for verify and download"
        )
    if arguments.action == "download":
        if arguments.local_output != MODAL_DOWNLOAD_OUTPUT_ROOT:
            raise ValueError(
                "download requires the frozen --local-output "
                f"{MODAL_DOWNLOAD_OUTPUT_ROOT}"
            )
        safe_relative_path(MODAL_DOWNLOAD_OUTPUT_ROOT)
    elif arguments.local_output:
        raise ValueError("--local-output is valid only for download")
    provider_fields = {
        "provider_cost_cap_usd": arguments.provider_cost_cap_usd,
        "provider_approval_plan_path": arguments.provider_approval_plan_path,
        "approval_plan_sha256": arguments.approval_plan_sha256,
        "provider_price_basis_path": arguments.provider_price_basis_path,
        "provider_price_basis_sha256": arguments.provider_price_basis_sha256,
    }
    if arguments.action in _PROVIDER_ACTIONS:
        missing = sorted(key for key, value in provider_fields.items() if not value)
        if missing:
            raise ValueError(
                "provider canary approval fields are required: " + ", ".join(missing)
            )
        _canonical_decimal_amount(
            arguments.provider_cost_cap_usd,
            "provider_cost_cap_usd",
            require_positive=True,
        )
        safe_relative_path(arguments.provider_approval_plan_path)
        safe_relative_path(arguments.provider_price_basis_path)
        for field in ("approval_plan_sha256", "provider_price_basis_sha256"):
            if _SHA256.fullmatch(getattr(arguments, field)) is None:
                raise ValueError(f"{field} must be a lowercase SHA-256")
    elif any(value for value in provider_fields.values()):
        raise ValueError(
            "provider cost, plan, and price-basis fields are valid only for "
            "provider canaries"
        )


def _validate_download_destination(
    arguments: argparse.Namespace,
    *,
    project_root: Path,
) -> None:
    if arguments.action != "download":
        return
    relative = safe_relative_path(arguments.local_output)
    current = project_root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError("download path may not traverse symbolic links")
        if current.exists() and not current.is_dir():
            raise ValueError("existing download path components must be directories")
    destination = current / arguments.run_id
    if destination.exists() or destination.is_symlink():
        raise ValueError("local run download destination already exists")


def _canonical_decimal_amount(
    value: object,
    field: str,
    *,
    require_positive: bool,
) -> Decimal:
    if (
        not isinstance(value, str)
        or len(value) > 64
        or _CANONICAL_DECIMAL.fullmatch(value) is None
    ):
        raise ValueError(f"{field} must be a canonical decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"{field} must be a canonical decimal string") from error
    if (
        not parsed.is_finite()
        or parsed.is_signed()
        or parsed < 0
        or (require_positive and parsed <= 0)
        or format(parsed, "f") != value
    ):
        qualifier = "positive" if require_positive else "non-negative"
        raise ValueError(f"{field} must be canonical, finite, and {qualifier}")
    return parsed


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key: {key}")
        result[key] = value
    return result


def _open_nofollow_local_directory(
    path: Path,
    *,
    create_missing: bool = False,
) -> int:
    """Open, and optionally create, a directory through no-follow descriptors."""

    no_follow = getattr(os, "O_NOFOLLOW", None)
    directory_only = getattr(os, "O_DIRECTORY", None)
    if no_follow is None or directory_only is None:
        raise ValueError("platform cannot enforce no-follow approval reads")
    absolute = Path(os.path.abspath(os.fspath(path)))
    flags = os.O_RDONLY | os.O_CLOEXEC | no_follow | directory_only
    descriptor = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            created = False
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create_missing:
                    raise
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    os.fsync(descriptor)
                    created = True
                except FileExistsError:
                    pass
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ValueError("local approval path contains an unsafe component")
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                if created:
                    os.fchmod(next_descriptor, 0o700)
                opened = os.fstat(next_descriptor)
                if (
                    (before.st_dev, before.st_ino)
                    != (opened.st_dev, opened.st_ino)
                    or not stat.S_ISDIR(opened.st_mode)
                    or (created and opened.st_uid != os.getuid())
                    or (created and stat.S_IMODE(opened.st_mode) != 0o700)
                ):
                    raise ValueError(
                        "local approval parent changed while it was opened"
                    )
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _launch_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
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


def _sha256_held_descriptor(
    descriptor: int,
    *,
    expected_size: int,
    maximum_bytes: int,
    label: str,
) -> str:
    if expected_size > maximum_bytes:
        raise ValueError(f"{label} exceeds its byte limit")
    digest = hashlib.sha256()
    offset = 0
    while offset < expected_size:
        try:
            chunk = os.pread(
                descriptor,
                min(64 * 1024, expected_size - offset),
                offset,
            )
        except InterruptedError:
            continue
        if not chunk:
            raise ValueError(f"{label} changed while it was hashed")
        digest.update(chunk)
        offset += len(chunk)
    if os.pread(descriptor, 1, expected_size):
        raise ValueError(f"{label} changed while it was hashed")
    return digest.hexdigest()


def _open_held_launch_file(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
    hash_content: bool,
    require_owner_executable: bool,
    require_current_uid: bool = True,
    required_mode: int | None = None,
    require_stable_ctime: bool = True,
) -> _HeldLaunchFileBinding:
    """Open one stable, owned, non-symlink launch dependency by descriptor."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_descriptor = _open_nofollow_local_directory(absolute.parent)
    descriptor: int | None = None
    try:
        before = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        mode = stat.S_IMODE(before.st_mode)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (require_current_uid and before.st_uid != os.getuid())
            or mode & 0o022
            or (required_mode is not None and mode != required_mode)
            or (require_owner_executable and not mode & stat.S_IXUSR)
            or before.st_size > maximum_bytes
        ):
            raise ValueError(f"{label} metadata is unsafe")
        descriptor = os.open(
            absolute.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            _launch_file_identity(opened) != _launch_file_identity(before)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (require_current_uid and opened.st_uid != os.getuid())
        ):
            raise ValueError(f"{label} changed while it was opened")
        digest = (
            _sha256_held_descriptor(
                descriptor,
                expected_size=opened.st_size,
                maximum_bytes=maximum_bytes,
                label=label,
            )
            if hash_content
            else None
        )
        after = os.fstat(descriptor)
        final_path = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            _launch_file_identity(after) != _launch_file_identity(opened)
            or _launch_file_identity(final_path) != _launch_file_identity(opened)
            or final_path.st_nlink != 1
            or (require_current_uid and final_path.st_uid != os.getuid())
        ):
            raise ValueError(f"{label} changed while it was bound")
        binding = _HeldLaunchFileBinding(
            label=label,
            canonical_path=absolute,
            descriptor=descriptor,
            device=after.st_dev,
            inode=after.st_ino,
            size_bytes=after.st_size,
            mode=stat.S_IMODE(after.st_mode),
            owner_uid=after.st_uid,
            mtime_ns=after.st_mtime_ns,
            ctime_ns=after.st_ctime_ns,
            sha256=digest,
            maximum_bytes=maximum_bytes,
            require_owner_executable=require_owner_executable,
            require_current_uid=require_current_uid,
            required_mode=required_mode,
            require_stable_ctime=require_stable_ctime,
        )
        descriptor = None
        return binding
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)


def _require_held_launch_file_binding(binding: _HeldLaunchFileBinding) -> None:
    if binding.descriptor < 0:
        raise ValueError(f"{binding.label} descriptor is closed")
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
    observed = _launch_file_identity(opened)
    if (
        observed[:-1] != expected[:-1]
        or (binding.require_stable_ctime and observed[-1] != expected[-1])
        or not stat.S_ISREG(opened.st_mode)
        or opened.st_nlink != 1
        or opened.st_uid != binding.owner_uid
        or (binding.require_current_uid and opened.st_uid != os.getuid())
        or stat.S_IMODE(opened.st_mode) & 0o022
        or (
            binding.required_mode is not None
            and stat.S_IMODE(opened.st_mode) != binding.required_mode
        )
        or (
            binding.require_owner_executable
            and not stat.S_IMODE(opened.st_mode) & stat.S_IXUSR
        )
    ):
        raise ValueError(f"{binding.label} descriptor changed")
    if binding.sha256 is not None and _sha256_held_descriptor(
        binding.descriptor,
        expected_size=binding.size_bytes,
        maximum_bytes=binding.maximum_bytes,
        label=binding.label,
    ) != binding.sha256:
        raise ValueError(f"{binding.label} bytes changed")
    parent_descriptor = _open_nofollow_local_directory(binding.canonical_path.parent)
    try:
        current = os.stat(
            binding.canonical_path.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise ValueError(f"{binding.label} path was removed") from None
    finally:
        os.close(parent_descriptor)
    observed_path = _launch_file_identity(current)
    if (
        observed_path[:-1] != expected[:-1]
        or (
            binding.require_stable_ctime
            and observed_path[-1] != expected[-1]
        )
        or stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or current.st_nlink != 1
        or current.st_uid != binding.owner_uid
        or (binding.require_current_uid and current.st_uid != os.getuid())
        or (
            binding.required_mode is not None
            and stat.S_IMODE(current.st_mode) != binding.required_mode
        )
    ):
        raise ValueError(f"{binding.label} path changed")


def _require_exact_modal_version(
    version_lookup: Callable[[str], str] | None = None,
) -> None:
    lookup = importlib.metadata.version if version_lookup is None else version_lookup
    try:
        installed_version = lookup("modal")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("the pinned Modal package is not installed") from error
    if installed_version != MODAL_VERSION or MODAL_VERSION != "1.5.3":
        raise ValueError("paid launch requires exactly Modal 1.5.3")


def _open_modal_executable_binding(
    modal_executable: Path | None = None,
    *,
    version_lookup: Callable[[str], str] | None = None,
) -> _HeldLaunchFileBinding:
    _require_exact_modal_version(version_lookup)
    selected = (
        Path(sys.executable).with_name("modal")
        if modal_executable is None
        else modal_executable
    )
    return _open_held_launch_file(
        selected,
        label="pinned Modal executable",
        maximum_bytes=_MAX_MODAL_CONSOLE_SCRIPT_BYTES,
        hash_content=True,
        require_owner_executable=True,
    )


def _resolved_venv_python_executable() -> Path:
    invoked = Path(os.path.abspath(sys.executable))
    try:
        venv_root = Path(sys.prefix).resolve(strict=True)
        base_root = Path(sys.base_prefix).resolve(strict=True)
        invoked_venv_root = invoked.parent.parent.resolve(strict=True)
        resolved = invoked.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("the active Python executable cannot be resolved") from error
    if venv_root == base_root or invoked_venv_root != venv_root:
        raise ValueError("paid launch requires the active project virtual environment")
    return resolved


def _canonical_venv_site_packages() -> Path:
    try:
        venv_root = Path(sys.prefix).resolve(strict=True)
        purelib = Path(sysconfig.get_path("purelib")).resolve(strict=True)
        purelib.relative_to(venv_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "virtual-environment site-packages cannot be resolved"
        ) from error
    descriptor = _open_nofollow_local_directory(purelib)
    os.close(descriptor)
    return purelib


def _open_python_executable_binding(
    python_executable: Path | None = None,
) -> _HeldLaunchFileBinding:
    selected = (
        _resolved_venv_python_executable()
        if python_executable is None
        else Path(python_executable).resolve(strict=True)
    )
    return _open_held_launch_file(
        selected,
        label="resolved Python executable",
        maximum_bytes=_MAX_PYTHON_EXECUTABLE_BYTES,
        hash_content=True,
        require_owner_executable=True,
        require_current_uid=False,
    )


def _canonical_passwd_home(
    passwd_lookup: Callable[[int], Any] | None = None,
) -> Path:
    lookup = pwd.getpwuid if passwd_lookup is None else passwd_lookup
    record = lookup(os.getuid())
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
    descriptor = _open_nofollow_local_directory(canonical)
    os.close(descriptor)
    return canonical


def _open_modal_config_binding(
    *,
    passwd_lookup: Callable[[int], Any] | None = None,
) -> _HeldLaunchFileBinding:
    canonical_home = _canonical_passwd_home(passwd_lookup)
    return _open_held_launch_file(
        canonical_home / ".modal.toml",
        label="canonical Modal configuration",
        maximum_bytes=_MAX_MODAL_CONFIG_BYTES,
        hash_content=False,
        require_owner_executable=False,
    )


def _open_modal_launch_bindings() -> _ModalLaunchBindings:
    python_executable = _open_python_executable_binding()
    try:
        executable = _open_modal_executable_binding()
    except BaseException:
        python_executable.close()
        raise
    try:
        config = _open_modal_config_binding()
    except BaseException:
        executable.close()
        python_executable.close()
        raise
    bindings = _ModalLaunchBindings(
        python_executable=python_executable,
        modal_executable=executable,
        modal_config=config,
    )
    try:
        bindings.require_current()
    except BaseException:
        bindings.close()
        raise
    return bindings


class _DarwinTimeval(ctypes.Structure):
    _fields_ = (("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_int))


class _DarwinTimespec(ctypes.Structure):
    _fields_ = (("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long))


class _DarwinProcBSDInfo(ctypes.Structure):
    _fields_ = (
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    )


def _darwin_machine_identity() -> bytes:
    """Read the stable host UUID without persisting or formatting it."""

    libc = ctypes.CDLL(None, use_errno=True)
    gethostuuid = libc.gethostuuid
    gethostuuid.argtypes = (
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.POINTER(_DarwinTimespec),
    )
    gethostuuid.restype = ctypes.c_int
    value = (ctypes.c_ubyte * 16)()
    wait = _DarwinTimespec(5, 0)
    if gethostuuid(value, ctypes.byref(wait)) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "local machine identity could not be read")
    return bytes(value)


def _linux_machine_identity() -> bytes:
    """Read the root-owned machine-id without following a replacement link."""

    descriptor = os.open(
        "/etc/machine-id",
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or metadata.st_size > 256
        ):
            raise ValueError("Linux machine-identity source metadata is unsafe")
        raw = os.read(descriptor, 257)
        if os.read(descriptor, 1):
            raise ValueError("Linux machine-identity source is unexpectedly large")
    finally:
        os.close(descriptor)
    selected = raw.strip()
    if re.fullmatch(rb"[0-9a-f]{32}", selected) is None:
        raise ValueError("Linux machine-identity source is invalid")
    return selected


def _default_machine_identity_provider() -> bytes:
    if sys.platform == "darwin":
        return _darwin_machine_identity()
    if sys.platform.startswith("linux"):
        return _linux_machine_identity()
    raise ValueError("platform has no approved local machine-identity provider")


def _validated_machine_identity(value: object) -> bytes:
    if not isinstance(value, bytes) or not 16 <= len(value) <= 256:
        raise ValueError("local machine identity must be bounded raw bytes")
    if not any(value):
        raise ValueError("local machine identity may not be all-zero bytes")
    return value


def _local_machine_binding_sha256(anchor_id: str, machine_identity: object) -> str:
    if not isinstance(anchor_id, str) or _SHA256.fullmatch(anchor_id) is None:
        raise ValueError("local host-anchor identifier is invalid")
    identity = _validated_machine_identity(machine_identity)
    return hmac.new(
        identity,
        b"RL4RL ModalLocalHostAnchor machine binding v1\0"
        + anchor_id.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


def _validated_boot_identity(value: object) -> bytes:
    if not isinstance(value, bytes) or len(value) != 16:
        raise ValueError("local OS boot identity must be exactly 16 raw bytes")
    if not any(value):
        raise ValueError("local OS boot identity may not be all-zero bytes")
    return value


def _parsed_boot_uuid(raw: bytes, label: str) -> bytes:
    selected = raw.rstrip(b"\x00").strip()
    if _BOOT_UUID_TEXT.fullmatch(selected) is None:
        raise ValueError(f"{label} is not a canonical boot UUID")
    try:
        parsed = uuid.UUID(selected.decode("ascii")).bytes
    except (UnicodeDecodeError, ValueError) as error:  # pragma: no cover - regex guard
        raise ValueError(f"{label} is not a canonical boot UUID") from error
    return _validated_boot_identity(parsed)


def _darwin_boot_identity() -> bytes:
    """Read ``kern.bootsessionuuid`` directly and retain only raw UUID bytes."""

    libc = ctypes.CDLL(None, use_errno=True)
    sysctlbyname = libc.sysctlbyname
    sysctlbyname.argtypes = (
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    sysctlbyname.restype = ctypes.c_int
    size = ctypes.c_size_t()
    if sysctlbyname(
        b"kern.bootsessionuuid",
        None,
        ctypes.byref(size),
        None,
        0,
    ) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "Darwin boot identity could not be read")
    if size.value < 36 or size.value > 64:
        raise ValueError("Darwin boot identity has an invalid size")
    value = (ctypes.c_ubyte * size.value)()
    observed_size = ctypes.c_size_t(size.value)
    if sysctlbyname(
        b"kern.bootsessionuuid",
        ctypes.byref(value),
        ctypes.byref(observed_size),
        None,
        0,
    ) != 0 or observed_size.value != size.value:
        error_number = ctypes.get_errno()
        raise OSError(error_number, "Darwin boot identity could not be read")
    return _parsed_boot_uuid(bytes(value), "Darwin boot identity")


def _linux_boot_identity() -> bytes:
    """Read the kernel boot UUID from procfs without following a link."""

    descriptor = os.open(
        "/proc/sys/kernel/random/boot_id",
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError("Linux boot-identity source metadata is unsafe")
        raw = os.read(descriptor, 65)
        if os.read(descriptor, 1):
            raise ValueError("Linux boot-identity source is unexpectedly large")
    finally:
        os.close(descriptor)
    return _parsed_boot_uuid(raw, "Linux boot identity")


def _default_boot_identity_provider() -> bytes:
    if sys.platform == "darwin":
        return _darwin_boot_identity()
    if sys.platform.startswith("linux"):
        return _linux_boot_identity()
    raise ValueError("platform has no approved local OS boot-identity provider")


def _darwin_boot_started_at_unix_microseconds() -> int:
    """Read the stable whole-second portion of Darwin's ``kern.boottime``.

    Darwin may adjust ``timeval.tv_usec`` within the same boot (for example
    after sleep or a wall-clock correction) while keeping both ``tv_sec`` and
    ``kern.bootsessionuuid`` stable.  The subsecond field therefore cannot be
    used as a boot-session identity invariant.  Linux's ``btime`` is already
    whole-second precision, so normalizing Darwin here gives both providers
    the same contract.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    sysctlbyname = libc.sysctlbyname
    sysctlbyname.argtypes = (
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    )
    sysctlbyname.restype = ctypes.c_int
    value = _DarwinTimeval()
    size = ctypes.c_size_t(ctypes.sizeof(value))
    result = sysctlbyname(
        b"kern.boottime",
        ctypes.byref(value),
        ctypes.byref(size),
        None,
        0,
    )
    if result != 0 or size.value != ctypes.sizeof(value):
        error_number = ctypes.get_errno()
        raise OSError(error_number, "kern.boottime could not be read")
    if value.tv_usec < 0 or value.tv_usec >= 1_000_000:
        raise ValueError("Darwin boot-session microseconds are invalid")
    return value.tv_sec * 1_000_000


def _linux_boot_started_at_unix_microseconds() -> int:
    """Read the kernel's stable boot-start second from procfs without a command."""

    descriptor = os.open(
        "/proc/stat",
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != 0:
            raise ValueError("Linux boot-session source metadata is unsafe")
        chunks: list[bytes] = []
        size = 0
        while chunk := os.read(descriptor, 64 * 1024):
            size += len(chunk)
            if size > 1024 * 1024:
                raise ValueError("Linux boot-session source is unexpectedly large")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    matches = [
        line.removeprefix(b"btime ")
        for line in b"".join(chunks).splitlines()
        if line.startswith(b"btime ")
    ]
    if len(matches) != 1 or re.fullmatch(rb"[1-9][0-9]*", matches[0]) is None:
        raise ValueError("Linux boot-session source is invalid")
    return int(matches[0]) * 1_000_000


def _default_boot_session_provider() -> int:
    if sys.platform == "darwin":
        return _darwin_boot_started_at_unix_microseconds()
    if sys.platform.startswith("linux"):
        return _linux_boot_started_at_unix_microseconds()
    raise ValueError("platform has no approved local boot-session provider")


def _validated_boot_started_at_unix_microseconds(value: object) -> int:
    if type(value) is not int:
        raise ValueError("local boot-session start must be an exact integer")
    upper = (
        time.time_ns() // 1_000
        + _MAX_BOOT_FUTURE_SKEW_MICROSECONDS
    )
    if value < _MIN_BOOT_STARTED_AT_UNIX_MICROSECONDS or value > upper:
        raise ValueError("local boot-session start is outside the valid time range")
    return value


def _boot_started_at_unix_second(value: object) -> int:
    """Return the validated boot-start second for stable identity checks."""

    return _validated_boot_started_at_unix_microseconds(value) // 1_000_000


def _local_boot_session_sha256(
    host_anchor_sha256: str,
    boot_identity: object,
) -> str:
    if (
        not isinstance(host_anchor_sha256, str)
        or _SHA256.fullmatch(host_anchor_sha256) is None
    ):
        raise ValueError("local host-anchor SHA-256 is invalid")
    identity = _validated_boot_identity(boot_identity)
    return hashlib.sha256(
        b"RL4RL ModalLocalBootSessionBinding v2\0"
        + bytes.fromhex(host_anchor_sha256)
        + identity
    ).hexdigest()


def _validate_local_containment_fields(payload: Mapping[str, Any]) -> None:
    if payload["local_host_anchor_path"] != modal_local_host_anchor_path().as_posix():
        raise ValueError("local host-anchor path is not canonical")
    host_sha256 = payload["local_host_anchor_sha256"]
    boot_started = payload["local_boot_started_at_unix_microseconds"]
    session_sha256 = payload["local_boot_session_sha256"]
    if not isinstance(host_sha256, str) or _SHA256.fullmatch(host_sha256) is None:
        raise ValueError("local host-anchor SHA-256 is invalid")
    if (
        not isinstance(session_sha256, str)
        or _SHA256.fullmatch(session_sha256) is None
    ):
        raise ValueError("local boot-session SHA-256 is invalid")
    _validated_boot_started_at_unix_microseconds(boot_started)


def modal_local_boot_session_relation(
    *,
    local_host_anchor_sha256: str,
    local_boot_started_at_unix_microseconds: int,
    local_boot_session_sha256: str,
    boot_session_provider: Callable[[], int] | None = None,
    boot_identity_provider: Callable[[], bytes] | None = None,
) -> str:
    """Classify the current OS boot, rejecting UUID/time contradictions."""

    expected = {
        "local_host_anchor_path": modal_local_host_anchor_path().as_posix(),
        "local_host_anchor_sha256": local_host_anchor_sha256,
        "local_boot_started_at_unix_microseconds": (
            local_boot_started_at_unix_microseconds
        ),
        "local_boot_session_sha256": local_boot_session_sha256,
    }
    _validate_local_containment_fields(expected)
    start_provider = (
        _default_boot_session_provider
        if boot_session_provider is None
        else boot_session_provider
    )
    identity_provider = (
        _default_boot_identity_provider
        if boot_identity_provider is None
        else boot_identity_provider
    )
    current_start = _validated_boot_started_at_unix_microseconds(start_provider())
    recorded_start_second = _boot_started_at_unix_second(
        local_boot_started_at_unix_microseconds
    )
    current_start_second = _boot_started_at_unix_second(current_start)
    current_session_sha256 = _local_boot_session_sha256(
        local_host_anchor_sha256,
        identity_provider(),
    )
    if current_session_sha256 == local_boot_session_sha256:
        if current_start_second != recorded_start_second:
            raise ValueError("same OS boot identity has a changed start time")
        return "same_boot_session"
    if current_start_second <= recorded_start_second:
        raise ValueError("changed OS boot identity does not have a later start time")
    return "different_boot_session"


def _open_or_create_private_child_directory(
    parent_descriptor: int,
    name: str,
) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("local containment directory name is invalid")
    created = False
    try:
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        try:
            os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
            os.fsync(parent_descriptor)
            created = True
        except FileExistsError:
            pass
        before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
    ):
        raise ValueError("local containment directory metadata is unsafe")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        if created:
            os.fchmod(descriptor, 0o700)
        opened = os.fstat(descriptor)
        rebound = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or (opened.st_dev, opened.st_ino) != (rebound.st_dev, rebound.st_ino)
            or not stat.S_ISDIR(opened.st_mode)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o700
        ):
            raise ValueError("local containment directory changed while opening")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _local_containment_parent(
    project_root: Path,
    receipt_directory: str | Path | None,
) -> Path:
    if receipt_directory is None:
        return project_root / "outputs" / "readiness"
    supplied = Path(receipt_directory)
    directory = supplied if supplied.is_absolute() else project_root / supplied
    absolute = Path(os.path.abspath(directory))
    descriptor = _open_nofollow_local_directory(
        absolute,
        create_missing=True,
    )
    os.close(descriptor)
    return absolute


def _prepare_local_containment_directory(
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    include_process_starts: bool,
) -> None:
    parent = _local_containment_parent(project_root, receipt_directory)
    parent_descriptor = _open_nofollow_local_directory(parent)
    containment_descriptor: int | None = None
    process_descriptor: int | None = None
    try:
        parent_metadata = os.fstat(parent_descriptor)
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or parent_metadata.st_uid != os.getuid()
            or stat.S_IMODE(parent_metadata.st_mode) & 0o022
        ):
            raise ValueError("local containment parent metadata is unsafe")
        containment_descriptor = _open_or_create_private_child_directory(
            parent_descriptor,
            MODAL_LOCAL_CONTAINMENT_ROOT.name,
        )
        if include_process_starts:
            process_descriptor = _open_or_create_private_child_directory(
                containment_descriptor,
                "process_starts",
            )
    finally:
        if process_descriptor is not None:
            os.close(process_descriptor)
        if containment_descriptor is not None:
            os.close(containment_descriptor)
        os.close(parent_descriptor)


def _local_containment_destination(
    logical_path: str,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
) -> Path:
    logical = safe_relative_path(logical_path)
    containment_root = safe_relative_path(MODAL_LOCAL_CONTAINMENT_ROOT.as_posix())
    try:
        relative = logical.relative_to(containment_root)
    except ValueError as error:
        raise ValueError("local containment path is not canonical") from error
    if receipt_directory is None:
        return project_root.joinpath(*logical.parts)
    return _local_containment_parent(project_root, receipt_directory).joinpath(
        MODAL_LOCAL_CONTAINMENT_ROOT.name,
        *relative.parts,
    )


def _validate_local_host_anchor_payload(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema_name",
        "schema_version",
        "anchor_id",
        "machine_binding_sha256",
    }:
        raise ValueError("local host anchor has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalLocalHostAnchor"
        or payload["schema_version"] != "1.0"
        or not isinstance(payload["anchor_id"], str)
        or _SHA256.fullmatch(payload["anchor_id"]) is None
        or not isinstance(payload["machine_binding_sha256"], str)
        or _SHA256.fullmatch(payload["machine_binding_sha256"]) is None
    ):
        raise ValueError("local host anchor has the wrong contract")


def _open_or_create_local_containment_binding(
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    host_anchor_id_factory: Callable[[int], str],
    machine_identity_provider: Callable[[], bytes],
    boot_session_provider: Callable[[], int],
    boot_identity_provider: Callable[[], bytes],
) -> _ModalLocalContainmentBinding:
    _prepare_local_containment_directory(
        project_root=project_root,
        receipt_directory=receipt_directory,
        include_process_starts=False,
    )
    logical = modal_local_host_anchor_path().as_posix()
    destination = _local_containment_destination(
        logical,
        project_root=project_root,
        receipt_directory=receipt_directory,
    )
    anchor_id = host_anchor_id_factory(32)
    if not isinstance(anchor_id, str) or _SHA256.fullmatch(anchor_id) is None:
        raise ValueError("local host-anchor factory returned an invalid identifier")
    machine_binding_sha256 = _local_machine_binding_sha256(
        anchor_id,
        machine_identity_provider(),
    )
    with contextlib.suppress(FileExistsError):
        create_json_exclusive(
            destination,
            {
                "schema_name": "ModalLocalHostAnchor",
                "schema_version": "1.0",
                "anchor_id": anchor_id,
                "machine_binding_sha256": machine_binding_sha256,
            },
        )
    anchor = _open_held_launch_file(
        destination,
        label="local host anchor",
        maximum_bytes=_MAX_LOCAL_HOST_ANCHOR_BYTES,
        hash_content=True,
        require_owner_executable=False,
        required_mode=0o600,
        # macOS may add com.apple.provenance after create-only publication.
        # The anchor is still inode-, path-, metadata-, and byte-hash-bound.
        require_stable_ctime=False,
    )
    try:
        payload, _raw, observed_sha256 = _read_private_json_path(
            destination,
            "local_host_anchor",
        )
        _validate_local_host_anchor_payload(payload)
        if anchor.sha256 != observed_sha256:
            raise ValueError("local host-anchor bytes changed while binding")
        if payload["machine_binding_sha256"] != _local_machine_binding_sha256(
            payload["anchor_id"],
            machine_identity_provider(),
        ):
            raise ValueError("local host anchor belongs to another machine")
        boot_started = _validated_boot_started_at_unix_microseconds(
            boot_session_provider()
        )
        boot_identity = _validated_boot_identity(boot_identity_provider())
        if anchor.sha256 is None:  # pragma: no cover - construction invariant
            raise AssertionError("local host anchor lacks its raw SHA-256")
        binding = _ModalLocalContainmentBinding(
            anchor=anchor,
            host_anchor_path=logical,
            host_anchor_sha256=anchor.sha256,
            host_anchor_id=payload["anchor_id"],
            machine_binding_sha256=payload["machine_binding_sha256"],
            boot_started_at_unix_microseconds=boot_started,
            boot_session_sha256=_local_boot_session_sha256(
                anchor.sha256,
                boot_identity,
            ),
            machine_identity_provider=machine_identity_provider,
            boot_session_provider=boot_session_provider,
            boot_identity_provider=boot_identity_provider,
        )
        binding.require_current()
        return binding
    except BaseException:
        anchor.close()
        raise


def validate_current_modal_local_host_anchor(
    project_root: str | Path,
    *,
    expected_path: str,
    expected_sha256: str,
    machine_identity_provider: Callable[[], bytes] | None = None,
) -> None:
    """Cost-free proof that the current private anchor matches recorded bytes."""

    if expected_path != modal_local_host_anchor_path().as_posix():
        raise ValueError("recorded local host-anchor path is not canonical")
    if _SHA256.fullmatch(expected_sha256) is None:
        raise ValueError("recorded local host-anchor SHA-256 is invalid")
    root = Path(os.path.abspath(os.fspath(project_root)))
    binding = _open_held_launch_file(
        root.joinpath(*safe_relative_path(expected_path).parts),
        label="local host anchor",
        maximum_bytes=_MAX_LOCAL_HOST_ANCHOR_BYTES,
        hash_content=True,
        require_owner_executable=False,
        required_mode=0o600,
        require_stable_ctime=False,
    )
    try:
        payload, _raw, observed_sha256 = _read_project_json_file(
            root,
            expected_path,
            "local_host_anchor",
        )
        _validate_local_host_anchor_payload(payload)
        provider = (
            _default_machine_identity_provider
            if machine_identity_provider is None
            else machine_identity_provider
        )
        if payload["machine_binding_sha256"] != _local_machine_binding_sha256(
            payload["anchor_id"],
            provider(),
        ):
            raise ValueError("current local host anchor belongs to another machine")
        if binding.sha256 != expected_sha256 or observed_sha256 != expected_sha256:
            raise ValueError("current local host anchor differs from recorded bytes")
        binding.require_current()
    finally:
        binding.close()


def _open_owned_private_runtime_directory(
    parent_descriptor: int,
    name: str,
    *,
    require_new: bool,
) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise ValueError("private runtime directory name is invalid")
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_descriptor)
        created = True
        os.fsync(parent_descriptor)
    except FileExistsError:
        if require_new:
            raise ValueError(
                "private Python runtime directory already exists"
            ) from None
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) != 0o700
        or (require_new and not created)
    ):
        raise ValueError("private Python runtime directory metadata is unsafe")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    opened = os.fstat(descriptor)
    if (
        (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.getuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise ValueError("private Python runtime directory changed while opening")
    return descriptor


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        try:
            written = os.write(descriptor, view)
        except InterruptedError:
            continue
        if written <= 0:
            raise OSError("private Python executable copy made no progress")
        view = view[written:]


def _remove_partial_python_execution_copy(
    *,
    runtime_descriptor: int,
    attempt_descriptor: int,
    attempt_id: str,
    attempt_identity: tuple[int, int],
    destination_identity: tuple[int, int] | None,
) -> None:
    """Remove only partial objects still bound to the held creation descriptors."""

    if destination_identity is not None:
        leaf = os.stat(
            "python",
            dir_fd=attempt_descriptor,
            follow_symlinks=False,
        )
        if (
            (leaf.st_dev, leaf.st_ino) != destination_identity
            or stat.S_ISLNK(leaf.st_mode)
            or not stat.S_ISREG(leaf.st_mode)
            or leaf.st_uid != os.getuid()
            or leaf.st_nlink != 1
        ):
            raise ValueError("partial Python execution copy was replaced")
        os.unlink("python", dir_fd=attempt_descriptor)
        os.fsync(attempt_descriptor)
    if os.listdir(attempt_descriptor):
        raise ValueError("partial Python execution directory is not empty")
    rebound = os.stat(
        attempt_id,
        dir_fd=runtime_descriptor,
        follow_symlinks=False,
    )
    if (
        (rebound.st_dev, rebound.st_ino) != attempt_identity
        or stat.S_ISLNK(rebound.st_mode)
        or not stat.S_ISDIR(rebound.st_mode)
    ):
        raise ValueError("partial Python execution directory was replaced")
    os.rmdir(attempt_id, dir_fd=runtime_descriptor)
    os.fsync(runtime_descriptor)


def _materialize_python_execution_copy(
    source: _HeldLaunchFileBinding,
    *,
    project_root: Path,
    attempt_id: str,
) -> _PrivatePythonExecutionCopy:
    """Create one private executable copy of the already-bound Python bytes."""

    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("Python runtime copy requires an exact attempt ID")
    source.require_current()
    if source.sha256 is None or source.size_bytes > _MAX_PYTHON_EXECUTABLE_BYTES:
        raise ValueError("resolved Python executable lacks a bounded content digest")
    root = Path(os.path.abspath(project_root))
    readiness_descriptor = _open_nofollow_local_directory(
        root.joinpath(*_MODAL_PYTHON_RUNTIME_ROOT.parent.parts)
    )
    runtime_descriptor: int | None = None
    attempt_descriptor: int | None = None
    destination_descriptor: int | None = None
    destination_identity: tuple[int, int] | None = None
    attempt_identity: tuple[int, int] | None = None
    binding: _HeldLaunchFileBinding | None = None
    destination = root.joinpath(
        *_MODAL_PYTHON_RUNTIME_ROOT.parts,
        attempt_id,
        "python",
    )
    digest = hashlib.sha256()
    try:
        runtime_descriptor = _open_owned_private_runtime_directory(
            readiness_descriptor,
            _MODAL_PYTHON_RUNTIME_ROOT.name,
            require_new=False,
        )
        attempt_descriptor = _open_owned_private_runtime_directory(
            runtime_descriptor,
            attempt_id,
            require_new=True,
        )
        attempt_metadata = os.fstat(attempt_descriptor)
        attempt_identity = (attempt_metadata.st_dev, attempt_metadata.st_ino)
        destination_descriptor = os.open(
            destination.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_CLOEXEC
            | os.O_NOFOLLOW,
            0o500,
            dir_fd=attempt_descriptor,
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
        os.fsync(attempt_descriptor)
        source.require_current()
        binding = _open_held_launch_file(
            destination,
            label="private Python execution copy",
            maximum_bytes=_MAX_PYTHON_EXECUTABLE_BYTES,
            hash_content=True,
            require_owner_executable=True,
            require_stable_ctime=False,
        )
        if binding.sha256 != source.sha256:
            raise ValueError("private Python executable copy digest changed")
        runtime_metadata = os.fstat(runtime_descriptor)
        result = _PrivatePythonExecutionCopy(
            binding=binding,
            runtime_directory_descriptor=runtime_descriptor,
            attempt_directory_descriptor=attempt_descriptor,
            runtime_directory_identity=(
                runtime_metadata.st_dev,
                runtime_metadata.st_ino,
            ),
            attempt_directory_identity=attempt_identity,
            attempt_id=attempt_id,
        )
        binding = None
        runtime_descriptor = None
        attempt_descriptor = None
        return result
    except BaseException:
        cleanup_error: BaseException | None = None
        if destination_descriptor is not None:
            os.close(destination_descriptor)
            destination_descriptor = None
        if binding is not None:
            binding.close()
            binding = None
        if (
            runtime_descriptor is not None
            and attempt_descriptor is not None
            and attempt_identity is not None
        ):
            try:
                _remove_partial_python_execution_copy(
                    runtime_descriptor=runtime_descriptor,
                    attempt_descriptor=attempt_descriptor,
                    attempt_id=attempt_id,
                    attempt_identity=attempt_identity,
                    destination_identity=destination_identity,
                )
            except BaseException as error:
                cleanup_error = error
        if cleanup_error is not None:
            raise RuntimeError(
                "partial private Python execution copy cleanup failed"
            ) from cleanup_error
        raise
    finally:
        if binding is not None:
            binding.close()
        if attempt_descriptor is not None:
            os.close(attempt_descriptor)
        if runtime_descriptor is not None:
            os.close(runtime_descriptor)
        os.close(readiness_descriptor)


def _read_json_leaf_from_directory(
    parent_descriptor: int,
    filename: str,
    field: str,
) -> tuple[dict[str, Any], bytes, str]:
    """Read one private immutable JSON leaf through an already-open parent."""

    before = os.stat(
        filename,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or stat.S_IMODE(before.st_mode) & 0o077
    ):
        raise ValueError(f"{field} must be a private owned single-link regular file")
    if before.st_size > _MAX_LOCAL_APPROVAL_BYTES:
        raise ValueError(f"{field} exceeds the local approval-file limit")
    descriptor = os.open(
        filename,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    try:
        opened = os.fstat(descriptor)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (opened.st_dev, opened.st_ino, opened.st_size)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) & 0o077
        ):
            raise ValueError(f"{field} changed before it was opened")
        chunks: list[bytes] = []
        observed_size = 0
        while chunk := os.read(descriptor, 64 * 1024):
            observed_size += len(chunk)
            if observed_size > _MAX_LOCAL_APPROVAL_BYTES:
                raise ValueError(f"{field} exceeds the local approval-file limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino, opened.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or observed_size != opened.st_size
        ):
            raise ValueError(f"{field} changed while it was read")
    finally:
        os.close(descriptor)
    final_leaf = os.stat(
        filename,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    if (
        stat.S_ISLNK(final_leaf.st_mode)
        or (before.st_dev, before.st_ino, before.st_size)
        != (final_leaf.st_dev, final_leaf.st_ino, final_leaf.st_size)
    ):
        raise ValueError(f"{field} changed after it was read")
    raw = b"".join(chunks)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_json_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must contain one JSON object")
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _read_private_json_path(
    path: Path,
    field: str,
) -> tuple[dict[str, Any], bytes, str]:
    """Secure-read one bounded private JSON path through stable descriptors."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    parent_path = absolute.parent
    parent_descriptor = _open_nofollow_local_directory(parent_path)
    try:
        parent_identity = os.fstat(parent_descriptor)
        result = _read_json_leaf_from_directory(
            parent_descriptor,
            absolute.name,
            field,
        )
        reopened_parent = _open_nofollow_local_directory(parent_path)
        try:
            observed_parent = os.fstat(reopened_parent)
            if (parent_identity.st_dev, parent_identity.st_ino) != (
                observed_parent.st_dev,
                observed_parent.st_ino,
            ):
                raise ValueError(f"{field} parent changed while it was read")
        finally:
            os.close(reopened_parent)
        return result
    finally:
        os.close(parent_descriptor)


def _read_project_json_file(
    project_root: Path,
    logical_path: str,
    field: str,
) -> tuple[dict[str, Any], bytes, str]:
    """Secure-read one bounded, private, project-relative JSON approval."""

    relative = safe_relative_path(logical_path)
    root = Path(os.path.abspath(os.fspath(project_root)))
    return _read_private_json_path(
        root.joinpath(*relative.parts),
        field,
    )


def _require_path_absent_secure(
    path: Path,
    field: str,
    *,
    missing_parent_ok: bool = False,
) -> None:
    """Require a leaf to remain absent under one stable no-follow parent."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        parent_descriptor = _open_nofollow_local_directory(absolute.parent)
    except FileNotFoundError:
        if missing_parent_ok:
            return
        raise
    try:
        parent_identity = os.fstat(parent_descriptor)
        try:
            os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"{field} already exists")
        reopened = _open_nofollow_local_directory(absolute.parent)
        try:
            observed = os.fstat(reopened)
            if (parent_identity.st_dev, parent_identity.st_ino) != (
                observed.st_dev,
                observed.st_ino,
            ):
                raise ValueError(f"{field} parent changed during absence check")
        finally:
            os.close(reopened)
    finally:
        os.close(parent_descriptor)


def _require_live_cohort_unsealed(
    *,
    project_root: Path,
    identity: ModalLiveCohortIdentity,
) -> None:
    logical = modal_migration_lineage_path(identity)
    _require_path_absent_secure(
        project_root.joinpath(*logical.parts),
        "final migration lineage seal",
        missing_parent_ok=True,
    )


def _validate_price_basis(
    payload: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
) -> tuple[Decimal, Decimal, Decimal]:
    if set(payload) != _PRICE_BASIS_FIELDS:
        raise ValueError("provider price basis has an invalid exact schema")
    if (
        payload["schema_name"] != "ProviderPriceBasis"
        or payload["schema_version"] != "1.0"
        or payload["model"] != TARGET_MODEL
    ):
        raise ValueError("provider price basis has the wrong contract or model")
    source_url = payload["official_source_url"]
    if (
        not isinstance(source_url, str)
        or _OFFICIAL_OPENAI_PRICE_URL.fullmatch(source_url) is None
    ):
        raise ValueError("provider price basis must cite an official OpenAI URL")
    timestamp = payload["retrieved_at_utc"]
    if not isinstance(timestamp, str) or re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z",
        timestamp,
    ) is None:
        raise ValueError("provider price basis timestamp must be canonical UTC Z-form")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("provider price basis timestamp is invalid") from error
    if parsed_timestamp.tzinfo is None:
        raise ValueError("provider price basis timestamp must include a timezone")
    if parsed_timestamp.isoformat().replace("+00:00", "Z") != timestamp:
        raise ValueError("provider price basis timestamp is not canonical UTC")
    observed_now = _now_utc() if now_utc is None else now_utc
    if observed_now.tzinfo is None:
        raise ValueError("provider price-basis validation time must include a timezone")
    parsed_timestamp = parsed_timestamp.astimezone(UTC)
    observed_now = observed_now.astimezone(UTC)
    if parsed_timestamp > observed_now + PROVIDER_PRICE_BASIS_FUTURE_SKEW:
        raise ValueError("provider price basis timestamp is too far in the future")
    if observed_now - parsed_timestamp > PROVIDER_PRICE_BASIS_MAX_AGE:
        raise ValueError("provider price basis is older than 48 hours")
    input_rate = _canonical_decimal_amount(
        payload["uncached_input_usd_per_million_tokens"],
        "uncached_input_usd_per_million_tokens",
        require_positive=True,
    )
    output_rate = _canonical_decimal_amount(
        payload["output_usd_per_million_tokens"],
        "output_usd_per_million_tokens",
        require_positive=True,
    )
    request_fee = _canonical_decimal_amount(
        payload["per_request_fee_usd"],
        "per_request_fee_usd",
        require_positive=False,
    )
    return input_rate, output_rate, request_fee


def _read_modal_price_basis(
    *,
    project_root: Path,
    logical_path: str,
    expected_raw_sha256: str,
    image_source_sha256: str,
    require_freshness: bool,
) -> dict[str, Any]:
    payload, _raw, raw_sha256 = _read_project_json_file(
        project_root,
        logical_path,
        "modal_price_basis_path",
    )
    if raw_sha256 != expected_raw_sha256:
        raise ValueError("Modal price-basis raw SHA-256 changed")
    modal_readiness.validate_modal_price_basis_payload(
        payload,
        expected_image_source_sha256=image_source_sha256,
        require_freshness=require_freshness,
    )
    expected_path = modal_readiness.modal_price_basis_logical_path(
        image_source_sha256,
        payload["retrieved_at_utc"],
    ).as_posix()
    if logical_path != expected_path:
        raise ValueError("Modal price basis is outside its canonical source path")
    return payload


def _validate_modal_approval_inputs(
    arguments: argparse.Namespace,
    *,
    project_root: Path,
    image_source_sha256: str,
) -> dict[str, Any]:
    price_basis = _read_modal_price_basis(
        project_root=project_root,
        logical_path=arguments.modal_price_basis_path,
        expected_raw_sha256=arguments.modal_price_basis_sha256,
        image_source_sha256=image_source_sha256,
        require_freshness=True,
    )
    resource_profile = modal_resource_profile(
        arguments.action,
        arguments.harness,
    )
    estimate = modal_readiness.derive_modal_action_cost_estimate(
        action=arguments.action,
        harness=arguments.harness or None,
        resource_profile=resource_profile,
        price_basis=price_basis,
    )
    approved_cap = _canonical_decimal_amount(
        arguments.modal_cost_cap_usd,
        "modal_cost_cap_usd",
        require_positive=True,
    )
    required_estimate = _canonical_decimal_amount(
        estimate["action_estimate_usd"],
        "modal_cost_estimate.action_estimate_usd",
        require_positive=True,
    )
    if approved_cap < required_estimate:
        raise ValueError(
            "Modal cost cap is below the source-bound action estimate"
        )
    return {
        "modal_cost_cap_usd": arguments.modal_cost_cap_usd,
        "modal_resource_profile": resource_profile,
        "modal_price_basis_path": arguments.modal_price_basis_path,
        "modal_price_basis_sha256": arguments.modal_price_basis_sha256,
        "modal_cost_estimate": estimate,
    }


def _validate_provider_approval_inputs(
    arguments: argparse.Namespace,
    *,
    project_root: Path,
    image_source_sha256: str,
    identity: ModalLiveCohortIdentity,
    accepted_cuda_environment_run_id: str,
) -> dict[str, str]:
    plan, _plan_raw, _plan_file_sha256 = _read_project_json_file(
        project_root,
        arguments.provider_approval_plan_path,
        "provider_approval_plan_path",
    )
    plan_sha256 = verify_provider_canary_approval_plan(plan)
    if plan_sha256 != arguments.approval_plan_sha256:
        raise ValueError("approved provider plan SHA-256 differs from its file")
    expected_plan = build_provider_canary_approval_plan(
        project_root,
        source_tree_sha256=identity.source_tree_sha256,
        cohort_id=identity.cohort_id,
        candidate_resume_preflight_receipt_path=(
            arguments.candidate_resume_preflight_receipt_path
        ),
        candidate_resume_preflight_receipt_sha256=(
            arguments.candidate_resume_preflight_receipt_sha256
        ),
    )
    if not modal_readiness.exact_json_equal(plan, expected_plan):
        raise ValueError(
            "provider approval plan differs from the current exact source, model, "
            "settings, or prompt constructors"
        )
    if (
        plan.get("source_tree_sha256") != identity.source_tree_sha256
        or plan.get("image_source_sha256") != image_source_sha256
        or plan.get("cohort_id") != identity.cohort_id
        or plan.get("candidate_resume_preflight_receipt")
        != {
            "path": arguments.candidate_resume_preflight_receipt_path,
            "sha256": arguments.candidate_resume_preflight_receipt_sha256,
        }
    ):
        raise ValueError(
            "provider approval plan differs from its source, cohort, or preflight"
        )

    validate_run_id(accepted_cuda_environment_run_id)
    expected_price_path = modal_readiness.modal_provider_price_basis_path(
        identity
    ).as_posix()
    if arguments.provider_price_basis_path != expected_price_path:
        raise ValueError(
            "provider price basis path is not bound to the accepted CUDA run"
        )
    price_basis, _price_raw, price_file_sha256 = _read_project_json_file(
        project_root,
        arguments.provider_price_basis_path,
        "provider_price_basis_path",
    )
    if price_file_sha256 != arguments.provider_price_basis_sha256:
        raise ValueError("provider price-basis SHA-256 differs from its file")
    input_rate, output_rate, request_fee = _validate_price_basis(price_basis)

    harnesses = plan["harnesses"]
    if not isinstance(harnesses, list):
        raise ValueError("provider approval plan harness roster is invalid")
    selected = (
        harnesses
        if arguments.action == "canaries"
        else [item for item in harnesses if item.get("harness") == arguments.harness]
    )
    expected_count = len(CANARY_ORDER) if arguments.action == "canaries" else 1
    if len(selected) != expected_count:
        raise ValueError("provider approval plan does not bind the selected harnesses")
    input_tokens = sum(
        int(item["first_opportunity"]["conservative_input_token_ceiling"])
        for item in selected
    )
    output_tokens = sum(
        int(item["request_settings"]["max_completion_tokens"])
        for item in selected
    )
    request_count = sum(int(item["maximum_attempts"]) for item in selected)
    required_cost_bound = (
        Decimal(input_tokens) * input_rate / Decimal(1_000_000)
        + Decimal(output_tokens) * output_rate / Decimal(1_000_000)
        + Decimal(request_count) * request_fee
    )
    approved_provider_cap = _canonical_decimal_amount(
        arguments.provider_cost_cap_usd,
        "provider_cost_cap_usd",
        require_positive=True,
    )
    if approved_provider_cap < required_cost_bound:
        raise ValueError(
            "provider cost cap is below the source-bound conservative token-price "
            "ceiling"
        )
    return {
        "provider_cost_cap_usd": arguments.provider_cost_cap_usd,
        "provider_approval_plan_path": arguments.provider_approval_plan_path,
        "approval_plan_sha256": plan_sha256,
        "provider_price_basis_path": arguments.provider_price_basis_path,
        "provider_price_basis_sha256": price_file_sha256,
    }


def _validated_receipt_binding(
    *,
    gate: str,
    logical_path: str,
    expected_sha256: str,
    validator: Callable[[], Any],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    before, raw_before, digest_before = _read_project_json_file(
        project_root,
        logical_path,
        f"{gate}.path",
    )
    if digest_before != expected_sha256:
        raise ValueError(f"{gate} receipt differs from its approved raw SHA-256")
    validator()
    after, raw_after, digest_after = _read_project_json_file(
        project_root,
        logical_path,
        f"{gate}.path",
    )
    if (
        not modal_readiness.exact_json_equal(before, after)
        or raw_before != raw_after
        or digest_before != digest_after
    ):
        raise ValueError(f"{gate} receipt changed during validation")
    if digest_after != expected_sha256:
        raise ValueError(f"{gate} receipt differs from its approved raw SHA-256")
    return after, {
        "gate": gate,
        "path": logical_path,
        "sha256": digest_after,
    }


def _validate_receipt_binding_roster(value: object, field: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    observed: set[tuple[str, str]] = set()
    for index, record in enumerate(value):
        if not isinstance(record, dict) or set(record) != {"gate", "path", "sha256"}:
            raise ValueError(f"{field}[{index}] has an invalid exact schema")
        gate = record["gate"]
        if not isinstance(gate, str) or not gate or not gate.replace("_", "").isalnum():
            raise ValueError(f"{field}[{index}].gate is invalid")
        path = record["path"]
        safe_relative_path(path)
        if _SHA256.fullmatch(record["sha256"]) is None:
            raise ValueError(f"{field}[{index}].sha256 is invalid")
        key = (gate, path)
        if key in observed:
            raise ValueError(f"{field} contains a duplicate binding")
        observed.add(key)


def _expected_intent_concrete_run_ids(
    *,
    action: str,
    run_id: str,
    verifier_run_id: str | None,
) -> list[str]:
    selected_run_id = validate_run_id(run_id)
    if action in _VERIFIER_ACTIONS:
        if verifier_run_id is None:
            raise ValueError("verifier action intent lacks its destination run ID")
        return [validate_run_id(verifier_run_id)]
    if action == "canaries":
        return [
            f"{selected_run_id}-{_CANARY_RUN_SUFFIXES[harness]}"
            for harness in CANARY_ORDER
        ]
    return [selected_run_id]


def _validate_action_intent_contract(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    identity: ModalLiveCohortIdentity,
    project_root: Path,
    expected_action: str | None = None,
    expected_run_id: str | None = None,
    expected_source_run_id: str | None = None,
    expected_verifier_run_id: str | None = None,
    expected_harness: str | None = None,
    launch_nonce: str | None = None,
    modal_command_python_executable: str | Path | None = None,
) -> None:
    """Validate the immutable identity, command, and capability intent core."""

    expected_fields = {field.name for field in dataclass_fields(ModalActionIntent)}
    if set(payload) != expected_fields:
        raise ValueError("Modal action intent has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalActionIntent"
        or payload["schema_version"] != "1.6"
        or payload["attempt_id"] != attempt_id
    ):
        raise ValueError("Modal action intent has the wrong contract")
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("Modal action intent attempt ID is invalid")
    if (
        payload["source_tree_sha256"] != identity.source_tree_sha256
        or payload["cohort_id"] != identity.cohort_id
        or payload["approved_image_source_sha256"]
        != identity.image_source_sha256
    ):
        raise ValueError("Modal action intent belongs to a different live cohort")
    action = payload["action"]
    if action not in _ACTIONS:
        raise ValueError("Modal action intent action is invalid")
    run_id = validate_run_id(payload["run_id"])
    source_run_id = payload["source_run_id"]
    verifier_run_id = payload["verifier_run_id"]
    harness = payload["harness"]
    if source_run_id is not None:
        validate_run_id(source_run_id)
    if verifier_run_id is not None:
        validate_run_id(verifier_run_id)
    if action == "checkpoint-resume":
        if source_run_id is None or source_run_id == run_id:
            raise ValueError("resume intent source and destination IDs are invalid")
    elif source_run_id is not None:
        raise ValueError("non-resume intent contains a source run ID")
    if action in _VERIFIER_ACTIONS:
        if verifier_run_id is None or verifier_run_id == run_id:
            raise ValueError("verifier intent source and destination IDs are invalid")
    elif verifier_run_id is not None:
        raise ValueError("non-verifier intent contains a verifier run ID")
    if action == "canary":
        if harness not in CANARY_ORDER or run_id != (
            f"{run_id.removesuffix('-' + _CANARY_RUN_SUFFIXES[harness])}-"
            f"{_CANARY_RUN_SUFFIXES[harness]}"
        ):
            raise ValueError("single-canary intent harness/run identity is invalid")
    elif harness is not None:
        raise ValueError("non-single-canary intent contains a harness")
    concrete = payload["concrete_remote_run_ids"]
    expected_concrete = _expected_intent_concrete_run_ids(
        action=action,
        run_id=run_id,
        verifier_run_id=verifier_run_id,
    )
    if (
        not isinstance(concrete, list)
        or concrete != expected_concrete
        or len(concrete) != len(set(concrete))
    ):
        raise ValueError("Modal action intent concrete run roster is invalid")
    if payload["modal_profile"] != MODAL_PROFILE:
        raise ValueError("Modal action intent uses the wrong profile")
    if payload["modal_environment"] != MODAL_ENVIRONMENT:
        raise ValueError("Modal action intent uses the wrong Modal environment")
    if payload["outer_cli_timeout_seconds"] != expected_outer_cli_timeout_seconds(
        action
    ):
        raise ValueError("Modal action intent timeout differs from its action")
    if not modal_readiness.exact_json_equal(
        payload["modal_resource_profile"],
        modal_resource_profile(action, harness or ""),
    ):
        raise ValueError("Modal action intent resource profile changed")
    if payload["modal_cost_approved"] is not True:
        raise ValueError("Modal action intent lacks Modal cost approval")
    provider_action = action in _PROVIDER_ACTIONS
    if payload["provider_cost_approved"] is not provider_action:
        raise ValueError(
            "Modal action intent provider approval differs from its action"
        )
    if type(payload["source_evidence_recovery"]) is not bool or (
        payload["source_evidence_recovery"] and action not in _VERIFIER_ACTIONS
    ):
        raise ValueError("Modal action intent evidence-recovery flag is invalid")
    _validate_receipt_binding_roster(
        payload["predecessor_receipts"],
        "intent.predecessor_receipts",
    )
    if _SHA256.fullmatch(payload["modal_command_sha256"]) is None:
        raise ValueError("Modal action intent command digest is invalid")
    if _SHA256.fullmatch(payload["launch_capability_sha256"]) is None:
        raise ValueError("Modal action intent capability digest is invalid")
    if launch_nonce is not None and payload["launch_capability_sha256"] != (
        _launch_capability_sha256(launch_nonce)
    ):
        raise ValueError("Modal action intent does not bind the launch capability")
    _validate_local_containment_fields(payload)
    expected_reservations = [
        binding
        for binding, _reservation in _remote_run_reservation_specs(
            concrete_remote_run_ids=concrete,
            attempt_id=attempt_id,
            action=action,
            identity=identity,
            created_at_utc=payload["created_at_utc"],
            launch_capability_sha256=payload["launch_capability_sha256"],
            local_host_anchor_path=payload["local_host_anchor_path"],
            local_host_anchor_sha256=payload["local_host_anchor_sha256"],
            local_boot_started_at_unix_microseconds=payload[
                "local_boot_started_at_unix_microseconds"
            ],
            local_boot_session_sha256=payload["local_boot_session_sha256"],
        )
    ]
    if payload["remote_run_reservations"] != expected_reservations:
        raise ValueError("Modal action intent reservation roster is not canonical")
    expected_command = build_modal_cli_command(
        python_executable=(
            sys.executable
            if modal_command_python_executable is None
            else modal_command_python_executable
        ),
        project_root=project_root,
        action=action,
        run_id=run_id,
        source_run_id=source_run_id,
        verifier_run_id=verifier_run_id,
        harness=harness,
        source_tree_sha256=identity.source_tree_sha256,
        cohort_id=identity.cohort_id,
        image_source_sha256=identity.image_source_sha256,
        provider_approved=provider_action,
    )
    if payload["modal_command_sha256"] != _modal_command_sha256(expected_command):
        raise ValueError("Modal action intent command digest is not canonical")
    created_at = payload["created_at_utc"]
    if not isinstance(created_at, str):
        raise ValueError("Modal action intent timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Modal action intent timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("Modal action intent timestamp lacks a timezone")
    expected_values = {
        "action": expected_action,
        "run_id": expected_run_id,
        "source_run_id": expected_source_run_id,
        "verifier_run_id": expected_verifier_run_id,
        "harness": expected_harness,
    }
    for field, expected in expected_values.items():
        if expected_action is not None and payload[field] != expected:
            raise ValueError(f"Modal action intent {field} differs from the invocation")


def validate_local_action_intent_for_entrypoint(
    *,
    project_root: str | Path,
    identity: ModalLiveCohortIdentity,
    attempt_id: str,
    expected_intent_sha256: str,
    launch_nonce: str,
    action: str,
    run_id: str,
    source_run_id: str | None,
    verifier_run_id: str | None,
    harness: str | None,
    machine_identity_provider: Callable[[], bytes] | None = None,
    boot_session_provider: Callable[[], int] | None = None,
    boot_identity_provider: Callable[[], bytes] | None = None,
    modal_command_python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Secure-read and bind the durable intent before any remote invocation."""

    if _SHA256.fullmatch(expected_intent_sha256) is None:
        raise ValueError("durable action intent raw SHA-256 is invalid")
    logical = modal_action_intent_receipt_path(identity, attempt_id).as_posix()
    root = Path(os.path.abspath(os.fspath(project_root)))
    payload, _raw, observed_sha256 = _read_project_json_file(
        root,
        logical,
        "durable_action_intent",
    )
    if observed_sha256 != expected_intent_sha256:
        raise ValueError("durable action intent differs from its launch-time bytes")
    _validate_action_intent_contract(
        payload,
        attempt_id=attempt_id,
        identity=identity,
        project_root=root,
        expected_action=action,
        expected_run_id=run_id,
        expected_source_run_id=source_run_id,
        expected_verifier_run_id=verifier_run_id,
        expected_harness=harness,
        launch_nonce=launch_nonce,
        modal_command_python_executable=modal_command_python_executable,
    )
    validate_current_modal_local_host_anchor(
        root,
        expected_path=payload["local_host_anchor_path"],
        expected_sha256=payload["local_host_anchor_sha256"],
        machine_identity_provider=machine_identity_provider,
    )
    if modal_local_boot_session_relation(
        local_host_anchor_sha256=payload["local_host_anchor_sha256"],
        local_boot_started_at_unix_microseconds=payload[
            "local_boot_started_at_unix_microseconds"
        ],
        local_boot_session_sha256=payload["local_boot_session_sha256"],
        boot_session_provider=boot_session_provider,
        boot_identity_provider=boot_identity_provider,
    ) != "same_boot_session":
        raise ValueError("durable action intent belongs to another boot session")
    _validate_global_remote_run_reservations(
        payload,
        project_root=root,
        receipt_directory=None,
        identity=identity,
    )
    _require_live_cohort_unsealed(project_root=root, identity=identity)
    terminal_logical = modal_action_terminal_receipt_path(
        identity,
        attempt_id,
    )
    _require_path_absent_secure(
        root.joinpath(*terminal_logical.parts),
        "action terminal receipt",
    )
    return payload


def _expected_source_concrete_run_ids(payload: Mapping[str, Any]) -> list[str]:
    action = payload["action"]
    run_id = validate_run_id(payload["run_id"])
    if action == "canaries":
        return [
            f"{run_id}-{_CANARY_RUN_SUFFIXES[harness]}" for harness in CANARY_ORDER
        ]
    return [run_id]


def _validate_source_action_intent(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    identity: ModalLiveCohortIdentity,
    project_root: Path,
) -> None:
    _validate_action_intent_contract(
        payload,
        attempt_id=attempt_id,
        identity=identity,
        project_root=project_root,
    )
    expected_fields = {field.name for field in dataclass_fields(ModalActionIntent)}
    if set(payload) != expected_fields:
        raise ValueError("source action intent has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalActionIntent"
        or payload["schema_version"] != "1.6"
        or payload["attempt_id"] != attempt_id
        or payload["action"] not in _SOURCE_PRODUCING_ACTIONS
    ):
        raise ValueError("source action intent has the wrong contract")
    if (
        payload["source_tree_sha256"] != identity.source_tree_sha256
        or payload["cohort_id"] != identity.cohort_id
        or payload["approved_image_source_sha256"]
        != identity.image_source_sha256
    ):
        raise ValueError("source action intent uses a stale image source")
    if _SHA256.fullmatch(payload["modal_command_sha256"]) is None:
        raise ValueError("source action intent command digest is invalid")
    if payload["modal_profile"] != MODAL_PROFILE:
        raise ValueError("source action intent used the wrong Modal profile")
    if payload["outer_cli_timeout_seconds"] != expected_outer_cli_timeout_seconds(
        payload["action"]
    ):
        raise ValueError("source action intent timeout differs from its action")
    if not modal_readiness.exact_json_equal(
        payload["modal_resource_profile"],
        modal_resource_profile(payload["action"], payload["harness"] or ""),
    ):
        raise ValueError("source action intent resource profile changed")
    _canonical_decimal_amount(
        payload["modal_cost_cap_usd"],
        "source_intent.modal_cost_cap_usd",
        require_positive=True,
    )
    if payload["modal_cost_approved"] is not True:
        raise ValueError("source action intent lacks Modal cost approval")
    modal_price_basis = _read_modal_price_basis(
        project_root=project_root,
        logical_path=payload["modal_price_basis_path"],
        expected_raw_sha256=payload["modal_price_basis_sha256"],
        image_source_sha256=identity.image_source_sha256,
        require_freshness=False,
    )
    expected_modal_estimate = modal_readiness.derive_modal_action_cost_estimate(
        action=payload["action"],
        harness=payload["harness"],
        resource_profile=payload["modal_resource_profile"],
        price_basis=modal_price_basis,
    )
    if not modal_readiness.exact_json_equal(
        payload["modal_cost_estimate"], expected_modal_estimate
    ):
        raise ValueError("source action intent Modal cost estimate changed")
    if _canonical_decimal_amount(
        payload["modal_cost_cap_usd"],
        "source_intent.modal_cost_cap_usd",
        require_positive=True,
    ) < _canonical_decimal_amount(
        expected_modal_estimate["action_estimate_usd"],
        "source_intent.modal_cost_estimate.action_estimate_usd",
        require_positive=True,
    ):
        raise ValueError("source action intent Modal cap is below its estimate")
    provider_action = payload["action"] in _PROVIDER_ACTIONS
    if payload["provider_cost_approved"] is not provider_action:
        raise ValueError("source action intent provider approval is invalid")
    provider_fields = (
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
    )
    if provider_action:
        _canonical_decimal_amount(
            payload["provider_cost_cap_usd"],
            "source_intent.provider_cost_cap_usd",
            require_positive=True,
        )
        safe_relative_path(payload["provider_approval_plan_path"])
        safe_relative_path(payload["provider_price_basis_path"])
        for field in ("approval_plan_sha256", "provider_price_basis_sha256"):
            if _SHA256.fullmatch(payload[field]) is None:
                raise ValueError(f"source action intent {field} is invalid")
    elif any(payload[field] is not None for field in provider_fields):
        raise ValueError("non-provider source intent contains provider fields")
    if payload["source_evidence_recovery"] is not False:
        raise ValueError("source-producing intent cannot be evidence recovery")
    _validate_receipt_binding_roster(
        payload["predecessor_receipts"],
        "source_intent.predecessor_receipts",
    )
    local_freeze_bindings = validate_local_freeze_evidence(
        project_root,
        expected_image_source_sha256=identity.image_source_sha256,
    )
    if not modal_readiness.exact_json_equal(
        tuple(payload["predecessor_receipts"][:3]), local_freeze_bindings
    ):
        raise ValueError("source action intent local freeze bindings changed")
    expected_predecessor_gates = {
        "cuda-environment": [],
        "offline-smoke": ["modal_cuda_environment_validated"],
        "candidate-smoke": [
            "modal_cuda_environment_validated",
            "modal_offline_smoke_validated",
        ],
        "checkpoint-resume": ["modal_artifact_round_trip_validated"],
        "canary": ["candidate_resume_preflight_validated"],
        "canaries": ["candidate_resume_preflight_validated"],
    }[payload["action"]]
    expected_predecessor_gates = [
        *LOCAL_ENGINEERING_FREEZE_GATES,
        *expected_predecessor_gates,
    ]
    if not modal_readiness.exact_json_equal(
        [item["gate"] for item in payload["predecessor_receipts"]],
        expected_predecessor_gates,
    ):
        raise ValueError("source action intent predecessor order is invalid")
    concrete = payload["concrete_remote_run_ids"]
    if concrete != _expected_source_concrete_run_ids(payload):
        raise ValueError("source action intent concrete run roster is invalid")
    action = payload["action"]
    harness = payload["harness"]
    if action == "canary":
        if harness not in CANARY_ORDER or payload["run_id"] != concrete[0]:
            raise ValueError("single-canary source intent is invalid")
    elif harness is not None:
        raise ValueError("non-single-canary source intent has a harness")
    if action == "checkpoint-resume":
        source_run_id = payload["source_run_id"]
        if not isinstance(source_run_id, str):
            raise ValueError("resume source intent lacks its source run")
        validate_run_id(source_run_id)
    elif payload["source_run_id"] is not None:
        raise ValueError("non-resume source intent has a source run")
    if payload["verifier_run_id"] is not None:
        raise ValueError("source-producing intent has a verifier run ID")
    expected_command = build_modal_cli_command(
        python_executable=sys.executable,
        project_root=project_root,
        action=payload["action"],
        run_id=payload["run_id"],
        source_run_id=payload["source_run_id"],
        verifier_run_id=None,
        harness=payload["harness"],
        source_tree_sha256=identity.source_tree_sha256,
        cohort_id=identity.cohort_id,
        image_source_sha256=identity.image_source_sha256,
        provider_approved=provider_action,
    )
    if payload["modal_command_sha256"] != _modal_command_sha256(expected_command):
        raise ValueError("source action intent command digest is not canonical")
    created_at = payload["created_at_utc"]
    if not isinstance(created_at, str):
        raise ValueError("source action intent timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("source action intent timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("source action intent timestamp lacks a timezone")


def _validate_source_action_terminal(
    payload: Mapping[str, Any],
    *,
    intent: Mapping[str, Any],
    attempt_id: str,
) -> None:
    expected_fields = {
        field.name for field in dataclass_fields(ModalActionAttemptReceipt)
    }
    if set(payload) != expected_fields:
        raise ValueError("source action terminal receipt has an invalid exact schema")
    if (
        payload["schema_name"] != "ModalActionAttemptReceipt"
        or payload["schema_version"] != "3.6"
        or payload["attempt_id"] != attempt_id
    ):
        raise ValueError("source action terminal receipt has the wrong contract")
    shared_fields = (
        "action",
        "run_id",
        "concrete_remote_run_ids",
        "remote_run_reservations",
        "local_host_anchor_path",
        "local_host_anchor_sha256",
        "local_boot_started_at_unix_microseconds",
        "local_boot_session_sha256",
        "source_run_id",
        "verifier_run_id",
        "harness",
        "source_tree_sha256",
        "cohort_id",
        "approved_image_source_sha256",
        "modal_command_sha256",
        "launch_capability_sha256",
        "modal_profile",
        "modal_environment",
        "outer_cli_timeout_seconds",
        "modal_cost_cap_usd",
        "modal_resource_profile",
        "modal_price_basis_path",
        "modal_price_basis_sha256",
        "modal_cost_estimate",
        "modal_cost_approved",
        "provider_cost_approved",
        "provider_cost_cap_usd",
        "provider_approval_plan_path",
        "approval_plan_sha256",
        "provider_price_basis_path",
        "provider_price_basis_sha256",
        "predecessor_receipts",
        "source_evidence_recovery",
    )
    if any(
        not modal_readiness.exact_json_equal(payload[field], intent[field])
        for field in shared_fields
    ):
        raise ValueError("source action intent and terminal receipt differ")
    if payload["started_at_utc"] != intent["created_at_utc"]:
        raise ValueError("source action intent and terminal start time differ")
    if (
        payload["modal_cli_process_started"] is not True
        or payload["remote_execution_state"] != "may_have_started"
    ):
        raise ValueError("source action did not start a possibly remote process")
    marker_path = payload["local_process_start_receipt_path"]
    marker_sha256 = payload["local_process_start_receipt_sha256"]
    if (
        marker_path
        != modal_local_process_start_receipt_path(attempt_id).as_posix()
        or not isinstance(marker_sha256, str)
        or _SHA256.fullmatch(marker_sha256) is None
    ):
        raise ValueError("source action lacks its local process-start receipt")
    process_id = _positive_process_identity(
        payload["local_process_id"],
        "source_terminal.local_process_id",
    )
    if (
        payload["local_process_group_id"] != process_id
        or payload["local_session_id"] != process_id
    ):
        raise ValueError("source action local process identity changed")
    status = payload["status"]
    if status not in {
        "succeeded",
        "failed",
        "timed_out",
        "interrupted",
        "cli_failed",
        "cleanup_failed",
    }:
        raise ValueError("source action terminal status cannot attribute a remote run")
    returncode = payload["returncode"]
    closed = payload["process_group_closed"]
    if status == "succeeded":
        if (
            payload["failure_kind"] is not None
            or returncode != 0
            or closed is not True
        ):
            raise ValueError("successful source terminal fields do not reconcile")
    elif status == "failed":
        if (
            payload["failure_kind"] != "modal_cli_exit"
            or not isinstance(returncode, int)
            or isinstance(returncode, bool)
            or returncode == 0
            or closed is not True
        ):
            raise ValueError("failed source terminal fields do not reconcile")
    elif status == "timed_out":
        if (
            payload["failure_kind"] != "outer_cli_timeout"
            or returncode is not None
            or closed is not True
        ):
            raise ValueError("timed-out source terminal fields do not reconcile")
    elif status == "interrupted":
        if (
            payload["failure_kind"] != "interrupt"
            or returncode is not None
            or closed is not True
        ):
            raise ValueError("interrupted source terminal fields do not reconcile")
    elif status == "cli_failed":
        if (
            payload["failure_kind"] not in {"process_launch", "modal_cli"}
            or returncode is not None
            or closed is not True
        ):
            raise ValueError("CLI-failed source terminal fields do not reconcile")
    elif status == "cleanup_failed":
        cleanup_kind = payload["failure_kind"]
        if cleanup_kind == "process_group_cleanup":
            reconciled = closed is False
        elif cleanup_kind == "python_execution_cleanup":
            reconciled = closed is None or closed is True
        elif cleanup_kind == "process_group_and_python_execution_cleanup":
            reconciled = closed is False
        else:
            reconciled = False
        if returncode is not None or not reconciled:
            raise ValueError(
                "cleanup-failed source terminal fields do not reconcile"
            )
    timestamps: list[datetime] = []
    for field in ("started_at_utc", "finished_at_utc"):
        value = payload[field]
        if not isinstance(value, str):
            raise ValueError(f"source terminal {field} is invalid")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"source terminal {field} is invalid") from error
        if parsed.tzinfo is None:
            raise ValueError(f"source terminal {field} lacks a timezone")
        timestamps.append(parsed)
    if timestamps[1] < timestamps[0]:
        raise ValueError("source action terminal finished before it started")


def _validate_aggregate_outcome_payload(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    run_id_prefix: str,
    identity: ModalLiveCohortIdentity,
    expected_all_succeeded: bool,
) -> None:
    validated = validate_provider_canary_aggregate_outcome_receipt(
        payload,
        expected_attempt_id=attempt_id,
        expected_run_id_prefix=run_id_prefix,
        expected_source_tree_sha256=identity.source_tree_sha256,
        expected_image_source_sha256=identity.image_source_sha256,
        expected_cohort_id=identity.cohort_id,
    )
    if validated["all_succeeded"] is not expected_all_succeeded:
        raise ValueError("provider aggregate outcome receipt is not terminal-bound")


def _validate_source_action_attribution(
    arguments: argparse.Namespace,
    *,
    project_root: Path,
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, str], ...]:
    terminal_logical = arguments.source_action_attempt_receipt_path
    terminal_relative = safe_relative_path(terminal_logical)
    if terminal_relative.parent != modal_action_attempt_directory(identity):
        raise ValueError("source terminal receipt is outside the canonical directory")
    match = re.fullmatch(r"([0-9a-f]{32})\.json", terminal_relative.name)
    if match is None:
        raise ValueError("source terminal receipt filename is invalid")
    attempt_id = match.group(1)
    if terminal_relative != modal_action_terminal_receipt_path(identity, attempt_id):
        raise ValueError("source terminal receipt path is not cohort-canonical")
    intent_logical = modal_action_intent_receipt_path(
        identity,
        attempt_id,
    ).as_posix()
    intent, _intent_raw, intent_sha256 = _read_project_json_file(
        project_root,
        intent_logical,
        "source_action_intent",
    )
    terminal, _terminal_raw, terminal_sha256 = _read_project_json_file(
        project_root,
        terminal_logical,
        "source_action_terminal",
    )
    if terminal_sha256 != arguments.source_action_attempt_receipt_sha256:
        raise ValueError(
            "source terminal receipt differs from its approved raw SHA-256"
        )
    _validate_source_action_intent(
        intent,
        attempt_id=attempt_id,
        identity=identity,
        project_root=project_root,
    )
    _validate_source_action_terminal(
        terminal,
        intent=intent,
        attempt_id=attempt_id,
    )
    marker_logical = terminal["local_process_start_receipt_path"]
    marker, _marker_raw, marker_sha256 = _read_project_json_file(
        project_root,
        marker_logical,
        "source_local_process_start",
    )
    marker = _validate_modal_local_process_start_receipt(
        marker,
        expected_attempt_id=attempt_id,
    )
    if marker_sha256 != terminal["local_process_start_receipt_sha256"]:
        raise ValueError("source local process-start receipt bytes changed")
    expected_marker_fields = {
        "intent_sha256": intent_sha256,
        "action": terminal["action"],
        "run_id": terminal["run_id"],
        "source_tree_sha256": terminal["source_tree_sha256"],
        "image_source_sha256": terminal["approved_image_source_sha256"],
        "cohort_id": terminal["cohort_id"],
        "modal_command_sha256": terminal["modal_command_sha256"],
        "launch_capability_sha256": terminal["launch_capability_sha256"],
        "modal_cost_cap_usd": terminal["modal_cost_cap_usd"],
        "provider_cost_cap_usd": terminal["provider_cost_cap_usd"],
        "local_host_anchor_path": terminal["local_host_anchor_path"],
        "local_host_anchor_sha256": terminal["local_host_anchor_sha256"],
        "local_boot_started_at_unix_microseconds": terminal[
            "local_boot_started_at_unix_microseconds"
        ],
        "local_boot_session_sha256": terminal["local_boot_session_sha256"],
        "process_id": terminal["local_process_id"],
        "expected_process_group_id": terminal["local_process_group_id"],
        "expected_session_id": terminal["local_session_id"],
    }
    if any(
        marker[field] != expected
        for field, expected in expected_marker_fields.items()
    ):
        raise ValueError("source local process-start receipt differs from terminal")
    source_run_id = arguments.run_id
    if source_run_id not in terminal["concrete_remote_run_ids"]:
        raise ValueError("download source is outside its producing action roster")
    if terminal["action"] != "canaries" and terminal["run_id"] != source_run_id:
        raise ValueError("download source differs from its producing action run ID")

    bindings = [
        {
            "gate": "source_action_intent",
            "path": intent_logical,
            "sha256": intent_sha256,
        },
        {
            "gate": "source_action_attempt_terminal",
            "path": terminal_logical,
            "sha256": terminal_sha256,
        },
        {
            "gate": "source_local_process_start",
            "path": marker_logical,
            "sha256": marker_sha256,
        },
    ]
    aggregate_complete = terminal["action"] == "canaries" and (
        (terminal["status"] == "succeeded" and terminal["returncode"] == 0)
        or (terminal["status"] == "failed" and terminal["returncode"] == 2)
    )
    if aggregate_complete:
        if arguments.source_evidence_recovery:
            raise ValueError(
                "complete aggregate outcome must not use evidence recovery"
            )
        aggregate_logical = provider_canary_aggregate_outcome_receipt_path(
            identity,
            attempt_id,
        ).as_posix()
        aggregate, _aggregate_raw, aggregate_sha256 = _read_project_json_file(
            project_root,
            aggregate_logical,
            "provider_canary_aggregate_outcomes",
        )
        _validate_aggregate_outcome_payload(
            aggregate,
            attempt_id=attempt_id,
            run_id_prefix=terminal["run_id"],
            identity=identity,
            expected_all_succeeded=terminal["status"] == "succeeded",
        )
        bindings.append(
            {
                "gate": "provider_canary_aggregate_outcomes",
                "path": aggregate_logical,
                "sha256": aggregate_sha256,
            }
        )
    elif terminal["status"] == "succeeded":
        if arguments.source_evidence_recovery:
            raise ValueError("successful source action must not use evidence recovery")
    elif not arguments.source_evidence_recovery:
        raise ValueError(
            "uncertain or failed source action requires explicit evidence recovery"
        )
    return tuple(bindings)


def _validate_predecessor_receipts(
    arguments: argparse.Namespace,
    *,
    project_root: Path,
    identity: ModalLiveCohortIdentity,
) -> tuple[dict[str, str], ...]:
    bindings = list(
        validate_local_freeze_evidence(
            project_root,
            expected_image_source_sha256=identity.image_source_sha256,
        )
    )

    if arguments.action in _VERIFIER_ACTIONS:
        bindings.extend(
            _validate_source_action_attribution(
                arguments,
                project_root=project_root,
                identity=identity,
            )
        )
        return tuple(bindings)

    def require_cohort_identity(
        payload: Mapping[str, Any],
        *,
        field: str,
    ) -> None:
        try:
            observed = ModalLiveCohortIdentity(
                source_tree_sha256=payload["source_tree_sha256"],
                image_source_sha256=payload["image_source_sha256"],
                cohort_id=payload["cohort_id"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{field} lacks a canonical cohort identity") from error
        if observed != identity:
            raise ValueError(f"{field} belongs to a different live cohort")

    def require_component_path(logical_path: str, gate: str) -> None:
        expected = modal_readiness.modal_component_receipt_path(
            identity,
            gate,
        ).as_posix()
        if logical_path != expected:
            raise ValueError(f"{gate} path is outside the current live cohort")

    def bind_cuda() -> None:
        require_component_path(
            arguments.cuda_receipt_path,
            "modal_cuda_environment_validated",
        )
        payload, binding = _validated_receipt_binding(
            gate="modal_cuda_environment_validated",
            logical_path=arguments.cuda_receipt_path,
            expected_sha256=arguments.cuda_receipt_sha256,
            validator=lambda: modal_readiness.validate_modal_readiness_receipt(
                "modal_cuda_environment_validated",
                project_root / arguments.cuda_receipt_path,
                root=project_root,
            ),
            project_root=project_root,
        )
        require_cohort_identity(payload, field="CUDA predecessor receipt")
        bindings.append(binding)

    if arguments.action in {"offline-smoke", "candidate-smoke"}:
        bind_cuda()
    if arguments.action == "candidate-smoke":
        require_component_path(
            arguments.offline_smoke_receipt_path,
            "modal_offline_smoke_validated",
        )
        offline_validator = getattr(
            modal_readiness,
            "validate_offline_smoke_validation_receipt",
            None,
        )
        if not callable(offline_validator):
            raise ValueError("downloaded-offline predecessor validator is unavailable")
        payload, binding = _validated_receipt_binding(
            gate="modal_offline_smoke_validated",
            logical_path=arguments.offline_smoke_receipt_path,
            expected_sha256=arguments.offline_smoke_receipt_sha256,
            validator=lambda: offline_validator(
                project_root / arguments.offline_smoke_receipt_path,
                root=project_root,
            ),
            project_root=project_root,
        )
        require_cohort_identity(payload, field="offline predecessor receipt")
        bindings.append(binding)
    elif arguments.action == "checkpoint-resume":
        require_component_path(
            arguments.artifact_round_trip_receipt_path,
            "modal_artifact_round_trip_validated",
        )
        payload, binding = _validated_receipt_binding(
            gate="modal_artifact_round_trip_validated",
            logical_path=arguments.artifact_round_trip_receipt_path,
            expected_sha256=arguments.artifact_round_trip_receipt_sha256,
            validator=lambda: modal_readiness.validate_modal_readiness_receipt(
                "modal_artifact_round_trip_validated",
                project_root / arguments.artifact_round_trip_receipt_path,
                root=project_root,
            ),
            project_root=project_root,
        )
        if payload.get("source_run_id") != arguments.source_run_id:
            raise ValueError(
                "resume source differs from the candidate round-trip receipt"
            )
        require_cohort_identity(payload, field="artifact round-trip receipt")
        downloaded = payload.get("downloaded_run_path")
        if not isinstance(downloaded, str):
            raise ValueError("candidate round-trip downloaded path is invalid")
        source_manifest, _raw, _digest = _read_project_json_file(
            project_root,
            f"{downloaded}/image_source_manifest.json",
            "candidate_image_source_manifest",
        )
        if canonical_sha256(source_manifest) != identity.image_source_sha256:
            raise ValueError("candidate round-trip receipt uses a stale image source")
        bindings.append(binding)
    elif arguments.action in _PROVIDER_ACTIONS:
        payload, binding = _validated_receipt_binding(
            gate="candidate_resume_preflight_validated",
            logical_path=arguments.candidate_resume_preflight_receipt_path,
            expected_sha256=(
                arguments.candidate_resume_preflight_receipt_sha256
            ),
            validator=lambda: (
                modal_readiness.validate_candidate_resume_preflight_receipt(
                    project_root
                    / arguments.candidate_resume_preflight_receipt_path,
                    root=project_root
                )
            ),
            project_root=project_root,
        )
        require_cohort_identity(payload, field="candidate-resume preflight")
        expected_preflight_path = (
            modal_readiness.modal_candidate_resume_preflight_receipt_path(
                identity,
                payload.get("binding_sha256"),
            ).as_posix()
        )
        if arguments.candidate_resume_preflight_receipt_path != (
            expected_preflight_path
        ):
            raise ValueError(
                "candidate-resume preflight path is outside the current live cohort"
            )
        bindings.append(binding)
    return tuple(bindings)


def _validate_approval_chain(
    arguments: argparse.Namespace,
    *,
    project_root: Path,
    identity: ModalLiveCohortIdentity,
) -> ValidatedApprovalChain:
    modal = _validate_modal_approval_inputs(
        arguments,
        project_root=project_root,
        image_source_sha256=identity.image_source_sha256,
    )
    provider: dict[str, str | None] = {
        "provider_cost_cap_usd": None,
        "provider_approval_plan_path": None,
        "approval_plan_sha256": None,
        "provider_price_basis_path": None,
        "provider_price_basis_sha256": None,
    }
    predecessor_receipts = _validate_predecessor_receipts(
        arguments,
        project_root=project_root,
        identity=identity,
    )
    if arguments.action in _PROVIDER_ACTIONS:
        cuda_run_id = _accepted_cuda_run_id_from_preflight(
            project_root,
            arguments.candidate_resume_preflight_receipt_path,
        )
        provider.update(
            _validate_provider_approval_inputs(
                arguments,
                project_root=project_root,
                image_source_sha256=identity.image_source_sha256,
                identity=identity,
                accepted_cuda_environment_run_id=cuda_run_id,
            )
        )
    return ValidatedApprovalChain(
        source_tree_sha256=identity.source_tree_sha256,
        cohort_id=identity.cohort_id,
        modal_cost_cap_usd=modal["modal_cost_cap_usd"],
        modal_resource_profile=modal["modal_resource_profile"],
        modal_price_basis_path=modal["modal_price_basis_path"],
        modal_price_basis_sha256=modal["modal_price_basis_sha256"],
        modal_cost_estimate=modal["modal_cost_estimate"],
        provider_cost_cap_usd=provider["provider_cost_cap_usd"],
        provider_approval_plan_path=provider["provider_approval_plan_path"],
        approval_plan_sha256=provider["approval_plan_sha256"],
        provider_price_basis_path=provider["provider_price_basis_path"],
        provider_price_basis_sha256=provider["provider_price_basis_sha256"],
        predecessor_receipts=predecessor_receipts,
    )


def _accepted_cuda_run_id_from_preflight(
    project_root: Path,
    logical_path: str,
) -> str:
    preflight, _raw, _sha256 = _read_project_json_file(
        project_root,
        logical_path,
        "candidate_resume_preflight_receipt",
    )
    cuda_execution = preflight.get("cuda_environment")
    if not isinstance(cuda_execution, dict):
        raise ValueError("candidate-resume preflight lacks its CUDA execution")
    cuda_run_id = cuda_execution.get("run_id")
    if not isinstance(cuda_run_id, str):
        raise ValueError("candidate-resume preflight CUDA run ID is invalid")
    return validate_run_id(cuda_run_id)


def validate_local_freeze_evidence(
    project_root: str | Path = ROOT,
    *,
    expected_image_source_sha256: str | None = None,
) -> tuple[dict[str, str], ...]:
    """Return exact current component and aggregate bindings before spend."""

    try:
        return local_engineering_freeze_predecessor_bindings(
            root=project_root,
            expected_image_source_sha256=expected_image_source_sha256,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "paid Modal launch requires a fresh current-source local engineering "
            "freeze with unit, offline-smoke, and aggregate receipts"
        ) from error


def validated_live_cohort_identity(
    arguments: argparse.Namespace,
    *,
    project_root: str | Path = ROOT,
    expected_image_source_sha256: str,
) -> ModalLiveCohortIdentity:
    """Derive live identity only from the fully revalidated local freeze."""

    try:
        payload = validate_local_engineering_freeze_receipt(
            root=project_root,
            expected_image_source_sha256=expected_image_source_sha256,
        )
    except (
        FileNotFoundError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "paid Modal launch requires a validated current-source local freeze"
        ) from error
    source_tree_sha256 = payload.get("source_tree_sha256")
    image_source_sha256 = payload.get("image_source_sha256")
    if image_source_sha256 != expected_image_source_sha256:
        raise ValueError("local freeze image source differs from the approved launch")
    return ModalLiveCohortIdentity(
        source_tree_sha256=source_tree_sha256,
        image_source_sha256=image_source_sha256,
        cohort_id=arguments.cohort_id,
    )


def _build_validated_launch(
    arguments: argparse.Namespace,
    *,
    project_root: str | Path = ROOT,
    environment: Mapping[str, str] | None = None,
    nonce_factory: Callable[[int], str] = secrets.token_hex,
    modal_executable_binding: _HeldLaunchFileBinding | None = None,
) -> tuple[list[str], dict[str, str], ValidatedApprovalChain]:
    """Build one exact Modal CLI invocation after all cost gates pass."""

    _validate_arguments(arguments)
    raw_root = Path(project_root)
    if raw_root.is_symlink():
        raise ValueError("project root may not be a symlink")
    root = raw_root.resolve()
    if not root.is_dir():
        raise ValueError("project root is missing")
    _validate_download_destination(arguments, project_root=root)
    validate_applied_patch_bundle(root)
    source = build_image_source_manifest(root)
    if arguments.expected_image_source_sha256 != source.manifest_sha256:
        raise ValueError(
            "approved image-source SHA-256 differs from the current local plan"
        )
    identity = validated_live_cohort_identity(
        arguments,
        project_root=root,
        expected_image_source_sha256=source.manifest_sha256,
    )
    approval_chain = _validate_approval_chain(
        arguments,
        project_root=root,
        identity=identity,
    )
    del environment  # Paid CLI state is never inherited from the ambient shell.
    owned_binding: _HeldLaunchFileBinding | None = None
    if modal_executable_binding is None:
        owned_binding = _open_modal_executable_binding()
        selected_binding = owned_binding
    else:
        selected_binding = modal_executable_binding
    try:
        modal_executable = Path(sys.executable).with_name("modal")
        if selected_binding.canonical_path != modal_executable:
            raise ValueError("pinned Modal executable path differs from this Python")
        command = list(
            build_modal_cli_command(
                python_executable=sys.executable,
                project_root=root,
                action=arguments.action,
                run_id=arguments.run_id,
                source_run_id=arguments.source_run_id or None,
                verifier_run_id=arguments.verifier_run_id or None,
                harness=arguments.harness or None,
                source_tree_sha256=identity.source_tree_sha256,
                cohort_id=identity.cohort_id,
                image_source_sha256=arguments.expected_image_source_sha256,
                provider_approved=arguments.provider_approved,
            )
        )
        if Path(command[0]) != selected_binding.canonical_path:
            raise AssertionError("canonical Modal executable path drifted")
    finally:
        if owned_binding is not None:
            owned_binding.close()

    child_environment = dict(_PAID_MODAL_BASE_ENVIRONMENT)
    nonce = nonce_factory(32)
    if _NONCE.fullmatch(nonce) is None:
        raise ValueError("launch nonce factory returned an invalid capability")
    child_environment[MODAL_LAUNCH_NONCE_ENV] = nonce
    child_environment[MODAL_LAUNCH_SOURCE_ENV] = source.manifest_sha256
    child_environment[MODAL_LAUNCH_SOURCE_TREE_ENV] = identity.source_tree_sha256
    child_environment[MODAL_LAUNCH_COHORT_ENV] = identity.cohort_id
    child_environment[MODAL_PROFILE_ENV] = MODAL_PROFILE
    child_environment[MODAL_ENVIRONMENT_ENV] = MODAL_ENVIRONMENT
    child_environment["PYTHONPATH"] = os.pathsep.join(
        (str(root), str(_canonical_venv_site_packages()))
    )
    return command, child_environment, approval_chain


def build_launch(
    arguments: argparse.Namespace,
    *,
    project_root: str | Path = ROOT,
    environment: Mapping[str, str] | None = None,
    nonce_factory: Callable[[int], str] = secrets.token_hex,
) -> tuple[list[str], dict[str, str]]:
    """Build one exact Modal CLI invocation after all cost gates pass."""

    command, child_environment, _approval_chain = _build_validated_launch(
        arguments,
        project_root=project_root,
        environment=environment,
        nonce_factory=nonce_factory,
    )
    return command, child_environment


def _utc_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("attempt receipt timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _sanitized_run_id(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        validate_run_id(value)
    except ValueError:
        return None
    return value


def _sanitized_sha256(value: object) -> str | None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        return None
    return value


def _sanitized_timeout(value: object) -> int | None:
    if type(value) is not int or value <= 0:
        return None
    return value


def _sanitized_decimal_cap(value: object) -> str | None:
    try:
        _canonical_decimal_amount(value, "cost_cap", require_positive=True)
    except ValueError:
        return None
    return value if isinstance(value, str) else None


def _sanitized_relative_path(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        safe_relative_path(value)
    except ValueError:
        return None
    return value


def _modal_command_sha256(command: Sequence[str]) -> str:
    encoded = json.dumps(
        list(command),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _descriptor_bound_modal_execution_command(
    canonical_command: Sequence[str],
    modal_binding: _HeldLaunchFileBinding,
    python_binding: _HeldLaunchFileBinding,
) -> list[str]:
    """Transform receipt argv into the descriptor-backed console-script argv."""

    if (
        not canonical_command
        or Path(canonical_command[0]) != modal_binding.canonical_path
    ):
        raise ValueError("canonical Modal command differs from its held executable")
    return [
        os.fspath(python_binding.canonical_path),
        modal_binding.execution_path,
        *canonical_command[1:],
    ]


def _write_attempt_payload(
    payload: Mapping[str, Any],
    *,
    attempt_id: str,
    filename: str,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity | None = None,
) -> Path:
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("attempt receipt ID is invalid")
    if filename not in {f"{attempt_id}.json", f"{attempt_id}.intent.json"}:
        raise ValueError("attempt journal filename is invalid")
    if receipt_directory is not None:
        supplied = Path(receipt_directory)
        directory = supplied if supplied.is_absolute() else project_root / supplied
        destination = Path(os.path.abspath(directory)) / filename
    elif filename.endswith(".intent.json"):
        if identity is None:
            raise ValueError("validated action intent requires a cohort identity")
        logical = modal_action_intent_receipt_path(identity, attempt_id)
        destination = project_root.joinpath(*logical.parts)
    elif identity is not None:
        logical = modal_action_terminal_receipt_path(identity, attempt_id)
        destination = project_root.joinpath(*logical.parts)
    else:
        logical = modal_launch_rejection_receipt_path(attempt_id)
        destination = project_root.joinpath(*logical.parts)
    create_json_exclusive(destination, dict(payload))
    return destination


def _write_attempt_receipt(
    receipt: ModalActionAttemptReceipt,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity | None = None,
) -> Path:
    if identity is None:
        if (
            receipt.modal_cli_process_started
            or receipt.remote_execution_state != "definitely_not_started"
            or receipt.returncode is not None
            or receipt.process_group_closed is not None
        ):
            raise ValueError(
                "global launch rejection may not claim a started Modal process"
            )
    elif (
        receipt.source_tree_sha256 != identity.source_tree_sha256
        or receipt.approved_image_source_sha256
        != identity.image_source_sha256
        or receipt.cohort_id != identity.cohort_id
    ):
        raise ValueError("terminal receipt differs from its cohort identity")
    return _write_attempt_payload(
        asdict(receipt),
        attempt_id=receipt.attempt_id,
        filename=f"{receipt.attempt_id}.json",
        project_root=project_root,
        receipt_directory=receipt_directory,
        identity=identity,
    )


def _write_action_intent(
    intent: ModalActionIntent,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity,
) -> Path:
    if (
        intent.source_tree_sha256 != identity.source_tree_sha256
        or intent.approved_image_source_sha256
        != identity.image_source_sha256
        or intent.cohort_id != identity.cohort_id
    ):
        raise ValueError("action intent differs from its cohort identity")
    return _write_attempt_payload(
        asdict(intent),
        attempt_id=intent.attempt_id,
        filename=f"{intent.attempt_id}.intent.json",
        project_root=project_root,
        receipt_directory=receipt_directory,
        identity=identity,
    )


def _positive_process_identity(value: object, field: str) -> int:
    if type(value) is not int or value <= 1:
        raise ValueError(f"{field} must be an exact positive process identifier")
    return value


def _validated_process_birth_identity(value: object) -> bytes:
    if not isinstance(value, bytes) or not 8 <= len(value) <= 256:
        raise ValueError("process birth identity must be bounded raw bytes")
    if not any(value):
        raise ValueError("process birth identity may not be all-zero bytes")
    return value


def _darwin_process_birth_identity(process_id: int) -> bytes:
    """Read a process start instant through libproc without a subprocess."""

    selected_pid = _positive_process_identity(process_id, "process_id")
    libc = ctypes.CDLL(None, use_errno=True)
    proc_pidinfo = libc.proc_pidinfo
    proc_pidinfo.argtypes = (
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    )
    proc_pidinfo.restype = ctypes.c_int
    info = _DarwinProcBSDInfo()
    observed = proc_pidinfo(
        selected_pid,
        3,  # PROC_PIDTBSDINFO
        0,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if observed <= 0:
        error_number = ctypes.get_errno()
        if error_number in {0, errno.ESRCH}:
            raise ProcessLookupError(
                errno.ESRCH,
                "Darwin process birth identity is unavailable",
            )
        raise OSError(error_number, "Darwin process birth identity could not be read")
    if observed != ctypes.sizeof(info) or info.pbi_pid != selected_pid:
        raise ValueError("Darwin process birth identity changed while it was read")
    if info.pbi_start_tvusec >= 1_000_000 or info.pbi_start_tvsec == 0:
        raise ValueError("Darwin process birth identity is invalid")
    return (
        selected_pid.to_bytes(8, "big")
        + info.pbi_start_tvsec.to_bytes(8, "big")
        + info.pbi_start_tvusec.to_bytes(8, "big")
    )


def _linux_process_birth_identity(process_id: int) -> bytes:
    """Read Linux procfs start ticks for one exact PID without following links."""

    selected_pid = _positive_process_identity(process_id, "process_id")
    try:
        descriptor = os.open(
            f"/proc/{selected_pid}/stat",
            os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
        )
    except FileNotFoundError:
        raise ProcessLookupError(
            errno.ESRCH,
            "Linux process birth identity is unavailable",
        ) from None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) & 0o022
        ):
            raise ValueError("Linux process birth source metadata is unsafe")
        raw = os.read(descriptor, 8193)
        if os.read(descriptor, 1):
            raise ValueError("Linux process birth source is unexpectedly large")
    finally:
        os.close(descriptor)
    prefix = f"{selected_pid} (".encode("ascii")
    closing = raw.rfind(b") ")
    if not raw.startswith(prefix) or closing < len(prefix):
        raise ValueError("Linux process birth source is invalid")
    fields = raw[closing + 2 :].split()
    if len(fields) < 20 or re.fullmatch(rb"[0-9]+", fields[19]) is None:
        raise ValueError("Linux process birth start ticks are invalid")
    start_ticks = int(fields[19])
    if start_ticks <= 0:
        raise ValueError("Linux process birth start ticks are invalid")
    return selected_pid.to_bytes(8, "big") + start_ticks.to_bytes(16, "big")


def _default_process_birth_identity_provider(process_id: int) -> bytes:
    if sys.platform == "darwin":
        return _darwin_process_birth_identity(process_id)
    if sys.platform.startswith("linux"):
        return _linux_process_birth_identity(process_id)
    raise ValueError("platform has no approved process birth-identity provider")


def _process_birth_identity_sha256(
    *,
    local_boot_session_sha256: str,
    process_id: int,
    process_birth_identity: object,
) -> str:
    if (
        not isinstance(local_boot_session_sha256, str)
        or _SHA256.fullmatch(local_boot_session_sha256) is None
    ):
        raise ValueError("process birth binding has an invalid boot-session digest")
    selected_pid = _positive_process_identity(process_id, "process_id")
    identity = _validated_process_birth_identity(process_birth_identity)
    return hashlib.sha256(
        b"RL4RL ModalLocalProcessBirthIdentity v1\0"
        + bytes.fromhex(local_boot_session_sha256)
        + selected_pid.to_bytes(8, "big")
        + identity
    ).hexdigest()


def _terminate_untrusted_started_process(process: Any) -> None:
    """Kill/reap a direct child whose group identity cannot be trusted."""

    process.kill()
    process.wait(timeout=1.0)


def _validate_modal_local_process_start_receipt(
    payload: Mapping[str, Any],
    *,
    expected_attempt_id: str | None = None,
) -> dict[str, Any]:
    expected_fields = {
        field.name for field in dataclass_fields(ModalLocalProcessStartReceipt)
    }
    if set(payload) != expected_fields:
        raise ValueError("local process-start receipt has an invalid exact schema")
    attempt_id = payload["attempt_id"]
    if (
        payload["schema_name"] != "ModalLocalProcessStart"
        or payload["schema_version"] != "1.1"
        or not isinstance(attempt_id, str)
        or _ATTEMPT_ID.fullmatch(attempt_id) is None
        or (expected_attempt_id is not None and attempt_id != expected_attempt_id)
    ):
        raise ValueError("local process-start receipt has the wrong contract")
    created_at = payload["created_at_utc"]
    if not isinstance(created_at, str):
        raise ValueError("local process-start timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("local process-start timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise ValueError("local process-start timestamp lacks a timezone")
    action = payload["action"]
    if action not in _ACTIONS:
        raise ValueError("local process-start action is invalid")
    validate_run_id(payload["run_id"])
    identity = ModalLiveCohortIdentity(
        source_tree_sha256=payload["source_tree_sha256"],
        image_source_sha256=payload["image_source_sha256"],
        cohort_id=payload["cohort_id"],
    )
    expected_intent_path = modal_action_intent_receipt_path(
        identity,
        attempt_id,
    ).as_posix()
    if payload["intent_path"] != expected_intent_path:
        raise ValueError("local process-start intent path is not canonical")
    for field in (
        "intent_sha256",
        "modal_command_sha256",
        "launch_capability_sha256",
    ):
        if _SHA256.fullmatch(payload[field]) is None:
            raise ValueError(f"local process-start {field} is invalid")
    _canonical_decimal_amount(
        payload["modal_cost_cap_usd"],
        "local_process_start.modal_cost_cap_usd",
        require_positive=True,
    )
    provider_cap = payload["provider_cost_cap_usd"]
    if provider_cap is not None:
        _canonical_decimal_amount(
            provider_cap,
            "local_process_start.provider_cost_cap_usd",
            require_positive=True,
        )
    if (action in _PROVIDER_ACTIONS) is not (provider_cap is not None):
        raise ValueError("local process-start provider cap differs from its action")
    _validate_local_containment_fields(payload)
    process_id = _positive_process_identity(payload["process_id"], "process_id")
    process_group_id = _positive_process_identity(
        payload["expected_process_group_id"],
        "expected_process_group_id",
    )
    session_id = _positive_process_identity(
        payload["expected_session_id"],
        "expected_session_id",
    )
    if process_id != process_group_id or process_id != session_id:
        raise ValueError(
            "local process-start identity is not an isolated session leader"
        )
    birth_sha256 = payload["process_birth_identity_sha256"]
    if not isinstance(birth_sha256, str) or _SHA256.fullmatch(birth_sha256) is None:
        raise ValueError("local process-start birth-identity SHA-256 is invalid")
    return dict(payload)


def _publish_modal_local_process_start(
    receipt: ModalLocalProcessStartReceipt,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
) -> _HeldModalLocalProcessStart:
    _validate_modal_local_process_start_receipt(
        asdict(receipt),
        expected_attempt_id=receipt.attempt_id,
    )
    _prepare_local_containment_directory(
        project_root=project_root,
        receipt_directory=receipt_directory,
        include_process_starts=True,
    )
    logical = modal_local_process_start_receipt_path(
        receipt.attempt_id
    ).as_posix()
    destination = _local_containment_destination(
        logical,
        project_root=project_root,
        receipt_directory=receipt_directory,
    )
    try:
        create_json_exclusive(destination, asdict(receipt))
    except BaseException as error:
        message = (
            "local process-start receipt attempt ID is already used"
            if isinstance(error, FileExistsError)
            else "local process-start receipt could not be created"
        )
        raise ModalProcessStartReceiptError(message) from error
    try:
        binding = _open_held_launch_file(
            destination,
            label="local process-start receipt",
            maximum_bytes=_MAX_LOCAL_PROCESS_START_BYTES,
            hash_content=True,
            require_owner_executable=False,
            required_mode=0o600,
            # macOS may add com.apple.provenance after publication. The held
            # marker remains inode-, path-, metadata-, and byte-hash-bound.
            require_stable_ctime=False,
        )
    except BaseException as error:
        raise ModalProcessStartReceiptError(
            "local process-start receipt could not be held"
        ) from error
    try:
        payload, _raw, observed_sha256 = _read_private_json_path(
            destination,
            "local_process_start_receipt",
        )
        validated = _validate_modal_local_process_start_receipt(
            payload,
            expected_attempt_id=receipt.attempt_id,
        )
        if (
            validated != asdict(receipt)
            or binding.sha256 is None
            or binding.sha256 != observed_sha256
        ):
            raise ValueError("published local process-start receipt changed")
        held = _HeldModalLocalProcessStart(
            logical_path=logical,
            receipt=receipt,
            binding=binding,
        )
        held.require_current()
        return held
    except BaseException as error:
        binding.close()
        if isinstance(error, ModalProcessStartReceiptError):
            raise
        raise ModalProcessStartReceiptError(
            "local process-start receipt failed revalidation"
        ) from error


def probe_same_boot_modal_process_group(
    project_root: str | Path,
    *,
    process_start_receipt_path: str,
    process_start_receipt_sha256: str,
    machine_identity_provider: Callable[[], bytes] | None = None,
    boot_session_provider: Callable[[], int] | None = None,
    boot_identity_provider: Callable[[], bytes] | None = None,
    process_birth_identity_provider: Callable[[int], bytes] | None = None,
    process_group_lookup: Callable[[int], int] | None = None,
    session_lookup: Callable[[int], int] | None = None,
    signal_zero: Callable[[int, int], None] | None = None,
) -> str:
    """Observe a verified same-boot process group using signal zero only."""

    if (
        not isinstance(process_start_receipt_sha256, str)
        or _SHA256.fullmatch(process_start_receipt_sha256) is None
    ):
        raise ValueError("process-start receipt SHA-256 is invalid")
    relative = safe_relative_path(process_start_receipt_path)
    match = re.fullmatch(r"([0-9a-f]{32})\.json", relative.name)
    if match is None or relative != modal_local_process_start_receipt_path(
        match.group(1)
    ):
        raise ValueError("process-start receipt path is not canonical")
    root = Path(os.path.abspath(os.fspath(project_root)))
    payload, _raw, observed_sha256 = _read_project_json_file(
        root,
        process_start_receipt_path,
        "local_process_start_receipt",
    )
    if observed_sha256 != process_start_receipt_sha256:
        raise ValueError("process-start receipt differs from recorded bytes")
    marker = _validate_modal_local_process_start_receipt(
        payload,
        expected_attempt_id=match.group(1),
    )
    validate_current_modal_local_host_anchor(
        root,
        expected_path=marker["local_host_anchor_path"],
        expected_sha256=marker["local_host_anchor_sha256"],
        machine_identity_provider=machine_identity_provider,
    )
    relation = modal_local_boot_session_relation(
        local_host_anchor_sha256=marker["local_host_anchor_sha256"],
        local_boot_started_at_unix_microseconds=marker[
            "local_boot_started_at_unix_microseconds"
        ],
        local_boot_session_sha256=marker["local_boot_session_sha256"],
        boot_session_provider=boot_session_provider,
        boot_identity_provider=boot_identity_provider,
    )
    if relation == "different_boot_session":
        return relation
    get_process_group = (
        os.getpgid if process_group_lookup is None else process_group_lookup
    )
    get_session = os.getsid if session_lookup is None else session_lookup
    probe = os.killpg if signal_zero is None else signal_zero
    process_id = marker["process_id"]
    expected_group = marker["expected_process_group_id"]
    expected_session = marker["expected_session_id"]
    leader_absent = False
    try:
        observed_group = get_process_group(process_id)
        observed_session = get_session(process_id)
    except ProcessLookupError:
        # A session leader can exit while other members leave its process
        # group alive.  The recorded group therefore still needs a
        # non-killing existence probe before absence can be claimed.
        leader_absent = True
    if not leader_absent and (
        type(observed_group) is not int
        or type(observed_session) is not int
        or observed_group != expected_group
        or observed_session != expected_session
    ):
        # A reused PID must never authorize even a signal-zero observation of
        # the old group as though it belonged to the replacement process.
        return "same_boot_process_identity_changed"
    if not leader_absent:
        birth_provider = (
            _default_process_birth_identity_provider
            if process_birth_identity_provider is None
            else process_birth_identity_provider
        )
        try:
            observed_birth_sha256 = _process_birth_identity_sha256(
                local_boot_session_sha256=marker["local_boot_session_sha256"],
                process_id=process_id,
                process_birth_identity=birth_provider(process_id),
            )
        except ProcessLookupError:
            leader_absent = True
        else:
            if observed_birth_sha256 != marker["process_birth_identity_sha256"]:
                return "same_boot_process_identity_changed"
    try:
        probe(expected_group, 0)
    except ProcessLookupError:
        return "same_boot_process_group_absent"
    except PermissionError:
        return "same_boot_process_group_exists"
    return "same_boot_process_group_exists"


def _exclusive_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _remote_run_reservation_specs(
    *,
    concrete_remote_run_ids: Sequence[str],
    attempt_id: str,
    action: str,
    identity: ModalLiveCohortIdentity,
    created_at_utc: str,
    launch_capability_sha256: str,
    local_host_anchor_path: str,
    local_host_anchor_sha256: str,
    local_boot_started_at_unix_microseconds: int,
    local_boot_session_sha256: str,
) -> tuple[tuple[dict[str, str], dict[str, Any]], ...]:
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError("remote run reservation attempt ID is invalid")
    if action not in _ACTIONS:
        raise ValueError("remote run reservation action is invalid")
    if _SHA256.fullmatch(launch_capability_sha256) is None:
        raise ValueError("remote run reservation capability digest is invalid")
    if not isinstance(created_at_utc, str) or not created_at_utc:
        raise ValueError("remote run reservation timestamp is invalid")
    containment_fields = {
        "local_host_anchor_path": local_host_anchor_path,
        "local_host_anchor_sha256": local_host_anchor_sha256,
        "local_boot_started_at_unix_microseconds": (
            local_boot_started_at_unix_microseconds
        ),
        "local_boot_session_sha256": local_boot_session_sha256,
    }
    _validate_local_containment_fields(containment_fields)
    specs: list[tuple[dict[str, str], dict[str, Any]]] = []
    for remote_run_id in concrete_remote_run_ids:
        selected_run_id = validate_run_id(remote_run_id)
        logical = modal_remote_run_reservation_path(selected_run_id).as_posix()
        payload = {
            "schema_name": "ModalRemoteRunReservation",
            "schema_version": "1.2",
            "remote_run_id": selected_run_id,
            "owner_attempt_id": attempt_id,
            "action": action,
            "source_tree_sha256": identity.source_tree_sha256,
            "image_source_sha256": identity.image_source_sha256,
            "cohort_id": identity.cohort_id,
            "modal_environment": MODAL_ENVIRONMENT,
            "created_at_utc": created_at_utc,
            "launch_capability_sha256": launch_capability_sha256,
            **containment_fields,
        }
        binding = {
            "run_id": selected_run_id,
            "path": logical,
            "sha256": hashlib.sha256(_exclusive_json_bytes(payload)).hexdigest(),
        }
        specs.append((binding, payload))
    if len(specs) != len({item[0]["run_id"] for item in specs}):
        raise ValueError("remote run reservation roster contains duplicate IDs")
    return tuple(specs)


def _remote_run_reservation_destination(
    logical_path: str,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
) -> Path:
    logical = safe_relative_path(logical_path)
    if receipt_directory is None:
        return project_root.joinpath(*logical.parts)
    supplied = Path(receipt_directory)
    directory = supplied if supplied.is_absolute() else project_root / supplied
    return (
        Path(os.path.abspath(directory))
        / "remote_run_reservations"
        / logical.name
    )


def _publish_global_remote_run_reservations(
    intent: ModalActionIntent,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity,
    require_current: Callable[[], None] | None = None,
) -> None:
    specs = _remote_run_reservation_specs(
        concrete_remote_run_ids=intent.concrete_remote_run_ids,
        attempt_id=intent.attempt_id,
        action=intent.action,
        identity=identity,
        created_at_utc=intent.created_at_utc,
        launch_capability_sha256=intent.launch_capability_sha256,
        local_host_anchor_path=intent.local_host_anchor_path,
        local_host_anchor_sha256=intent.local_host_anchor_sha256,
        local_boot_started_at_unix_microseconds=(
            intent.local_boot_started_at_unix_microseconds
        ),
        local_boot_session_sha256=intent.local_boot_session_sha256,
    )
    expected_bindings = tuple(binding for binding, _payload in specs)
    if intent.remote_run_reservations != expected_bindings:
        raise ValueError("action intent remote run reservation roster is not canonical")
    for binding, payload in specs:
        if require_current is not None:
            require_current()
        if binding["run_id"] in _LEGACY_RESERVED_REMOTE_RUN_IDS:
            raise ValueError(
                f"remote run ID {binding['run_id']} is a preserved legacy ID"
            )
        destination = _remote_run_reservation_destination(
            binding["path"],
            project_root=project_root,
            receipt_directory=receipt_directory,
        )
        try:
            create_json_exclusive(destination, payload)
        except FileExistsError as error:
            raise ValueError(
                f"remote run ID {binding['run_id']} is already globally reserved"
            ) from error
        observed, _raw, observed_sha256 = _read_private_json_path(
            destination,
            f"remote_run_reservation[{binding['run_id']}]",
        )
        if observed != payload or observed_sha256 != binding["sha256"]:
            raise ValueError("published remote run reservation changed unexpectedly")
        if require_current is not None:
            require_current()


def _validate_global_remote_run_reservations(
    payload: Mapping[str, Any],
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity,
) -> None:
    specs = _remote_run_reservation_specs(
        concrete_remote_run_ids=payload["concrete_remote_run_ids"],
        attempt_id=payload["attempt_id"],
        action=payload["action"],
        identity=identity,
        created_at_utc=payload["created_at_utc"],
        launch_capability_sha256=payload["launch_capability_sha256"],
        local_host_anchor_path=payload["local_host_anchor_path"],
        local_host_anchor_sha256=payload["local_host_anchor_sha256"],
        local_boot_started_at_unix_microseconds=payload[
            "local_boot_started_at_unix_microseconds"
        ],
        local_boot_session_sha256=payload["local_boot_session_sha256"],
    )
    expected_bindings = [binding for binding, _reservation in specs]
    observed_bindings = payload["remote_run_reservations"]
    if not isinstance(observed_bindings, (list, tuple)) or (
        list(observed_bindings) != expected_bindings
    ):
        raise ValueError("action intent remote run reservation roster changed")
    for binding, expected_reservation in specs:
        destination = _remote_run_reservation_destination(
            binding["path"],
            project_root=project_root,
            receipt_directory=receipt_directory,
        )
        observed, _raw, observed_sha256 = _read_private_json_path(
            destination,
            f"remote_run_reservation[{binding['run_id']}]",
        )
        if (
            observed != expected_reservation
            or observed_sha256 != binding["sha256"]
        ):
            raise ValueError("remote run reservation owner or raw bytes changed")


def _action_attempt_directory_path(
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity,
) -> Path:
    if receipt_directory is not None:
        supplied = Path(receipt_directory)
        directory = supplied if supplied.is_absolute() else project_root / supplied
        return Path(os.path.abspath(directory))
    logical = modal_action_attempt_directory(identity)
    return project_root.joinpath(*logical.parts)


def _scan_validated_cohort_intents(
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity,
) -> dict[str, tuple[dict[str, Any], str]]:
    """Return all stable exact-schema intents in one locked cohort journal."""

    directory = _action_attempt_directory_path(
        project_root=project_root,
        receipt_directory=receipt_directory,
        identity=identity,
    )
    try:
        descriptor = _open_nofollow_local_directory(directory)
    except FileNotFoundError:
        return {}
    try:
        directory_identity = os.fstat(descriptor)
        before_names = tuple(sorted(os.listdir(descriptor)))
        records: dict[str, tuple[dict[str, Any], str]] = {}
        for filename in before_names:
            match = _ACTION_INTENT_FILENAME.fullmatch(filename)
            if match is None:
                if filename.endswith(".intent.json"):
                    raise ValueError("cohort journal contains an invalid intent name")
                continue
            attempt_id = match.group(1)
            payload, _raw, raw_sha256 = _read_json_leaf_from_directory(
                descriptor,
                filename,
                f"cohort_action_intent[{attempt_id}]",
            )
            _validate_action_intent_contract(
                payload,
                attempt_id=attempt_id,
                identity=identity,
                project_root=project_root,
            )
            records[attempt_id] = (payload, raw_sha256)
        after_names = tuple(sorted(os.listdir(descriptor)))
        if before_names != after_names:
            raise ValueError("cohort action journal changed while it was scanned")
        reopened = _open_nofollow_local_directory(directory)
        try:
            observed = os.fstat(reopened)
            if (directory_identity.st_dev, directory_identity.st_ino) != (
                observed.st_dev,
                observed.st_ino,
            ):
                raise ValueError("cohort action journal changed while it was scanned")
        finally:
            os.close(reopened)
        return records
    finally:
        os.close(descriptor)


def _assert_fresh_remote_run_reservation(
    intent: ModalActionIntent,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity,
    require_current_intent: bool,
) -> str | None:
    """Reject destination reuse and verify the new intent is its reservation."""

    records = _scan_validated_cohort_intents(
        project_root=project_root,
        receipt_directory=receipt_directory,
        identity=identity,
    )
    owners: dict[str, str] = {}
    for existing_attempt_id, (payload, _raw_sha256) in records.items():
        for remote_run_id in payload["concrete_remote_run_ids"]:
            previous = owners.setdefault(remote_run_id, existing_attempt_id)
            if previous != existing_attempt_id:
                raise ValueError(
                    "cohort journal reuses concrete remote run ID "
                    f"{remote_run_id}"
                )
    current_record = records.get(intent.attempt_id)
    if require_current_intent:
        if current_record is None:
            raise ValueError("new action intent is absent from its cohort journal")
        current_payload, current_sha256 = current_record
        if canonical_sha256(current_payload) != canonical_sha256(asdict(intent)):
            raise ValueError("new action intent changed after durable reservation")
        for remote_run_id in intent.concrete_remote_run_ids:
            if owners.get(remote_run_id) != intent.attempt_id:
                raise ValueError("new action intent did not reserve its remote run ID")
        return current_sha256
    if current_record is not None:
        raise ValueError("action attempt ID is already present in the cohort journal")
    reused = sorted(set(intent.concrete_remote_run_ids) & set(owners))
    if reused:
        raise ValueError(
            "fresh live action cannot reuse concrete remote run ID(s): "
            + ", ".join(reused)
        )
    return None


def _inspect_current_action_intent_ownership(
    intent: ModalActionIntent,
    *,
    project_root: Path,
    receipt_directory: str | Path | None,
    identity: ModalLiveCohortIdentity,
) -> str | None:
    """Return the exact durable intent digest, or ``None`` for proven absence.

    This is deliberately suitable for the exception path of
    :func:`_write_action_intent`.  A create-only publisher can raise after the
    complete bytes reached stable storage.  Only the exact canonical bytes,
    their complete reservation roster, and an unsealed cohort establish
    journal ownership.  Any malformed, partial, or unstable file raises rather
    than being mistaken for absence.
    """

    destination = _action_attempt_directory_path(
        project_root=project_root,
        receipt_directory=receipt_directory,
        identity=identity,
    ) / f"{intent.attempt_id}.intent.json"
    try:
        payload, raw, raw_sha256 = _read_private_json_path(
            destination,
            "current_action_intent",
        )
    except FileNotFoundError:
        _require_path_absent_secure(
            destination,
            "current action intent",
            missing_parent_ok=True,
        )
        return None

    _validate_action_intent_contract(
        payload,
        attempt_id=intent.attempt_id,
        identity=identity,
        project_root=project_root,
    )
    expected_raw = _exclusive_json_bytes(asdict(intent))
    if raw != expected_raw or raw_sha256 != hashlib.sha256(expected_raw).hexdigest():
        raise ValueError("current action intent differs from its canonical bytes")
    journal_sha256 = _assert_fresh_remote_run_reservation(
        intent,
        project_root=project_root,
        receipt_directory=receipt_directory,
        identity=identity,
        require_current_intent=True,
    )
    if journal_sha256 != raw_sha256:
        raise ValueError("current action intent digest changed during ownership proof")
    _validate_global_remote_run_reservations(
        payload,
        project_root=project_root,
        receipt_directory=receipt_directory,
        identity=identity,
    )
    _require_live_cohort_unsealed(
        project_root=project_root,
        identity=identity,
    )
    return raw_sha256


def _receipt_fields(arguments: argparse.Namespace) -> dict[str, Any]:
    raw_action = getattr(arguments, "action", None)
    action = raw_action if raw_action in _ACTIONS else None
    raw_harness = getattr(arguments, "harness", None)
    harness = raw_harness if raw_harness in CANARY_ORDER else None
    run_id = _sanitized_run_id(getattr(arguments, "run_id", None))
    verifier_run_id = _sanitized_run_id(
        getattr(arguments, "verifier_run_id", None)
    )
    if action in _VERIFIER_ACTIONS and verifier_run_id is not None:
        concrete_remote_run_ids = (verifier_run_id,)
    elif action == "canaries" and run_id is not None:
        concrete_remote_run_ids = tuple(
            f"{run_id}-{_CANARY_RUN_SUFFIXES[harness_id]}"
            for harness_id in CANARY_ORDER
        )
    elif action is not None and run_id is not None:
        concrete_remote_run_ids = (run_id,)
    else:
        concrete_remote_run_ids = ()
    provider_action = action in _PROVIDER_ACTIONS
    try:
        resource_profile = (
            modal_resource_profile(action, harness or "")
            if action is not None
            else None
        )
    except (KeyError, ValueError):
        resource_profile = None
    return {
        "action": action,
        "run_id": run_id,
        "concrete_remote_run_ids": concrete_remote_run_ids,
        "remote_run_reservations": (),
        "local_host_anchor_path": None,
        "local_host_anchor_sha256": None,
        "local_boot_started_at_unix_microseconds": None,
        "local_boot_session_sha256": None,
        "source_run_id": _sanitized_run_id(getattr(arguments, "source_run_id", None)),
        "verifier_run_id": verifier_run_id,
        "harness": harness,
        "source_tree_sha256": None,
        "cohort_id": _sanitized_run_id(getattr(arguments, "cohort_id", None)),
        "approved_image_source_sha256": _sanitized_sha256(
            getattr(arguments, "expected_image_source_sha256", None)
        ),
        "launch_capability_sha256": None,
        "modal_environment": MODAL_ENVIRONMENT,
        "outer_cli_timeout_seconds": _sanitized_timeout(
            getattr(arguments, "outer_cli_timeout_seconds", None)
        ),
        "modal_cost_cap_usd": _sanitized_decimal_cap(
            getattr(arguments, "modal_cost_cap_usd", None)
        ),
        "modal_resource_profile": resource_profile,
        "modal_price_basis_path": _sanitized_relative_path(
            getattr(arguments, "modal_price_basis_path", None)
        ),
        "modal_price_basis_sha256": _sanitized_sha256(
            getattr(arguments, "modal_price_basis_sha256", None)
        ),
        "modal_cost_estimate": None,
        "modal_cost_approved": getattr(arguments, "approved", None) is True,
        "provider_cost_approved": (
            getattr(arguments, "provider_approved", None) is True
        ),
        "provider_cost_cap_usd": (
            _sanitized_decimal_cap(
                getattr(arguments, "provider_cost_cap_usd", None)
            )
            if provider_action
            else None
        ),
        "provider_approval_plan_path": (
            _sanitized_relative_path(
                getattr(arguments, "provider_approval_plan_path", None)
            )
            if provider_action
            else None
        ),
        "approval_plan_sha256": (
            _sanitized_sha256(getattr(arguments, "approval_plan_sha256", None))
            if provider_action
            else None
        ),
        "provider_price_basis_path": (
            _sanitized_relative_path(
                getattr(arguments, "provider_price_basis_path", None)
            )
            if provider_action
            else None
        ),
        "provider_price_basis_sha256": (
            _sanitized_sha256(
                getattr(arguments, "provider_price_basis_sha256", None)
            )
            if provider_action
            else None
        ),
        "predecessor_receipts": (),
        "source_evidence_recovery": (
            action in _VERIFIER_ACTIONS
            and getattr(arguments, "source_evidence_recovery", None) is True
        ),
    }


def run(
    arguments: argparse.Namespace,
    *,
    runner: Callable[..., Any] = subprocess.Popen,
    project_root: str | Path = ROOT,
    receipt_directory: str | Path | None = None,
    attempt_id_factory: Callable[[int], str] = secrets.token_hex,
    host_anchor_id_factory: Callable[[int], str] = secrets.token_hex,
    machine_identity_provider: Callable[[], bytes] | None = None,
    boot_session_provider: Callable[[], int] | None = None,
    boot_identity_provider: Callable[[], bytes] | None = None,
    process_birth_identity_provider: Callable[[int], bytes] | None = None,
    now_factory: Callable[[], datetime] = _now_utc,
    process_group_capture: Callable[[Any], int] = capture_isolated_process_group,
    process_group_terminator: Callable[..., None] = terminate_process_group,
) -> int:
    raw_root = Path(project_root)
    receipt_root = Path(os.path.abspath(raw_root))
    started_at = now_factory()
    requested_attempt_id = getattr(arguments, "attempt_id", "")
    if requested_attempt_id and not isinstance(requested_attempt_id, str):
        raise ValueError("reviewed attempt ID must be text")
    attempt_id = requested_attempt_id or attempt_id_factory(16)
    if _ATTEMPT_ID.fullmatch(attempt_id) is None:
        raise ValueError(
            "attempt ID must contain exactly 32 lowercase hexadecimal digits"
        )
    fields = _receipt_fields(arguments)

    status = "preflight_failed"
    failure_kind: str | None = None
    pending_error: BaseException | None = None
    command: list[str] | None = None
    returncode: int | None = None
    process: Any | None = None
    local_process_id: int | None = None
    process_group_id: int | None = None
    local_session_id: int | None = None
    process_group_closed: bool | None = None
    process_start_failed = False
    popen_attempted = False
    launcher_lock: int | None = None
    launch_bindings: _ModalLaunchBindings | None = None
    local_containment: _ModalLocalContainmentBinding | None = None
    process_start_binding: _HeldModalLocalProcessStart | None = None
    python_execution: _PrivatePythonExecutionCopy | None = None
    live_identity: ModalLiveCohortIdentity | None = None
    cohort_journal_identity: ModalLiveCohortIdentity | None = None
    intent: ModalActionIntent | None = None
    intent_sha256: str | None = None
    intent_persistence_failure_kind = "action_intent_persistence"

    selected_machine_identity_provider = (
        _default_machine_identity_provider
        if machine_identity_provider is None
        else machine_identity_provider
    )
    selected_boot_session_provider = (
        _default_boot_session_provider
        if boot_session_provider is None
        else boot_session_provider
    )
    selected_boot_identity_provider = (
        _default_boot_identity_provider
        if boot_identity_provider is None
        else boot_identity_provider
    )
    selected_process_birth_identity_provider = (
        _default_process_birth_identity_provider
        if process_birth_identity_provider is None
        else process_birth_identity_provider
    )

    def require_current_prestart_bindings() -> None:
        if launcher_lock is None or local_containment is None:
            raise ValueError("local containment bindings are incomplete")
        _assert_launcher_lock_identity(launcher_lock)
        local_containment.require_current()
        if launch_bindings is not None:
            launch_bindings.require_current()

    # Lock contention and a blocked or malformed global journal are pure
    # pre-ownership failures.  They must not create another journal record:
    # doing so without the lock would race the owner, while doing so after a
    # blocked scan would mutate the very evidence that requires recovery.
    launcher_lock = _acquire_launcher_lock(receipt_root)
    try:
        global_journal = _scan_modal_global_action_journal(
            lock_descriptor=launcher_lock,
            process_probe=lambda locked_root, marker_path, marker_sha256: (
                probe_same_boot_modal_process_group(
                    locked_root,
                    process_start_receipt_path=marker_path,
                    process_start_receipt_sha256=marker_sha256,
                    machine_identity_provider=selected_machine_identity_provider,
                    boot_session_provider=selected_boot_session_provider,
                    boot_identity_provider=selected_boot_identity_provider,
                    process_birth_identity_provider=(
                        selected_process_birth_identity_provider
                    ),
                )
            ),
        )
        _require_modal_global_action_gate_clear(
            global_journal,
            candidate_attempt_id=attempt_id,
        )
    except BaseException:
        _release_launcher_lock(launcher_lock)
        launcher_lock = None
        raise

    try:
        local_containment = _open_or_create_local_containment_binding(
            project_root=receipt_root,
            receipt_directory=receipt_directory,
            host_anchor_id_factory=host_anchor_id_factory,
            machine_identity_provider=selected_machine_identity_provider,
            boot_session_provider=selected_boot_session_provider,
            boot_identity_provider=selected_boot_identity_provider,
        )
        fields.update(
            {
                "local_host_anchor_path": local_containment.host_anchor_path,
                "local_host_anchor_sha256": local_containment.host_anchor_sha256,
                "local_boot_started_at_unix_microseconds": (
                    local_containment.boot_started_at_unix_microseconds
                ),
                "local_boot_session_sha256": (
                    local_containment.boot_session_sha256
                ),
            }
        )
        require_current_prestart_bindings()
        launch_bindings = _open_modal_launch_bindings()
        require_current_prestart_bindings()
        command, environment, approval_chain = _build_validated_launch(
            arguments,
            project_root=project_root,
            modal_executable_binding=launch_bindings.modal_executable,
        )
        environment["MODAL_CONFIG_PATH"] = launch_bindings.modal_config.execution_path
        environment[MODAL_ACTION_ATTEMPT_ID_ENV] = attempt_id
        fields.update(asdict(approval_chain))
        fields["launch_capability_sha256"] = _launch_capability_sha256(
            environment[MODAL_LAUNCH_NONCE_ENV]
        )
        live_identity = ModalLiveCohortIdentity(
            source_tree_sha256=approval_chain.source_tree_sha256,
            image_source_sha256=fields["approved_image_source_sha256"],
            cohort_id=approval_chain.cohort_id,
        )
        _require_live_cohort_unsealed(
            project_root=receipt_root,
            identity=live_identity,
        )
        root = raw_root.resolve()
        if (
            fields["action"] is None
            or fields["run_id"] is None
            or fields["source_tree_sha256"] is None
            or fields["cohort_id"] is None
            or fields["approved_image_source_sha256"] is None
            or fields["outer_cli_timeout_seconds"] is None
            or fields["modal_cost_cap_usd"] is None
            or fields["modal_resource_profile"] is None
            or fields["modal_price_basis_path"] is None
            or fields["modal_price_basis_sha256"] is None
            or fields["modal_cost_estimate"] is None
        ):
            raise AssertionError("validated paid action lacks durable intent fields")
        reservation_specs = _remote_run_reservation_specs(
            concrete_remote_run_ids=fields["concrete_remote_run_ids"],
            attempt_id=attempt_id,
            action=fields["action"],
            identity=live_identity,
            created_at_utc=_utc_timestamp(started_at),
            launch_capability_sha256=fields["launch_capability_sha256"],
            local_host_anchor_path=fields["local_host_anchor_path"],
            local_host_anchor_sha256=fields["local_host_anchor_sha256"],
            local_boot_started_at_unix_microseconds=fields[
                "local_boot_started_at_unix_microseconds"
            ],
            local_boot_session_sha256=fields["local_boot_session_sha256"],
        )
        fields["remote_run_reservations"] = tuple(
            binding for binding, _payload in reservation_specs
        )
        intent = ModalActionIntent(
            schema_name="ModalActionIntent",
            schema_version="1.6",
            attempt_id=attempt_id,
            created_at_utc=_utc_timestamp(started_at),
            action=fields["action"],
            run_id=fields["run_id"],
            concrete_remote_run_ids=fields["concrete_remote_run_ids"],
            remote_run_reservations=fields["remote_run_reservations"],
            local_host_anchor_path=fields["local_host_anchor_path"],
            local_host_anchor_sha256=fields["local_host_anchor_sha256"],
            local_boot_started_at_unix_microseconds=fields[
                "local_boot_started_at_unix_microseconds"
            ],
            local_boot_session_sha256=fields["local_boot_session_sha256"],
            source_run_id=fields["source_run_id"],
            verifier_run_id=fields["verifier_run_id"],
            harness=fields["harness"],
            source_tree_sha256=fields["source_tree_sha256"],
            cohort_id=fields["cohort_id"],
            approved_image_source_sha256=(
                fields["approved_image_source_sha256"]
            ),
            modal_command_sha256=_modal_command_sha256(command),
            launch_capability_sha256=fields["launch_capability_sha256"],
            modal_profile=MODAL_PROFILE,
            modal_environment=fields["modal_environment"],
            outer_cli_timeout_seconds=fields["outer_cli_timeout_seconds"],
            modal_cost_cap_usd=fields["modal_cost_cap_usd"],
            modal_resource_profile=fields["modal_resource_profile"],
            modal_price_basis_path=fields["modal_price_basis_path"],
            modal_price_basis_sha256=fields["modal_price_basis_sha256"],
            modal_cost_estimate=fields["modal_cost_estimate"],
            modal_cost_approved=fields["modal_cost_approved"],
            provider_cost_approved=fields["provider_cost_approved"],
            provider_cost_cap_usd=fields["provider_cost_cap_usd"],
            provider_approval_plan_path=fields["provider_approval_plan_path"],
            approval_plan_sha256=fields["approval_plan_sha256"],
            provider_price_basis_path=fields["provider_price_basis_path"],
            provider_price_basis_sha256=fields["provider_price_basis_sha256"],
            predecessor_receipts=fields["predecessor_receipts"],
            source_evidence_recovery=fields["source_evidence_recovery"],
        )
        _assert_fresh_remote_run_reservation(
            intent,
            project_root=receipt_root,
            receipt_directory=receipt_directory,
            identity=live_identity,
            require_current_intent=False,
        )
        require_current_prestart_bindings()
        _publish_global_remote_run_reservations(
            intent,
            project_root=receipt_root,
            receipt_directory=receipt_directory,
            identity=live_identity,
            require_current=require_current_prestart_bindings,
        )
        require_current_prestart_bindings()
        try:
            intent_path = _write_action_intent(
                intent,
                project_root=receipt_root,
                receipt_directory=receipt_directory,
                identity=live_identity,
            )
        except BaseException as error:
            try:
                recovered_intent_sha256 = (
                    _inspect_current_action_intent_ownership(
                        intent,
                        project_root=receipt_root,
                        receipt_directory=receipt_directory,
                        identity=live_identity,
                    )
                )
            except BaseException as inspection_error:
                intent_persistence_failure_kind = (
                    "action_intent_persistence_uncertain"
                )
                raise ModalAttemptReceiptError(
                    "immutable Modal action intent persistence is uncertain"
                ) from inspection_error
            if recovered_intent_sha256 is not None:
                intent_sha256 = recovered_intent_sha256
                cohort_journal_identity = live_identity
                intent_persistence_failure_kind = (
                    "action_intent_post_persistence"
                )
            raise ModalAttemptReceiptError(
                "immutable Modal action intent could not be created"
            ) from error
        intent_sha256 = _inspect_current_action_intent_ownership(
            intent,
            project_root=receipt_root,
            receipt_directory=receipt_directory,
            identity=live_identity,
        )
        if intent_sha256 is None:  # pragma: no cover - exhaustive contract guard
            raise AssertionError("durable Modal action intent lacks a raw digest")
        cohort_journal_identity = live_identity
        environment[MODAL_ACTION_INTENT_SHA256_ENV] = intent_sha256
        require_current_prestart_bindings()
        revalidated_chain = _validate_approval_chain(
            arguments,
            project_root=root,
            identity=live_identity,
        )
        if not modal_readiness.exact_json_equal(
            asdict(revalidated_chain), asdict(approval_chain)
        ):
            raise ValueError("approval chain changed after action intent persistence")
        final_intent_sha256 = _assert_fresh_remote_run_reservation(
            intent,
            project_root=receipt_root,
            receipt_directory=receipt_directory,
            identity=live_identity,
            require_current_intent=True,
        )
        if final_intent_sha256 != intent_sha256:
            raise ValueError("durable action intent changed before Modal process start")
        _require_live_cohort_unsealed(
            project_root=receipt_root,
            identity=live_identity,
        )
        _validate_global_remote_run_reservations(
            asdict(intent),
            project_root=receipt_root,
            receipt_directory=receipt_directory,
            identity=live_identity,
        )
        if receipt_directory is None:
            validate_local_action_intent_for_entrypoint(
                project_root=receipt_root,
                identity=live_identity,
                attempt_id=attempt_id,
                expected_intent_sha256=intent_sha256,
                launch_nonce=environment[MODAL_LAUNCH_NONCE_ENV],
                action=fields["action"],
                run_id=fields["run_id"],
                source_run_id=fields["source_run_id"],
                verifier_run_id=fields["verifier_run_id"],
                harness=fields["harness"],
                machine_identity_provider=selected_machine_identity_provider,
                boot_session_provider=selected_boot_session_provider,
                boot_identity_provider=selected_boot_identity_provider,
            )
        require_current_prestart_bindings()
        print(f"Modal action intent: {intent_path}", file=sys.stderr)
    except ModalAttemptReceiptError as error:
        status = "preflight_failed"
        failure_kind = intent_persistence_failure_kind
        pending_error = error
    except KeyboardInterrupt as error:
        status = "interrupted"
        failure_kind = "interrupt"
        pending_error = error
    except (FileNotFoundError, ValueError) as error:
        status = "preflight_rejected"
        failure_kind = "preflight"
        pending_error = error
    except BaseException as error:
        status = "preflight_failed"
        failure_kind = "preflight"
        pending_error = error
    else:
        try:
            require_current_prestart_bindings()
            python_execution = _materialize_python_execution_copy(
                launch_bindings.python_executable,
                project_root=receipt_root,
                attempt_id=attempt_id,
            )
            require_current_prestart_bindings()
            python_execution.require_current()
            execution_command = _descriptor_bound_modal_execution_command(
                command,
                launch_bindings.modal_executable,
                python_execution.binding,
            )
            popen_attempted = True
            process = runner(
                execution_command,
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                executable=os.fspath(python_execution.canonical_path),
                pass_fds=launch_bindings.pass_fds,
            )
            try:
                local_process_id = _positive_process_identity(
                    process.pid,
                    "Popen.pid",
                )
                # Popen completed ``start_new_session=True`` before returning,
                # so the child is the expected PGID and SID leader. Persist
                # these expected identities before normal observation/wait.
                process_group_id = local_process_id
                local_session_id = local_process_id
                process_group_closed = False
                if intent_sha256 is None or live_identity is None:
                    raise AssertionError(
                        "started process lacks its durable intent identity"
                    )
                process_birth_sha256 = _process_birth_identity_sha256(
                    local_boot_session_sha256=fields[
                        "local_boot_session_sha256"
                    ],
                    process_id=local_process_id,
                    process_birth_identity=(
                        selected_process_birth_identity_provider(
                            local_process_id
                        )
                    ),
                )
                process_start_receipt = ModalLocalProcessStartReceipt(
                    schema_name="ModalLocalProcessStart",
                    schema_version="1.1",
                    attempt_id=attempt_id,
                    created_at_utc=_utc_timestamp(now_factory()),
                    action=fields["action"],
                    run_id=fields["run_id"],
                    intent_path=modal_action_intent_receipt_path(
                        live_identity,
                        attempt_id,
                    ).as_posix(),
                    intent_sha256=intent_sha256,
                    source_tree_sha256=fields["source_tree_sha256"],
                    image_source_sha256=fields["approved_image_source_sha256"],
                    cohort_id=fields["cohort_id"],
                    modal_command_sha256=_modal_command_sha256(command),
                    launch_capability_sha256=fields[
                        "launch_capability_sha256"
                    ],
                    modal_cost_cap_usd=fields["modal_cost_cap_usd"],
                    provider_cost_cap_usd=fields["provider_cost_cap_usd"],
                    local_host_anchor_path=fields["local_host_anchor_path"],
                    local_host_anchor_sha256=fields[
                        "local_host_anchor_sha256"
                    ],
                    local_boot_started_at_unix_microseconds=fields[
                        "local_boot_started_at_unix_microseconds"
                    ],
                    local_boot_session_sha256=fields[
                        "local_boot_session_sha256"
                    ],
                    process_id=local_process_id,
                    expected_process_group_id=process_group_id,
                    expected_session_id=local_session_id,
                    process_birth_identity_sha256=process_birth_sha256,
                )
                process_start_binding = _publish_modal_local_process_start(
                    process_start_receipt,
                    project_root=receipt_root,
                    receipt_directory=receipt_directory,
                )
                process_start_binding.require_current()
                require_current_prestart_bindings()
                python_execution.require_current()
            except BaseException as error:
                process_start_failed = True
                if isinstance(error, ModalProcessStartReceiptError):
                    raise
                raise ModalProcessStartReceiptError(
                    "started process could not be bound to its local marker"
                ) from error
            observed_process_group_id = process_group_capture(process)
            if (
                type(observed_process_group_id) is not int
                or observed_process_group_id != process_group_id
            ):
                raise ValueError("captured process-group identity changed")
            try:
                returncode = int(
                    process.wait(
                        timeout=arguments.outer_cli_timeout_seconds,
                    )
                )
            except subprocess.TimeoutExpired:
                status = "timed_out"
                failure_kind = "outer_cli_timeout"
                pending_error = ModalCLITimeoutError(
                    "Modal CLI exceeded its approved outer deadline of "
                    f"{arguments.outer_cli_timeout_seconds} seconds"
                )
            except KeyboardInterrupt as error:
                status = "interrupted"
                failure_kind = "interrupt"
                pending_error = error
            except BaseException as error:
                status = "cli_failed"
                failure_kind = "modal_cli"
                pending_error = error
            else:
                if returncode == 0:
                    status = "succeeded"
                else:
                    status = "failed"
                    failure_kind = "modal_cli_exit"
        except ModalProcessStartReceiptError as error:
            process_start_failed = True
            status = "cli_failed"
            failure_kind = "process_start_receipt_persistence"
            returncode = None
            pending_error = error
        except KeyboardInterrupt as error:
            status = "interrupted"
            failure_kind = "interrupt"
            pending_error = error
        except (FileNotFoundError, ValueError) as error:
            if not popen_attempted:
                status = "preflight_rejected"
                failure_kind = "preflight"
            elif process is None:
                status = "cli_failed"
                failure_kind = "process_launch"
            else:
                status = "cli_failed"
                failure_kind = "modal_cli"
            pending_error = error
        except BaseException as error:
            if not popen_attempted:
                status = "preflight_failed"
                failure_kind = "preflight"
            elif process is None:
                status = "cli_failed"
                failure_kind = "process_launch"
            else:
                status = "cli_failed"
                failure_kind = "process_launch"
            pending_error = error
        finally:
            if process is not None and process_group_id is not None:
                try:
                    process_group_terminator(
                        process,
                        process_group_id=process_group_id,
                    )
                except BaseException as error:
                    process_group_closed = False
                    status = "cleanup_failed"
                    failure_kind = (
                        "process_start_receipt_and_process_group_cleanup"
                        if process_start_failed
                        else "process_group_cleanup"
                    )
                    returncode = None
                    pending_error = error
                else:
                    process_group_closed = True
            elif process is not None:
                try:
                    _terminate_untrusted_started_process(process)
                except BaseException as error:
                    process_group_closed = False
                    status = "cleanup_failed"
                    failure_kind = (
                        "process_start_receipt_and_process_group_cleanup"
                    )
                    returncode = None
                    pending_error = error
                else:
                    process_group_closed = False
                    status = "cleanup_failed"
                    failure_kind = (
                        "process_start_receipt_and_process_group_cleanup"
                    )
                    returncode = None
            if (
                process is not None
                and launch_bindings is not None
                and python_execution is not None
            ):
                try:
                    require_current_prestart_bindings()
                    python_execution.require_current()
                    if process_start_binding is not None:
                        process_start_binding.require_current()
                except BaseException as error:
                    if process_start_binding is not None:
                        process_start_failed = True
                    if status != "cleanup_failed":
                        status = "cli_failed"
                        failure_kind = (
                            "process_start_receipt_persistence"
                            if process_start_failed
                            else "modal_cli"
                        )
                        returncode = None
                        pending_error = error

    if python_execution is not None:
        pre_cleanup_error: BaseException | None = None
        try:
            require_current_prestart_bindings()
            python_execution.require_current()
            if process_start_binding is not None:
                process_start_binding.require_current()
        except BaseException as error:
            pre_cleanup_error = error
            if process_start_binding is not None:
                try:
                    process_start_binding.require_current()
                except BaseException:
                    process_start_failed = True
        cleanup_error: BaseException | None = None
        try:
            python_execution.close_and_remove()
        except BaseException as error:
            cleanup_error = error
        finally:
            python_execution = None
        if cleanup_error is not None:
            if process_start_failed and (
                process is not None and process_group_closed is not True
            ):
                failure_kind = (
                    "process_start_receipt_process_group_and_"
                    "python_execution_cleanup"
                )
            elif process_start_failed:
                failure_kind = "process_start_receipt_and_python_execution_cleanup"
            elif failure_kind == "process_group_cleanup" or (
                process is not None and process_group_closed is not True
            ):
                failure_kind = "process_group_and_python_execution_cleanup"
            else:
                failure_kind = "python_execution_cleanup"
            status = "cleanup_failed"
            returncode = None
            pending_error = cleanup_error
        elif pre_cleanup_error is not None and status != "cleanup_failed":
            status = "cli_failed"
            failure_kind = (
                "process_start_receipt_persistence"
                if process_start_failed
                else "modal_cli"
            )
            returncode = None
            pending_error = pre_cleanup_error

    terminal_binding_error: BaseException | None = None
    if launcher_lock is not None:
        try:
            _assert_launcher_lock_identity(launcher_lock)
            if local_containment is not None:
                local_containment.require_current()
            if launch_bindings is not None:
                launch_bindings.require_current()
        except BaseException as error:
            terminal_binding_error = error
    if process_start_binding is not None:
        try:
            process_start_binding.require_current()
        except BaseException as error:
            process_start_failed = True
            terminal_binding_error = error
    if terminal_binding_error is not None and status != "cleanup_failed":
        if process is None:
            status = "preflight_rejected"
            failure_kind = "preflight"
        else:
            status = "cli_failed"
            failure_kind = (
                "process_start_receipt_persistence"
                if process_start_failed
                else "modal_cli"
            )
        returncode = None
        pending_error = terminal_binding_error

    process_start_logical = (
        modal_local_process_start_receipt_path(attempt_id).as_posix()
        if process is not None
        else None
    )
    process_start_sha256 = (
        process_start_binding.sha256
        if process_start_binding is not None
        else None
    )

    receipt = ModalActionAttemptReceipt(
        schema_name="ModalActionAttemptReceipt",
        schema_version="3.6",
        attempt_id=attempt_id,
        started_at_utc=_utc_timestamp(started_at),
        finished_at_utc=_utc_timestamp(now_factory()),
        status=status,
        failure_kind=failure_kind,
        action=fields["action"],
        run_id=fields["run_id"],
        concrete_remote_run_ids=fields["concrete_remote_run_ids"],
        remote_run_reservations=fields["remote_run_reservations"],
        local_host_anchor_path=fields["local_host_anchor_path"],
        local_host_anchor_sha256=fields["local_host_anchor_sha256"],
        local_boot_started_at_unix_microseconds=fields[
            "local_boot_started_at_unix_microseconds"
        ],
        local_boot_session_sha256=fields["local_boot_session_sha256"],
        source_run_id=fields["source_run_id"],
        verifier_run_id=fields["verifier_run_id"],
        harness=fields["harness"],
        source_tree_sha256=fields["source_tree_sha256"],
        cohort_id=fields["cohort_id"],
        approved_image_source_sha256=fields["approved_image_source_sha256"],
        modal_command_sha256=(
            _modal_command_sha256(command) if command is not None else None
        ),
        launch_capability_sha256=fields["launch_capability_sha256"],
        modal_profile=MODAL_PROFILE,
        modal_environment=fields["modal_environment"],
        outer_cli_timeout_seconds=fields["outer_cli_timeout_seconds"],
        modal_cost_cap_usd=fields["modal_cost_cap_usd"],
        modal_resource_profile=fields["modal_resource_profile"],
        modal_price_basis_path=fields["modal_price_basis_path"],
        modal_price_basis_sha256=fields["modal_price_basis_sha256"],
        modal_cost_estimate=fields["modal_cost_estimate"],
        modal_cost_approved=fields["modal_cost_approved"],
        provider_cost_approved=fields["provider_cost_approved"],
        provider_cost_cap_usd=fields["provider_cost_cap_usd"],
        provider_approval_plan_path=fields["provider_approval_plan_path"],
        approval_plan_sha256=fields["approval_plan_sha256"],
        provider_price_basis_path=fields["provider_price_basis_path"],
        provider_price_basis_sha256=fields["provider_price_basis_sha256"],
        predecessor_receipts=fields["predecessor_receipts"],
        source_evidence_recovery=fields["source_evidence_recovery"],
        local_process_start_receipt_path=process_start_logical,
        local_process_start_receipt_sha256=process_start_sha256,
        local_process_id=local_process_id,
        local_process_group_id=process_group_id,
        local_session_id=local_session_id,
        modal_cli_process_started=process is not None,
        remote_execution_state=(
            "may_have_started"
            if process is not None
            else "definitely_not_started"
        ),
        returncode=returncode,
        process_group_closed=process_group_closed,
    )
    try:
        try:
            receipt_path = _write_attempt_receipt(
                receipt,
                project_root=receipt_root,
                receipt_directory=receipt_directory,
                identity=cohort_journal_identity,
            )
            if launcher_lock is not None:
                _assert_launcher_lock_identity(launcher_lock)
            if local_containment is not None:
                local_containment.require_current()
            if launch_bindings is not None:
                launch_bindings.require_current()
            if process_start_binding is not None:
                process_start_binding.require_current()
        except BaseException as error:
            raise ModalAttemptReceiptError(
                "immutable Modal action attempt receipt could not be created"
            ) from error
    finally:
        if process_start_binding is not None:
            process_start_binding.close()
        if launch_bindings is not None:
            launch_bindings.close()
        if local_containment is not None:
            local_containment.close()
        if launcher_lock is not None:
            _release_launcher_lock(launcher_lock)
    print(f"Modal action attempt receipt: {receipt_path}", file=sys.stderr)

    if pending_error is not None:
        raise pending_error
    if returncode is None:  # pragma: no cover - exhaustive state guard
        raise RuntimeError("Modal CLI completed without a return code")
    return returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except ModalLaunchLockContentionError as error:
        print(f"launch_modal.py: {error}", file=sys.stderr)
        return 75
    except ModalActionJournalBlockedError as error:
        print(f"launch_modal.py: {error}", file=sys.stderr)
        return 75
    except ModalActionJournalIntegrityError as error:
        print(f"launch_modal.py: {error}", file=sys.stderr)
        return 75
    except ModalCLITimeoutError as error:
        print(f"launch_modal.py: {error}", file=sys.stderr)
        return 124
    except ProcessGroupClosureError:
        print(
            "launch_modal.py: Modal CLI process-group cleanup failed",
            file=sys.stderr,
        )
        return 125
    except (ModalAttemptReceiptError, ModalProcessStartReceiptError) as error:
        print(f"launch_modal.py: {error}", file=sys.stderr)
        return 125
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 2  # pragma: no cover - argparse.error raises SystemExit


if __name__ == "__main__":
    raise SystemExit(main())
