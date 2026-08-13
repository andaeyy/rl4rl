#!/usr/bin/env python3
# ruff: noqa: E402
"""Create immutable, source-versioned local engineering freeze evidence.

The receipts created here are local self-reports. They establish complete
Phase-2 cost-free validation, local unit-test, and provider-free smoke evidence
against one source/lock/image identity. They do not attest scientific custody,
authorize a scientific launch, or substitute for live accelerator evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import selectors
import shutil
import stat
import subprocess
import sys
import sysconfig
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.process_control import (
    capture_isolated_process_group,
    terminate_process_group,
)
from modal_boundary import (
    MODAL_VERSION,
    build_image_source_manifest,
    safe_relative_path,
)
from study.serialization import content_hash, create_json_exclusive

LOCAL_ENGINEERING_FREEZE_ROOT = Path(
    "outputs/readiness/modal_only_final/local_engineering_freezes"
)
LOCAL_ENGINEERING_FREEZE_RECEIPT_FILENAME = "local_engineering_freeze_receipt.json"
LOCAL_ENGINEERING_FREEZE_RECEIPT_CONTRACT = {
    "schema_name": "LocalEngineeringFreezeReceipt",
    "schema_version": "1.0",
}
LOCAL_PHASE2_VALIDATION_RECEIPT_FILENAME = (
    "phase2_local_validation_evidence_receipt.json"
)
LOCAL_PHASE2_VALIDATION_RECEIPT_CONTRACT = {
    "schema_name": "Phase2LocalValidationEvidenceReceipt",
    "schema_version": "1.0",
}
LOCAL_ENGINEERING_FREEZE_GATES = (
    "local_unit_tested",
    "local_offline_smoke_tested",
    "local_engineering_freeze_validated",
)
EXECUTION_SOURCE_MANIFEST_FILENAME = "execution_source_manifest.json"
VALIDATION_INPUT_MANIFEST_FILENAME = "validation_input_manifest.json"
EXECUTION_ENVIRONMENT_MANIFEST_FILENAME = "execution_environment_manifest.json"
OFFLINE_ARTIFACT_MANIFEST_FILENAME = "offline_smoke_artifact_manifest.json"
EXECUTION_SOURCE_MANIFEST_CONTRACT = {
    "schema_name": "LocalExecutionSourceManifest",
    "schema_version": "1.0",
}
VALIDATION_INPUT_MANIFEST_CONTRACT = {
    "schema_name": "LocalValidationInputManifest",
    "schema_version": "1.0",
}
EXECUTION_ENVIRONMENT_MANIFEST_CONTRACT = {
    "schema_name": "LocalExecutionEnvironmentManifest",
    "schema_version": "1.0",
}
OFFLINE_ARTIFACT_MANIFEST_CONTRACT = {
    "schema_name": "LocalOfflineArtifactTreeManifest",
    "schema_version": "1.0",
}

# These paths and contracts describe the superseded one-shot evidence. They
# remain readable historical files if present, but no current freeze validator
# or paid launcher accepts them.
LEGACY_LOCAL_ENGINEERING_RECEIPT_CONTRACTS = {
    "unit_tested": {
        "receipt_path": (
            "outputs/readiness/modal_only_final/unit_test_evidence_receipt.json"
        ),
        "receipt_contract": {
            "schema_name": "LocalEngineeringEvidenceReceipt",
            "schema_version": "1.0",
            "evidence_kind": "unit_test_suite",
        },
    },
    "offline_smoke_tested": {
        "receipt_path": (
            "outputs/readiness/modal_only_final/offline_smoke_evidence_receipt.json"
        ),
        "receipt_contract": {
            "schema_name": "LocalEngineeringEvidenceReceipt",
            "schema_version": "1.0",
            "evidence_kind": "offline_smoke",
        },
    },
}

LOCAL_ENGINEERING_RECEIPT_CONTRACTS = {
    "unit_tested": {
        "receipt_filename": "unit_test_evidence_receipt.json",
        "result_filename": "unit_test_result.txt",
        "receipt_contract": {
            "schema_name": "LocalEngineeringEvidenceReceipt",
            "schema_version": "2.0",
            "evidence_kind": "unit_test_suite",
        },
    },
    "offline_smoke_tested": {
        "receipt_filename": "offline_smoke_evidence_receipt.json",
        "result_filename": "offline_smoke_result.json",
        "artifact_directory_filename": "offline_smoke_artifacts",
        "receipt_contract": {
            "schema_name": "LocalEngineeringEvidenceReceipt",
            "schema_version": "2.0",
            "evidence_kind": "offline_smoke",
        },
    },
}

_STRIPPED_EXECUTION_ENVIRONMENT = frozenset(
    {
        "DISCOVERY_API_BASE",
        "DISCOVERY_API_KEY",
        "DISCOVERY_MODEL",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "TINKER_API_KEY",
        "TML_API_KEY",
        "PYTEST_ADDOPTS",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "PYTHONINSPECT",
    }
)

_COMMON_IDENTITY_FIELDS = frozenset(
    {
        "validation_identity_sha256",
        "execution_source_manifest_path",
        "execution_source_manifest_sha256",
        "validation_input_manifest_path",
        "validation_input_manifest_sha256",
        "execution_environment_manifest_path",
        "execution_environment_manifest_sha256",
    }
)
_ACTION_ACCOUNTING_FIELDS = frozenset(
    {
        "provider_calls",
        "remote_actions",
        "remote_training_runs",
        "scientific_runs",
        "local_fixture_training_permitted",
        "scientific",
        "externally_attested",
        "passed",
    }
)

_RECEIPT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "evidence_kind",
        "recorded_at_utc",
        "source_revision",
        "source_tree_sha256",
        "source_tree_sha256_before_command",
        "source_tree_sha256_after_command",
        "image_source_sha256",
        "command",
        "command_sha256",
        "resolved_command",
        "resolved_command_sha256",
        "execution_environment",
        "execution_environment_sha256",
        "result_path",
        "result_sha256",
        "artifact_manifest_path",
        "artifact_manifest_sha256",
        "checks_completed",
    }
) | _COMMON_IDENTITY_FIELDS | _ACTION_ACCOUNTING_FIELDS
_FREEZE_RECEIPT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "recorded_at_utc",
        "source_revision",
        "source_tree_sha256",
        "image_source_sha256",
        "component_receipts",
    }
) | _COMMON_IDENTITY_FIELDS | _ACTION_ACCOUNTING_FIELDS
_PHASE2_RECEIPT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "recorded_at_utc",
        "source_revision",
        "source_tree_sha256",
        "source_tree_sha256_before_commands",
        "source_tree_sha256_after_commands",
        "image_source_sha256",
        "dependency_lock_sha256",
        "image_source_file_count",
        "image_source_total_bytes",
        "image_source_two_copy_upper_bound_bytes",
        "mandatory_component_ids",
        "component_receipt_coverage",
        "executed_components",
    }
) | _COMMON_IDENTITY_FIELDS | _ACTION_ACCOUNTING_FIELDS
_PHASE2_COMPONENT_FIELDS = frozenset(
    {
        "component_id",
        "command",
        "resolved_command",
        "cwd",
        "environment_overrides",
        "execution_environment",
        "timeout_seconds",
        "returncode",
        "stdout",
        "stdout_sha256",
        "stdout_bytes",
        "stderr",
        "stderr_sha256",
        "stderr_bytes",
        "checks_completed",
        "passed",
    }
)
MANDATORY_PHASE2_VALIDATION_COMPONENTS = (
    "root_make_check",
    "complete_architecture_pytest",
    "migration_ruff",
    "git_diff_check",
    "vendored_openevolve_diff_check",
    "configuration_validation",
    "compile_validation",
    "environment_import_validation",
    "dependency_lock_validation",
    "modal_boundary_launcher_security_device_resume_timeout",
    "fresh_c0_c3_no_search_offline_smoke",
    "sealed_layer_b_c_synthetic",
    "reconstruction_reporting_synthetic",
    "four_controller_static_validation",
    "modal_cost_free_plan",
)
_PHASE2_COMPONENT_RECEIPT_COVERAGE = {
    "complete_architecture_pytest": "unit_tested",
    "fresh_c0_c3_no_search_offline_smoke": "offline_smoke_tested",
}
_MAX_PHASE2_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_PHASE2_TOTAL_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_LOCAL_COMMAND_OUTPUT_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_MANIFEST_FILES = 10_000
_MAX_SOURCE_MANIFEST_FILE_BYTES = 256 * 1024 * 1024
_MAX_SOURCE_MANIFEST_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_VALIDATION_INPUT_FILE_BYTES = 32 * 1024 * 1024
_MAX_VALIDATION_INPUT_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_OFFLINE_ARTIFACT_FILES = 10_000
_MAX_OFFLINE_ARTIFACT_FILE_BYTES = 32 * 1024 * 1024
_MAX_OFFLINE_ARTIFACT_TOTAL_BYTES = 1024 * 1024 * 1024
_MAX_LOCAL_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
_MAX_MODAL_CONSOLE_SCRIPT_BYTES = 64 * 1024
_WORKSPACE_SOURCE_ROOT_FILES = (
    "Makefile",
    "pyproject.toml",
    "uv.lock",
)
_WORKSPACE_SOURCE_DIRECTORIES = (
    "src",
    "tests",
    "configs",
    "schemas",
    "data",
)
_SOURCE_ROOT_FILES = (
    "experiment_manifest.yaml",
    "modal_action_journal.py",
    "modal_app.py",
    "modal_boundary.py",
    "modal_image_build.py",
    "pyproject.toml",
    "scientific_decisions.yaml",
    "uv.lock",
    "vendor/openevolve/README.md",
    "vendor/openevolve/pyproject.toml",
    "vendor/starting_model/checkpoints/best.pt",
    "vendor/starting_model/src/data.py",
    "vendor/starting_model/src/eval.py",
    "vendor/starting_model/src/model.py",
    "vendor/starting_model/src/train.py",
)
_SOURCE_DIRECTORIES = (
    "agents",
    "analysis",
    "architecture_ir",
    "artifacts",
    "audits",
    "baselines",
    "common",
    "containment",
    "evaluation",
    "mechanism",
    "novelty",
    "private_eval",
    "reconstruction",
    "replication",
    "reporting",
    "research_ledger",
    "review",
    "sealed_eval",
    "scripts",
    "study",
    "tests",
    "vendor/openevolve/openevolve",
    "vendor/openevolve/tests",
    "vendor_patches",
)
_VALIDATION_INPUT_FILES = (
    "README.md",
    "MODAL_MIGRATION_NOTES.md",
    "readiness_evidence.yaml",
)
_GENERATED_DIRECTORY_NAMES = frozenset({"__pycache__", ".pytest_cache"})
_GENERATED_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})
_SOURCE_SUFFIXES = frozenset(
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
_PYTEST_PASSED = re.compile(r"(?m)(?:^|\s)([1-9][0-9]*) passed(?:[,\s]|$)")
_SHA256_TEXT = re.compile(r"\A[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class _Phase2CommandSpec:
    component_id: str
    command: tuple[str, ...]
    cwd: str
    timeout_seconds: int
    result_kind: str
    environment_overrides: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class _ValidationIdentity:
    validation_identity_sha256: str
    source_revision: str
    source_manifest: dict[str, Any]
    source_manifest_sha256: str
    validation_input_manifest: dict[str, Any]
    validation_input_manifest_sha256: str
    execution_environment_manifest: dict[str, Any]
    execution_environment_manifest_sha256: str


_MIGRATION_RUFF_TARGETS = (
    "agents",
    "analysis",
    "architecture_ir",
    "artifacts",
    "audits",
    "baselines",
    "common",
    "containment",
    "evaluation",
    "mechanism",
    "novelty",
    "private_eval",
    "reconstruction",
    "replication",
    "reporting",
    "research_ledger",
    "review",
    "sealed_eval",
    "scripts",
    "study",
    "tests",
    "modal_action_journal.py",
    "modal_app.py",
    "modal_boundary.py",
    "modal_image_build.py",
)
_COMPILE_VALIDATION_TARGETS = (
    "common",
    "agents",
    "scripts",
    "tests",
    "study",
    "evaluation",
    "sealed_eval",
    "containment",
    "architecture_ir",
    "novelty",
    "review",
    "mechanism",
    "replication",
    "baselines",
    "analysis",
    "artifacts",
    "private_eval",
    "reconstruction",
    "research_ledger",
    "reporting",
    "modal_action_journal.py",
    "modal_app.py",
    "modal_boundary.py",
    "modal_image_build.py",
)
_MODAL_FOCUSED_TESTS = (
    "tests/test_modal_action_journal.py",
    "tests/test_modal_boundary.py",
    "tests/test_modal_image_build.py",
    "tests/test_modal_import_boundary.py",
    "tests/test_modal_launcher.py",
    "tests/test_modal_readiness_receipts.py",
    "tests/test_worker_environment.py",
    "tests/test_device_training.py",
    "tests/test_checkpoint_resume.py",
    "tests/test_resume_contract_verification.py",
    "tests/test_resume_progression_verification.py",
    "tests/test_process_control.py",
    "tests/test_openevolve_process_boundary.py",
    "tests/test_runtime_context.py",
    "tests/test_cuda_environment_diagnostic.py",
    "tests/test_accelerator_evidence_recording.py",
    "tests/test_training_invariants.py",
)


def _phase2_command_specs() -> tuple[_Phase2CommandSpec, ...]:
    """Return the exact ordered, provider-free Phase-2 command matrix."""

    no_bytecode = (("PYTHONDONTWRITEBYTECODE", "1"),)
    return (
        _Phase2CommandSpec(
            component_id="root_make_check",
            command=("make", "check"),
            cwd="workspace_root",
            timeout_seconds=300,
            result_kind="pytest",
            environment_overrides=(
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("UV_CACHE_DIR", "tmp/uv-cache"),
                ("UV_OFFLINE", "1"),
            ),
        ),
        _Phase2CommandSpec(
            component_id="migration_ruff",
            command=(
                "../.venv/bin/ruff",
                "check",
                "--isolated",
                "--select",
                "E4,E7,E9,F",
                "--ignore",
                "E402",
                "--target-version",
                "py312",
                "--line-length",
                "88",
                *_MIGRATION_RUFF_TARGETS,
            ),
            cwd="project_root",
            timeout_seconds=120,
            result_kind="ruff",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="git_diff_check",
            command=("git", "diff", "--check"),
            cwd="workspace_root",
            timeout_seconds=60,
            result_kind="zero_exit",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="vendored_openevolve_diff_check",
            command=(
                "git",
                "-C",
                "vendor/openevolve",
                "diff",
                "--check",
            ),
            cwd="project_root",
            timeout_seconds=60,
            result_kind="zero_exit",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="configuration_validation",
            command=("python", "scripts/validate_configs.py"),
            cwd="project_root",
            timeout_seconds=120,
            result_kind="configuration",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="compile_validation",
            command=(
                "python",
                "-m",
                "compileall",
                "-q",
                *_COMPILE_VALIDATION_TARGETS,
            ),
            cwd="project_root",
            timeout_seconds=180,
            result_kind="compile",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="environment_import_validation",
            command=("python", "scripts/check_environment.py"),
            cwd="project_root",
            timeout_seconds=120,
            result_kind="environment",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="dependency_lock_validation",
            command=("uv", "lock", "--check", "--offline"),
            cwd="project_root",
            timeout_seconds=180,
            result_kind="zero_exit",
            environment_overrides=(
                ("PYTHONDONTWRITEBYTECODE", "1"),
                ("UV_CACHE_DIR", "../tmp/uv-cache"),
                ("UV_OFFLINE", "1"),
            ),
        ),
        _Phase2CommandSpec(
            component_id=(
                "modal_boundary_launcher_security_device_resume_timeout"
            ),
            command=("python", "-m", "pytest", *_MODAL_FOCUSED_TESTS),
            cwd="project_root",
            timeout_seconds=600,
            result_kind="pytest",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="sealed_layer_b_c_synthetic",
            command=(
                "python",
                "-m",
                "pytest",
                "tests/test_sealed_post_search_orchestration.py",
                "tests/test_evaluation_firewall_records.py",
            ),
            cwd="project_root",
            timeout_seconds=180,
            result_kind="pytest",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="reconstruction_reporting_synthetic",
            command=(
                "python",
                "-m",
                "pytest",
                "tests/test_reporting_reconstruction.py",
                "tests/test_reporting_report.py",
                "tests/test_reconstruction_run.py",
            ),
            cwd="project_root",
            timeout_seconds=180,
            result_kind="pytest",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="four_controller_static_validation",
            command=("python", "scripts/validate_engineering_canaries.py"),
            cwd="project_root",
            timeout_seconds=180,
            result_kind="four_controller_static",
            environment_overrides=no_bytecode,
        ),
        _Phase2CommandSpec(
            component_id="modal_cost_free_plan",
            command=("python", "scripts/modal_plan.py"),
            cwd="project_root",
            timeout_seconds=120,
            result_kind="modal_plan",
            environment_overrides=no_bytecode,
        ),
    )


def _sha256_file(path: Path) -> str:
    """Hash one regular, singly linked file through its opened descriptor."""

    parent_descriptor, absolute_parent = _open_absolute_directory_nofollow(
        Path(path).parent,
        create=False,
    )
    try:
        record, _ = _manifest_record_from_parent(
            parent_descriptor,
            Path(path).name,
            logical_path=Path(path).name,
            field="required file",
            max_file_bytes=_MAX_SOURCE_MANIFEST_FILE_BYTES,
        )
        _require_same_directory_path(absolute_parent, parent_descriptor)
        return str(record["sha256"])
    finally:
        os.close(parent_descriptor)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _open_absolute_directory_nofollow(
    directory: Path,
    *,
    create: bool,
) -> tuple[int, Path]:
    """Open one absolute directory path without following any symlink."""

    absolute = Path(os.path.abspath(os.fspath(directory)))
    if not absolute.is_absolute() or not absolute.parts:
        raise ValueError("local evidence directory could not be anchored")
    descriptor = os.open(absolute.anchor, _directory_open_flags())
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ValueError(
                    "local evidence directory contains an unsafe path component"
                )
            if create:
                try:
                    os.mkdir(component, 0o755, dir_fd=descriptor)
                except FileExistsError:
                    pass
                else:
                    # Persist every newly created directory entry before it is
                    # used as the anchor for the next component.
                    os.fsync(descriptor)
            try:
                before = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"local evidence directory does not exist: {absolute}"
                ) from None
            if stat.S_ISLNK(before.st_mode):
                raise ValueError(
                    "local evidence directory path may not contain a symbolic link"
                )
            if not stat.S_ISDIR(before.st_mode):
                raise NotADirectoryError(
                    "local evidence directory component is not a directory: "
                    f"{component}"
                )
            try:
                child_descriptor = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise ValueError(
                    "local evidence directory changed while it was opened"
                ) from error
            opened = os.fstat(child_descriptor)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(child_descriptor)
                raise ValueError("local evidence directory changed while it was opened")
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor, absolute
    except BaseException:
        os.close(descriptor)
        raise


def _require_manifest_regular_file(
    observed: os.stat_result,
    *,
    field: str,
    max_file_bytes: int,
) -> None:
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{field} must be a regular file")
    if observed.st_nlink != 1:
        raise ValueError(f"{field} must have exactly one hard link")
    if observed.st_uid != os.getuid():
        raise ValueError(f"{field} must be owned by the current user")
    if observed.st_size < 0 or observed.st_size > max_file_bytes:
        raise ValueError(f"{field} exceeds the per-file manifest byte limit")


def _hash_open_manifest_file(
    descriptor: int,
    observed: os.stat_result,
    *,
    field: str,
) -> str:
    digest = hashlib.sha256()
    remaining = observed.st_size
    while remaining:
        try:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        except InterruptedError:
            continue
        if not chunk:
            raise ValueError(f"{field} ended before its descriptor-bound size")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ValueError(f"{field} grew while it was hashed")
    after = os.fstat(descriptor)
    identity = (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{field} changed while it was hashed")
    return digest.hexdigest()


def _manifest_record_from_parent(
    parent_descriptor: int,
    filename: str,
    *,
    logical_path: str,
    field: str,
    max_file_bytes: int,
) -> tuple[dict[str, Any], tuple[int, int]]:
    try:
        before = os.stat(
            filename,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"required manifest input is missing: {field}") from None
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{field} may not be a symbolic link")
    _require_manifest_regular_file(
        before,
        field=field,
        max_file_bytes=max_file_bytes,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError(f"{field} changed while it was opened") from error
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"{field} changed while it was opened")
        _require_manifest_regular_file(
            opened,
            field=field,
            max_file_bytes=max_file_bytes,
        )
        digest = _hash_open_manifest_file(descriptor, opened, field=field)
    finally:
        os.close(descriptor)
    after = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(after.st_mode) or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{field} changed while it was hashed")
    return (
        {
            "relative_path": logical_path,
            "sha256": digest,
            "size_bytes": opened.st_size,
        },
        (opened.st_dev, opened.st_ino),
    )


def _open_manifest_subdirectory(
    parent_descriptor: int,
    name: str,
    *,
    field: str,
) -> int:
    before = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{field} may not be a symbolic link")
    if not stat.S_ISDIR(before.st_mode):
        raise NotADirectoryError(f"{field} is not a directory")
    try:
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
    except OSError as error:
        raise ValueError(f"{field} changed while it was opened") from error
    opened = os.fstat(descriptor)
    if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
        os.close(descriptor)
        raise ValueError(f"{field} changed while it was opened")
    return descriptor


def _open_manifest_relative_directory(
    root_descriptor: int,
    relative: str,
    *,
    field: str,
) -> int:
    descriptor = os.dup(root_descriptor)
    try:
        for component in safe_relative_path(relative).parts:
            child = _open_manifest_subdirectory(
                descriptor,
                component,
                field=f"{field}/{component}",
            )
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_manifest_relative_parent(
    root_descriptor: int,
    relative: str,
    *,
    field: str,
) -> tuple[int, str, dict[str, tuple[int, int]]]:
    path = safe_relative_path(relative)
    descriptor = os.dup(root_descriptor)
    ancestors: dict[str, tuple[int, int]] = {}
    try:
        parts: list[str] = []
        for component in path.parts[:-1]:
            child = _open_manifest_subdirectory(
                descriptor,
                component,
                field=f"{field}/{component}",
            )
            os.close(descriptor)
            descriptor = child
            parts.append(component)
            opened = os.fstat(descriptor)
            ancestors["/".join(parts)] = (opened.st_dev, opened.st_ino)
        return descriptor, path.name, ancestors
    except BaseException:
        os.close(descriptor)
        raise


def _scan_manifest_scope_once(
    root: Path,
    *,
    namespace: str,
    root_files: tuple[str, ...],
    directories: tuple[str, ...],
    include_complete_root: bool,
    max_file_bytes: int,
    max_files: int,
    max_total_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, tuple[int, int]]]:
    descriptor, absolute_root = _open_absolute_directory_nofollow(
        root,
        create=False,
    )
    records: list[dict[str, Any]] = []
    identities: dict[str, tuple[int, int]] = {
        f"{namespace}/.": (
            os.fstat(descriptor).st_dev,
            os.fstat(descriptor).st_ino,
        )
    }
    seen: set[str] = set()
    total_bytes = 0

    def append_record(
        parent_descriptor: int,
        filename: str,
        logical: str,
    ) -> None:
        nonlocal total_bytes
        if logical in seen:
            raise ValueError(f"manifest input is duplicated: {logical}")
        record, identity = _manifest_record_from_parent(
            parent_descriptor,
            filename,
            logical_path=logical,
            field=logical,
            max_file_bytes=max_file_bytes,
        )
        if len(records) + 1 > max_files:
            raise ValueError("manifest exceeds its file-count limit")
        total_bytes += int(record["size_bytes"])
        if total_bytes > max_total_bytes:
            raise ValueError("manifest exceeds its total-byte limit")
        records.append(record)
        identities[logical] = identity
        seen.add(logical)

    def walk(directory_descriptor: int, relative: str) -> None:
        names = sorted(os.listdir(directory_descriptor))
        for name in names:
            if name in {"", ".", ".."}:
                raise ValueError("manifest tree contains an unsafe directory entry")
            child_relative = f"{relative}/{name}" if relative else name
            logical = f"{namespace}/{child_relative}"
            observed = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(f"{logical} may not be a symbolic link")
            if stat.S_ISDIR(observed.st_mode):
                if name in _GENERATED_DIRECTORY_NAMES:
                    continue
                child_descriptor = _open_manifest_subdirectory(
                    directory_descriptor,
                    name,
                    field=logical,
                )
                try:
                    opened = os.fstat(child_descriptor)
                    identities[f"{logical}/"] = (opened.st_dev, opened.st_ino)
                    walk(child_descriptor, child_relative)
                finally:
                    os.close(child_descriptor)
                after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if (opened.st_dev, opened.st_ino) != (
                    after.st_dev,
                    after.st_ino,
                ):
                    raise ValueError(f"{logical} changed while it was scanned")
            elif stat.S_ISREG(observed.st_mode):
                if Path(name).suffix.lower() in _GENERATED_FILE_SUFFIXES:
                    continue
                append_record(directory_descriptor, name, logical)
            else:
                raise ValueError(f"{logical} is not a regular file or directory")

    try:
        for relative in root_files:
            parent_descriptor, filename, ancestors = _open_manifest_relative_parent(
                descriptor,
                relative,
                field=f"{namespace}/{relative}",
            )
            try:
                for ancestor, identity in ancestors.items():
                    identities[f"{namespace}/{ancestor}/"] = identity
                append_record(
                    parent_descriptor,
                    filename,
                    f"{namespace}/{safe_relative_path(relative).as_posix()}",
                )
            finally:
                os.close(parent_descriptor)
        if include_complete_root:
            walk(descriptor, "")
        else:
            for relative in directories:
                directory_descriptor = _open_manifest_relative_directory(
                    descriptor,
                    relative,
                    field=f"{namespace}/{relative}",
                )
                try:
                    opened = os.fstat(directory_descriptor)
                    logical = f"{namespace}/{safe_relative_path(relative).as_posix()}/"
                    identities[logical] = (opened.st_dev, opened.st_ino)
                    walk(directory_descriptor, safe_relative_path(relative).as_posix())
                finally:
                    os.close(directory_descriptor)
        _require_same_directory_path(absolute_root, descriptor)
    finally:
        os.close(descriptor)
    return sorted(records, key=lambda item: str(item["relative_path"])), identities


def _scan_manifest_scope(
    root: Path,
    *,
    namespace: str,
    root_files: tuple[str, ...] = (),
    directories: tuple[str, ...] = (),
    include_complete_root: bool = False,
    max_file_bytes: int,
    max_files: int,
    max_total_bytes: int,
) -> list[dict[str, Any]]:
    first, first_identities = _scan_manifest_scope_once(
        root,
        namespace=namespace,
        root_files=root_files,
        directories=directories,
        include_complete_root=include_complete_root,
        max_file_bytes=max_file_bytes,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
    )
    second, second_identities = _scan_manifest_scope_once(
        root,
        namespace=namespace,
        root_files=root_files,
        directories=directories,
        include_complete_root=include_complete_root,
        max_file_bytes=max_file_bytes,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
    )
    if first != second or first_identities != second_identities:
        raise ValueError(f"{namespace} changed while its manifest was constructed")
    return first


def _build_file_manifest(
    contract: dict[str, str],
    records: list[dict[str, Any]],
    *,
    max_files: int,
    max_total_bytes: int,
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: str(item["relative_path"]))
    if len(ordered) > max_files:
        raise ValueError("combined manifest exceeds its file-count limit")
    total_bytes = sum(int(item["size_bytes"]) for item in ordered)
    if total_bytes > max_total_bytes:
        raise ValueError("combined manifest exceeds its total-byte limit")
    if len({str(item["relative_path"]) for item in ordered}) != len(ordered):
        raise ValueError("combined manifest contains duplicate logical paths")
    return {
        **contract,
        "files": ordered,
        "file_count": len(ordered),
        "total_bytes": total_bytes,
    }


def _require_same_directory_path(directory: Path, expected_descriptor: int) -> None:
    reopened_descriptor, _ = _open_absolute_directory_nofollow(
        directory,
        create=False,
    )
    try:
        expected = os.fstat(expected_descriptor)
        reopened = os.fstat(reopened_descriptor)
        if (expected.st_dev, expected.st_ino) != (
            reopened.st_dev,
            reopened.st_ino,
        ):
            raise ValueError("local evidence directory changed during immutable access")
    finally:
        os.close(reopened_descriptor)


def _require_safe_frozen_file(
    observed: os.stat_result,
    *,
    field: str,
    expected_size: int | None = None,
) -> None:
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"{field} must be a regular file")
    if observed.st_nlink != 1:
        raise ValueError(f"{field} must have exactly one hard link")
    if observed.st_uid != os.getuid():
        raise ValueError(f"{field} must be owned by the current user")
    if stat.S_IMODE(observed.st_mode) & 0o077:
        raise ValueError(f"{field} may not grant group or other permissions")
    if expected_size is not None and observed.st_size != expected_size:
        raise ValueError(f"{field} has an incomplete or unexpected size")


def _read_open_file_exactly(
    descriptor: int,
    observed: os.stat_result,
    *,
    field: str,
) -> bytes:
    remaining = observed.st_size
    chunks: list[bytes] = []
    while remaining:
        try:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
        except InterruptedError:
            continue
        if not chunk:
            raise ValueError(f"{field} ended before its recorded size")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ValueError(f"{field} grew while it was read")
    after = os.fstat(descriptor)
    if (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{field} changed while it was read")
    return b"".join(chunks)


def _read_named_frozen_file(
    parent_descriptor: int,
    filename: str,
    *,
    field: str,
) -> tuple[bytes, tuple[int, int]]:
    try:
        before = os.stat(
            filename,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            f"required local evidence file is missing: {field}"
        ) from None
    if stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{field} may not be a symbolic link")
    _require_safe_frozen_file(before, field=field)
    if before.st_size > _MAX_LOCAL_EVIDENCE_FILE_BYTES:
        raise ValueError(f"{field} exceeds the local evidence byte limit")
    read_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    read_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(filename, read_flags, dir_fd=parent_descriptor)
    except OSError as error:
        raise ValueError(f"{field} changed while it was opened") from error
    try:
        opened = os.fstat(descriptor)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError(f"{field} changed while it was opened")
        _require_safe_frozen_file(opened, field=field)
        payload = _read_open_file_exactly(descriptor, opened, field=field)
    finally:
        os.close(descriptor)
    after = os.stat(filename, dir_fd=parent_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(after.st_mode) or (before.st_dev, before.st_ino) != (
        after.st_dev,
        after.st_ino,
    ):
        raise ValueError(f"{field} changed while it was read")
    return payload, (before.st_dev, before.st_ino)


def _open_contained_parent(
    root: Path,
    relative: Path,
) -> tuple[int, Path]:
    root_descriptor, absolute_root = _open_absolute_directory_nofollow(
        root,
        create=False,
    )
    descriptor = root_descriptor
    try:
        for component in relative.parts[:-1]:
            before = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(before.st_mode):
                raise ValueError("local evidence path may not traverse symbolic links")
            if not stat.S_ISDIR(before.st_mode):
                raise NotADirectoryError(
                    f"local evidence parent is not a directory: {component}"
                )
            try:
                child_descriptor = os.open(
                    component,
                    _directory_open_flags(),
                    dir_fd=descriptor,
                )
            except OSError as error:
                raise ValueError(
                    "local evidence parent changed while it was opened"
                ) from error
            opened = os.fstat(child_descriptor)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                os.close(child_descriptor)
                raise ValueError("local evidence parent changed while it was opened")
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor, absolute_root
    except BaseException:
        os.close(descriptor)
        raise


def _read_contained_bytes(root: Path, logical: object, field: str) -> bytes:
    """Read immutable local evidence twice through no-follow directory FDs."""

    if not isinstance(logical, str):
        raise ValueError(f"{field} must be a project-relative path")
    try:
        relative = safe_relative_path(logical)
    except ValueError as error:
        raise ValueError(
            f"{field} must be normalized, project-relative, and non-traversing"
        ) from error
    first_parent, absolute_root = _open_contained_parent(Path(root), relative)
    try:
        first_payload, first_identity = _read_named_frozen_file(
            first_parent,
            relative.name,
            field=field,
        )
        first_parent_stat = os.fstat(first_parent)
    finally:
        os.close(first_parent)

    # Reopen the complete root/parent/target path. A rename or symlink swap
    # between validation stages must not let one byte stream be hashed while
    # another is parsed.
    second_parent, _ = _open_contained_parent(absolute_root, relative)
    try:
        second_parent_stat = os.fstat(second_parent)
        if (first_parent_stat.st_dev, first_parent_stat.st_ino) != (
            second_parent_stat.st_dev,
            second_parent_stat.st_ino,
        ):
            raise ValueError(f"{field} parent changed while it was validated")
        second_payload, second_identity = _read_named_frozen_file(
            second_parent,
            relative.name,
            field=field,
        )
        if first_identity != second_identity or first_payload != second_payload:
            raise ValueError(f"{field} changed while it was validated")
    finally:
        os.close(second_parent)
    return first_payload


def _create_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Durably create immutable result bytes without following path symlinks."""

    if not isinstance(payload, bytes):
        raise TypeError("local engineering result payload must be bytes")
    destination = Path(os.path.abspath(os.fspath(path)))
    if not destination.name or destination.name in {".", ".."}:
        raise ValueError("local engineering result must name a file")
    parent_descriptor, absolute_parent = _open_absolute_directory_nofollow(
        destination.parent,
        create=True,
    )
    published = False
    try:
        _require_same_directory_path(absolute_parent, parent_descriptor)
        write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        write_flags |= getattr(os, "O_CLOEXEC", 0)
        write_flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(
            destination.name,
            write_flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        published = True
        try:
            remaining = memoryview(payload)
            while remaining:
                try:
                    written = os.write(descriptor, remaining)
                except InterruptedError:
                    continue
                if written <= 0:
                    raise OSError("exclusive result write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            _require_safe_frozen_file(
                os.fstat(descriptor),
                field="local engineering result",
                expected_size=len(payload),
            )
        except BaseException:
            # A partial create-only result is deliberately retained as an
            # immutable quarantine marker. It may never be retried or replaced.
            with suppress(OSError):
                os.fsync(descriptor)
            raise
        finally:
            os.close(descriptor)
        os.fsync(parent_descriptor)

        reopened, _ = _read_named_frozen_file(
            parent_descriptor,
            destination.name,
            field="local engineering result",
        )
        if reopened != payload:
            raise ValueError("published local engineering result bytes differ")
        _require_same_directory_path(absolute_parent, parent_descriptor)
    finally:
        if published:
            with suppress(OSError):
                os.fsync(parent_descriptor)
        os.close(parent_descriptor)


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256_TEXT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _exact_bool(value: object, field: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")


def _exact_int(value: object, field: str, expected: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    if expected is not None and value != expected:
        raise ValueError(f"{field} must be exactly {expected}")
    return value


def _valid_utc(value: object, field: str = "recorded_at_utc") -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an explicit UTC timestamp") from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must use UTC")
    return parsed


def _same_lexical_absolute_path(left: str | Path, right: str | Path) -> bool:
    """Compare paths without accepting a symbolic-link alias."""

    return os.path.abspath(os.fspath(left)) == os.path.abspath(os.fspath(right))


class _ProcessOutputLimitError(RuntimeError):
    """A frozen command exceeded its pre-accumulation output budget."""


def _run_streaming_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: float,
    max_output_bytes: int,
) -> subprocess.CompletedProcess[bytes]:
    """Run one command in an isolated group with bounded streaming capture.

    The byte budget is shared by stdout and stderr and checked before a chunk
    is appended. The isolated group is verified closed on every path, including
    the case where the leader exits successfully while a descendant survives.
    """

    if not command or not all(isinstance(item, str) and item for item in command):
        raise ValueError("bounded command argv must contain nonempty text")
    if timeout_seconds <= 0 or max_output_bytes <= 0:
        raise ValueError("bounded command limits must be positive")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    process_group_id = capture_isolated_process_group(process)
    if process.stdout is None or process.stderr is None:  # pragma: no cover
        terminate_process_group(process, process_group_id=process_group_id)
        raise RuntimeError("bounded command pipes were not created")
    selector = selectors.DefaultSelector()
    stdout_descriptor = process.stdout.fileno()
    stderr_descriptor = process.stderr.fileno()
    streams = {
        stdout_descriptor: ("stdout", bytearray()),
        stderr_descriptor: ("stderr", bytearray()),
    }
    for descriptor in streams:
        os.set_blocking(descriptor, False)
        selector.register(descriptor, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    total_bytes = 0
    leader_returncode: int | None = None
    closure_error: BaseException | None = None
    group_closed = False
    try:
        while selector.get_map() or process.poll() is None:
            observed = process.poll()
            if observed is not None and leader_returncode is None:
                leader_returncode = observed
                try:
                    terminate_process_group(
                        process,
                        process_group_id=process_group_id,
                    )
                    group_closed = True
                except BaseException as error:
                    closure_error = error
                    raise
            remaining_time = deadline - time.monotonic()
            if remaining_time <= 0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            events = selector.select(min(0.05, remaining_time))
            for key, _ in events:
                descriptor = int(key.fd)
                stream_name, captured = streams[descriptor]
                remaining_budget = max_output_bytes - total_bytes
                try:
                    chunk = os.read(
                        descriptor,
                        min(64 * 1024, remaining_budget + 1),
                    )
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                if len(chunk) > remaining_budget:
                    raise _ProcessOutputLimitError(
                        "bounded command output exceeded "
                        f"{max_output_bytes} bytes while reading {stream_name}"
                    )
                captured.extend(chunk)
                total_bytes += len(chunk)
        if leader_returncode is None:
            leader_returncode = process.wait(timeout=0)
    except BaseException:
        raise
    finally:
        if not group_closed:
            try:
                terminate_process_group(
                    process,
                    process_group_id=process_group_id,
                )
                group_closed = True
            except BaseException as error:
                closure_error = error
        selector.close()
        process.stdout.close()
        process.stderr.close()
        if closure_error is not None:
            raise closure_error
    if leader_returncode is None:  # pragma: no cover - exhaustive guard
        raise RuntimeError("bounded command did not expose its leader result")
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=leader_returncode,
        stdout=bytes(streams[stdout_descriptor][1]),
        stderr=bytes(streams[stderr_descriptor][1]),
    )


def _minimal_execution_environment(project_root: Path) -> dict[str, str]:
    """Return the exact small environment inherited by local validation."""

    del project_root  # The mapping intentionally contains no workspace alias.
    environment = {
        "PATH": os.environ.get("PATH") or os.defpath,
        "LANG": os.environ.get("LANG") or "C",
        "LC_ALL": os.environ.get("LC_ALL") or "C",
        "NO_COLOR": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUTF8": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
    }
    temporary_root = os.environ.get("TMPDIR")
    if temporary_root:
        environment["TMPDIR"] = temporary_root
    for key in _STRIPPED_EXECUTION_ENVIRONMENT:
        environment.pop(key, None)
    return dict(sorted(environment.items()))


def _resolve_executable(executable: str, *, environment: dict[str, str]) -> str:
    if "/" in executable:
        candidate = Path(executable)
        if not candidate.is_absolute():
            raise ValueError("relative executable resolution requires an explicit cwd")
        invoked = Path(os.path.abspath(os.fspath(candidate)))
    else:
        found = shutil.which(executable, path=environment["PATH"])
        if found is None:
            raise FileNotFoundError(f"required executable is missing: {executable}")
        invoked = Path(os.path.abspath(found))
    if not invoked.exists():
        raise FileNotFoundError(f"required executable is missing: {invoked}")
    return os.fspath(invoked)


def _hash_executable(path: str | Path) -> dict[str, Any]:
    invoked = Path(os.path.abspath(os.fspath(path)))
    invoked_before = os.lstat(invoked)
    if not (stat.S_ISREG(invoked_before.st_mode) or stat.S_ISLNK(invoked_before.st_mode)):
        raise ValueError(f"executable is not a regular file or symlink: {invoked}")
    resolved = Path(os.path.realpath(invoked))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(resolved, flags)
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"resolved executable is not a regular file: {resolved}")
        digest = _hash_open_manifest_file(
            descriptor,
            observed,
            field=f"executable {invoked}",
        )
    finally:
        os.close(descriptor)
    resolved_after = os.stat(resolved, follow_symlinks=False)
    invoked_after = os.lstat(invoked)
    if (
        invoked_before.st_dev,
        invoked_before.st_ino,
        invoked_before.st_mtime_ns,
        invoked_before.st_ctime_ns,
    ) != (
        invoked_after.st_dev,
        invoked_after.st_ino,
        invoked_after.st_mtime_ns,
        invoked_after.st_ctime_ns,
    ) or (
        observed.st_dev,
        observed.st_ino,
        observed.st_size,
        observed.st_mtime_ns,
        observed.st_ctime_ns,
    ) != (
        resolved_after.st_dev,
        resolved_after.st_ino,
        resolved_after.st_size,
        resolved_after.st_mtime_ns,
        resolved_after.st_ctime_ns,
    ):
        raise ValueError(f"executable changed while it was identified: {invoked}")
    return {
        "invoked_path": os.fspath(invoked),
        "resolved_path": os.fspath(resolved),
        "sha256": digest,
        "size_bytes": observed.st_size,
        "symlink_target": os.readlink(invoked) if stat.S_ISLNK(invoked_before.st_mode) else None,
    }


def _modal_cli_installation_proof(
    modal_executable: str | Path | None = None,
    *,
    version_lookup: Any = None,
) -> dict[str, Any]:
    """Bind the exact non-symlink Modal console script used by paid launches."""

    lookup = importlib.metadata.version if version_lookup is None else version_lookup
    try:
        installed_version = lookup("modal")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("the pinned Modal package is not installed") from error
    if installed_version != MODAL_VERSION or MODAL_VERSION != "1.5.3":
        raise ValueError("local freeze requires exactly Modal 1.5.3")
    selected = (
        Path(sys.executable).with_name("modal")
        if modal_executable is None
        else Path(modal_executable)
    )
    absolute = Path(os.path.abspath(os.fspath(selected)))
    before = os.lstat(absolute)
    mode = stat.S_IMODE(before.st_mode)
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.getuid()
        or mode & 0o022
        or not mode & stat.S_IXUSR
        or before.st_size > _MAX_MODAL_CONSOLE_SCRIPT_BYTES
    ):
        raise ValueError("pinned Modal executable metadata is unsafe")
    identity = _hash_executable(absolute)
    after = os.lstat(absolute)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        stat.S_IMODE(before.st_mode),
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        stat.S_IMODE(after.st_mode),
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        before_identity != after_identity
        or after.st_nlink != 1
        or after.st_uid != os.getuid()
        or identity["symlink_target"] is not None
    ):
        raise ValueError("pinned Modal executable changed while it was identified")
    return {
        **identity,
        "distribution_version": installed_version,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mode": stat.S_IMODE(after.st_mode),
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
    }


def _openevolve_installation_proof(project_root: Path) -> dict[str, Any]:
    try:
        venv_root = Path(sys.prefix).resolve(strict=True)
        purelib = Path(sysconfig.get_path("purelib")).resolve(strict=True)
        purelib.relative_to(venv_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            "installed OpenEvolve environment cannot be resolved"
        ) from error
    distributions = tuple(
        distribution
        for distribution in importlib.metadata.distributions(
            path=[os.fspath(purelib)]
        )
        if re.sub(
            r"[-_.]+",
            "-",
            str(distribution.metadata.get("Name", "")).lower(),
        )
        == "openevolve"
    )
    if len(distributions) != 1:
        raise ValueError(
            "installed OpenEvolve distribution is unavailable or ambiguous"
        )
    installed_root = purelib / "openevolve"
    vendor_root = project_root / "vendor" / "openevolve" / "openevolve"
    installed = _scan_manifest_scope(
        installed_root,
        namespace="installed_openevolve",
        include_complete_root=True,
        max_file_bytes=_MAX_SOURCE_MANIFEST_FILE_BYTES,
        max_files=_MAX_SOURCE_MANIFEST_FILES,
        max_total_bytes=_MAX_SOURCE_MANIFEST_TOTAL_BYTES,
    )
    vendor = _scan_manifest_scope(
        vendor_root,
        namespace="vendor_openevolve",
        include_complete_root=True,
        max_file_bytes=_MAX_SOURCE_MANIFEST_FILE_BYTES,
        max_files=_MAX_SOURCE_MANIFEST_FILES,
        max_total_bytes=_MAX_SOURCE_MANIFEST_TOTAL_BYTES,
    )
    installed_normalized = [
        {
            **record,
            "relative_path": str(record["relative_path"]).removeprefix(
                "installed_openevolve/"
            ),
        }
        for record in installed
    ]
    vendor_normalized = [
        {
            **record,
            "relative_path": str(record["relative_path"]).removeprefix(
                "vendor_openevolve/"
            ),
        }
        for record in vendor
    ]
    if installed_normalized != vendor_normalized:
        raise ValueError("installed OpenEvolve bytes differ from the vendored source")
    return {
        "distribution_version": distributions[0].version,
        "installed_root": os.fspath(installed_root),
        "vendor_root": "vendor/openevolve/openevolve",
        "file_count": len(vendor_normalized),
        "files_sha256": content_hash(vendor_normalized),
        "installed_matches_vendor": True,
    }


def execution_environment_manifest(
    root: str | Path = ROOT,
) -> dict[str, Any]:
    project_root = Path(os.path.abspath(os.fspath(root)))
    environment = _minimal_execution_environment(project_root)
    ruff_candidate = project_root.parent / ".venv" / "bin" / "ruff"
    if ruff_candidate.exists() or ruff_candidate.is_symlink():
        ruff_executable = os.fspath(Path(os.path.abspath(ruff_candidate)))
    else:
        ruff_executable = _resolve_executable("ruff", environment=environment)
    tool_paths = {
        "git": _resolve_executable("git", environment=environment),
        "make": _resolve_executable("make", environment=environment),
        "uv": _resolve_executable("uv", environment=environment),
        "ruff": ruff_executable,
    }
    workspace_python = project_root.parent / ".venv" / "bin" / "python"
    python_identity = _hash_executable(sys.executable)
    dependencies: dict[str, str] = {}
    for distribution in (
        "modal",
        "numpy",
        "openai",
        "openevolve",
        "pytest",
        "PyYAML",
        "torch",
    ):
        try:
            dependencies[distribution.lower()] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(
                f"required validation dependency is not installed: {distribution}"
            ) from error
    manifest: dict[str, Any] = {
        **EXECUTION_ENVIRONMENT_MANIFEST_CONTRACT,
        "environment": environment,
        "python": {
            **python_identity,
            "version": sys.version,
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
        },
        "tools": {
            name: _hash_executable(path)
            for name, path in sorted(tool_paths.items())
        },
        "dependencies": dict(sorted(dependencies.items())),
        "modal_cli": _modal_cli_installation_proof(),
        "openevolve_installation": _openevolve_installation_proof(project_root),
    }
    if workspace_python.exists() or workspace_python.is_symlink():
        manifest["workspace_python"] = _hash_executable(workspace_python)
    else:
        manifest["workspace_python"] = None
    return manifest


def _git_revision(root: Path) -> str:
    environment = _minimal_execution_environment(root)
    git_executable = _resolve_executable("git", environment=environment)
    try:
        completed = _run_streaming_bounded_process(
            [git_executable, "rev-parse", "HEAD"],
            cwd=root,
            environment=environment,
            timeout_seconds=5,
            max_output_bytes=64 * 1024,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("unable to resolve the source revision") from error
    try:
        revision = completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("source revision is not ASCII") from error
    if (
        completed.returncode != 0
        or completed.stderr
        or len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("source revision is not a lowercase Git digest")
    return revision


def execution_source_manifest(root: str | Path = ROOT) -> dict[str, Any]:
    """Build a complete descriptor-bound manifest of executed local inputs."""

    project_root = Path(os.path.abspath(os.fspath(root)))
    workspace_records = _scan_manifest_scope(
        project_root.parent,
        namespace="workspace",
        root_files=_WORKSPACE_SOURCE_ROOT_FILES,
        directories=_WORKSPACE_SOURCE_DIRECTORIES,
        max_file_bytes=_MAX_SOURCE_MANIFEST_FILE_BYTES,
        max_files=_MAX_SOURCE_MANIFEST_FILES,
        max_total_bytes=_MAX_SOURCE_MANIFEST_TOTAL_BYTES,
    )
    architecture_records = _scan_manifest_scope(
        project_root,
        namespace="architecture_discovery",
        root_files=_SOURCE_ROOT_FILES,
        directories=_SOURCE_DIRECTORIES,
        max_file_bytes=_MAX_SOURCE_MANIFEST_FILE_BYTES,
        max_files=_MAX_SOURCE_MANIFEST_FILES,
        max_total_bytes=_MAX_SOURCE_MANIFEST_TOTAL_BYTES,
    )
    return _build_file_manifest(
        EXECUTION_SOURCE_MANIFEST_CONTRACT,
        [*workspace_records, *architecture_records],
        max_files=_MAX_SOURCE_MANIFEST_FILES,
        max_total_bytes=_MAX_SOURCE_MANIFEST_TOTAL_BYTES,
    )


def source_tree_sha256(root: str | Path = ROOT) -> str:
    """Return the canonical digest of the persisted execution-source manifest."""

    return content_hash(execution_source_manifest(root))


def validation_input_manifest(root: str | Path = ROOT) -> dict[str, Any]:
    """Bind dynamic pre-live documents without treating them as execution source."""

    records = _scan_manifest_scope(
        Path(root),
        namespace="architecture_dynamic",
        root_files=_VALIDATION_INPUT_FILES,
        max_file_bytes=_MAX_VALIDATION_INPUT_FILE_BYTES,
        max_files=len(_VALIDATION_INPUT_FILES),
        max_total_bytes=_MAX_VALIDATION_INPUT_TOTAL_BYTES,
    )
    return _build_file_manifest(
        VALIDATION_INPUT_MANIFEST_CONTRACT,
        records,
        max_files=len(_VALIDATION_INPUT_FILES),
        max_total_bytes=_MAX_VALIDATION_INPUT_TOTAL_BYTES,
    )


def _current_validation_identity(root: str | Path = ROOT) -> _ValidationIdentity:
    project_root = Path(os.path.abspath(os.fspath(root)))
    source_manifest = execution_source_manifest(project_root)
    validation_manifest = validation_input_manifest(project_root)
    environment_manifest = execution_environment_manifest(project_root)
    source_digest = content_hash(source_manifest)
    validation_digest = content_hash(validation_manifest)
    environment_digest = content_hash(environment_manifest)
    revision = _git_revision(project_root)
    validation_identity = content_hash(
        {
            "schema_name": "LocalValidationIdentity",
            "schema_version": "1.0",
            "source_tree_sha256": source_digest,
            "source_revision": revision,
            "validation_input_manifest_sha256": validation_digest,
            "execution_environment_manifest_sha256": environment_digest,
        }
    )
    return _ValidationIdentity(
        validation_identity_sha256=validation_identity,
        source_revision=revision,
        source_manifest=source_manifest,
        source_manifest_sha256=source_digest,
        validation_input_manifest=validation_manifest,
        validation_input_manifest_sha256=validation_digest,
        execution_environment_manifest=environment_manifest,
        execution_environment_manifest_sha256=environment_digest,
    )


def _identity_manifest_specs(
    identity: _ValidationIdentity,
) -> tuple[tuple[str, dict[str, Any], str], ...]:
    return (
        (
            EXECUTION_SOURCE_MANIFEST_FILENAME,
            identity.source_manifest,
            identity.source_manifest_sha256,
        ),
        (
            VALIDATION_INPUT_MANIFEST_FILENAME,
            identity.validation_input_manifest,
            identity.validation_input_manifest_sha256,
        ),
        (
            EXECUTION_ENVIRONMENT_MANIFEST_FILENAME,
            identity.execution_environment_manifest,
            identity.execution_environment_manifest_sha256,
        ),
    )


def _ensure_identity_manifests(
    project_root: Path,
    identity: _ValidationIdentity,
    *,
    create_missing: bool,
) -> None:
    directory = local_engineering_freeze_directory(
        identity.validation_identity_sha256
    )
    for filename, expected, expected_digest in _identity_manifest_specs(identity):
        logical = (directory / filename).as_posix()
        absolute = project_root / logical
        if not absolute.exists() and not absolute.is_symlink():
            if not create_missing:
                raise FileNotFoundError(
                    f"validation identity manifest is missing: {logical}"
                )
            try:
                create_json_exclusive(absolute, expected)
            except FileExistsError:
                # A racing writer may only win if it published the exact bytes.
                pass
        raw = _read_contained_bytes(
            project_root,
            logical,
            f"validation_identity_manifest.{filename}",
        )
        observed = _load_exact_json_object(
            raw,
            field=f"validation identity manifest {logical}",
        )
        if observed != expected or content_hash(observed) != expected_digest:
            raise ValueError(f"validation identity manifest changed: {logical}")


def _identity_receipt_fields(identity: _ValidationIdentity) -> dict[str, str]:
    directory = local_engineering_freeze_directory(
        identity.validation_identity_sha256
    )
    return {
        "validation_identity_sha256": identity.validation_identity_sha256,
        "execution_source_manifest_path": (
            directory / EXECUTION_SOURCE_MANIFEST_FILENAME
        ).as_posix(),
        "execution_source_manifest_sha256": identity.source_manifest_sha256,
        "validation_input_manifest_path": (
            directory / VALIDATION_INPUT_MANIFEST_FILENAME
        ).as_posix(),
        "validation_input_manifest_sha256": identity.validation_input_manifest_sha256,
        "execution_environment_manifest_path": (
            directory / EXECUTION_ENVIRONMENT_MANIFEST_FILENAME
        ).as_posix(),
        "execution_environment_manifest_sha256": (
            identity.execution_environment_manifest_sha256
        ),
    }


def _validate_identity_receipt_fields(
    payload: dict[str, Any],
    identity: _ValidationIdentity,
) -> None:
    expected = _identity_receipt_fields(identity)
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"local validation {field} changed")


def _action_accounting() -> dict[str, bool | int]:
    return {
        "provider_calls": 0,
        "remote_actions": 0,
        "remote_training_runs": 0,
        "scientific_runs": 0,
        "local_fixture_training_permitted": True,
        "scientific": False,
        "externally_attested": False,
        "passed": True,
    }


def _validate_action_accounting(payload: dict[str, Any]) -> None:
    _exact_int(payload["provider_calls"], "provider_calls", 0)
    _exact_int(payload["remote_actions"], "remote_actions", 0)
    _exact_int(payload["remote_training_runs"], "remote_training_runs", 0)
    _exact_int(payload["scientific_runs"], "scientific_runs", 0)
    _exact_bool(
        payload["local_fixture_training_permitted"],
        "local_fixture_training_permitted",
        True,
    )
    _exact_bool(payload["scientific"], "scientific", False)
    _exact_bool(payload["externally_attested"], "externally_attested", False)
    _exact_bool(payload["passed"], "passed", True)


def local_engineering_freeze_directory(validation_identity_digest: str) -> Path:
    """Return the directory namespaced by source, HEAD, inputs, and environment."""

    return LOCAL_ENGINEERING_FREEZE_ROOT / _sha256(
        validation_identity_digest,
        "validation_identity_sha256",
    )


def local_engineering_receipt_path(
    level_name: str,
    *,
    validation_identity_digest: str,
) -> Path:
    contract = LOCAL_ENGINEERING_RECEIPT_CONTRACTS.get(level_name)
    if contract is None:
        raise ValueError(f"unknown local engineering level: {level_name}")
    return local_engineering_freeze_directory(validation_identity_digest) / str(
        contract["receipt_filename"]
    )


def local_engineering_result_path(
    level_name: str,
    *,
    validation_identity_digest: str,
) -> Path:
    contract = LOCAL_ENGINEERING_RECEIPT_CONTRACTS.get(level_name)
    if contract is None:
        raise ValueError(f"unknown local engineering level: {level_name}")
    return local_engineering_freeze_directory(validation_identity_digest) / str(
        contract["result_filename"]
    )


def local_engineering_freeze_receipt_path(validation_identity_digest: str) -> Path:
    return (
        local_engineering_freeze_directory(validation_identity_digest)
        / LOCAL_ENGINEERING_FREEZE_RECEIPT_FILENAME
    )


def local_phase2_validation_receipt_path(validation_identity_digest: str) -> Path:
    return (
        local_engineering_freeze_directory(validation_identity_digest)
        / LOCAL_PHASE2_VALIDATION_RECEIPT_FILENAME
    )


def current_local_engineering_receipt_path(
    level_name: str,
    *,
    root: str | Path = ROOT,
) -> Path:
    identity = _current_validation_identity(root)
    return local_engineering_receipt_path(
        level_name,
        validation_identity_digest=identity.validation_identity_sha256,
    )


def current_local_engineering_freeze_receipt_path(
    *,
    root: str | Path = ROOT,
) -> Path:
    identity = _current_validation_identity(root)
    return local_engineering_freeze_receipt_path(identity.validation_identity_sha256)


def frozen_local_engineering_command(
    level_name: str,
    *,
    validation_identity_digest: str,
) -> list[str]:
    """Return the exact logical command recorded for one source freeze."""

    _sha256(validation_identity_digest, "validation_identity_sha256")
    if level_name == "unit_tested":
        # pyproject.toml already supplies one -q. Repeating it suppresses the
        # final pass-count line that the receipt independently parses.
        return ["python", "-m", "pytest"]
    if level_name == "offline_smoke_tested":
        artifacts = local_engineering_freeze_directory(
            validation_identity_digest
        ) / str(
            LOCAL_ENGINEERING_RECEIPT_CONTRACTS[level_name][
                "artifact_directory_filename"
            ]
        )
        return [
            "python",
            "scripts/study_offline_smoke.py",
            "--output-dir",
            artifacts.as_posix(),
            "--study-id",
            f"readiness-offline-smoke-{validation_identity_digest[:16]}",
            "--study-seed",
            "7",
            "--blocks",
            "1",
            "--opportunities",
            "3",
        ]
    raise ValueError(f"unknown local engineering level: {level_name}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_exact_json_object(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} must be one UTF-8 JSON object") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{field} must be one UTF-8 JSON object")
    return payload


def _offline_artifact_root(
    project_root: Path,
    validation_identity_digest: str,
) -> Path:
    return (
        project_root
        / local_engineering_freeze_directory(validation_identity_digest)
        / str(
            LOCAL_ENGINEERING_RECEIPT_CONTRACTS["offline_smoke_tested"][
                "artifact_directory_filename"
            ]
        )
    )


def _build_offline_artifact_manifest(
    project_root: Path,
    validation_identity_digest: str,
) -> dict[str, Any]:
    artifact_root = _offline_artifact_root(
        project_root,
        validation_identity_digest,
    )
    records = _scan_manifest_scope(
        artifact_root,
        namespace="offline_smoke_artifacts",
        include_complete_root=True,
        max_file_bytes=_MAX_OFFLINE_ARTIFACT_FILE_BYTES,
        max_files=_MAX_OFFLINE_ARTIFACT_FILES,
        max_total_bytes=_MAX_OFFLINE_ARTIFACT_TOTAL_BYTES,
    )
    manifest = _build_file_manifest(
        OFFLINE_ARTIFACT_MANIFEST_CONTRACT,
        records,
        max_files=_MAX_OFFLINE_ARTIFACT_FILES,
        max_total_bytes=_MAX_OFFLINE_ARTIFACT_TOTAL_BYTES,
    )
    manifest["artifact_root"] = (
        artifact_root.relative_to(project_root).as_posix()
    )
    manifest["validation_identity_sha256"] = validation_identity_digest
    return manifest


def _validate_offline_artifact_tree(
    project_root: Path,
    validation_identity_digest: str,
    stdout_bytes: bytes,
) -> tuple[dict[str, Any], int]:
    """Validate the complete retained C0-C3/no-search fixture and manifest it."""

    summary = _load_exact_json_object(stdout_bytes, field="offline-smoke stdout")
    expected_study_id = f"readiness-offline-smoke-{validation_identity_digest[:16]}"
    artifact_root = _offline_artifact_root(
        project_root,
        validation_identity_digest,
    )
    before = _build_offline_artifact_manifest(
        project_root,
        validation_identity_digest,
    )
    expected_children = {
        expected_study_id,
        f"{expected_study_id}-no-search-smoke",
    }
    observed_children = {path.name for path in artifact_root.iterdir()}
    if observed_children != expected_children:
        raise ValueError("offline artifact root has an unexpected exact roster")
    primary_root = artifact_root / expected_study_id
    no_search_root = artifact_root / f"{expected_study_id}-no-search-smoke"
    from reconstruction.downloaded_offline import (  # noqa: PLC0415
        DownloadedOfflineValidationError,
        _validate_study,
    )

    try:
        plan, study_evidence, _ = _validate_study(primary_root, no_search_root)
    except DownloadedOfflineValidationError as error:
        raise ValueError(f"offline artifact tree is invalid: {error}") from error
    if plan.study_id != expected_study_id:
        raise ValueError("offline artifact study ID differs from its frozen command")
    retained_summary = _load_exact_json_object(
        _read_contained_bytes(
            artifact_root,
            f"{expected_study_id}/offline_smoke_summary.json",
            "offline_smoke_summary",
        ),
        field="retained offline-smoke summary",
    )
    if retained_summary != summary:
        raise ValueError("offline-smoke stdout differs from its retained summary")
    assigned_run_ids = [run.run_id for run in plan.runs]
    stdout_run_ids = [
        record.get("run_id")
        for record in summary.get("runs", [])
        if isinstance(record, dict)
    ]
    if stdout_run_ids != assigned_run_ids:
        raise ValueError("offline-smoke stdout run IDs differ from the frozen plan")
    evidence_runs = study_evidence.get("runs")
    if not isinstance(evidence_runs, list) or [
        record.get("run_id") for record in evidence_runs if isinstance(record, dict)
    ] != assigned_run_ids:
        raise ValueError("offline-smoke index evidence differs from frozen run IDs")
    after = _build_offline_artifact_manifest(
        project_root,
        validation_identity_digest,
    )
    if before != after:
        raise ValueError("offline artifact tree changed while it was validated")
    return before, len(assigned_run_ids)


def _run_frozen_command(
    level_name: str,
    *,
    project_root: Path,
    expected_identity: _ValidationIdentity,
) -> dict[str, Any]:
    validation_digest = expected_identity.validation_identity_sha256
    result = project_root / local_engineering_result_path(
        level_name,
        validation_identity_digest=validation_digest,
    )
    if result.exists() or result.is_symlink():
        raise FileExistsError(f"local engineering result already exists: {result}")
    if level_name == "offline_smoke_tested":
        artifacts = _offline_artifact_root(project_root, validation_digest)
        if artifacts.exists() or artifacts.is_symlink():
            raise FileExistsError(
                f"local engineering artifact directory already exists: {artifacts}"
            )
    logical_command = frozen_local_engineering_command(
        level_name,
        validation_identity_digest=validation_digest,
    )
    command = [os.path.abspath(sys.executable), *logical_command[1:]]
    environment = _minimal_execution_environment(project_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment = dict(sorted(environment.items()))
    timeout_seconds = 900 if level_name == "unit_tested" else 300
    before = _current_validation_identity(project_root)
    if before != expected_identity:
        raise RuntimeError("local validation identity changed before command start")
    completed: subprocess.CompletedProcess[bytes] | None = None
    command_error: BaseException | None = None
    try:
        completed = _run_streaming_bounded_process(
            command,
            cwd=project_root,
            environment=environment,
            timeout_seconds=timeout_seconds,
            max_output_bytes=_MAX_LOCAL_COMMAND_OUTPUT_BYTES,
        )
    except BaseException as error:
        command_error = error
    after = _current_validation_identity(project_root)
    if after != before:
        raise RuntimeError(
            "local validation identity changed while the frozen command ran"
        ) from command_error
    if command_error is not None:
        raise command_error
    if completed is None:  # pragma: no cover - exhaustive state guard
        raise RuntimeError("local engineering command produced no process result")
    if completed.returncode != 0:
        raise RuntimeError(
            f"local engineering command failed with exit {completed.returncode}"
        )
    artifact_manifest: dict[str, Any] | None = None
    if level_name == "offline_smoke_tested":
        if completed.stderr:
            raise ValueError("offline-smoke stderr must be exactly empty")
        artifact_manifest, _ = _validate_offline_artifact_tree(
            project_root,
            validation_digest,
            completed.stdout,
        )
        payload = completed.stdout
    else:
        payload = completed.stdout
        if completed.stderr:
            payload += b"\n--- sanitized stderr stream ---\n" + completed.stderr
    _create_bytes_exclusive(result, payload)
    return {
        "source_tree_sha256_before_command": before.source_manifest_sha256,
        "source_tree_sha256_after_command": after.source_manifest_sha256,
        "resolved_command": command,
        "execution_environment": environment,
        "artifact_manifest": artifact_manifest,
    }


def _validate_unit_result_bytes(payload: bytes) -> int:
    try:
        result = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("unit-test result must be UTF-8") from error
    match = _PYTEST_PASSED.search(result)
    if match is None:
        raise ValueError("unit-test result lacks a positive pytest pass count")
    lowered = result.lower()
    if re.search(r"\b(?:failed|error|errors)\b", lowered):
        raise ValueError("unit-test result contains a failure or error")
    return int(match.group(1))


def _validate_offline_result_bytes(raw: bytes) -> int:
    payload = _load_exact_json_object(raw, field="offline-smoke result")
    if set(payload) != {
        "schema_name",
        "schema_version",
        "mode",
        "provider_calls",
        "torch_training_runs",
        "scientific",
        "study_id",
        "assignment_hash",
        "scheduler",
        "runs",
        "artifact_index_manifest",
        "no_search",
    }:
        raise ValueError("offline-smoke result has an invalid exact schema")
    if (
        payload["schema_name"] != "OfflineStudySmokeSummary"
        or payload["schema_version"] != "1.0"
    ):
        raise ValueError("offline-smoke result has the wrong contract")
    expected = {
        "mode": "offline_synthetic_only",
        "provider_calls": 0,
        "torch_training_runs": 0,
        "scientific": False,
    }
    for field, expected_value in expected.items():
        observed = payload.get(field)
        if type(expected_value) is bool and type(observed) is not bool:
            raise ValueError(f"offline-smoke {field} must be boolean")
        if isinstance(expected_value, int) and not isinstance(expected_value, bool):
            _exact_int(observed, f"offline-smoke {field}", expected_value)
        if observed != expected_value:
            raise ValueError(f"offline-smoke {field} is invalid")
    if (
        not isinstance(payload["study_id"], str)
        or not payload["study_id"]
        or _SHA256_TEXT.fullmatch(str(payload["assignment_hash"])) is None
        or payload["artifact_index_manifest"] != "artifact_index_manifest.json"
    ):
        raise ValueError("offline-smoke identity fields are invalid")
    scheduler = payload["scheduler"]
    if not isinstance(scheduler, dict) or set(scheduler) != {
        "study_id",
        "assignment_hash",
        "accelerator_kind",
        "counts",
        "active_run_id",
        "revision",
    }:
        raise ValueError("offline-smoke scheduler has an invalid exact schema")
    counts = scheduler.get("counts")
    if not isinstance(counts, dict) or set(counts) != {
        "pending",
        "running",
        "completed",
        "interrupted",
    }:
        raise ValueError("offline-smoke scheduler counts are invalid")
    if (
        scheduler["study_id"] != payload["study_id"]
        or scheduler["assignment_hash"] != payload["assignment_hash"]
        or scheduler["accelerator_kind"] != "cpu"
        or scheduler["active_run_id"] is not None
        or counts != {
            "pending": 0,
            "running": 0,
            "completed": 4,
            "interrupted": 0,
        }
    ):
        raise ValueError("offline-smoke scheduler is not exactly terminal")
    _exact_int(scheduler["revision"], "offline-smoke scheduler revision")
    runs = payload["runs"]
    if not isinstance(runs, list) or len(runs) != 4:
        raise ValueError("offline-smoke must contain exactly four C0-C3 runs")
    conditions = []
    run_ids: list[str] = []
    for run in runs:
        if not isinstance(run, dict) or set(run) != {
            "run_id",
            "condition_id",
            "status",
            "seed_evaluations",
            "proposal_opportunities",
        }:
            raise ValueError("offline-smoke run has an invalid exact schema")
        if run.get("status") != "completed":
            raise ValueError("offline-smoke contains an incomplete run")
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("offline-smoke run IDs must be nonempty text")
        run_ids.append(run_id)
        condition = run.get("condition_id")
        if not isinstance(condition, str):
            raise ValueError("offline-smoke condition IDs must be text")
        conditions.append(condition)
        _exact_int(run["seed_evaluations"], "offline-smoke seed evaluations")
        _exact_int(
            run["proposal_opportunities"],
            "offline-smoke proposal opportunities",
        )
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("offline-smoke run IDs must be unique")
    if set(conditions) != {"C0", "C1", "C2", "C3"}:
        raise ValueError("offline-smoke does not cover C0-C3 exactly once")
    no_search = payload["no_search"]
    if not isinstance(no_search, dict) or set(no_search) != {
        "condition_id",
        "scientific",
        "adaptive_feedback_visible_to_backend",
        "provider_input_constant",
        "request_count",
        "ledger",
    }:
        raise ValueError("offline-smoke lacks no-search evidence")
    if no_search["condition_id"] != "NO_SEARCH":
        raise ValueError("offline-smoke no-search condition ID is invalid")
    for field, expected_value in {
        "scientific": False,
        "adaptive_feedback_visible_to_backend": False,
        "provider_input_constant": True,
    }.items():
        _exact_bool(no_search.get(field), f"no_search.{field}", expected_value)
    request_count = _exact_int(
        no_search["request_count"],
        "no_search.request_count",
    )
    ledger_payload = no_search["ledger"]
    if not isinstance(ledger_payload, dict):
        raise ValueError("offline-smoke no-search ledger must be an object")
    from study import BudgetLedger  # noqa: PLC0415

    try:
        ledger = BudgetLedger.from_dict(ledger_payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("offline-smoke no-search ledger is invalid") from error
    if (
        ledger.to_dict() != ledger_payload
        or ledger.provider_attempts != request_count
        or ledger.proposal_opportunities != request_count
        or ledger.terminal_opportunities != request_count
        or ledger.active_opportunity is not None
    ):
        raise ValueError("offline-smoke no-search ledger is inconsistent")
    return len(runs)


def _validate_result_bytes(level_name: str, payload: bytes) -> int:
    if level_name == "unit_tested":
        return _validate_unit_result_bytes(payload)
    if level_name == "offline_smoke_tested":
        return _validate_offline_result_bytes(payload)
    raise ValueError(f"unknown local engineering level: {level_name}")


def _current_source_identities(project_root: Path) -> tuple[str, str]:
    source_digest = source_tree_sha256(project_root)
    image_digest = build_image_source_manifest(project_root).manifest_sha256
    return source_digest, image_digest


def _require_validation_identity_current(
    project_root: Path,
    identity: _ValidationIdentity,
) -> None:
    source_manifest = execution_source_manifest(project_root)
    environment_manifest = execution_environment_manifest(project_root)
    source_digest = content_hash(source_manifest)
    environment_digest = content_hash(environment_manifest)
    revision = _git_revision(project_root)
    expected_identity_sha256 = content_hash(
        {
            "schema_name": "LocalValidationIdentity",
            "schema_version": "1.0",
            "source_tree_sha256": source_digest,
            "source_revision": revision,
            "validation_input_manifest_sha256": (
                identity.validation_input_manifest_sha256
            ),
            "execution_environment_manifest_sha256": environment_digest,
        }
    )
    if (
        identity.source_manifest != source_manifest
        or identity.source_manifest_sha256 != source_digest
        or identity.execution_environment_manifest != environment_manifest
        or identity.execution_environment_manifest_sha256 != environment_digest
        or identity.source_revision != revision
        or identity.validation_identity_sha256 != expected_identity_sha256
    ):
        raise ValueError(
            "historical local freeze execution source, Git revision, or "
            "environment changed"
        )


def _current_identity_and_image(
    project_root: Path,
    *,
    validation_identity: _ValidationIdentity | None = None,
) -> tuple[_ValidationIdentity, str]:
    identity = validation_identity or _current_validation_identity(project_root)
    if validation_identity is not None:
        _require_validation_identity_current(project_root, identity)
    image_digest = build_image_source_manifest(project_root).manifest_sha256
    return identity, image_digest


def _phase2_image_metrics(project_root: Path) -> dict[str, str | int]:
    manifest = build_image_source_manifest(project_root)
    total_bytes = sum(item.size_bytes for item in manifest.files)
    return {
        "dependency_lock_sha256": manifest.dependency_lock_sha256,
        "image_source_file_count": len(manifest.files),
        "image_source_total_bytes": total_bytes,
        "image_source_two_copy_upper_bound_bytes": 2 * total_bytes,
    }


def _phase2_execution_environment(
    spec: _Phase2CommandSpec,
    *,
    project_root: Path,
) -> dict[str, str]:
    """Build a minimal provider-free environment for a validation command."""

    environment = _minimal_execution_environment(project_root)
    environment.update(dict(spec.environment_overrides))
    for key in _STRIPPED_EXECUTION_ENVIRONMENT:
        environment.pop(key, None)
    return dict(sorted(environment.items()))


def _phase2_execution_command(
    spec: _Phase2CommandSpec,
    *,
    project_root: Path,
    environment: dict[str, str] | None = None,
) -> list[str]:
    if environment is None:
        environment = _phase2_execution_environment(
            spec,
            project_root=project_root,
        )
    command = list(spec.command)
    executable = command[0]
    if executable == "python":
        command[0] = os.path.abspath(sys.executable)
    elif "/" in executable:
        cwd = _phase2_component_cwd(spec, project_root)
        candidate = Path(os.path.abspath(cwd / executable))
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Phase-2 validation executable is missing: {executable}"
            )
        command[0] = os.fspath(candidate)
    else:
        command[0] = _resolve_executable(executable, environment=environment)
    return command


def _phase2_component_cwd(spec: _Phase2CommandSpec, project_root: Path) -> Path:
    if spec.cwd == "project_root":
        return project_root
    if spec.cwd == "workspace_root":
        return project_root.parent
    raise ValueError(f"unknown Phase-2 command cwd: {spec.cwd}")


def _decode_phase2_stream(payload: bytes, *, field: str) -> str:
    if len(payload) > _MAX_PHASE2_COMMAND_OUTPUT_BYTES:
        raise ValueError(f"{field} exceeds the local evidence byte limit")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{field} must be UTF-8") from error


def _execute_phase2_component(
    spec: _Phase2CommandSpec,
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Execute one isolated local validation command and capture exact output."""

    environment = _phase2_execution_environment(
        spec,
        project_root=project_root,
    )
    resolved_command = _phase2_execution_command(
        spec,
        project_root=project_root,
        environment=environment,
    )
    completed = _run_streaming_bounded_process(
        resolved_command,
        cwd=_phase2_component_cwd(spec, project_root),
        environment=environment,
        timeout_seconds=spec.timeout_seconds,
        max_output_bytes=_MAX_PHASE2_COMMAND_OUTPUT_BYTES,
    )
    stdout_bytes = completed.stdout
    stderr_bytes = completed.stderr
    stdout = _decode_phase2_stream(
        stdout_bytes,
        field=f"{spec.component_id}.stdout",
    )
    stderr = _decode_phase2_stream(
        stderr_bytes,
        field=f"{spec.component_id}.stderr",
    )
    record = {
        "component_id": spec.component_id,
        "command": list(spec.command),
        "resolved_command": resolved_command,
        "cwd": spec.cwd,
        "environment_overrides": dict(spec.environment_overrides),
        "execution_environment": environment,
        "timeout_seconds": spec.timeout_seconds,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stdout_bytes": len(stdout_bytes),
        "stderr": stderr,
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "stderr_bytes": len(stderr_bytes),
        "checks_completed": 0,
        "passed": completed.returncode == 0,
    }
    record["checks_completed"] = _phase2_component_checks(spec, record)
    return record


def _phase2_json_output(record: dict[str, Any], *, field: str) -> dict[str, Any]:
    return _load_exact_json_object(
        str(record["stdout"]).encode("utf-8"),
        field=field,
    )


def _phase2_component_checks(
    spec: _Phase2CommandSpec,
    record: dict[str, Any],
) -> int:
    returncode = record.get("returncode")
    if not isinstance(returncode, int) or isinstance(returncode, bool):
        raise ValueError(f"{spec.component_id} return code must be an integer")
    if returncode != 0:
        raise RuntimeError(
            f"mandatory Phase-2 component failed: {spec.component_id} "
            f"(exit {returncode})"
        )
    combined = f"{record.get('stdout', '')}\n{record.get('stderr', '')}"
    lowered = combined.lower()
    if spec.result_kind == "pytest":
        matches = [int(item) for item in _PYTEST_PASSED.findall(combined)]
        if not matches:
            raise ValueError(
                f"{spec.component_id} lacks a positive pytest pass count"
            )
        if re.search(r"\b(?:failed|error|errors)\b", lowered):
            raise ValueError(f"{spec.component_id} reported a pytest failure")
        return sum(matches)
    if spec.result_kind == "ruff":
        if "all checks passed" not in lowered:
            raise ValueError("migration Ruff did not report all checks passed")
        return 1
    if spec.result_kind == "configuration":
        if "configuration invariants: pass" not in lowered:
            raise ValueError("configuration validation did not report PASS")
        return 1
    if spec.result_kind == "compile":
        if "error compiling" in lowered or "traceback" in lowered:
            raise ValueError("compile validation reported an error")
        return 1
    if spec.result_kind == "environment":
        payload = _phase2_json_output(record, field=spec.component_id)
        _exact_bool(
            payload.get("scientific_cpu_fallback"),
            "environment.scientific_cpu_fallback",
            False,
        )
        credentials = payload.get("credentials")
        if not isinstance(credentials, dict) or set(credentials) != {
            "DISCOVERY_API_KEY",
            "DISCOVERY_API_BASE",
            "DISCOVERY_MODEL",
        }:
            raise ValueError("environment validation lacks exact credential fields")
        for name, present in credentials.items():
            _exact_bool(present, f"environment.credentials.{name}", False)
        return 1
    if spec.result_kind == "four_controller_static":
        payload = _phase2_json_output(record, field=spec.component_id)
        _exact_bool(
            payload.get("static_controller_surfaces_passed"),
            "static_controller_surfaces_passed",
            True,
        )
        _exact_int(payload.get("provider_calls"), "provider_calls", 0)
        _exact_int(payload.get("training_runs"), "training_runs", 0)
        _exact_int(
            payload.get("entrypoint_execution_runs"),
            "entrypoint_execution_runs",
            0,
        )
        surfaces = payload.get("static_controller_surfaces")
        harnesses = surfaces.get("harnesses") if isinstance(surfaces, dict) else None
        if not isinstance(harnesses, list) or len(harnesses) != 4:
            raise ValueError("static validation did not cover exactly four harnesses")
        return 4
    if spec.result_kind == "modal_plan":
        payload = _phase2_json_output(record, field=spec.component_id)
        if payload.get("schema_name") != "ModalExecutionPlan":
            raise ValueError("Modal plan has the wrong schema")
        _exact_int(payload.get("remote_calls_started"), "remote_calls_started", 0)
        functions = payload.get("functions")
        if not isinstance(functions, dict) or not functions:
            raise ValueError("Modal plan has no bounded function specifications")
        for name, function in functions.items():
            if not isinstance(name, str) or not isinstance(function, dict):
                raise ValueError("Modal plan function specification is invalid")
            _exact_int(function.get("max_containers"), f"{name}.max_containers", 1)
            _exact_int(function.get("min_containers"), f"{name}.min_containers", 0)
            _exact_int(function.get("retries"), f"{name}.retries", 0)
        return len(functions)
    if spec.result_kind == "zero_exit":
        return 1
    raise ValueError(f"unknown Phase-2 result kind: {spec.result_kind}")


def _validate_phase2_component_record(
    spec: _Phase2CommandSpec,
    record: object,
    *,
    project_root: Path,
) -> None:
    if not isinstance(record, dict) or set(record) != _PHASE2_COMPONENT_FIELDS:
        raise ValueError(
            f"Phase-2 component {spec.component_id} has an invalid exact schema"
        )
    if record["component_id"] != spec.component_id:
        raise ValueError("Phase-2 component order or identity changed")
    if record["command"] != list(spec.command):
        raise ValueError(f"{spec.component_id} command changed")
    if record["cwd"] != spec.cwd:
        raise ValueError(f"{spec.component_id} cwd changed")
    if record["environment_overrides"] != dict(spec.environment_overrides):
        raise ValueError(f"{spec.component_id} environment overrides changed")
    expected_environment = _phase2_execution_environment(
        spec,
        project_root=project_root,
    )
    if record["execution_environment"] != expected_environment:
        raise ValueError(f"{spec.component_id} exact execution environment changed")
    if record["resolved_command"] != _phase2_execution_command(
        spec,
        project_root=project_root,
        environment=expected_environment,
    ):
        raise ValueError(f"{spec.component_id} resolved command changed")
    _exact_int(record["timeout_seconds"], "timeout_seconds", spec.timeout_seconds)
    _exact_int(record["returncode"], "returncode", 0)
    _exact_bool(record["passed"], "passed", True)
    for stream_name in ("stdout", "stderr"):
        rendered = record[stream_name]
        if not isinstance(rendered, str):
            raise ValueError(f"{spec.component_id}.{stream_name} must be text")
        encoded = rendered.encode("utf-8")
        if len(encoded) > _MAX_PHASE2_COMMAND_OUTPUT_BYTES:
            raise ValueError(f"{spec.component_id}.{stream_name} is oversized")
        _exact_int(
            record[f"{stream_name}_bytes"],
            f"{stream_name}_bytes",
            len(encoded),
        )
        _sha256(record[f"{stream_name}_sha256"], f"{stream_name}_sha256")
        if record[f"{stream_name}_sha256"] != hashlib.sha256(encoded).hexdigest():
            raise ValueError(f"{spec.component_id}.{stream_name} digest mismatch")
    if int(record["stdout_bytes"]) + int(record["stderr_bytes"]) > (
        _MAX_PHASE2_COMMAND_OUTPUT_BYTES
    ):
        raise ValueError(f"{spec.component_id} combined output is oversized")
    expected_checks = _phase2_component_checks(spec, record)
    if _exact_int(record["checks_completed"], "checks_completed") != expected_checks:
        raise ValueError(f"{spec.component_id} check count changed")


def _run_phase2_validation_commands(
    project_root: Path,
    *,
    expected_identity: _ValidationIdentity,
    expected_image_source_sha256: str,
) -> dict[str, Any]:
    """Run every non-component Phase-2 command against one source identity."""

    identity_before, image_before = _current_identity_and_image(project_root)
    if (
        identity_before != expected_identity
        or image_before != expected_image_source_sha256
    ):
        raise RuntimeError("local validation identity changed before Phase-2 validation")
    metrics_before = _phase2_image_metrics(project_root)
    records: list[dict[str, Any]] = []
    total_output_bytes = 0
    for spec in _phase2_command_specs():
        record = _execute_phase2_component(spec, project_root=project_root)
        total_output_bytes += int(record["stdout_bytes"]) + int(
            record["stderr_bytes"]
        )
        if total_output_bytes > _MAX_PHASE2_TOTAL_OUTPUT_BYTES:
            raise ValueError("Phase-2 command matrix exceeds its total output limit")
        _validate_phase2_component_record(
            spec,
            record,
            project_root=project_root,
        )
        current_identity, current_image = _current_identity_and_image(project_root)
        if current_identity != identity_before or current_image != image_before:
            raise RuntimeError(
                "local validation identity changed during Phase-2 component "
                f"{spec.component_id}"
            )
        if _phase2_image_metrics(project_root) != metrics_before:
            raise RuntimeError(
                f"image or dependency identity changed during {spec.component_id}"
            )
        records.append(record)
    identity_after, image_after = _current_identity_and_image(project_root)
    if identity_after != identity_before or image_after != image_before:
        raise RuntimeError("local validation identity changed during Phase-2 validation")
    return {
        **LOCAL_PHASE2_VALIDATION_RECEIPT_CONTRACT,
        **_identity_receipt_fields(identity_before),
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_revision": identity_before.source_revision,
        "source_tree_sha256": identity_before.source_manifest_sha256,
        "source_tree_sha256_before_commands": identity_before.source_manifest_sha256,
        "source_tree_sha256_after_commands": identity_after.source_manifest_sha256,
        "image_source_sha256": image_before,
        **metrics_before,
        "mandatory_component_ids": list(MANDATORY_PHASE2_VALIDATION_COMPONENTS),
        "component_receipt_coverage": dict(_PHASE2_COMPONENT_RECEIPT_COVERAGE),
        "executed_components": records,
        **_action_accounting(),
    }


def validate_local_engineering_receipt(
    level_name: str,
    receipt_path: str | Path,
    *,
    root: str | Path = ROOT,
    expected_source_tree_sha256: str | None = None,
    expected_image_source_sha256: str | None = None,
    _validation_identity: _ValidationIdentity | None = None,
) -> str:
    """Revalidate one component against the exact current source and image."""

    contract = LOCAL_ENGINEERING_RECEIPT_CONTRACTS.get(level_name)
    if contract is None:
        raise ValueError(f"unknown local engineering level: {level_name}")
    project_root = Path(os.path.abspath(os.fspath(root)))
    identity, current_image = _current_identity_and_image(
        project_root,
        validation_identity=_validation_identity,
    )
    current_source = identity.source_manifest_sha256
    if (
        expected_source_tree_sha256 is not None
        and current_source != expected_source_tree_sha256
    ):
        raise ValueError("current local source differs from the expected freeze")
    if (
        expected_image_source_sha256 is not None
        and current_image != expected_image_source_sha256
    ):
        raise ValueError("current image source differs from the expected freeze")
    _ensure_identity_manifests(project_root, identity, create_missing=False)
    expected_logical = local_engineering_receipt_path(
        level_name,
        validation_identity_digest=identity.validation_identity_sha256,
    )
    expected_receipt = project_root / expected_logical
    if not _same_lexical_absolute_path(receipt_path, expected_receipt):
        raise ValueError("local engineering receipt path differs from its freeze")
    receipt_bytes = _read_contained_bytes(
        project_root,
        expected_logical.as_posix(),
        "receipt_path",
    )
    payload = _load_exact_json_object(
        receipt_bytes,
        field="local engineering receipt",
    )
    if not isinstance(payload, dict) or set(payload) != _RECEIPT_FIELDS:
        raise ValueError("local engineering receipt has an invalid exact schema")
    if any(
        payload[field] != expected
        for field, expected in contract["receipt_contract"].items()
    ):
        raise ValueError("local engineering receipt has the wrong schema contract")
    _valid_utc(payload["recorded_at_utc"])
    _validate_identity_receipt_fields(payload, identity)
    revision = identity.source_revision
    if payload["source_revision"] != revision:
        raise ValueError("local engineering receipt belongs to another revision")
    for field in (
        "source_tree_sha256",
        "source_tree_sha256_before_command",
        "source_tree_sha256_after_command",
    ):
        _sha256(payload[field], field)
        if payload[field] != current_source:
            raise ValueError(f"local engineering receipt {field} has changed")
    _sha256(payload["image_source_sha256"], "image_source_sha256")
    if payload["image_source_sha256"] != current_image:
        raise ValueError("local engineering receipt image source has changed")
    command = frozen_local_engineering_command(
        level_name,
        validation_identity_digest=identity.validation_identity_sha256,
    )
    if payload["command"] != command:
        raise ValueError("local engineering receipt command differs from its freeze")
    _sha256(payload["command_sha256"], "command_sha256")
    if payload["command_sha256"] != content_hash(command):
        raise ValueError("local engineering command digest mismatch")
    expected_environment = _minimal_execution_environment(project_root)
    expected_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    expected_environment = dict(sorted(expected_environment.items()))
    expected_resolved_command = [os.path.abspath(sys.executable), *command[1:]]
    if payload["resolved_command"] != expected_resolved_command:
        raise ValueError("local engineering resolved command changed")
    _sha256(payload["resolved_command_sha256"], "resolved_command_sha256")
    if payload["resolved_command_sha256"] != content_hash(expected_resolved_command):
        raise ValueError("local engineering resolved command digest mismatch")
    if payload["execution_environment"] != expected_environment:
        raise ValueError("local engineering exact execution environment changed")
    _sha256(payload["execution_environment_sha256"], "execution_environment_sha256")
    if payload["execution_environment_sha256"] != content_hash(expected_environment):
        raise ValueError("local engineering execution environment digest mismatch")
    result_logical = local_engineering_result_path(
        level_name,
        validation_identity_digest=identity.validation_identity_sha256,
    ).as_posix()
    if payload["result_path"] != result_logical:
        raise ValueError("local engineering result path differs from its freeze")
    result_bytes = _read_contained_bytes(
        project_root,
        result_logical,
        "result_path",
    )
    _sha256(payload["result_sha256"], "result_sha256")
    if payload["result_sha256"] != hashlib.sha256(result_bytes).hexdigest():
        raise ValueError("local engineering result digest mismatch")
    checks = _validate_result_bytes(level_name, result_bytes)
    if level_name == "offline_smoke_tested":
        artifact_logical = (
            local_engineering_freeze_directory(identity.validation_identity_sha256)
            / OFFLINE_ARTIFACT_MANIFEST_FILENAME
        ).as_posix()
        if payload["artifact_manifest_path"] != artifact_logical:
            raise ValueError("offline artifact manifest path changed")
        observed_artifact_manifest = _load_exact_json_object(
            _read_contained_bytes(
                project_root,
                artifact_logical,
                "offline_artifact_manifest_path",
            ),
            field="offline artifact manifest",
        )
        expected_artifact_manifest, artifact_checks = _validate_offline_artifact_tree(
            project_root,
            identity.validation_identity_sha256,
            result_bytes,
        )
        if observed_artifact_manifest != expected_artifact_manifest:
            raise ValueError("offline artifact manifest changed")
        _sha256(payload["artifact_manifest_sha256"], "artifact_manifest_sha256")
        if payload["artifact_manifest_sha256"] != content_hash(
            expected_artifact_manifest
        ):
            raise ValueError("offline artifact manifest digest mismatch")
        if artifact_checks != checks:
            raise ValueError("offline artifact run count differs from stdout")
    elif (
        payload["artifact_manifest_path"] is not None
        or payload["artifact_manifest_sha256"] is not None
    ):
        raise ValueError("unit receipt may not claim an artifact manifest")
    if _exact_int(payload["checks_completed"], "checks_completed") != checks:
        raise ValueError("local engineering check count differs from its result")
    _validate_action_accounting(payload)
    return (
        f"{expected_receipt} revision={revision} source_tree={current_source} "
        f"image_source={current_image} checks={checks}"
    )


def validate_phase2_validation_receipt(
    receipt_path: str | Path | None = None,
    *,
    root: str | Path = ROOT,
    expected_source_tree_sha256: str | None = None,
    expected_image_source_sha256: str | None = None,
    _validation_identity: _ValidationIdentity | None = None,
) -> dict[str, Any]:
    """Revalidate the exact local Phase-2 command matrix and its outputs."""

    project_root = Path(os.path.abspath(os.fspath(root)))
    identity, current_image = _current_identity_and_image(
        project_root,
        validation_identity=_validation_identity,
    )
    current_source = identity.source_manifest_sha256
    if (
        expected_source_tree_sha256 is not None
        and current_source != expected_source_tree_sha256
    ):
        raise ValueError("current local source differs from the Phase-2 freeze")
    if (
        expected_image_source_sha256 is not None
        and current_image != expected_image_source_sha256
    ):
        raise ValueError("current image source differs from the Phase-2 freeze")
    _ensure_identity_manifests(project_root, identity, create_missing=False)
    expected_logical = local_phase2_validation_receipt_path(
        identity.validation_identity_sha256
    )
    expected = project_root / expected_logical
    if receipt_path is not None and not _same_lexical_absolute_path(
        receipt_path,
        expected,
    ):
        raise ValueError("Phase-2 validation receipt path is not current")
    payload = _load_exact_json_object(
        _read_contained_bytes(
            project_root,
            expected_logical.as_posix(),
            "phase2_validation_receipt_path",
        ),
        field="Phase-2 validation receipt",
    )
    if not isinstance(payload, dict) or set(payload) != _PHASE2_RECEIPT_FIELDS:
        raise ValueError("Phase-2 validation receipt has an invalid exact schema")
    if any(
        payload[field] != value
        for field, value in LOCAL_PHASE2_VALIDATION_RECEIPT_CONTRACT.items()
    ):
        raise ValueError("Phase-2 validation receipt has the wrong contract")
    _valid_utc(payload["recorded_at_utc"])
    _validate_identity_receipt_fields(payload, identity)
    if payload["source_revision"] != identity.source_revision:
        raise ValueError("Phase-2 validation receipt belongs to another revision")
    for field in (
        "source_tree_sha256",
        "source_tree_sha256_before_commands",
        "source_tree_sha256_after_commands",
    ):
        _sha256(payload[field], field)
        if payload[field] != current_source:
            raise ValueError(f"Phase-2 validation {field} has changed")
    _sha256(payload["image_source_sha256"], "image_source_sha256")
    if payload["image_source_sha256"] != current_image:
        raise ValueError("Phase-2 validation image source has changed")
    metrics = _phase2_image_metrics(project_root)
    _sha256(payload["dependency_lock_sha256"], "dependency_lock_sha256")
    if payload["dependency_lock_sha256"] != metrics["dependency_lock_sha256"]:
        raise ValueError("Phase-2 dependency-lock identity has changed")
    for field in (
        "image_source_file_count",
        "image_source_total_bytes",
        "image_source_two_copy_upper_bound_bytes",
    ):
        observed = _exact_int(payload[field], field)
        if observed != metrics[field]:
            raise ValueError(f"Phase-2 {field} has changed")
    if payload["image_source_file_count"] <= 0:
        raise ValueError("Phase-2 image source must contain files")
    if payload["image_source_two_copy_upper_bound_bytes"] != (
        2 * payload["image_source_total_bytes"]
    ):
        raise ValueError("Phase-2 two-copy image bound is invalid")
    if payload["mandatory_component_ids"] != list(
        MANDATORY_PHASE2_VALIDATION_COMPONENTS
    ):
        raise ValueError("Phase-2 mandatory component roster changed")
    if payload["component_receipt_coverage"] != (
        _PHASE2_COMPONENT_RECEIPT_COVERAGE
    ):
        raise ValueError("Phase-2 component-receipt coverage changed")
    specs = _phase2_command_specs()
    records = payload["executed_components"]
    if not isinstance(records, list) or len(records) != len(specs):
        raise ValueError("Phase-2 executed component roster is incomplete")
    total_output_bytes = 0
    for spec, record in zip(specs, records, strict=True):
        _validate_phase2_component_record(
            spec,
            record,
            project_root=project_root,
        )
        total_output_bytes += int(record["stdout_bytes"]) + int(
            record["stderr_bytes"]
        )
    if total_output_bytes > _MAX_PHASE2_TOTAL_OUTPUT_BYTES:
        raise ValueError("Phase-2 receipt exceeds its total output limit")
    covered = {
        *(spec.component_id for spec in specs),
        *_PHASE2_COMPONENT_RECEIPT_COVERAGE,
    }
    if covered != set(MANDATORY_PHASE2_VALIDATION_COMPONENTS):
        raise ValueError("Phase-2 validation does not cover every mandatory gate")
    _validate_action_accounting(payload)
    return payload


def _component_receipt_records(
    project_root: Path,
    *,
    identity: _ValidationIdentity,
    image_digest: str,
) -> list[dict[str, str]]:
    source_digest = identity.source_manifest_sha256
    records: list[dict[str, str]] = []
    for level_name in ("unit_tested", "offline_smoke_tested"):
        logical = local_engineering_receipt_path(
            level_name,
            validation_identity_digest=identity.validation_identity_sha256,
        )
        validate_local_engineering_receipt(
            level_name,
            project_root / logical,
            root=project_root,
            expected_source_tree_sha256=source_digest,
            expected_image_source_sha256=image_digest,
            _validation_identity=identity,
        )
        records.append(
            {
                "level_name": level_name,
                "path": logical.as_posix(),
                "sha256": hashlib.sha256(
                    _read_contained_bytes(
                        project_root,
                        logical.as_posix(),
                        f"{level_name}.receipt_path",
                    )
                ).hexdigest(),
            }
        )
    phase2_logical = local_phase2_validation_receipt_path(
        identity.validation_identity_sha256
    )
    validate_phase2_validation_receipt(
        project_root / phase2_logical,
        root=project_root,
        expected_source_tree_sha256=source_digest,
        expected_image_source_sha256=image_digest,
        _validation_identity=identity,
    )
    records.append(
        {
            "level_name": "phase2_validated",
            "path": phase2_logical.as_posix(),
            "sha256": hashlib.sha256(
                _read_contained_bytes(
                    project_root,
                    phase2_logical.as_posix(),
                    "phase2_validated.receipt_path",
                )
            ).hexdigest(),
        }
    )
    return records


def validate_local_engineering_freeze_receipt(
    receipt_path: str | Path | None = None,
    *,
    root: str | Path = ROOT,
    expected_image_source_sha256: str | None = None,
    _validation_identity: _ValidationIdentity | None = None,
) -> dict[str, Any]:
    """Revalidate the aggregate and all three raw component bindings."""

    project_root = Path(os.path.abspath(os.fspath(root)))
    identity, image_digest = _current_identity_and_image(
        project_root,
        validation_identity=_validation_identity,
    )
    source_digest = identity.source_manifest_sha256
    if (
        expected_image_source_sha256 is not None
        and image_digest != expected_image_source_sha256
    ):
        raise ValueError("current image source differs from the approved launch")
    _ensure_identity_manifests(project_root, identity, create_missing=False)
    expected_logical = local_engineering_freeze_receipt_path(
        identity.validation_identity_sha256
    )
    expected = project_root / expected_logical
    if receipt_path is not None and not _same_lexical_absolute_path(
        receipt_path,
        expected,
    ):
        raise ValueError("local engineering freeze receipt path is not current")
    payload = _load_exact_json_object(
        _read_contained_bytes(
            project_root,
            expected_logical.as_posix(),
            "freeze_receipt_path",
        ),
        field="local engineering freeze receipt",
    )
    if not isinstance(payload, dict) or set(payload) != _FREEZE_RECEIPT_FIELDS:
        raise ValueError("local engineering freeze receipt has an invalid schema")
    if any(
        payload[field] != value
        for field, value in LOCAL_ENGINEERING_FREEZE_RECEIPT_CONTRACT.items()
    ):
        raise ValueError("local engineering freeze receipt has the wrong contract")
    aggregate_time = _valid_utc(payload["recorded_at_utc"])
    _validate_identity_receipt_fields(payload, identity)
    revision = identity.source_revision
    if payload["source_revision"] != revision:
        raise ValueError("local engineering freeze belongs to another revision")
    _sha256(payload["source_tree_sha256"], "source_tree_sha256")
    _sha256(payload["image_source_sha256"], "image_source_sha256")
    if payload["source_tree_sha256"] != source_digest:
        raise ValueError("local engineering freeze source tree has changed")
    if payload["image_source_sha256"] != image_digest:
        raise ValueError("local engineering freeze image source has changed")
    expected_components = _component_receipt_records(
        project_root,
        identity=identity,
        image_digest=image_digest,
    )
    if payload["component_receipts"] != expected_components:
        raise ValueError("local engineering freeze component bindings changed")
    for record in expected_components:
        component = _load_exact_json_object(
            _read_contained_bytes(
                project_root,
                record["path"],
                f"{record['level_name']}.receipt_path",
            ),
            field=f"{record['level_name']} receipt",
        )
        if aggregate_time < _valid_utc(
            component["recorded_at_utc"],
            f"{record['level_name']}.recorded_at_utc",
        ):
            raise ValueError("local engineering freeze predates a component receipt")
    _validate_action_accounting(payload)
    return payload


def historical_local_engineering_freeze_predecessor_bindings(
    bindings: list[dict[str, str]] | tuple[dict[str, str], ...],
    *,
    root: str | Path = ROOT,
    expected_image_source_sha256: str | None = None,
) -> tuple[dict[str, str], ...]:
    """Revalidate launch-time freeze bytes while permitting docs-only drift.

    The three validation-input documents rotate the freeze namespace before a
    launch, but edits to those documents after a durable action intent must not
    invalidate historical accounting. Execution source, dependency lock,
    image, Git revision, and stripped execution environment remain current.
    """

    if not isinstance(bindings, (list, tuple)) or len(bindings) != 3:
        raise ValueError("historical local freeze binding roster is incomplete")
    for index, (gate, record) in enumerate(
        zip(LOCAL_ENGINEERING_FREEZE_GATES, bindings, strict=True)
    ):
        if (
            not isinstance(record, dict)
            or set(record) != {"gate", "path", "sha256"}
            or record["gate"] != gate
        ):
            raise ValueError(
                f"historical local freeze binding {index} is invalid"
            )
        _sha256(record["sha256"], f"historical binding {index}.sha256")

    project_root = Path(os.path.abspath(os.fspath(root)))
    aggregate_binding = bindings[2]
    aggregate_relative = safe_relative_path(aggregate_binding["path"])
    if (
        aggregate_relative.name != LOCAL_ENGINEERING_FREEZE_RECEIPT_FILENAME
        or aggregate_relative.parent.parent != LOCAL_ENGINEERING_FREEZE_ROOT
    ):
        raise ValueError("historical local freeze aggregate path is not canonical")
    validation_identity_sha256 = _sha256(
        aggregate_relative.parent.name,
        "historical validation identity",
    )
    expected_aggregate = local_engineering_freeze_receipt_path(
        validation_identity_sha256
    ).as_posix()
    if aggregate_binding["path"] != expected_aggregate:
        raise ValueError("historical local freeze aggregate path changed")
    aggregate_raw = _read_contained_bytes(
        project_root,
        expected_aggregate,
        "historical_freeze_receipt_path",
    )
    if hashlib.sha256(aggregate_raw).hexdigest() != aggregate_binding["sha256"]:
        raise ValueError("historical local freeze aggregate bytes changed")
    aggregate = _load_exact_json_object(
        aggregate_raw,
        field="historical local engineering freeze receipt",
    )
    if not isinstance(aggregate, dict):
        raise ValueError("historical local freeze receipt is not an object")

    directory = local_engineering_freeze_directory(validation_identity_sha256)
    validation_logical = (directory / VALIDATION_INPUT_MANIFEST_FILENAME).as_posix()
    if aggregate.get("validation_input_manifest_path") != validation_logical:
        raise ValueError("historical validation-input manifest path changed")
    validation_raw = _read_contained_bytes(
        project_root,
        validation_logical,
        "historical_validation_input_manifest_path",
    )
    validation_manifest = _load_exact_json_object(
        validation_raw,
        field="historical validation-input manifest",
    )
    if not isinstance(validation_manifest, dict) or set(validation_manifest) != {
        "schema_name",
        "schema_version",
        "files",
        "file_count",
        "total_bytes",
    }:
        raise ValueError("historical validation-input manifest schema changed")
    if any(
        validation_manifest[field] != expected
        for field, expected in VALIDATION_INPUT_MANIFEST_CONTRACT.items()
    ):
        raise ValueError("historical validation-input manifest contract changed")
    files = validation_manifest["files"]
    if (
        not isinstance(files, list)
        or len(files) != len(_VALIDATION_INPUT_FILES)
        or validation_manifest["file_count"] != len(files)
        or any(
            not isinstance(record, dict)
            or set(record) != {"relative_path", "sha256", "size_bytes"}
            for record in files
        )
    ):
        raise ValueError("historical validation-input manifest contents changed")
    if (
        [record["relative_path"] for record in files]
        != sorted(
            f"architecture_dynamic/{name}" for name in _VALIDATION_INPUT_FILES
        )
        or any(
            not isinstance(record["size_bytes"], int)
            or isinstance(record["size_bytes"], bool)
            or record["size_bytes"] < 0
            or _SHA256_TEXT.fullmatch(str(record["sha256"])) is None
            for record in files
        )
        or validation_manifest["total_bytes"]
        != sum(record["size_bytes"] for record in files)
    ):
        raise ValueError("historical validation-input manifest contents changed")
    validation_digest = content_hash(validation_manifest)
    if (
        aggregate.get("validation_input_manifest_sha256")
        != validation_digest
    ):
        raise ValueError("historical validation-input manifest digest changed")

    source_manifest = execution_source_manifest(project_root)
    environment_manifest = execution_environment_manifest(project_root)
    source_digest = content_hash(source_manifest)
    environment_digest = content_hash(environment_manifest)
    revision = _git_revision(project_root)
    if aggregate.get("source_tree_sha256") != source_digest:
        raise ValueError("historical local freeze execution source changed")
    if aggregate.get("source_revision") != revision:
        raise ValueError("historical local freeze Git revision changed")
    if (
        aggregate.get("execution_environment_manifest_sha256")
        != environment_digest
    ):
        raise ValueError("historical local freeze execution environment changed")
    identity = _ValidationIdentity(
        validation_identity_sha256=content_hash(
            {
                "schema_name": "LocalValidationIdentity",
                "schema_version": "1.0",
                "source_tree_sha256": source_digest,
                "source_revision": revision,
                "validation_input_manifest_sha256": validation_digest,
                "execution_environment_manifest_sha256": environment_digest,
            }
        ),
        source_revision=revision,
        source_manifest=source_manifest,
        source_manifest_sha256=source_digest,
        validation_input_manifest=validation_manifest,
        validation_input_manifest_sha256=validation_digest,
        execution_environment_manifest=environment_manifest,
        execution_environment_manifest_sha256=environment_digest,
    )
    if identity.validation_identity_sha256 != validation_identity_sha256:
        raise ValueError("historical local freeze identity no longer reconciles")
    payload = validate_local_engineering_freeze_receipt(
        project_root / expected_aggregate,
        root=project_root,
        expected_image_source_sha256=expected_image_source_sha256,
        _validation_identity=identity,
    )
    records_by_level = {
        record["level_name"]: record for record in payload["component_receipts"]
    }
    if set(records_by_level) != {
        "unit_tested",
        "offline_smoke_tested",
        "phase2_validated",
    }:
        raise ValueError("historical local freeze component roster changed")
    expected_bindings = (
        {
            "gate": LOCAL_ENGINEERING_FREEZE_GATES[0],
            "path": records_by_level["unit_tested"]["path"],
            "sha256": records_by_level["unit_tested"]["sha256"],
        },
        {
            "gate": LOCAL_ENGINEERING_FREEZE_GATES[1],
            "path": records_by_level["offline_smoke_tested"]["path"],
            "sha256": records_by_level["offline_smoke_tested"]["sha256"],
        },
        dict(aggregate_binding),
    )
    if tuple(bindings) != expected_bindings:
        raise ValueError("historical local freeze predecessor bindings changed")
    return expected_bindings


def local_engineering_freeze_predecessor_bindings(
    *,
    root: str | Path = ROOT,
    expected_image_source_sha256: str | None = None,
) -> tuple[dict[str, str], ...]:
    """Return exact current component-plus-aggregate paid-action bindings."""

    project_root = Path(os.path.abspath(os.fspath(root)))
    payload = validate_local_engineering_freeze_receipt(
        root=project_root,
        expected_image_source_sha256=expected_image_source_sha256,
    )
    records = payload["component_receipts"]
    records_by_level = {record["level_name"]: record for record in records}
    if set(records_by_level) != {
        "unit_tested",
        "offline_smoke_tested",
        "phase2_validated",
    }:
        raise ValueError("local engineering component roster changed")
    bindings = [
        {
            "gate": gate,
            "path": record["path"],
            "sha256": record["sha256"],
        }
        for gate, record in zip(
            LOCAL_ENGINEERING_FREEZE_GATES[:2],
            (
                records_by_level["unit_tested"],
                records_by_level["offline_smoke_tested"],
            ),
            strict=True,
        )
    ]
    aggregate_logical = local_engineering_freeze_receipt_path(
        payload["validation_identity_sha256"]
    ).as_posix()
    bindings.append(
        {
            "gate": LOCAL_ENGINEERING_FREEZE_GATES[2],
            "path": aggregate_logical,
            "sha256": hashlib.sha256(
                _read_contained_bytes(
                    project_root,
                    aggregate_logical,
                    "freeze_receipt_path",
                )
            ).hexdigest(),
        }
    )
    return tuple(bindings)


def record_local_engineering_freeze_receipt(
    *,
    root: str | Path = ROOT,
    run_command: bool = False,
) -> dict[str, Any]:
    """Run/revalidate Phase 2 and create the immutable aggregate."""

    if not run_command:
        raise ValueError(
            "aggregate requires --run-command for explicit Phase-2 validation"
        )
    project_root = Path(os.path.abspath(os.fspath(root)))
    identity, image_digest = _current_identity_and_image(project_root)
    source_digest = identity.source_manifest_sha256
    _ensure_identity_manifests(project_root, identity, create_missing=True)
    output = project_root / local_engineering_freeze_receipt_path(
        identity.validation_identity_sha256
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"local engineering freeze receipt already exists: {output}"
        )
    for level_name in ("unit_tested", "offline_smoke_tested"):
        validate_local_engineering_receipt(
            level_name,
            project_root
            / local_engineering_receipt_path(
                level_name,
                validation_identity_digest=identity.validation_identity_sha256,
            ),
            root=project_root,
            expected_source_tree_sha256=source_digest,
            expected_image_source_sha256=image_digest,
        )
    phase2_logical = local_phase2_validation_receipt_path(
        identity.validation_identity_sha256
    )
    phase2_output = project_root / phase2_logical
    if phase2_output.exists() or phase2_output.is_symlink():
        validate_phase2_validation_receipt(
            phase2_output,
            root=project_root,
            expected_source_tree_sha256=source_digest,
            expected_image_source_sha256=image_digest,
        )
    else:
        phase2_payload = _run_phase2_validation_commands(
            project_root,
            expected_identity=identity,
            expected_image_source_sha256=image_digest,
        )
        create_json_exclusive(phase2_output, phase2_payload)
        validate_phase2_validation_receipt(
            phase2_output,
            root=project_root,
            expected_source_tree_sha256=source_digest,
            expected_image_source_sha256=image_digest,
        )
    components = _component_receipt_records(
        project_root,
        identity=identity,
        image_digest=image_digest,
    )
    identity_after, image_after = _current_identity_and_image(project_root)
    if identity_after != identity or image_after != image_digest:
        raise RuntimeError(
            "local validation identity changed while aggregating freeze evidence"
        )
    payload = {
        **LOCAL_ENGINEERING_FREEZE_RECEIPT_CONTRACT,
        **_identity_receipt_fields(identity),
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_revision": identity.source_revision,
        "source_tree_sha256": source_digest,
        "image_source_sha256": image_digest,
        "component_receipts": components,
        **_action_accounting(),
    }
    create_json_exclusive(output, payload)
    validate_local_engineering_freeze_receipt(output, root=project_root)
    return payload


def record_local_engineering_evidence(
    level_name: str,
    *,
    root: str | Path = ROOT,
    run_command: bool = False,
) -> dict[str, Any]:
    """Run one frozen command and create its immutable versioned component."""

    contract = LOCAL_ENGINEERING_RECEIPT_CONTRACTS.get(level_name)
    if contract is None:
        raise ValueError(f"unknown local engineering level: {level_name}")
    if not run_command:
        raise ValueError(
            "versioned local engineering evidence requires --run-command so "
            "source identity can be checked before and after execution"
        )
    project_root = Path(os.path.abspath(os.fspath(root)))
    identity_before, image_before = _current_identity_and_image(project_root)
    source_before = identity_before.source_manifest_sha256
    validation_digest = identity_before.validation_identity_sha256
    _ensure_identity_manifests(project_root, identity_before, create_missing=True)
    receipt_logical = local_engineering_receipt_path(
        level_name,
        validation_identity_digest=validation_digest,
    )
    receipt_output = project_root / receipt_logical
    if receipt_output.exists() or receipt_output.is_symlink():
        raise FileExistsError(
            f"local engineering receipt already exists: {receipt_output}"
        )
    command_evidence = _run_frozen_command(
        level_name,
        project_root=project_root,
        expected_identity=identity_before,
    )
    identity_after, image_after = _current_identity_and_image(project_root)
    if (
        command_evidence["source_tree_sha256_before_command"] != source_before
        or command_evidence["source_tree_sha256_after_command"] != source_before
        or identity_after != identity_before
        or image_after != image_before
    ):
        raise RuntimeError(
            "local validation identity or image changed during frozen validation"
        )
    result_logical = local_engineering_result_path(
        level_name,
        validation_identity_digest=validation_digest,
    )
    result_bytes = _read_contained_bytes(
        project_root,
        result_logical.as_posix(),
        "result_path",
    )
    checks = _validate_result_bytes(level_name, result_bytes)
    command = frozen_local_engineering_command(
        level_name,
        validation_identity_digest=validation_digest,
    )
    artifact_manifest = command_evidence["artifact_manifest"]
    if artifact_manifest is None:
        artifact_manifest_path: str | None = None
        artifact_manifest_sha256: str | None = None
    else:
        artifact_manifest_logical = (
            local_engineering_freeze_directory(validation_digest)
            / OFFLINE_ARTIFACT_MANIFEST_FILENAME
        )
        create_json_exclusive(
            project_root / artifact_manifest_logical,
            artifact_manifest,
        )
        artifact_manifest_path = artifact_manifest_logical.as_posix()
        artifact_manifest_sha256 = content_hash(artifact_manifest)
    payload = {
        **contract["receipt_contract"],
        **_identity_receipt_fields(identity_before),
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_revision": identity_before.source_revision,
        "source_tree_sha256": source_before,
        "source_tree_sha256_before_command": command_evidence[
            "source_tree_sha256_before_command"
        ],
        "source_tree_sha256_after_command": command_evidence[
            "source_tree_sha256_after_command"
        ],
        "image_source_sha256": image_before,
        "command": command,
        "command_sha256": content_hash(command),
        "resolved_command": command_evidence["resolved_command"],
        "resolved_command_sha256": content_hash(
            command_evidence["resolved_command"]
        ),
        "execution_environment": command_evidence["execution_environment"],
        "execution_environment_sha256": content_hash(
            command_evidence["execution_environment"]
        ),
        "result_path": result_logical.as_posix(),
        "result_sha256": hashlib.sha256(result_bytes).hexdigest(),
        "artifact_manifest_path": artifact_manifest_path,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "checks_completed": checks,
        **_action_accounting(),
    }
    create_json_exclusive(receipt_output, payload)
    validate_local_engineering_receipt(
        level_name,
        receipt_output,
        root=project_root,
        expected_source_tree_sha256=source_before,
        expected_image_source_sha256=image_before,
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record source-versioned local engineering freeze evidence."
    )
    parser.add_argument(
        "level",
        choices=(*LOCAL_ENGINEERING_RECEIPT_CONTRACTS, "aggregate"),
    )
    parser.add_argument(
        "--run-command",
        action="store_true",
        help=(
            "run the frozen provider-free command or complete Phase-2 matrix "
            "before creating its receipt"
        ),
    )
    arguments = parser.parse_args()
    if arguments.level == "aggregate":
        payload = record_local_engineering_freeze_receipt(
            run_command=arguments.run_command
        )
    else:
        payload = record_local_engineering_evidence(
            arguments.level,
            run_command=arguments.run_command,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
