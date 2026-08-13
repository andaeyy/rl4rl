#!/usr/bin/env python3
"""Record and revalidate a completed full-profile Modal/CUDA training run.

The command performs no training, provider request, Modal call, or network
operation. It reads an already-downloaded run, verifies every bound artifact,
and creates one immutable readiness receipt.
"""

# The direct-script entrypoint adds the project root before local imports.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.candidate_artifact import inspect_candidate_artifact
from common.device import AcceleratorFingerprint
from common.runtime_context import ExecutionContextV1
from common.training_config import FULL_TRAIN_CUDA_V2, TrainingSeedBundle
from modal_boundary import (
    ImageSourceManifestV1,
    SourceFileV1,
    canonical_sha256,
    safe_relative_path,
    volume_artifact_uri,
)
from study.serialization import create_json_exclusive

SCHEMA_NAME = "AcceleratorValidationEvidence"
SCHEMA_VERSION = "2.0"
EXPECTED_PROFILE_HASH = (
    "fb245e5d7eedc85b9fb79788c4a372e71c376d2283633103fd393b0d9ca1f70f"
)
_RUNTIME_HASH_FIELDS = frozenset(
    {
        "execution_context.json",
        "cuda_environment.json",
        "image_source_manifest.json",
    }
)
_DETERMINISM_FIELDS = frozenset(
    {
        "deterministic_algorithms",
        "cublas_workspace_config",
        "cudnn_deterministic",
        "cudnn_benchmark",
        "allow_tf32",
    }
)
EVIDENCE_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "recorded_at_utc",
        "code_revision",
        "dependency_lock_hash",
        "image_source_hash",
        "modal_image_id",
        "execution_backend",
        "artifact_root",
        "artifact_uri",
        "training_output_relative_path",
        "modal_app_name",
        "modal_function_name",
        "modal_app_id",
        "modal_function_id",
        "modal_call_id",
        "requested_gpu_kind",
        "observed_gpu_kind",
        "observed_gpu_name",
        "observed_gpu_count",
        "observed_gpu_compute_capability",
        "training_profile_name",
        "training_profile_version",
        "training_profile_hash",
        "candidate_artifact_hash",
        "candidate_graph_hash",
        "seed_bundle_hash",
        "training_manifest_hash",
        "training_summary_hash",
        "checkpoint_hash",
        "event_log_hash",
        "runtime_artifact_hashes",
        "deterministic_settings",
        "success",
        "scientific",
        "hardware_matched",
        "unsupported_operation_fallback",
        "cleanup_completed",
        "steps_completed",
    }
)


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular artifact is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return payload


def _exact_bool(payload: Mapping[str, Any], field: str, expected: bool) -> None:
    value = payload.get(field)
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be non-empty text or null")
    return value


def _relative_path(value: object, field: str, *, allow_dot: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    if allow_dot and value == ".":
        return value
    try:
        return safe_relative_path(value).as_posix()
    except ValueError as error:
        raise ValueError(
            f"{field} must be normalized, relative, and non-traversing"
        ) from error


def _contained_file(root: Path, relative: object, field: str) -> Path:
    logical = _relative_path(relative, field)
    path = root / PurePosixPath(logical)
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if (
        resolved_path.parent != resolved_root
        and resolved_root not in resolved_path.parents
    ):
        raise ValueError(f"{field} escapes the training artifact root")
    cursor = resolved_root
    for component in PurePosixPath(logical).parts:
        cursor /= component
        if cursor.is_symlink():
            raise ValueError(f"{field} may not traverse symbolic links")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return path


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not revision:
        return None
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        return None
    return revision


def _image_source_manifest(payload: dict[str, Any]) -> ImageSourceManifestV1:
    required = {
        "schema_name",
        "schema_version",
        "recipe_version",
        "python_version",
        "uv_version",
        "modal_version",
        "dependency_lock_sha256",
        "files",
    }
    if set(payload) != required:
        raise ValueError("image source manifest has unexpected or missing fields")
    if payload["schema_name"] != ImageSourceManifestV1.SCHEMA_NAME:
        raise ValueError("image source manifest has the wrong schema")
    if payload["schema_version"] != ImageSourceManifestV1.SCHEMA_VERSION:
        raise ValueError("image source manifest has an unsupported version")
    files = payload["files"]
    if not isinstance(files, list):
        raise ValueError("image source manifest files must be a list")
    entries: list[SourceFileV1] = []
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != SourceFileV1.FIELDS:
            raise ValueError("image source entry has unexpected or missing fields")
        relative_path = entry["relative_path"]
        sha256 = entry["sha256"]
        if not isinstance(relative_path, str) or not isinstance(sha256, str):
            raise ValueError("image source path and digest must be text")
        entries.append(
            SourceFileV1(
                relative_path=relative_path,
                sha256=sha256,
                size_bytes=_exact_int(entry["size_bytes"], "image source size"),
            )
        )
    manifest = ImageSourceManifestV1(
        dependency_lock_sha256=_sha256(
            payload["dependency_lock_sha256"], "dependency_lock_sha256"
        ),
        files=tuple(entries),
        recipe_version=str(payload["recipe_version"]),
        python_version=str(payload["python_version"]),
        uv_version=str(payload["uv_version"]),
        modal_version=str(payload["modal_version"]),
    )
    if manifest.to_dict() != payload:
        raise ValueError("image source manifest contains coerced field types")
    lock_entries = [
        entry for entry in manifest.files if entry.relative_path == "uv.lock"
    ]
    if len(lock_entries) != 1:
        raise ValueError("image source manifest must bind exactly one uv.lock")
    if lock_entries[0].sha256 != manifest.dependency_lock_sha256:
        raise ValueError("image source uv.lock digest differs from dependency identity")
    return manifest


def _inspect_artifacts(
    *, artifact_root: Path, training_output_dir: Path
) -> dict[str, Any]:
    if artifact_root.is_symlink() or training_output_dir.is_symlink():
        raise ValueError("artifact roots may not be symbolic links")
    root = artifact_root.resolve()
    training = training_output_dir.resolve()
    if not root.is_dir() or not training.is_dir():
        raise FileNotFoundError("artifact root and training output must be directories")
    if training != root and root not in training.parents:
        raise ValueError("training output must be contained by artifact_root")

    manifest_path = training / "training_manifest.json"
    summary_path = training / "training_summary.json"
    context_path = root / "execution_context.json"
    cuda_path = root / "cuda_environment.json"
    source_path = root / "image_source_manifest.json"
    manifest = _load_object(manifest_path)
    summary = _load_object(summary_path)
    context_payload = _load_object(context_path)
    cuda = _load_object(cuda_path)
    source_payload = _load_object(source_path)

    context = ExecutionContextV1.from_dict(context_payload)
    if context.execution_backend != "modal":
        raise ValueError("accelerator validation requires Modal execution")
    if context.app_name is None or context.function_name is None:
        raise ValueError("Modal app and function identity are required")
    if context.image_source_sha256 is None:
        raise ValueError("Modal image source identity is required")
    if context.artifact_uri is None or not context.artifact_uri.startswith("volume://"):
        raise ValueError("Modal Volume artifact identity is required")
    if context.artifact_uri != volume_artifact_uri(context.run_id):
        raise ValueError("Modal artifact URI does not identify the recorded run")

    source_manifest = _image_source_manifest(source_payload)
    image_source_hash = canonical_sha256(source_payload)
    if image_source_hash != source_manifest.manifest_sha256:
        raise ValueError("image source manifest canonical hash does not reconstruct")
    if image_source_hash != context.image_source_sha256:
        raise ValueError("execution context and image source manifest differ")

    if FULL_TRAIN_CUDA_V2.profile_hash != EXPECTED_PROFILE_HASH:
        raise RuntimeError("full_train_cuda_v2 profile hash changed unexpectedly")
    expected_profile = {
        "profile_name": FULL_TRAIN_CUDA_V2.name,
        "profile_version": FULL_TRAIN_CUDA_V2.version,
        "profile_hash": EXPECTED_PROFILE_HASH,
        "accelerator_kind": "cuda",
    }
    for field, expected in expected_profile.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"training summary {field} does not match full_train_cuda_v2"
            )
    if summary.get("schema_name") != "TrainingResult":
        raise ValueError("training summary must use the TrainingResult schema")
    if summary.get("schema_version") != "2.0":
        raise ValueError("training summary must use the portable v2 schema")
    summary_device = summary.get("device")
    if not isinstance(summary_device, str) or summary_device.split(":", 1)[0] != "cuda":
        raise ValueError("training summary did not select CUDA")
    for field, expected in {
        "success": True,
        "scientific": True,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
    }.items():
        _exact_bool(summary, field, expected)
    steps = _exact_int(summary.get("steps_completed"), "steps_completed")
    if steps != FULL_TRAIN_CUDA_V2.max_steps:
        raise ValueError("training did not complete the exact full_train_cuda_v2 steps")

    if manifest.get("profile_hash") != EXPECTED_PROFILE_HASH:
        raise ValueError(
            "training manifest profile hash does not match full_train_cuda_v2"
        )
    if manifest.get("schema_name") != "TrainingManifest":
        raise ValueError("training manifest must use the TrainingManifest schema")
    if manifest.get("schema_version") != "2.0":
        raise ValueError("training manifest must use the portable v2 schema")
    profile = manifest.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("training manifest profile must be an object")
    expected_profile_payload = FULL_TRAIN_CUDA_V2.to_dict()
    if (
        set(profile) != set(expected_profile_payload)
        or canonical_sha256(profile) != EXPECTED_PROFILE_HASH
    ):
        raise ValueError("training manifest does not contain the exact CUDA profile")
    requested_device = manifest.get("requested_device")
    if (
        not isinstance(requested_device, str)
        or requested_device.split(":", 1)[0] != "cuda"
    ):
        raise ValueError("training manifest did not request CUDA")
    selected_device = manifest.get("selected_device")
    if (
        not isinstance(selected_device, str)
        or selected_device.split(":", 1)[0] != "cuda"
    ):
        raise ValueError("training manifest did not observe CUDA")
    if selected_device != summary_device:
        raise ValueError("training manifest and summary selected different devices")
    _exact_bool(manifest, "allow_cpu_for_tests", False)
    _exact_bool(manifest, "hardware_matched_scientific_run", True)
    if manifest.get("execution_context") != context.to_dict():
        raise ValueError("training manifest execution context differs from the run")

    candidate_format = manifest.get("candidate_format")
    if candidate_format != "architecture_ir":
        raise ValueError("full accelerator evidence requires Architecture IR")
    candidate_relative = _relative_path(
        manifest.get("immutable_candidate_relative_path"),
        "immutable_candidate_relative_path",
    )
    if candidate_relative != "candidate_graph.json":
        raise ValueError("Architecture IR must use candidate_graph.json")
    if manifest.get("candidate_path") != candidate_relative:
        raise ValueError("training manifest candidate_path is not portable")
    candidate_path = _contained_file(
        training, candidate_relative, "immutable_candidate_relative_path"
    )
    candidate_hash = _sha256_file(candidate_path)
    for field in ("candidate_source_hash", "candidate_artifact_hash"):
        if _sha256(manifest.get(field), field) != candidate_hash:
            raise ValueError(f"training manifest {field} differs from the candidate")
    if (
        _sha256(summary.get("candidate_source_hash"), "candidate_source_hash")
        != candidate_hash
    ):
        raise ValueError("training summary differs from the immutable candidate")
    graph_hash = _sha256(manifest.get("candidate_graph_hash"), "candidate_graph_hash")
    candidate_inspection = inspect_candidate_artifact(candidate_path)
    if not candidate_inspection.valid:
        raise ValueError("receipt candidate is not valid Architecture IR")
    if candidate_inspection.graph_hash != graph_hash:
        raise ValueError("candidate graph hash does not reconstruct from the artifact")
    raw_seed_bundle = manifest.get("seed_bundle")
    seed_fields = {
        "model_initialization_seed",
        "training_data_seed",
        "development_set_seed",
        "dataloader_seed",
    }
    if not isinstance(raw_seed_bundle, dict) or set(raw_seed_bundle) != seed_fields:
        raise ValueError("training manifest seed_bundle has an invalid exact schema")
    seed_bundle = TrainingSeedBundle(
        **{
            field: _exact_int(raw_seed_bundle[field], f"seed_bundle.{field}")
            for field in seed_fields
        }
    )
    seed_hash = _sha256(manifest.get("seed_bundle_hash"), "seed_bundle_hash")
    if seed_bundle.bundle_hash != seed_hash:
        raise ValueError("seed bundle hash does not reconstruct from the manifest")
    for summary_field, seed_field in {
        "initialization_seed": "model_initialization_seed",
        "data_seed": "training_data_seed",
        "development_seed": "development_set_seed",
        "dataloader_seed": "dataloader_seed",
    }.items():
        if (
            _exact_int(summary.get(summary_field), summary_field)
            != raw_seed_bundle[seed_field]
        ):
            raise ValueError(
                f"training summary {summary_field} differs from seed bundle"
            )
    dependency_hash = _sha256(
        manifest.get("dependency_lock_hash"), "dependency_lock_hash"
    )
    if dependency_hash != source_manifest.dependency_lock_sha256:
        raise ValueError("training and image dependency-lock hashes differ")

    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("training manifest lacks runtime evidence")
    for field, expected in {
        "cuda_available": True,
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
    }.items():
        _exact_bool(runtime, field, expected)
    runtime_count = _exact_int(
        runtime.get("cuda_device_count"), "runtime.cuda_device_count", minimum=1
    )
    if runtime_count != 1:
        raise ValueError("accelerator validation requires exactly one visible GPU")
    if runtime.get("cublas_workspace_config") != ":4096:8":
        raise ValueError("runtime CUBLAS_WORKSPACE_CONFIG is not deterministic")
    if runtime.get("pytorch_enable_mps_fallback") not in {"", "0"}:
        raise ValueError("runtime requested unsupported fallback")
    fingerprint = runtime.get("accelerator_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("training manifest lacks accelerator fingerprint")
    if set(fingerprint) != AcceleratorFingerprint.FIELD_NAMES:
        raise ValueError("accelerator fingerprint has an invalid exact schema")
    if fingerprint.get("requested_device") != requested_device:
        raise ValueError("accelerator fingerprint requested device differs")
    if fingerprint.get("selected_device") != selected_device:
        raise ValueError("accelerator fingerprint selected device differs")
    if fingerprint.get("accelerator_kind") != "cuda":
        raise ValueError("observed accelerator kind is not CUDA")
    gpu_name = fingerprint.get("gpu_name")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise ValueError("observed GPU name is missing")
    gpu_count = _exact_int(fingerprint.get("gpu_count"), "gpu_count", minimum=1)
    if gpu_count != 1 or gpu_count != runtime_count:
        raise ValueError("GPU counts do not prove one-device execution")
    capability = fingerprint.get("compute_capability")
    if not isinstance(capability, str) or not capability:
        raise ValueError("observed GPU compute capability is missing")
    for field in ("cuda_runtime", "torch_version", "host_platform"):
        if not isinstance(fingerprint.get(field), str) or not fingerprint[field]:
            raise ValueError(f"accelerator fingerprint {field} is missing")
    driver = fingerprint.get("cuda_driver")
    if not isinstance(driver, str) or not driver.strip():
        raise ValueError("accelerator fingerprint cuda_driver is missing")
    try:
        training_fingerprint = AcceleratorFingerprint.from_dict(
            fingerprint
        ).validate_cuda(exact_gpu_count=1, require_driver=True)
    except ValueError as error:
        raise ValueError(
            f"training accelerator fingerprint is invalid: {error}"
        ) from error

    _exact_bool(cuda, "cuda_available", True)
    try:
        cuda_fingerprint = AcceleratorFingerprint.from_dict(
            cuda.get("accelerator_fingerprint")
        ).validate_cuda(exact_gpu_count=1, require_driver=True)
    except ValueError as error:
        raise ValueError(
            f"CUDA environment accelerator fingerprint is invalid: {error}"
        ) from error
    if cuda_fingerprint != training_fingerprint:
        raise ValueError(
            "CUDA environment accelerator fingerprint differs from training"
        )
    cuda_count = _exact_int(
        cuda.get("cuda_device_count"), "cuda_device_count", minimum=1
    )
    if cuda_count != 1 or cuda_count != gpu_count:
        raise ValueError("CUDA environment GPU count differs from training")
    if cuda.get("cuda_device_name") != gpu_name:
        raise ValueError("CUDA environment GPU name differs from training")
    raw_capability = cuda.get("cuda_compute_capability")
    if (
        not isinstance(raw_capability, list)
        or len(raw_capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_capability
        )
    ):
        raise ValueError("CUDA compute capability must be two exact integers")
    cuda_capability = f"{raw_capability[0]}.{raw_capability[1]}"
    if cuda_capability != capability:
        raise ValueError("CUDA compute capability differs from training")
    if cuda.get("execution_context") != context.to_dict():
        raise ValueError("CUDA environment execution context differs from the run")
    for cuda_field, fingerprint_field in {
        "cuda_runtime": "cuda_runtime",
        "cuda_driver": "cuda_driver",
        "torch": "torch_version",
        "platform": "host_platform",
    }.items():
        if cuda.get(cuda_field) != fingerprint[fingerprint_field]:
            raise ValueError(
                f"CUDA environment {cuda_field} differs from training fingerprint"
            )

    summary_fingerprint = summary.get("accelerator_fingerprint")
    if summary_fingerprint != fingerprint:
        raise ValueError(
            "training summary accelerator fingerprint differs from manifest"
        )
    checkpoint_path = _contained_file(
        training, summary.get("checkpoint_path"), "checkpoint_path"
    )
    event_path = _contained_file(
        training, summary.get("event_log_path"), "event_log_path"
    )
    checkpoint_hash = _sha256_file(checkpoint_path)
    if (
        _sha256(summary.get("checkpoint_sha256"), "checkpoint_sha256")
        != checkpoint_hash
    ):
        raise ValueError("checkpoint hash does not match the training summary")

    return {
        "dependency_lock_hash": dependency_hash,
        "image_source_hash": image_source_hash,
        "modal_image_id": context.modal_image_id,
        "execution_backend": "modal",
        "artifact_uri": context.artifact_uri,
        "modal_app_name": context.app_name,
        "modal_function_name": context.function_name,
        "modal_app_id": context.modal_app_id,
        "modal_function_id": context.modal_function_id,
        "modal_call_id": context.modal_call_id,
        "requested_gpu_kind": "cuda",
        "observed_gpu_kind": "cuda",
        "observed_gpu_name": gpu_name,
        "observed_gpu_count": gpu_count,
        "observed_gpu_compute_capability": capability,
        "training_profile_name": FULL_TRAIN_CUDA_V2.name,
        "training_profile_version": FULL_TRAIN_CUDA_V2.version,
        "training_profile_hash": EXPECTED_PROFILE_HASH,
        "candidate_artifact_hash": candidate_hash,
        "candidate_graph_hash": graph_hash,
        "seed_bundle_hash": seed_hash,
        "training_manifest_hash": _sha256_file(manifest_path),
        "training_summary_hash": _sha256_file(summary_path),
        "checkpoint_hash": checkpoint_hash,
        "event_log_hash": _sha256_file(event_path),
        "runtime_artifact_hashes": {
            "execution_context.json": _sha256_file(context_path),
            "cuda_environment.json": _sha256_file(cuda_path),
            "image_source_manifest.json": _sha256_file(source_path),
        },
        "deterministic_settings": {
            "deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
        },
        "success": True,
        "scientific": True,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
        "steps_completed": steps,
    }


def _is_project_local(path: Path) -> bool:
    resolved = path.resolve()
    return resolved == ROOT or ROOT in resolved.parents


def _logical_artifact_root(path: Path, receipt_path: Path) -> str:
    resolved = path.resolve()
    if _is_project_local(receipt_path):
        if not _is_project_local(resolved):
            raise ValueError(
                "project-local accelerator evidence may only reference "
                "project artifacts"
            )
        relative = resolved.relative_to(ROOT).as_posix()
    else:
        relative = Path(
            os.path.relpath(resolved, receipt_path.resolve().parent)
        ).as_posix()
    return _relative_path(relative, "artifact_root", allow_dot=True)


def _resolve_artifact_root(receipt_path: Path, relative: str) -> Path:
    base = ROOT if _is_project_local(receipt_path) else receipt_path.parent
    return base if relative == "." else base / relative


def _validate_code_revision(value: object) -> str | None:
    revision = _optional_text(value, "code_revision")
    if revision is not None and (
        len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("code_revision must be a lowercase Git/source digest or null")
    return revision


def record_accelerator_validation(
    *,
    training_output_dir: str | Path,
    output_path: str | Path,
    artifact_root: str | Path | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Validate downloaded artifacts and create one immutable v2 receipt."""

    raw_training = Path(training_output_dir)
    if raw_training.is_symlink():
        raise ValueError("training output may not be a symbolic link")
    training = raw_training.resolve()
    if artifact_root is None:
        candidates = (training, training.parent)
        root = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "execution_context.json").is_file()
            ),
            training,
        )
    else:
        raw_root = Path(artifact_root)
        if raw_root.is_symlink():
            raise ValueError("artifact root may not be a symbolic link")
        root = raw_root.resolve()
    output = Path(output_path).resolve()
    inspected = _inspect_artifacts(
        artifact_root=root,
        training_output_dir=training,
    )
    evidence = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "code_revision": _validate_code_revision(
            _git_revision() if code_revision is None else code_revision
        ),
        "artifact_root": _logical_artifact_root(root, output),
        "training_output_relative_path": (
            "."
            if training == root
            else _relative_path(
                training.relative_to(root).as_posix(),
                "training_output_relative_path",
            )
        ),
        **inspected,
    }
    if set(evidence) != EVIDENCE_FIELDS:
        raise RuntimeError("internal accelerator evidence schema mismatch")
    create_json_exclusive(output, evidence)
    return evidence


def validate_accelerator_validation_evidence(
    evidence_path: str | Path,
) -> dict[str, Any]:
    """Revalidate a stored v2 receipt and every artifact it commits to."""

    path = Path(evidence_path).resolve()
    payload = _load_object(path)
    if set(payload) != EVIDENCE_FIELDS:
        raise ValueError("accelerator evidence has unexpected or missing fields")
    if payload["schema_name"] != SCHEMA_NAME:
        raise ValueError("accelerator evidence has the wrong schema")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("accelerator evidence has an unsupported version")
    recorded = payload["recorded_at_utc"]
    if not isinstance(recorded, str) or not recorded.endswith("Z"):
        raise ValueError("recorded_at_utc must be an explicit UTC timestamp")
    try:
        datetime.fromisoformat(recorded.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("recorded_at_utc must be an explicit UTC timestamp") from error
    _validate_code_revision(payload["code_revision"])
    for field in (
        "success",
        "scientific",
        "hardware_matched",
        "unsupported_operation_fallback",
        "cleanup_completed",
    ):
        expected = field != "unsupported_operation_fallback"
        _exact_bool(payload, field, expected)
    if (
        _exact_int(payload["steps_completed"], "steps_completed")
        != FULL_TRAIN_CUDA_V2.max_steps
    ):
        raise ValueError("accelerator evidence has the wrong full-profile step count")
    if (
        not isinstance(payload["runtime_artifact_hashes"], dict)
        or set(payload["runtime_artifact_hashes"]) != _RUNTIME_HASH_FIELDS
    ):
        raise ValueError("runtime_artifact_hashes has an invalid exact schema")
    for field, digest in payload["runtime_artifact_hashes"].items():
        _sha256(digest, f"runtime_artifact_hashes.{field}")
    settings = payload["deterministic_settings"]
    if not isinstance(settings, dict) or set(settings) != _DETERMINISM_FIELDS:
        raise ValueError("deterministic_settings has an invalid exact schema")
    for field, expected in {
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
    }.items():
        _exact_bool(settings, field, expected)
    if settings["cublas_workspace_config"] != ":4096:8":
        raise ValueError("deterministic_settings has the wrong CUBLAS configuration")
    for field in (
        "dependency_lock_hash",
        "image_source_hash",
        "training_profile_hash",
        "candidate_artifact_hash",
        "candidate_graph_hash",
        "seed_bundle_hash",
        "training_manifest_hash",
        "training_summary_hash",
        "checkpoint_hash",
        "event_log_hash",
    ):
        _sha256(payload[field], field)
    artifact_relative = _relative_path(
        payload["artifact_root"], "artifact_root", allow_dot=True
    )
    training_relative = _relative_path(
        payload["training_output_relative_path"],
        "training_output_relative_path",
        allow_dot=True,
    )
    root = _resolve_artifact_root(path, artifact_relative)
    training = root if training_relative == "." else root / training_relative
    inspected = _inspect_artifacts(
        artifact_root=root,
        training_output_dir=training,
    )
    mismatches = {
        field: {"expected": payload[field], "observed": observed}
        for field, observed in inspected.items()
        if payload.get(field) != observed
    }
    if mismatches:
        raise ValueError(f"accelerator evidence artifact mismatch: {mismatches}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record an existing full_train_cuda_v2 Modal/CUDA run without "
            "training, provider, network, or Modal calls."
        )
    )
    parser.add_argument("--training-output-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--code-revision")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "readiness"
            / "full_train_cuda_v2_accelerator_evidence.json"
        ),
    )
    arguments = parser.parse_args()
    evidence = record_accelerator_validation(
        training_output_dir=arguments.training_output_dir,
        artifact_root=arguments.artifact_root,
        output_path=arguments.output,
        code_revision=arguments.code_revision,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
