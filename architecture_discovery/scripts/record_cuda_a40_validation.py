#!/usr/bin/env python3
"""Validate and record a trusted, non-scientific CUDA/A40 smoke run.

This command does not train, contact a provider, or accept provider credentials.
It hashes and validates an already completed ``smoke_train_cuda_v1`` output
directory.  A passing receipt is engineering evidence only and never authorizes
scientific execution or cross-backend pooling.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.training_config import (  # noqa: E402
    SMOKE_TRAIN_CUDA_V1,
    TrainingSeedBundle,
)
from common.cuda_contract import (  # noqa: E402
    cuda_allocation_gpu_counts,
    cuda_allocation_proves_exactly_one,
    is_nvidia_a40_name,
)
from common.trusted_candidate import (  # noqa: E402
    TRUSTED_INITIAL_CANDIDATE_PATH,
    TRUSTED_INITIAL_CANDIDATE_SHA256,
    validate_trusted_initial_candidate,
)
from study.serialization import create_json_exclusive  # noqa: E402
from study.budget import (  # noqa: E402
    AcceleratorBackend,
    AcceleratorResourceLedger,
)


SCHEMA_NAME = "CudaA40ValidationReceipt"
SCHEMA_VERSION = "1.0"
TELEMETRY_SCHEMA_VERSION = "accelerator_telemetry_v1"
TRUSTED_CANDIDATE = TRUSTED_INITIAL_CANDIDATE_PATH
_CREDENTIAL_MARKERS = (
    "DISCOVERY_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "PASSWORD",
    "PRIVATE_KEY",
)
_TEXT_ARTIFACT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".log",
    ".out",
    ".err",
    ".txt",
    ".py",
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_bool(payload: Mapping[str, Any], field: str, expected: bool) -> None:
    value = payload.get(field)
    if type(value) is not bool or value is not expected:
        raise ValueError(f"{field} must be exactly {expected}")


def _exact_int(
    payload: Mapping[str, Any],
    field: str,
    *,
    minimum: int = 0,
) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _contained_file(output_dir: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = output_dir / path
    path = path.resolve()
    if not path.is_relative_to(output_dir):
        raise ValueError(f"{field} escapes the training output directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_identity(identity: object) -> dict[str, Any]:
    if not isinstance(identity, dict):
        raise ValueError("CUDA telemetry identity must be a mapping")
    name = identity.get("name")
    if not is_nvidia_a40_name(name):
        raise ValueError("CUDA smoke must run on an NVIDIA A40")
    total_memory = _exact_int(identity, "total_memory_bytes", minimum=1)
    capability = identity.get("compute_capability")
    if (
        not isinstance(capability, (list, tuple))
        or len(capability) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in capability
        )
    ):
        raise ValueError("CUDA compute capability must be two non-negative integers")
    uuid_value = identity.get("uuid")
    if uuid_value is not None and (
        not isinstance(uuid_value, str) or not uuid_value
    ):
        raise ValueError("CUDA UUID must be non-empty text when available")
    return {
        "name": name,
        "uuid": uuid_value,
        "total_memory_bytes": total_memory,
        "compute_capability": list(capability),
    }


def _validate_runtime(runtime: object) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise ValueError("CUDA runtime telemetry must be a mapping")
    required = (
        "torch_version",
        "cuda_runtime_version",
        "cuda_driver_version",
        "cudnn_version",
    )
    for field in required:
        value = runtime.get(field)
        if value is None or isinstance(value, bool) or not str(value).strip():
            raise ValueError(f"CUDA runtime field {field} must be recorded")
    return {field: runtime[field] for field in required}


def _validate_memory(memory: object) -> dict[str, int]:
    if not isinstance(memory, dict):
        raise ValueError("CUDA memory telemetry must be a mapping")
    result = {
        field: _exact_int(memory, field)
        for field in (
            "allocated_bytes",
            "reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
        )
    }
    if result["peak_allocated_bytes"] <= 0 or result["peak_reserved_bytes"] <= 0:
        raise ValueError("CUDA peak memory telemetry must show positive allocation")
    return result


def _validate_telemetry(
    telemetry: object,
    *,
    expected_device: str,
) -> dict[str, Any]:
    if not isinstance(telemetry, dict):
        raise ValueError("accelerator_telemetry must be a mapping")
    expected = {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "backend": "cuda",
        "device": expected_device,
    }
    for field, value in expected.items():
        if telemetry.get(field) != value:
            raise ValueError(f"CUDA telemetry {field} mismatch")
    return {
        **expected,
        "identity": _validate_identity(telemetry.get("identity")),
        "runtime": _validate_runtime(telemetry.get("runtime")),
        "memory": _validate_memory(telemetry.get("memory")),
    }


def _validate_allocation(allocation: object) -> dict[str, Any]:
    if not isinstance(allocation, dict):
        raise ValueError("training manifest lacks CUDA scheduler allocation evidence")
    if allocation.get("scheduler") != "slurm":
        raise ValueError("CUDA/A40 validation requires a Slurm allocation")
    job_id = allocation.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("CUDA allocation lacks a Slurm job ID")
    visible = allocation.get("visible_devices")
    if not isinstance(visible, str) or len(
        [item for item in visible.split(",") if item.strip()]
    ) != 1:
        raise ValueError("CUDA allocation must expose exactly one visible GPU")
    markers = allocation.get("allocation_markers")
    if not isinstance(markers, dict) or not cuda_allocation_proves_exactly_one(
        markers
    ):
        raise ValueError(
            "CUDA allocation must prove exactly one scheduler GPU; observed "
            f"{cuda_allocation_gpu_counts(markers if isinstance(markers, dict) else {})}"
        )
    return {
        "scheduler": "slurm",
        "job_id": job_id,
        "visible_devices": visible,
        "allocation_markers": dict(markers),
    }


def _validate_determinism(determinism: object) -> dict[str, Any]:
    if not isinstance(determinism, dict):
        raise ValueError("training manifest lacks CUDA determinism evidence")
    exact = {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "float32_matmul_precision": "highest",
    }
    for field, expected in exact.items():
        observed = determinism.get(field)
        if type(expected) is bool and type(observed) is not bool:
            raise ValueError(f"CUDA determinism field {field} must be boolean")
        if observed != expected:
            raise ValueError(f"CUDA determinism field {field} mismatch")
    return exact


def _scan_credential_markers(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
            continue
        relative = str(path.relative_to(output_dir))
        data = path.read_bytes()
        text = data.decode("utf-8", errors="replace").upper()
        found = sorted(marker for marker in _CREDENTIAL_MARKERS if marker in text)
        if found:
            raise ValueError(
                f"credential marker names found in {relative}: {', '.join(found)}"
            )
        hashes[relative] = hashlib.sha256(data).hexdigest()
    if not hashes:
        raise ValueError("no text artifacts were available for credential scanning")
    return hashes


def _validate_training_output(
    output_dir: Path,
    *,
    expected_candidate: Path,
    expected_seed: int,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    expected_candidate = expected_candidate.resolve()
    summary_path = output_dir / "training_summary.json"
    manifest_path = output_dir / "training_manifest.json"
    resource_path = output_dir / "accelerator_resource_usage.json"
    candidate_path = output_dir / "candidate_source.py"
    for path in (
        summary_path,
        manifest_path,
        resource_path,
        candidate_path,
        expected_candidate,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(manifest, dict):
        raise ValueError("training summary and manifest must be JSON objects")

    for field, expected in {
        "success": True,
        "scientific": False,
        "hardware_matched": True,
        "unsupported_operation_fallback": False,
        "cleanup_completed": True,
        "timing_synchronized": True,
        "parameters_changed": True,
    }.items():
        _exact_bool(summary, field, expected)
    if summary.get("failure_stage") not in {None, ""} or summary.get("error") not in {
        None,
        "",
    }:
        raise ValueError("successful CUDA smoke summary contains failure text")
    if _exact_int(summary, "steps_completed") != SMOKE_TRAIN_CUDA_V1.max_steps:
        raise ValueError("CUDA smoke must complete exactly 10 optimizer steps")
    _exact_int(summary, "examples_processed", minimum=1)
    best_step = _exact_int(summary, "best_development_step", minimum=1)
    if best_step > SMOKE_TRAIN_CUDA_V1.max_steps:
        raise ValueError("selected checkpoint step exceeds completed optimizer steps")
    for field in (
        "best_development_exact_match_accuracy",
        "best_development_loss",
        "final_training_loss",
        "train_seconds",
    ):
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"summary field {field} must be numeric")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"summary field {field} must be finite and non-negative")

    expected_summary = {
        "profile_name": SMOKE_TRAIN_CUDA_V1.name,
        "profile_version": SMOKE_TRAIN_CUDA_V1.version,
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "device": "cuda:0",
        "dtype": "float32",
        "accelerator_telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
    }
    for field, expected in expected_summary.items():
        if summary.get(field) != expected:
            raise ValueError(f"training summary {field} does not match CUDA smoke")
    expected_seeds = TrainingSeedBundle.from_run_seed(expected_seed)
    for field, expected in {
        "initialization_seed": expected_seeds.model_initialization_seed,
        "data_seed": expected_seeds.training_data_seed,
        "development_seed": expected_seeds.development_set_seed,
        "dataloader_seed": expected_seeds.dataloader_seed,
    }.items():
        if summary.get(field) != expected:
            raise ValueError(f"training summary {field} does not match seed")
    initial_parameters = summary.get("initial_parameter_sha256")
    final_parameters = summary.get("final_parameter_sha256")
    if not _sha256_digest(initial_parameters) or not _sha256_digest(final_parameters):
        raise ValueError("parameter-state hashes must be lowercase SHA-256 digests")
    if initial_parameters == final_parameters:
        raise ValueError("parameters did not change from initialization")
    summary_telemetry = _validate_telemetry(
        summary.get("accelerator_telemetry"),
        expected_device="cuda:0",
    )

    _exact_bool(manifest, "allow_cpu_for_tests", False)
    _exact_bool(manifest, "hardware_matched_scientific_run", False)
    if manifest.get("requested_device") != "cuda":
        raise ValueError("training manifest did not request logical device cuda")
    if manifest.get("selected_device") != "cuda:0":
        raise ValueError("training manifest did not resolve to cuda:0")
    if manifest.get("candidate_source_hash") != summary.get(
        "candidate_source_hash"
    ):
        raise ValueError("manifest and summary candidate hashes differ")
    if manifest.get("profile_hash") != SMOKE_TRAIN_CUDA_V1.profile_hash:
        raise ValueError("training manifest profile hash is not CUDA smoke")
    profile = manifest.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("training manifest lacks serialized profile")
    for field, expected in {
        "name": SMOKE_TRAIN_CUDA_V1.name,
        "version": SMOKE_TRAIN_CUDA_V1.version,
        "max_steps": 10,
        "device_requirement": "cuda",
        "scientific": False,
        "cpu_fallback": False,
        "mixed_precision": False,
        "torch_compile": False,
        "automatic_batch_size_reduction": False,
    }.items():
        if profile.get(field) != expected:
            raise ValueError(f"manifest profile {field} mismatch")
    if manifest.get("parameter_count_role") != "descriptive_metadata_only":
        raise ValueError("parameter count is not recorded as descriptive metadata")
    _exact_int(summary, "parameter_count_metadata", minimum=1)
    for field in ("controller_source_hash", "dependency_lock_hash"):
        if not _sha256_digest(manifest.get(field)):
            raise ValueError(f"training manifest {field} must be a SHA-256 digest")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("training manifest lacks runtime evidence")
    manifest_accelerator = runtime.get("accelerator")
    if not isinstance(manifest_accelerator, dict):
        raise ValueError("training manifest lacks accelerator metadata")
    manifest_telemetry = {
        "schema_version": manifest_accelerator.get("schema_version"),
        "backend": manifest_accelerator.get("backend"),
        "device": manifest_accelerator.get("device"),
        "identity": manifest_accelerator.get("identity"),
        "runtime": manifest_accelerator.get("runtime"),
        "memory": summary_telemetry["memory"],
    }
    validated_manifest_telemetry = _validate_telemetry(
        manifest_telemetry,
        expected_device="cuda:0",
    )
    if validated_manifest_telemetry["identity"] != summary_telemetry["identity"]:
        raise ValueError("manifest and summary CUDA identities differ")
    if validated_manifest_telemetry["runtime"] != summary_telemetry["runtime"]:
        raise ValueError("manifest and summary CUDA runtime metadata differ")
    allocation = _validate_allocation(manifest_accelerator.get("allocation"))
    determinism = _validate_determinism(runtime.get("cuda_determinism"))

    resource_record = json.loads(resource_path.read_text(encoding="utf-8"))
    if not isinstance(resource_record, dict):
        raise ValueError("CUDA accelerator resource record must be a mapping")
    if resource_record.get("schema_name") != "AcceleratorTrainingResourceRecord":
        raise ValueError("CUDA accelerator resource record has the wrong schema")
    if resource_record.get("schema_version") != "2.0":
        raise ValueError("unsupported CUDA accelerator resource record version")
    for field, expected in {
        "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
        "candidate_source_hash": summary.get("candidate_source_hash"),
        "selected_device": "cuda:0",
        "training_summary_sha256": _sha256_file(summary_path),
    }.items():
        if resource_record.get(field) != expected:
            raise ValueError(f"CUDA accelerator resource field {field} mismatch")
    accelerator_key = resource_record.get("accelerator_key")
    if not isinstance(accelerator_key, str) or not accelerator_key.startswith("cuda:"):
        raise ValueError("CUDA accelerator resource record lacks a CUDA lease key")
    resource_ledger = AcceleratorResourceLedger.from_dict(
        resource_record.get("resource_ledger")
    )
    expected_condition = (
        f"nvidia_a40:{SMOKE_TRAIN_CUDA_V1.name}:"
        f"{SMOKE_TRAIN_CUDA_V1.profile_hash}"
    )
    if (
        resource_ledger.spec.backend is not AcceleratorBackend.CUDA
        or resource_ledger.spec.hardware_condition != expected_condition
        or resource_ledger.accelerator_seconds != float(summary.get("train_seconds"))
    ):
        raise ValueError("CUDA accelerator resource accounting does not reconstruct")

    candidate_hash = _sha256_file(candidate_path)
    expected_candidate_hash = validate_trusted_initial_candidate(expected_candidate)
    if expected_candidate_hash != TRUSTED_INITIAL_CANDIDATE_SHA256:
        raise ValueError("trusted candidate constant does not reconstruct")
    if candidate_hash != TRUSTED_INITIAL_CANDIDATE_SHA256:
        raise ValueError("CUDA smoke did not use trusted common/initial_candidate.py")
    if candidate_hash != summary.get("candidate_source_hash"):
        raise ValueError("candidate source hash does not match training summary")
    checkpoint_path = _contained_file(
        output_dir,
        summary.get("checkpoint_path"),
        "checkpoint_path",
    )
    checkpoint_hash = _sha256_file(checkpoint_path)
    if checkpoint_hash != summary.get("checkpoint_sha256"):
        raise ValueError("checkpoint hash does not match training summary")
    event_path = _contained_file(
        output_dir,
        summary.get("event_log_path"),
        "event_log_path",
    )
    credential_scan_hashes = _scan_credential_markers(output_dir)
    accelerator_identity = summary_telemetry["identity"]
    gpu_key = (
        f"cuda:uuid:{accelerator_identity['uuid']}"
        if accelerator_identity["uuid"]
        else "cuda:slurm:"
        + allocation["job_id"]
        + ":"
        + allocation["visible_devices"]
    )
    return {
        "summary": summary,
        "manifest": manifest,
        "telemetry": summary_telemetry,
        "allocation": allocation,
        "determinism": determinism,
        "gpu_key": gpu_key,
        "artifact_hashes": {
            "candidate_source_sha256": candidate_hash,
            "training_manifest_sha256": _sha256_file(manifest_path),
            "training_summary_sha256": _sha256_file(summary_path),
            "accelerator_resource_usage_sha256": _sha256_file(resource_path),
            "checkpoint_sha256": checkpoint_hash,
            "event_log_sha256": _sha256_file(event_path),
        },
        "credential_scan_hashes": credential_scan_hashes,
    }


def _receipt_without_hash(
    *,
    output_dir: Path,
    expected_candidate: Path,
    expected_seed: int,
    validated: Mapping[str, Any],
    recorded_at_utc: str,
    recorder_hostname: str,
) -> dict[str, Any]:
    summary = validated["summary"]
    manifest = validated["manifest"]
    telemetry = validated["telemetry"]
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": recorded_at_utc,
        "validation_kind": "trusted_checked_in_candidate_non_scientific_smoke",
        "scientific_readiness": False,
        "formal_scientific_execution_authorized": False,
        "cross_backend_pooling_authorized": False,
        "provider_calls": 0,
        "training_runs_created_by_recorder": 0,
        "profile": {
            "name": SMOKE_TRAIN_CUDA_V1.name,
            "version": SMOKE_TRAIN_CUDA_V1.version,
            "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
            "optimizer_steps": SMOKE_TRAIN_CUDA_V1.max_steps,
            "scientific": False,
        },
        "execution": {
            "requested_device": "cuda",
            "selected_device": "cuda:0",
            "seed": expected_seed,
            "success": True,
            "steps_completed": summary["steps_completed"],
            "examples_processed": summary["examples_processed"],
            "selected_checkpoint_step": summary["best_development_step"],
            "validation_completed": True,
            "parameters_changed": True,
            "timing_synchronized": True,
            "unsupported_operation_fallback": False,
            "cpu_fallback": False,
        },
        "accelerator": {
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "gpu_key": validated["gpu_key"],
            "identity": telemetry["identity"],
            "runtime": telemetry["runtime"],
            "memory": telemetry["memory"],
            "allocation": validated["allocation"],
            "determinism": validated["determinism"],
            "recorder_hostname": recorder_hostname,
        },
        "artifacts": {
            "training_output_dir": str(output_dir),
            "trusted_candidate_path": str(expected_candidate),
            **validated["artifact_hashes"],
            "profile_hash": SMOKE_TRAIN_CUDA_V1.profile_hash,
            "controller_source_hash": manifest.get("controller_source_hash"),
            "dependency_lock_hash": manifest.get("dependency_lock_hash"),
            "initial_parameter_sha256": summary["initial_parameter_sha256"],
            "final_parameter_sha256": summary["final_parameter_sha256"],
        },
        "credential_scan": {
            "passed": True,
            "credential_values_supplied_to_recorder": False,
            "scanned_artifact_sha256s": validated["credential_scan_hashes"],
        },
        "limitations": [
            "This ten-step smoke is non-scientific and cannot rank architectures.",
            "It does not prove arbitrary-Python containment or authorize paid runs.",
            "CUDA and MPS are separate hardware conditions and are not pooled.",
        ],
    }


def validate_cuda_a40_training_output(
    training_output_dir: str | Path,
    *,
    expected_seed: int = 1,
) -> dict[str, Any]:
    """Revalidate one trusted smoke output without creating a receipt."""

    if not isinstance(expected_seed, int) or isinstance(expected_seed, bool):
        raise ValueError("expected_seed must be an integer")
    validate_trusted_initial_candidate(TRUSTED_CANDIDATE)
    return _validate_training_output(
        Path(training_output_dir).resolve(),
        expected_candidate=TRUSTED_CANDIDATE,
        expected_seed=expected_seed,
    )


def record_cuda_a40_validation(
    *,
    training_output_dir: str | Path,
    output_path: str | Path,
    expected_candidate: str | Path = TRUSTED_CANDIDATE,
    expected_seed: int = 1,
) -> dict[str, Any]:
    """Create one exclusive hash-linked receipt for a completed CUDA smoke."""

    if not isinstance(expected_seed, int) or isinstance(expected_seed, bool):
        raise ValueError("expected_seed must be an integer")
    output_dir = Path(training_output_dir).resolve()
    candidate = Path(expected_candidate).resolve()
    if candidate != TRUSTED_CANDIDATE.resolve():
        raise ValueError(
            "CUDA/A40 smoke validation is restricted to "
            "common/initial_candidate.py"
        )
    validated = validate_cuda_a40_training_output(
        output_dir,
        expected_seed=expected_seed,
    )
    payload = _receipt_without_hash(
        output_dir=output_dir,
        expected_candidate=candidate,
        expected_seed=expected_seed,
        validated=validated,
        recorded_at_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        recorder_hostname=socket.gethostname(),
    )
    payload["receipt_payload_sha256"] = _canonical_sha256(payload)
    resolved_output_path = Path(output_path).resolve()
    if resolved_output_path.is_relative_to(output_dir):
        raise ValueError("CUDA/A40 receipt must be stored outside the training output")
    create_json_exclusive(resolved_output_path, payload)
    return payload


def validate_cuda_a40_receipt(
    receipt_path: str | Path,
    *,
    expected_candidate: str | Path = TRUSTED_CANDIDATE,
    expected_seed: int = 1,
) -> dict[str, Any]:
    """Revalidate a receipt and every linked training artifact from source."""

    path = Path(receipt_path).resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("CUDA/A40 validation receipt must be a JSON object")
    if receipt.get("schema_name") != SCHEMA_NAME:
        raise ValueError("CUDA/A40 validation receipt has the wrong schema")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported CUDA/A40 validation receipt version")
    recorded = receipt.get("recorded_at_utc")
    if not isinstance(recorded, str) or not recorded.endswith("Z"):
        raise ValueError("CUDA/A40 receipt lacks a UTC recording timestamp")
    expected_hash = receipt.get("receipt_payload_sha256")
    without_hash = {
        key: value for key, value in receipt.items() if key != "receipt_payload_sha256"
    }
    if not _sha256_digest(expected_hash) or expected_hash != _canonical_sha256(
        without_hash
    ):
        raise ValueError("CUDA/A40 receipt payload hash is invalid")
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("CUDA/A40 receipt lacks artifact links")
    output_dir_value = artifacts.get("training_output_dir")
    if not isinstance(output_dir_value, str) or not output_dir_value:
        raise ValueError("CUDA/A40 receipt lacks a training output directory")
    output_dir = Path(output_dir_value).resolve()
    accelerator = receipt.get("accelerator")
    if not isinstance(accelerator, dict):
        raise ValueError("CUDA/A40 receipt lacks accelerator evidence")
    recorder_hostname = accelerator.get("recorder_hostname")
    if not isinstance(recorder_hostname, str) or not recorder_hostname:
        raise ValueError("CUDA/A40 receipt lacks its recording hostname")
    candidate = Path(expected_candidate).resolve()
    if candidate != TRUSTED_CANDIDATE.resolve():
        raise ValueError(
            "CUDA/A40 smoke validation is restricted to "
            "common/initial_candidate.py"
        )
    validated = _validate_training_output(
        output_dir,
        expected_candidate=candidate,
        expected_seed=expected_seed,
    )
    rebuilt = _receipt_without_hash(
        output_dir=output_dir,
        expected_candidate=candidate,
        expected_seed=expected_seed,
        validated=validated,
        recorded_at_utc=recorded,
        recorder_hostname=recorder_hostname,
    )
    if receipt != {**rebuilt, "receipt_payload_sha256": _canonical_sha256(rebuilt)}:
        raise ValueError("CUDA/A40 receipt does not match revalidated artifacts")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record an existing ten-step trusted CUDA/A40 smoke without training "
            "or provider calls."
        )
    )
    parser.add_argument("--training-output-dir", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="new receipt path; existing files are never overwritten",
    )
    parser.add_argument("--expected-seed", type=int, default=1)
    arguments = parser.parse_args()
    evidence = record_cuda_a40_validation(
        training_output_dir=arguments.training_output_dir,
        output_path=arguments.output,
        expected_seed=arguments.expected_seed,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
