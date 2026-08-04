"""Machine-readable audit of candidate-execution boundary capabilities.

The audit is intentionally conservative.  Finding a command, Python hook, or
platform feature is not proof that a candidate process is contained by it.
Only externally produced, artifact-bound adversarial-test evidence may move a
boundary control to ``proven``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import torch

from common.cuda_contract import cuda_allocation_proves_exactly_one


SCHEMA_NAME = "candidate_containment_capability_audit"
SCHEMA_VERSION = "2.0"


class CapabilityState(StrEnum):
    """Strength of evidence for one boundary control."""

    ABSENT = "absent"
    DETECTED = "detected_not_enforced"
    NOT_PROVEN = "not_proven"
    PROVEN = "proven_by_adversarial_test"


@dataclass(frozen=True)
class ControlEvidence:
    control: str
    state: CapabilityState
    method: str
    detail: str
    artifact_hash: str | None = None

    @property
    def proven(self) -> bool:
        return self.state is CapabilityState.PROVEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "control": self.control,
            "state": self.state.value,
            "method": self.method,
            "detail": self.detail,
            "artifact_hash": self.artifact_hash,
        }


REQUIRED_STRONG_CONTROLS = (
    "filesystem_allowlist",
    "network_isolation",
    "credential_isolation",
    "child_process_isolation",
    "resource_limits",
    "unprivileged_identity",
    "platform_sandbox",
)

REQUIRED_ADVERSARIAL_TESTS: Mapping[str, str] = {
    "filesystem_allowlist": "cannot_read_outside_allowlist",
    "network_isolation": "cannot_open_network_socket",
    "credential_isolation": "cannot_observe_parent_credentials",
    "child_process_isolation": "cannot_spawn_child_process",
    "resource_limits": "resource_limit_terminates_candidate",
    "unprivileged_identity": "candidate_runs_as_dedicated_unprivileged_identity",
    "platform_sandbox": "sandbox_boundary_survives_python_bypass_attempts",
}


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BoundaryAttestation:
    """Trusted-runner report binding adversarial results to an artifact.

    This object is a transport schema, not a signature verifier.  The trusted
    study runner must authenticate its provenance before passing it here.
    Environment variables and candidate-controlled files are never accepted as
    attestations by :func:`audit_runtime`.
    """

    runner: str
    candidate_artifact_hash: str
    report_artifact_hash: str
    created_at_utc: str
    test_results: Mapping[str, bool]
    authenticated_by_trusted_runner: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "test_results", MappingProxyType(dict(self.test_results)))

    def validate(self) -> tuple[str, ...]:
        problems: list[str] = []
        if not self.authenticated_by_trusted_runner:
            problems.append("boundary attestation provenance was not authenticated")
        for field_name, value in (
            ("candidate_artifact_hash", self.candidate_artifact_hash),
            ("report_artifact_hash", self.report_artifact_hash),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                problems.append(f"{field_name} is not a lowercase SHA-256 digest")
        required = set(REQUIRED_ADVERSARIAL_TESTS.values())
        missing = required.difference(self.test_results)
        if missing:
            problems.append(f"attestation omitted required tests: {sorted(missing)}")
        failed = sorted(
            test_name
            for test_name in required
            if self.test_results.get(test_name) is not True
        )
        if failed:
            problems.append(f"attestation did not pass required tests: {failed}")
        return tuple(problems)


@dataclass(frozen=True)
class CapabilityAudit:
    created_at_utc: str
    platform_system: str
    platform_release: str
    machine: str
    python_implementation: str
    mps_built: bool
    mps_available: bool
    mps_fallback_requested: bool
    cuda_compiled_runtime: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_visible_devices: str | None
    cuda_scheduler_job_id: str | None
    cuda_scheduler_gpu_assignment: str | None
    cuda_allocation_validated: bool
    cuda_devices: tuple[Mapping[str, Any], ...]
    detected_container_runtimes: tuple[str, ...]
    visible_credential_names: tuple[str, ...]
    controls: Mapping[str, ControlEvidence]
    attested_candidate_artifact_hash: str | None = None
    attestation_report_artifact_hash: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "visible_credential_names", tuple(self.visible_credential_names))
        object.__setattr__(
            self,
            "cuda_devices",
            tuple(MappingProxyType(dict(device)) for device in self.cuda_devices),
        )
        object.__setattr__(
            self,
            "detected_container_runtimes",
            tuple(self.detected_container_runtimes),
        )
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))
        object.__setattr__(self, "notes", tuple(self.notes))

    @property
    def strong_containment_proven(self) -> bool:
        return all(self.controls[name].proven for name in REQUIRED_STRONG_CONTROLS)

    @property
    def audit_hash(self) -> str:
        return _sha256_text(_canonical_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "platform": {
                "system": self.platform_system,
                "release": self.platform_release,
                "machine": self.machine,
                "python_implementation": self.python_implementation,
            },
            "mps": {
                "built": self.mps_built,
                "available": self.mps_available,
                "fallback_requested": self.mps_fallback_requested,
            },
            "cuda": {
                "compiled_runtime": self.cuda_compiled_runtime,
                "available": self.cuda_available,
                "device_count": self.cuda_device_count,
                "cuda_visible_devices": self.cuda_visible_devices,
                "scheduler_job_id": self.cuda_scheduler_job_id,
                "scheduler_gpu_assignment": self.cuda_scheduler_gpu_assignment,
                "allocation_validated": self.cuda_allocation_validated,
                "devices": [dict(device) for device in self.cuda_devices],
            },
            "detected_container_runtimes": list(self.detected_container_runtimes),
            "visible_credential_names": list(self.visible_credential_names),
            "attested_candidate_artifact_hash": self.attested_candidate_artifact_hash,
            "attestation_report_artifact_hash": self.attestation_report_artifact_hash,
            "controls": {
                name: self.controls[name].to_dict() for name in sorted(self.controls)
            },
            "strong_containment_proven": self.strong_containment_proven,
            "notes": list(self.notes),
        }
        if include_hash:
            payload["audit_hash"] = self.audit_hash
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


_CREDENTIAL_MARKERS = (
    "API_KEY",
    "ACCESS_TOKEN",
    "AUTH_TOKEN",
    "PASSWORD",
    "SECRET",
    "PRIVATE_KEY",
)


def _visible_credential_names(environment: Mapping[str, str]) -> tuple[str, ...]:
    """Return names only.  Secret values never enter the audit artifact."""

    return tuple(
        sorted(
            name
            for name in environment
            if any(marker in name.upper() for marker in _CREDENTIAL_MARKERS)
        )
    )


def _base_controls() -> dict[str, ControlEvidence]:
    sandbox_exec = shutil.which("sandbox-exec")
    container_runtimes = {
        name: shutil.which(name)
        for name in ("apptainer", "singularity", "podman")
    }
    detected_sandboxes = {
        name: path for name, path in container_runtimes.items() if path is not None
    }
    sandbox_details = []
    if sandbox_exec:
        sandbox_details.append(f"sandbox-exec={sandbox_exec}")
    sandbox_details.extend(
        f"{name}={path}" for name, path in sorted(detected_sandboxes.items())
    )
    try:
        import resource  # noqa: PLC0415 - capability probe is platform-dependent

        resource_detail = f"resource module exposes {len(dir(resource))} attributes"
        resource_state = CapabilityState.DETECTED
    except ImportError:
        resource_detail = "Python resource module unavailable"
        resource_state = CapabilityState.ABSENT

    try:
        effective_uid = os.geteuid()
        identity_detail = f"effective uid is {effective_uid}; no dedicated-worker proof"
        identity_state = CapabilityState.DETECTED
    except AttributeError:
        identity_detail = "effective uid inspection unavailable"
        identity_state = CapabilityState.ABSENT

    return {
        "filesystem_allowlist": ControlEvidence(
            "filesystem_allowlist",
            CapabilityState.NOT_PROVEN,
            "runtime introspection",
            "no kernel-enforced candidate filesystem allowlist was attested",
        ),
        "network_isolation": ControlEvidence(
            "network_isolation",
            CapabilityState.NOT_PROVEN,
            "runtime introspection",
            "Python socket monkeypatches are bypassable and do not prove OS isolation",
        ),
        "credential_isolation": ControlEvidence(
            "credential_isolation",
            CapabilityState.NOT_PROVEN,
            "runtime introspection",
            "environment scrubbing requires an adversarial child-process test",
        ),
        "child_process_isolation": ControlEvidence(
            "child_process_isolation",
            CapabilityState.NOT_PROVEN,
            "runtime introspection",
            "no kernel-enforced prohibition on fork, exec, or spawn was attested",
        ),
        "resource_limits": ControlEvidence(
            "resource_limits",
            resource_state,
            "resource-module discovery",
            resource_detail + "; availability is not proof that limits were applied",
        ),
        "unprivileged_identity": ControlEvidence(
            "unprivileged_identity",
            identity_state,
            "effective-identity discovery",
            identity_detail,
        ),
        "platform_sandbox": ControlEvidence(
            "platform_sandbox",
            (
                CapabilityState.DETECTED
                if sandbox_exec or detected_sandboxes
                else CapabilityState.ABSENT
            ),
            "executable discovery",
            (
                "sandbox/container tools detected: " + ", ".join(sandbox_details)
                + "; candidate-bound enforcement not attested"
                if sandbox_details
                else "no supported platform sandbox or container executable detected"
            ),
        ),
    }


def _detected_container_runtimes() -> tuple[str, ...]:
    """Return executable names only; discovery is never containment proof."""

    return tuple(
        name
        for name in ("apptainer", "singularity", "podman")
        if shutil.which(name) is not None
    )


def _cuda_allocation_evidence(
    environment: Mapping[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Read scheduler allocation markers without treating visibility as a GPU test."""

    job_id = environment.get("SLURM_JOB_ID") or environment.get("SLURM_JOBID")
    assignment = next(
        (
            environment[name]
            for name in (
                "SLURM_STEP_GPUS",
                "SLURM_JOB_GPUS",
                "SLURM_GPUS_ON_NODE",
            )
            if environment.get(name)
        ),
        None,
    )
    visible = environment.get("CUDA_VISIBLE_DEVICES")
    return job_id, assignment, visible


def _cuda_devices(cuda_available: bool, device_count: int) -> tuple[dict[str, Any], ...]:
    """Collect non-secret CUDA identity metadata when PyTorch can access devices."""

    if not cuda_available:
        return ()
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        try:
            properties = torch.cuda.get_device_properties(index)
            capability = torch.cuda.get_device_capability(index)
            uuid_value = getattr(properties, "uuid", None)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "uuid": str(uuid_value) if uuid_value else None,
                    "total_memory_bytes": int(properties.total_memory),
                    "compute_capability": [int(capability[0]), int(capability[1])],
                }
            )
        except (AssertionError, RuntimeError) as error:
            devices.append(
                {
                    "index": index,
                    "metadata_error": f"{type(error).__name__}: {error}",
                }
            )
    return tuple(devices)


def _apply_attestation(
    controls: Mapping[str, ControlEvidence],
    attestation: BoundaryAttestation,
) -> tuple[dict[str, ControlEvidence], tuple[str, ...], str | None, str | None]:
    problems = attestation.validate()
    if problems:
        return dict(controls), problems, None, None

    updated = dict(controls)
    for control, test_name in REQUIRED_ADVERSARIAL_TESTS.items():
        updated[control] = ControlEvidence(
            control=control,
            state=CapabilityState.PROVEN,
            method=f"trusted boundary adversarial test: {test_name}",
            detail=f"passed by {attestation.runner} for bound candidate artifact",
            artifact_hash=attestation.report_artifact_hash,
        )
    return (
        updated,
        (),
        attestation.candidate_artifact_hash,
        attestation.report_artifact_hash,
    )


def audit_runtime(
    *,
    environment: Mapping[str, str] | None = None,
    trusted_attestation: BoundaryAttestation | None = None,
) -> CapabilityAudit:
    """Audit the current runtime without performing network or filesystem attacks."""

    environment = os.environ if environment is None else environment
    controls = _base_controls()
    notes = [
        "detected_not_enforced never satisfies the scientific containment gate",
        "static source inspection and Python monkeypatches are defense in depth only",
    ]
    attested_candidate_hash: str | None = None
    attestation_report_hash: str | None = None
    if trusted_attestation is not None:
        (
            controls,
            attestation_problems,
            attested_candidate_hash,
            attestation_report_hash,
        ) = _apply_attestation(controls, trusted_attestation)
        notes.extend(attestation_problems)

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())
    fallback = str(environment.get("PYTORCH_ENABLE_MPS_FALLBACK", "")).lower()
    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
    scheduler_job_id, scheduler_assignment, cuda_visible_devices = (
        _cuda_allocation_evidence(environment)
    )
    cuda_allocation_validated = bool(
        cuda_available
        and cuda_device_count == 1
        and scheduler_job_id
        and cuda_allocation_proves_exactly_one(
            {
                "SLURM_JOB_GPUS": environment.get("SLURM_JOB_GPUS", ""),
                "SLURM_STEP_GPUS": environment.get("SLURM_STEP_GPUS", ""),
                "SLURM_GPUS_ON_NODE": environment.get("SLURM_GPUS_ON_NODE", ""),
            }
        )
        and cuda_visible_devices not in {None, "", "-1"}
    )
    cuda_devices = _cuda_devices(cuda_available, cuda_device_count)
    if cuda_available and not cuda_allocation_validated:
        notes.append(
            "CUDA visibility is not proof of a scheduler GPU allocation; "
            "scientific CUDA execution remains blocked"
        )
    return CapabilityAudit(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        python_implementation=platform.python_implementation(),
        mps_built=mps_built,
        mps_available=mps_available,
        mps_fallback_requested=fallback in {"1", "true", "yes", "on"},
        cuda_compiled_runtime=torch.version.cuda,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_visible_devices=cuda_visible_devices,
        cuda_scheduler_job_id=scheduler_job_id,
        cuda_scheduler_gpu_assignment=scheduler_assignment,
        cuda_allocation_validated=cuda_allocation_validated,
        cuda_devices=cuda_devices,
        detected_container_runtimes=_detected_container_runtimes(),
        visible_credential_names=_visible_credential_names(environment),
        controls=controls,
        attested_candidate_artifact_hash=attested_candidate_hash,
        attestation_report_artifact_hash=attestation_report_hash,
        notes=tuple(notes),
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description="Audit candidate containment capabilities")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    arguments = parser.parse_args()
    report = audit_runtime()
    rendered = report.to_json() + "\n"
    if arguments.output is None:
        print(rendered, end="")
    else:
        arguments.output.write_text(rendered, encoding="utf-8")
    return 0 if report.strong_containment_proven else 2


if __name__ == "__main__":
    raise SystemExit(_main())
