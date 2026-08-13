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


SCHEMA_NAME = "candidate_containment_capability_audit"
LEGACY_SCHEMA_VERSION = "1.0"
SCHEMA_VERSION = "2.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({LEGACY_SCHEMA_VERSION, SCHEMA_VERSION})


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

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlEvidence":
        artifact_hash = payload.get("artifact_hash")
        if artifact_hash is not None:
            artifact_hash = str(artifact_hash)
        return cls(
            control=str(payload["control"]),
            state=CapabilityState(str(payload["state"])),
            method=str(payload["method"]),
            detail=str(payload["detail"]),
            artifact_hash=artifact_hash,
        )


@dataclass(frozen=True)
class CUDADeviceMetadata:
    """Stable, non-secret identity and capacity metadata for one CUDA device."""

    index: int
    name: str
    compute_capability: tuple[int, int] | None = None
    total_memory_bytes: int | None = None
    multi_processor_count: int | None = None
    device_uuid: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("CUDA device index cannot be negative")
        if not self.name:
            raise ValueError("CUDA device name cannot be empty")
        if self.compute_capability is not None:
            capability = tuple(int(item) for item in self.compute_capability)
            if len(capability) != 2 or any(item < 0 for item in capability):
                raise ValueError("compute_capability must contain nonnegative major/minor")
            object.__setattr__(self, "compute_capability", capability)
        for field_name in ("total_memory_bytes", "multi_processor_count"):
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "compute_capability": (
                list(self.compute_capability)
                if self.compute_capability is not None
                else None
            ),
            "total_memory_bytes": self.total_memory_bytes,
            "multi_processor_count": self.multi_processor_count,
            "device_uuid": self.device_uuid,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CUDADeviceMetadata":
        raw_capability = payload.get("compute_capability")
        capability: tuple[int, int] | None = None
        if raw_capability is not None:
            if not isinstance(raw_capability, (list, tuple)) or len(raw_capability) != 2:
                raise ValueError("CUDA compute_capability must be a major/minor pair")
            capability = (int(raw_capability[0]), int(raw_capability[1]))
        raw_total_memory = payload.get("total_memory_bytes")
        raw_multi_processor_count = payload.get("multi_processor_count")
        raw_uuid = payload.get("device_uuid")
        return cls(
            index=int(payload["index"]),
            name=str(payload["name"]),
            compute_capability=capability,
            total_memory_bytes=(
                int(raw_total_memory) if raw_total_memory is not None else None
            ),
            multi_processor_count=(
                int(raw_multi_processor_count)
                if raw_multi_processor_count is not None
                else None
            ),
            device_uuid=str(raw_uuid) if raw_uuid is not None else None,
        )


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
    visible_credential_names: tuple[str, ...]
    controls: Mapping[str, ControlEvidence]
    attested_candidate_artifact_hash: str | None = None
    attestation_report_artifact_hash: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION
    cpu_available: bool = True
    cuda_built: bool = False
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_runtime_version: str | None = None
    cuda_driver_version: str | None = None
    cuda_devices: tuple[CUDADeviceMetadata, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "visible_credential_names", tuple(self.visible_credential_names))
        object.__setattr__(self, "controls", MappingProxyType(dict(self.controls)))
        object.__setattr__(self, "cuda_devices", tuple(self.cuda_devices))
        object.__setattr__(self, "notes", tuple(self.notes))
        if self.schema_name != SCHEMA_NAME:
            raise ValueError(f"unsupported capability-audit schema {self.schema_name!r}")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported capability-audit version {self.schema_version!r}"
            )
        if self.cuda_device_count < 0:
            raise ValueError("cuda_device_count cannot be negative")
        if any(device.index >= self.cuda_device_count for device in self.cuda_devices):
            raise ValueError("CUDA metadata index falls outside cuda_device_count")

    @property
    def strong_containment_proven(self) -> bool:
        return all(self.controls[name].proven for name in REQUIRED_STRONG_CONTROLS)

    @property
    def audit_hash(self) -> str:
        return _sha256_text(_canonical_json(self.to_dict(include_hash=False)))

    def accelerator_available(self, accelerator: str) -> bool:
        """Return audited availability for a normalized accelerator kind."""

        normalized = accelerator.strip().lower().split(":", maxsplit=1)[0]
        if normalized == "cpu":
            return self.cpu_available
        if normalized == "mps":
            return self.mps_built and self.mps_available
        if normalized == "cuda":
            return (
                self.cuda_built
                and self.cuda_available
                and self.cuda_device_count > 0
            )
        return False

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
            "visible_credential_names": list(self.visible_credential_names),
            "attested_candidate_artifact_hash": self.attested_candidate_artifact_hash,
            "attestation_report_artifact_hash": self.attestation_report_artifact_hash,
            "controls": {
                name: self.controls[name].to_dict() for name in sorted(self.controls)
            },
            "strong_containment_proven": self.strong_containment_proven,
            "notes": list(self.notes),
        }
        if self.schema_version == LEGACY_SCHEMA_VERSION:
            # This is intentionally the exact v1 shape.  New accelerator fields
            # must never perturb hashes of historical MPS audit artifacts.
            payload["mps"] = {
                "built": self.mps_built,
                "available": self.mps_available,
                "fallback_requested": self.mps_fallback_requested,
            }
        else:
            payload["accelerators"] = {
                "cpu": {"available": self.cpu_available},
                "mps": {
                    "built": self.mps_built,
                    "available": self.mps_available,
                    "fallback_requested": self.mps_fallback_requested,
                },
                "cuda": {
                    "built": self.cuda_built,
                    "available": self.cuda_available,
                    "device_count": self.cuda_device_count,
                    "runtime_version": self.cuda_runtime_version,
                    "driver_version": self.cuda_driver_version,
                    "devices": [device.to_dict() for device in self.cuda_devices],
                },
            }
        if include_hash:
            payload["audit_hash"] = self.audit_hash
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CapabilityAudit":
        """Decode v2 audits and historical v1 MPS audits without rewriting them."""

        schema_name = str(payload.get("schema_name", ""))
        schema_version = str(payload.get("schema_version", ""))
        if schema_name != SCHEMA_NAME:
            raise ValueError(f"unsupported capability-audit schema {schema_name!r}")
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"unsupported capability-audit version {schema_version!r}"
            )

        raw_platform = payload.get("platform")
        raw_controls = payload.get("controls")
        if not isinstance(raw_platform, Mapping) or not isinstance(raw_controls, Mapping):
            raise ValueError("capability audit requires platform and controls mappings")
        controls: dict[str, ControlEvidence] = {}
        for name, raw_evidence in raw_controls.items():
            if not isinstance(raw_evidence, Mapping):
                raise ValueError(f"control evidence {name!r} must be a mapping")
            evidence = ControlEvidence.from_dict(raw_evidence)
            if evidence.control != str(name):
                raise ValueError(f"control evidence {name!r} has a mismatched name")
            controls[str(name)] = evidence

        if schema_version == LEGACY_SCHEMA_VERSION:
            raw_mps = payload.get("mps")
            if not isinstance(raw_mps, Mapping):
                raise ValueError("v1 capability audit requires an mps mapping")
            raw_cpu: Mapping[str, Any] = {"available": True}
            raw_cuda: Mapping[str, Any] = {
                "built": False,
                "available": False,
                "device_count": 0,
                "runtime_version": None,
                "driver_version": None,
                "devices": (),
            }
        else:
            raw_accelerators = payload.get("accelerators")
            if not isinstance(raw_accelerators, Mapping):
                raise ValueError("v2 capability audit requires an accelerators mapping")
            raw_cpu = raw_accelerators.get("cpu", {})
            raw_mps = raw_accelerators.get("mps", {})
            raw_cuda = raw_accelerators.get("cuda", {})
            if not all(
                isinstance(item, Mapping) for item in (raw_cpu, raw_mps, raw_cuda)
            ):
                raise ValueError("accelerator capability entries must be mappings")

        raw_devices = raw_cuda.get("devices", ())
        if not isinstance(raw_devices, (list, tuple)):
            raise ValueError("CUDA devices must be a list")
        devices = tuple(
            CUDADeviceMetadata.from_dict(item)
            for item in raw_devices
            if isinstance(item, Mapping)
        )
        if len(devices) != len(raw_devices):
            raise ValueError("each CUDA device metadata entry must be a mapping")

        raw_runtime = raw_cuda.get("runtime_version")
        raw_driver = raw_cuda.get("driver_version")
        raw_attested_hash = payload.get("attested_candidate_artifact_hash")
        raw_report_hash = payload.get("attestation_report_artifact_hash")
        audit = cls(
            created_at_utc=str(payload["created_at_utc"]),
            platform_system=str(raw_platform["system"]),
            platform_release=str(raw_platform["release"]),
            machine=str(raw_platform["machine"]),
            python_implementation=str(raw_platform["python_implementation"]),
            mps_built=_exact_bool(raw_mps, "built"),
            mps_available=_exact_bool(raw_mps, "available"),
            mps_fallback_requested=_exact_bool(raw_mps, "fallback_requested"),
            visible_credential_names=tuple(
                str(item) for item in payload.get("visible_credential_names", ())
            ),
            controls=controls,
            cpu_available=_exact_bool(raw_cpu, "available"),
            cuda_built=_exact_bool(raw_cuda, "built"),
            cuda_available=_exact_bool(raw_cuda, "available"),
            cuda_device_count=int(raw_cuda.get("device_count", 0)),
            cuda_runtime_version=(
                str(raw_runtime) if raw_runtime is not None else None
            ),
            cuda_driver_version=str(raw_driver) if raw_driver is not None else None,
            cuda_devices=devices,
            attested_candidate_artifact_hash=(
                str(raw_attested_hash) if raw_attested_hash is not None else None
            ),
            attestation_report_artifact_hash=(
                str(raw_report_hash) if raw_report_hash is not None else None
            ),
            notes=tuple(str(item) for item in payload.get("notes", ())),
            schema_name=schema_name,
            schema_version=schema_version,
        )
        expected_hash = payload.get("audit_hash")
        if expected_hash is not None and str(expected_hash) != audit.audit_hash:
            raise ValueError("capability audit hash does not match its contents")
        expected_containment = payload.get("strong_containment_proven")
        if (
            expected_containment is not None
            and expected_containment is not audit.strong_containment_proven
        ):
            raise ValueError("capability audit containment summary is inconsistent")
        return audit

    @classmethod
    def from_json(cls, rendered: str) -> "CapabilityAudit":
        payload = json.loads(rendered)
        if not isinstance(payload, Mapping):
            raise ValueError("capability audit JSON must contain an object")
        return cls.from_dict(payload)


def _exact_bool(payload: Mapping[str, Any], name: str) -> bool:
    value = payload.get(name)
    if type(value) is not bool:
        raise ValueError(f"{name} must be boolean")
    return value


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
            CapabilityState.DETECTED if sandbox_exec else CapabilityState.ABSENT,
            "executable discovery",
            (
                f"sandbox tool detected at {sandbox_exec}; enforcement not attested"
                if sandbox_exec
                else "no supported platform sandbox executable detected"
            ),
        ),
    }


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


def _probe_cuda(
    notes: list[str],
) -> tuple[
    bool,
    bool,
    int,
    str | None,
    str | None,
    tuple[CUDADeviceMetadata, ...],
]:
    """Probe CUDA defensively without treating an incomplete probe as availability."""

    version_module = getattr(torch, "version", None)
    raw_runtime_version = getattr(version_module, "cuda", None)
    runtime_version = (
        str(raw_runtime_version) if raw_runtime_version is not None else None
    )
    cuda_built = runtime_version is not None
    cuda_api = getattr(torch, "cuda", None)
    try:
        cuda_available = bool(cuda_api and cuda_api.is_available())
    except Exception as error:  # pragma: no cover - depends on broken driver state
        notes.append(
            f"CUDA availability probe failed: {type(error).__name__}: {error}"
        )
        cuda_available = False

    driver_version: str | None = None
    if cuda_built or cuda_available:
        driver_getter = getattr(cuda_api, "driver_version", None)
        if not callable(driver_getter):
            driver_getter = getattr(
                getattr(torch, "_C", None), "_cuda_getDriverVersion", None
            )
        if callable(driver_getter):
            try:
                raw_driver_version = driver_getter()
                if raw_driver_version is not None:
                    driver_version = str(raw_driver_version)
            except Exception as error:  # pragma: no cover - driver-specific
                notes.append(
                    f"CUDA driver-version probe failed: {type(error).__name__}: {error}"
                )

    if not cuda_available:
        return cuda_built, False, 0, runtime_version, driver_version, ()

    try:
        device_count = int(cuda_api.device_count())
    except Exception as error:  # pragma: no cover - depends on broken driver state
        notes.append(f"CUDA device-count probe failed: {type(error).__name__}: {error}")
        return cuda_built, False, 0, runtime_version, driver_version, ()
    if device_count <= 0:
        notes.append("CUDA reported available but exposed no devices")
        return cuda_built, False, 0, runtime_version, driver_version, ()

    devices: list[CUDADeviceMetadata] = []
    for index in range(device_count):
        properties = None
        try:
            properties = cuda_api.get_device_properties(index)
        except Exception as error:  # pragma: no cover - driver-specific
            notes.append(
                f"CUDA device {index} properties probe failed: "
                f"{type(error).__name__}: {error}"
            )

        raw_name = getattr(properties, "name", None)
        if raw_name is None:
            try:
                raw_name = cuda_api.get_device_name(index)
            except Exception as error:  # pragma: no cover - driver-specific
                notes.append(
                    f"CUDA device {index} name probe failed: "
                    f"{type(error).__name__}: {error}"
                )
                raw_name = f"cuda:{index}"

        raw_major = getattr(properties, "major", None)
        raw_minor = getattr(properties, "minor", None)
        if raw_major is None or raw_minor is None:
            try:
                raw_major, raw_minor = cuda_api.get_device_capability(index)
            except Exception:  # pragma: no cover - optional metadata only
                raw_major = raw_minor = None
        capability = (
            (int(raw_major), int(raw_minor))
            if raw_major is not None and raw_minor is not None
            else None
        )
        raw_total_memory = getattr(properties, "total_memory", None)
        raw_multi_processor_count = getattr(
            properties, "multi_processor_count", None
        )
        raw_uuid = getattr(properties, "uuid", None)
        devices.append(
            CUDADeviceMetadata(
                index=index,
                name=str(raw_name),
                compute_capability=capability,
                total_memory_bytes=(
                    int(raw_total_memory) if raw_total_memory is not None else None
                ),
                multi_processor_count=(
                    int(raw_multi_processor_count)
                    if raw_multi_processor_count is not None
                    else None
                ),
                device_uuid=str(raw_uuid) if raw_uuid is not None else None,
            )
        )
    return (
        cuda_built,
        cuda_available,
        device_count,
        runtime_version,
        driver_version,
        tuple(devices),
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
    (
        cuda_built,
        cuda_available,
        cuda_device_count,
        cuda_runtime_version,
        cuda_driver_version,
        cuda_devices,
    ) = _probe_cuda(notes)
    fallback = str(environment.get("PYTORCH_ENABLE_MPS_FALLBACK", "")).lower()
    return CapabilityAudit(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        python_implementation=platform.python_implementation(),
        mps_built=mps_built,
        mps_available=mps_available,
        mps_fallback_requested=fallback in {"1", "true", "yes", "on"},
        visible_credential_names=_visible_credential_names(environment),
        controls=controls,
        cpu_available=True,
        cuda_built=cuda_built,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_runtime_version=cuda_runtime_version,
        cuda_driver_version=cuda_driver_version,
        cuda_devices=cuda_devices,
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
