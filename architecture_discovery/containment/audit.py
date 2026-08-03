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
SCHEMA_VERSION = "1.0"


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
    visible_credential_names: tuple[str, ...]
    controls: Mapping[str, ControlEvidence]
    attested_candidate_artifact_hash: str | None = None
    attestation_report_artifact_hash: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    schema_name: str = SCHEMA_NAME
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "visible_credential_names", tuple(self.visible_credential_names))
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
