"""Scientific execution policy built on explicit containment evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from containment.audit import CapabilityAudit, REQUIRED_STRONG_CONTROLS


class CandidateFormat(StrEnum):
    ARBITRARY_PYTHON = "arbitrary_python"
    ARCHITECTURE_IR = "architecture_ir"


class GatePhase(StrEnum):
    PRE_EXECUTION = "pre_execution"
    POST_EXECUTION = "post_execution"


@dataclass(frozen=True)
class ScientificExecutionRequest:
    candidate_format: CandidateFormat
    requested_device: str
    phase: GatePhase = GatePhase.PRE_EXECUTION
    scientific: bool = True
    ir_validated: bool = False
    trusted_ir_interpreter: bool = False
    runtime_validity_passed: bool = False
    candidate_artifact_hash: str | None = None
    cpu_fallback_allowed: bool = False
    required_cuda_device_name: str | None = None


@dataclass(frozen=True)
class ScientificGateDecision:
    allowed: bool
    scientific: bool
    phase: GatePhase
    candidate_format: CandidateFormat
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    audit_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "scientific": self.scientific,
            "phase": self.phase.value,
            "candidate_format": self.candidate_format.value,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "audit_hash": self.audit_hash,
        }


def assess_scientific_execution(
    audit: CapabilityAudit,
    request: ScientificExecutionRequest,
) -> ScientificGateDecision:
    """Return a fail-closed decision for candidate construction or qualification."""

    blockers: list[str] = []
    warnings = [
        "source scanning and in-process guards do not establish an OS boundary"
    ]
    if not request.scientific:
        warnings.append("engineering execution is not scientific validation")
        return ScientificGateDecision(
            allowed=True,
            scientific=False,
            phase=request.phase,
            candidate_format=request.candidate_format,
            blockers=(),
            warnings=tuple(warnings),
            audit_hash=audit.audit_hash,
        )

    requested_device = request.requested_device.lower()
    requested_backend = requested_device.split(":", maxsplit=1)[0]
    if request.cpu_fallback_allowed:
        blockers.append("scientific candidate training forbids CPU fallback")
    if audit.mps_fallback_requested:
        blockers.append(
            "PYTORCH_ENABLE_MPS_FALLBACK requests an untracked CPU fallback"
        )
    if requested_backend == "mps":
        if requested_device != "mps":
            blockers.append("MPS scientific execution must request exactly 'mps'")
        if not audit.mps_built or not audit.mps_available:
            blockers.append(
                "MPS is not both built and available in the audited execution environment"
            )
    elif requested_backend == "cuda":
        if requested_device != "cuda":
            blockers.append(
                "CUDA scientific execution must request logical device 'cuda'; "
                "the worker resolves cuda:0 inside its allocation"
            )
        if not audit.cuda_available or audit.cuda_device_count < 1:
            blockers.append("CUDA is unavailable in the audited execution environment")
        if not audit.cuda_allocation_validated:
            blockers.append(
                "CUDA scientific execution requires a validated scheduler GPU allocation"
            )
        if request.required_cuda_device_name is not None:
            observed_names = {
                str(device.get("name", "")) for device in audit.cuda_devices
            }
            if not any(
                request.required_cuda_device_name.lower() in name.lower()
                for name in observed_names
            ):
                blockers.append(
                    "allocated CUDA device does not match required device name "
                    f"{request.required_cuda_device_name!r}"
                )
        warnings.append(
            "CUDA and MPS are distinct hardware conditions and require an explicit "
            "analysis plan before any cross-backend comparison"
        )
    else:
        blockers.append(
            "scientific candidate training requires requested_device='mps' or 'cuda'"
        )

    if request.candidate_format is CandidateFormat.ARBITRARY_PYTHON:
        unproven = [
            name for name in REQUIRED_STRONG_CONTROLS if not audit.controls[name].proven
        ]
        if unproven:
            blockers.append(
                "arbitrary Python requires proven OS containment controls: "
                + ", ".join(unproven)
            )
        elif (
            request.candidate_artifact_hash is None
            or audit.attested_candidate_artifact_hash
            != request.candidate_artifact_hash
        ):
            blockers.append(
                "OS-boundary attestation is not bound to this exact candidate artifact"
            )
    elif request.candidate_format is CandidateFormat.ARCHITECTURE_IR:
        if not request.ir_validated:
            blockers.append("architecture IR did not pass typed graph validation")
        if not request.trusted_ir_interpreter:
            blockers.append("architecture IR is not bound to a trusted interpreter")
    else:  # pragma: no cover - enum makes this unreachable without type abuse
        blockers.append("unknown candidate format")

    if request.phase is GatePhase.POST_EXECUTION and not request.runtime_validity_passed:
        blockers.append("runtime transformer-validity evidence did not pass")

    return ScientificGateDecision(
        allowed=not blockers,
        scientific=True,
        phase=request.phase,
        candidate_format=request.candidate_format,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        audit_hash=audit.audit_hash,
    )
