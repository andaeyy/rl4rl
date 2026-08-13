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
    # Profiles should bind this explicitly.  The compatibility path infers a
    # non-CPU requested accelerator for older callers that predate this field.
    required_accelerator: str | None = None


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

    requested = _accelerator_kind(request.requested_device)
    required_name = request.required_accelerator or request.requested_device
    required = _accelerator_kind(required_name)
    if requested is None:
        blockers.append(
            f"unsupported requested accelerator {request.requested_device!r}"
        )
    if required is None:
        blockers.append(
            f"unsupported scientific profile accelerator {required_name!r}"
        )
    elif required == "cpu":
        blockers.append(
            "scientific candidate training requires requested_device='mps' or "
            "requested_device='cuda'; CPU execution or fallback is forbidden"
        )
    elif requested != required:
        blockers.append(
            f"scientific profile requires requested_device='{required}'"
        )
    if requested == "cpu" and required != "cpu":
        blockers.append(
            "scientific candidate training requires requested_device='mps' or "
            "requested_device='cuda'; CPU execution or fallback is forbidden"
        )

    if required == "mps":
        if not audit.accelerator_available("mps"):
            blockers.append(
                "MPS is not both built and available in the audited execution environment"
            )
    elif required == "cuda":
        if not audit.accelerator_available("cuda"):
            blockers.append(
                "CUDA is not built, available, and backed by a visible device in the "
                "audited execution environment"
            )
        requested_index = _cuda_device_index(request.requested_device)
        if requested_index is not None and requested_index >= audit.cuda_device_count:
            blockers.append(
                f"requested CUDA device index {requested_index} is not visible in the "
                "audited execution environment"
            )

    # Preserve the historical fail-closed rule even when a new backend is
    # requested: a scientific worker must never inherit a silent MPS-to-CPU
    # escape hatch.
    if audit.mps_fallback_requested:
        blockers.append(
            "PYTORCH_ENABLE_MPS_FALLBACK requests an untracked CPU fallback"
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


def _accelerator_kind(device: str) -> str | None:
    normalized = device.strip().lower()
    if normalized in {"cpu", "mps", "cuda"}:
        return normalized
    if normalized.startswith("cuda:"):
        suffix = normalized.removeprefix("cuda:")
        if suffix.isdigit():
            return "cuda"
    return None


def _cuda_device_index(device: str) -> int | None:
    normalized = device.strip().lower()
    if normalized == "cuda" or not normalized.startswith("cuda:"):
        return None
    suffix = normalized.removeprefix("cuda:")
    return int(suffix) if suffix.isdigit() else None
