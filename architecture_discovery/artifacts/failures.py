"""Predeclared failure taxonomy and infrastructure-only rerun authorization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from artifacts.records import content_sha256, require_identifier


class FailureDomain(StrEnum):
    SCIENTIFIC = "scientific"
    CANDIDATE = "candidate"
    INFRASTRUCTURE = "infrastructure"


class FailureClass(StrEnum):
    NONQUALIFYING_RESULT = "nonqualifying_result"
    SCIENTIFIC_BUDGET_EXHAUSTED = "scientific_budget_exhausted"
    PROPOSAL_PARSE = "proposal_parse"
    CANDIDATE_IMPORT = "candidate_import"
    INVALID_TRANSFORMER = "invalid_transformer"
    TRAINING_DIVERGENCE = "training_divergence"
    PROVIDER_TRANSIENT = "provider_transient"
    WORKER_CRASH = "worker_crash"
    MPS_DRIVER_FAILURE = "mps_driver_failure"
    CUDA_DRIVER_FAILURE = "cuda_driver_failure"
    CUDA_DETERMINISTIC_KERNEL_UNAVAILABLE = (
        "cuda_deterministic_kernel_unavailable"
    )
    ACCELERATOR_CLEANUP_FAILURE = "accelerator_cleanup_failure"
    FILESYSTEM_IO = "filesystem_io"
    POWER_INTERRUPTION = "power_interruption"
    MPS_UNAVAILABLE = "mps_unavailable"
    CUDA_UNAVAILABLE = "cuda_unavailable"
    MODAL_INFRASTRUCTURE_FAILURE = "modal_infrastructure_failure"
    CONTAINMENT_UNAVAILABLE = "containment_unavailable"


_DOMAIN_BY_CLASS = {
    FailureClass.NONQUALIFYING_RESULT: FailureDomain.SCIENTIFIC,
    FailureClass.SCIENTIFIC_BUDGET_EXHAUSTED: FailureDomain.SCIENTIFIC,
    FailureClass.PROPOSAL_PARSE: FailureDomain.CANDIDATE,
    FailureClass.CANDIDATE_IMPORT: FailureDomain.CANDIDATE,
    FailureClass.INVALID_TRANSFORMER: FailureDomain.CANDIDATE,
    FailureClass.TRAINING_DIVERGENCE: FailureDomain.CANDIDATE,
    FailureClass.PROVIDER_TRANSIENT: FailureDomain.INFRASTRUCTURE,
    FailureClass.WORKER_CRASH: FailureDomain.INFRASTRUCTURE,
    FailureClass.MPS_DRIVER_FAILURE: FailureDomain.INFRASTRUCTURE,
    FailureClass.CUDA_DRIVER_FAILURE: FailureDomain.INFRASTRUCTURE,
    FailureClass.CUDA_DETERMINISTIC_KERNEL_UNAVAILABLE: FailureDomain.INFRASTRUCTURE,
    FailureClass.ACCELERATOR_CLEANUP_FAILURE: FailureDomain.INFRASTRUCTURE,
    FailureClass.FILESYSTEM_IO: FailureDomain.INFRASTRUCTURE,
    FailureClass.POWER_INTERRUPTION: FailureDomain.INFRASTRUCTURE,
    FailureClass.MPS_UNAVAILABLE: FailureDomain.INFRASTRUCTURE,
    FailureClass.CUDA_UNAVAILABLE: FailureDomain.INFRASTRUCTURE,
    FailureClass.MODAL_INFRASTRUCTURE_FAILURE: FailureDomain.INFRASTRUCTURE,
    FailureClass.CONTAINMENT_UNAVAILABLE: FailureDomain.INFRASTRUCTURE,
}


DEFAULT_RERUNNABLE_INFRASTRUCTURE_CLASSES: frozenset[FailureClass] = frozenset(
    {
        FailureClass.PROVIDER_TRANSIENT,
        FailureClass.WORKER_CRASH,
        FailureClass.MPS_DRIVER_FAILURE,
        FailureClass.CUDA_DRIVER_FAILURE,
        FailureClass.MODAL_INFRASTRUCTURE_FAILURE,
        FailureClass.FILESYSTEM_IO,
        FailureClass.POWER_INTERRUPTION,
    }
)


@dataclass(frozen=True)
class FailureRecord:
    failure_id: str
    attempt_id: str
    failure_class: FailureClass
    failure_domain: FailureDomain
    stage: str
    terminal: bool

    def __post_init__(self) -> None:
        require_identifier(self.failure_id, "failure_id")
        require_identifier(self.attempt_id, "attempt_id")
        require_identifier(self.stage, "stage")
        if not isinstance(self.terminal, bool):
            raise ValueError("terminal must be boolean")
        if _DOMAIN_BY_CLASS[self.failure_class] is not self.failure_domain:
            raise ValueError("failure class does not belong to its declared domain")

    @classmethod
    def create(
        cls,
        *,
        attempt_id: str,
        failure_class: FailureClass | str,
        stage: str,
        terminal: bool = True,
    ) -> FailureRecord:
        resolved_class = FailureClass(failure_class)
        identity = {
            "attempt_id": attempt_id,
            "failure_class": resolved_class.value,
            "stage": stage,
            "terminal": terminal,
        }
        return cls(
            failure_id=f"failure-{content_sha256(identity)[:24]}",
            attempt_id=attempt_id,
            failure_class=resolved_class,
            failure_domain=_DOMAIN_BY_CLASS[resolved_class],
            stage=stage,
            terminal=terminal,
        )

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "attempt_id": self.attempt_id,
            "failure_class": self.failure_class.value,
            "failure_domain": self.failure_domain.value,
            "stage": self.stage,
            "terminal": self.terminal,
        }


@dataclass(frozen=True)
class RerunPolicy:
    rerunnable_classes: frozenset[FailureClass] = (
        DEFAULT_RERUNNABLE_INFRASTRUCTURE_CLASSES
    )
    max_linked_attempts: int = 2
    schema_name: str = "RerunPolicy"
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rerunnable_classes",
            frozenset(self.rerunnable_classes),
        )
        if self.max_linked_attempts < 1:
            raise ValueError("max_linked_attempts must be positive")
        if any(
            _DOMAIN_BY_CLASS[item] is not FailureDomain.INFRASTRUCTURE
            for item in self.rerunnable_classes
        ):
            raise ValueError("rerun policy may authorize infrastructure classes only")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "rerunnable_classes": sorted(
                item.value for item in self.rerunnable_classes
            ),
            "max_linked_attempts": self.max_linked_attempts,
        }


@dataclass(frozen=True)
class RerunAuthorization:
    authorization_id: str
    assigned_run_id: str
    previous_attempt_id: str
    rerun_attempt_id: str
    attempt_number: int
    triggering_failure_id: str
    triggering_failure_class: FailureClass
    policy_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "authorization_id",
            "assigned_run_id",
            "previous_attempt_id",
            "rerun_attempt_id",
            "triggering_failure_id",
        ):
            require_identifier(getattr(self, name), name)
        if self.attempt_number < 1:
            raise ValueError("attempt_number must be positive")
        if len(self.policy_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.policy_sha256
        ):
            raise ValueError("policy_sha256 must be a lowercase SHA-256 digest")

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "authorization_id": self.authorization_id,
            "assigned_run_id": self.assigned_run_id,
            "previous_attempt_id": self.previous_attempt_id,
            "rerun_attempt_id": self.rerun_attempt_id,
            "attempt_number": self.attempt_number,
            "triggering_failure_id": self.triggering_failure_id,
            "triggering_failure_class": self.triggering_failure_class.value,
            "policy_sha256": self.policy_sha256,
        }


class RerunNotAuthorized(ValueError):
    pass


def authorize_rerun(
    *,
    assigned_run_id: str,
    previous_attempt_id: str,
    attempt_number: int,
    failure: FailureRecord,
    policy: RerunPolicy,
) -> RerunAuthorization:
    require_identifier(assigned_run_id, "assigned_run_id")
    require_identifier(previous_attempt_id, "previous_attempt_id")
    if not failure.terminal:
        raise RerunNotAuthorized("a nonterminal failure cannot authorize a rerun")
    if failure.attempt_id != previous_attempt_id:
        raise RerunNotAuthorized("failure does not belong to the previous attempt")
    if failure.failure_domain is not FailureDomain.INFRASTRUCTURE:
        raise RerunNotAuthorized(
            "candidate and scientific failures remain ITT outcomes"
        )
    if failure.failure_class not in policy.rerunnable_classes:
        raise RerunNotAuthorized("infrastructure failure class was not preregistered")
    if attempt_number < 1 or attempt_number > policy.max_linked_attempts:
        raise RerunNotAuthorized("linked rerun attempt exceeds the frozen policy")
    identity = {
        "assigned_run_id": assigned_run_id,
        "previous_attempt_id": previous_attempt_id,
        "attempt_number": attempt_number,
        "failure_id": failure.failure_id,
        "policy_sha256": policy.policy_sha256,
    }
    digest = content_sha256(identity)
    return RerunAuthorization(
        authorization_id=f"rerun-auth-{digest[:24]}",
        assigned_run_id=assigned_run_id,
        previous_attempt_id=previous_attempt_id,
        rerun_attempt_id=f"{assigned_run_id}-rerun-{attempt_number}-{digest[:10]}",
        attempt_number=attempt_number,
        triggering_failure_id=failure.failure_id,
        triggering_failure_class=failure.failure_class,
        policy_sha256=policy.policy_sha256,
    )
