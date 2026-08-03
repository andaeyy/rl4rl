"""Intent-to-treat ledger retaining every preselected seed and attempt."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from mechanism.validation import (
    require_bool,
    require_finite,
    require_identifier,
    require_nonnegative_int,
    require_sha256,
)
from replication.policy import ReplicationPolicy


class ReplicationStatus(StrEnum):
    PLANNED = "planned"
    SUCCEEDED = "succeeded"
    CANDIDATE_FAILURE = "candidate_failure"
    SCIENTIFIC_FAILURE = "scientific_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@dataclass(frozen=True)
class ReplicationAttemptRecord:
    policy_id: str
    seed_id: str
    attempt_index: int
    status: ReplicationStatus
    build_id: str | None
    metric_name: str | None
    metric_value: float | None
    final_state_sha256: str | None
    error_class: str | None
    error_message: str | None
    counts_in_intent_to_treat: bool = True

    def __post_init__(self) -> None:
        require_bool(self.counts_in_intent_to_treat, "counts_in_intent_to_treat")
        require_identifier(self.policy_id, "policy_id")
        require_identifier(self.seed_id, "seed_id")
        require_nonnegative_int(self.attempt_index, "attempt_index")
        if not self.counts_in_intent_to_treat:
            raise ValueError("every replication attempt must remain in intent-to-treat")
        if self.status is ReplicationStatus.PLANNED:
            raise ValueError("planned assignments are not execution attempts")
        if self.status is ReplicationStatus.SUCCEEDED:
            if self.build_id is None or self.metric_name is None:
                raise ValueError("successful attempt needs build and metric identifiers")
            require_identifier(self.build_id, "build_id")
            require_identifier(self.metric_name, "metric_name")
            if self.metric_value is None:
                raise ValueError("successful attempt needs a metric value")
            require_finite(self.metric_value, "metric_value")
            if self.final_state_sha256 is None:
                raise ValueError("successful attempt needs a final-state hash")
            require_sha256(self.final_state_sha256, "final_state_sha256")
            if self.error_class is not None or self.error_message is not None:
                raise ValueError("successful attempt cannot contain an error")
        else:
            if self.error_class is None or self.error_message is None:
                raise ValueError("failed attempt needs an error class and message")
            require_identifier(self.error_class, "error_class")
            if self.metric_value is not None or self.metric_name is not None:
                raise ValueError("failed attempt cannot contribute an outcome metric")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "seed_id": self.seed_id,
            "attempt_index": self.attempt_index,
            "status": self.status.value,
            "build_id": self.build_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "final_state_sha256": self.final_state_sha256,
            "error_class": self.error_class,
            "error_message": self.error_message,
            "counts_in_intent_to_treat": self.counts_in_intent_to_treat,
        }


class IntentToTreatLedger:
    """Fixed seed assignments with no API for adding or replacing a seed."""

    def __init__(self, policy: ReplicationPolicy) -> None:
        self.policy = policy
        self._attempts: dict[str, list[ReplicationAttemptRecord]] = {
            seed.seed_id: [] for seed in policy.seeds
        }

    @property
    def seed_ids(self) -> tuple[str, ...]:
        return tuple(seed.seed_id for seed in self.policy.seeds)

    def attempts_for(self, seed_id: str) -> tuple[ReplicationAttemptRecord, ...]:
        try:
            return tuple(self._attempts[seed_id])
        except KeyError as error:
            raise ValueError("seed is not part of the frozen policy") from error

    def terminal_status(self, seed_id: str) -> ReplicationStatus | None:
        attempts = self.attempts_for(seed_id)
        if not attempts:
            return None
        last = attempts[-1]
        if last.status is ReplicationStatus.INFRASTRUCTURE_FAILURE:
            infrastructure_count = sum(
                item.status is ReplicationStatus.INFRASTRUCTURE_FAILURE
                for item in attempts
            )
            if infrastructure_count <= self.policy.failure_policy.max_infrastructure_retries:
                return None
        return last.status

    def record(self, attempt: ReplicationAttemptRecord) -> None:
        if attempt.policy_id != self.policy.policy_id:
            raise ValueError("attempt belongs to a different replication policy")
        if attempt.seed_id not in self._attempts:
            raise ValueError("cannot add or replace a seed after policy freeze")
        if self.terminal_status(attempt.seed_id) is not None:
            raise ValueError("replication seed already has a terminal outcome")
        expected_index = len(self._attempts[attempt.seed_id])
        if attempt.attempt_index != expected_index:
            raise ValueError("attempt indices must be contiguous within a seed")
        if attempt.status is ReplicationStatus.INFRASTRUCTURE_FAILURE:
            if attempt.error_class not in set(
                self.policy.failure_policy.infrastructure_retry_classes
            ):
                raise ValueError("infrastructure failure class was not preregistered")
        self._attempts[attempt.seed_id].append(attempt)

    @property
    def complete(self) -> bool:
        return all(self.terminal_status(seed_id) is not None for seed_id in self.seed_ids)

    def successful_seed_count(self) -> int:
        if not self.complete:
            raise RuntimeError("cannot assess replication success before every seed is terminal")
        rule = self.policy.success_rule.replication_metric
        count = 0
        for seed_id in self.seed_ids:
            attempts = self.attempts_for(seed_id)
            terminal = attempts[-1]
            if (
                terminal.status is ReplicationStatus.SUCCEEDED
                and terminal.metric_name == rule.metric_name
                and terminal.metric_value is not None
                and rule.evaluate(terminal.metric_value)
            ):
                count += 1
        return count

    def meets_success_rule(self) -> bool:
        return (
            self.successful_seed_count()
            >= self.policy.success_rule.minimum_successful_seeds
        )

    def to_dict(self) -> dict[str, Any]:
        assignments = []
        for seed in self.policy.seeds:
            attempts = self.attempts_for(seed.seed_id)
            terminal = self.terminal_status(seed.seed_id)
            assignments.append(
                {
                    "seed": seed.to_dict(),
                    "assignment_status": (
                        ReplicationStatus.PLANNED.value
                        if terminal is None
                        else terminal.value
                    ),
                    "attempts": [attempt.to_dict() for attempt in attempts],
                }
            )
        return {
            "schema_name": "IntentToTreatReplicationLedger",
            "schema_version": "1.0",
            "policy_id": self.policy.policy_id,
            "policy_hash": self.policy.policy_hash,
            "seed_replacement_policy": self.policy.failure_policy.seed_replacement_policy,
            "assignments": assignments,
            "complete": self.complete,
        }
