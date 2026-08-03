"""Preregistered replication policy and immutable policy receipts."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable

from mechanism.plans import TrainingBudget
from mechanism.validation import (
    require_bool,
    require_finite,
    require_identifier,
    require_nonnegative_int,
    require_positive_int,
    require_sha256,
    require_text,
)
from study.serialization import content_hash, create_json_exclusive, read_json


class Comparator(StrEnum):
    GREATER_EQUAL = ">="
    GREATER = ">"
    LESS_EQUAL = "<="
    LESS = "<"

    @property
    def function(self) -> Callable[[float, float], bool]:
        return {
            Comparator.GREATER_EQUAL: operator.ge,
            Comparator.GREATER: operator.gt,
            Comparator.LESS_EQUAL: operator.le,
            Comparator.LESS: operator.lt,
        }[self]


@dataclass(frozen=True)
class MetricRule:
    metric_name: str
    comparator: Comparator
    threshold: float

    def __post_init__(self) -> None:
        require_identifier(self.metric_name, "metric_name")
        require_finite(self.threshold, "threshold")

    def evaluate(self, value: float) -> bool:
        return self.comparator.function(require_finite(value, self.metric_name), self.threshold)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "comparator": self.comparator.value,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetricRule":
        return cls(
            metric_name=require_identifier(payload["metric_name"], "metric_name"),
            comparator=Comparator(payload["comparator"]),
            threshold=require_finite(payload["threshold"], "threshold"),
        )


@dataclass(frozen=True)
class PromotionRule:
    layer_b_qualification_required: bool
    mechanism_claim_required: bool
    qualification_metric: MetricRule

    def __post_init__(self) -> None:
        require_bool(
            self.layer_b_qualification_required,
            "layer_b_qualification_required",
        )
        require_bool(self.mechanism_claim_required, "mechanism_claim_required")
        if not self.layer_b_qualification_required:
            raise ValueError("replication promotion must require Layer B qualification")
        if not self.mechanism_claim_required:
            raise ValueError("replication promotion must require a mechanism claim")

    def evaluate(
        self,
        *,
        layer_b_qualified: bool,
        mechanism_claim_present: bool,
        qualification_metrics: dict[str, float],
    ) -> bool:
        if self.layer_b_qualification_required and not layer_b_qualified:
            return False
        if self.mechanism_claim_required and not mechanism_claim_present:
            return False
        try:
            value = qualification_metrics[self.qualification_metric.metric_name]
        except KeyError as error:
            raise ValueError("promotion metric is missing") from error
        return self.qualification_metric.evaluate(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_b_qualification_required": self.layer_b_qualification_required,
            "mechanism_claim_required": self.mechanism_claim_required,
            "qualification_metric": self.qualification_metric.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromotionRule":
        return cls(
            layer_b_qualification_required=require_bool(
                payload["layer_b_qualification_required"],
                "layer_b_qualification_required",
            ),
            mechanism_claim_required=require_bool(
                payload["mechanism_claim_required"], "mechanism_claim_required"
            ),
            qualification_metric=MetricRule.from_dict(
                payload["qualification_metric"]
            ),
        )


@dataclass(frozen=True)
class SuccessRule:
    replication_metric: MetricRule
    minimum_successful_seeds: int
    require_all_preselected_seeds_terminal: bool = True

    def __post_init__(self) -> None:
        require_bool(
            self.require_all_preselected_seeds_terminal,
            "require_all_preselected_seeds_terminal",
        )
        require_positive_int(self.minimum_successful_seeds, "minimum_successful_seeds")
        if not self.require_all_preselected_seeds_terminal:
            raise ValueError("success cannot be decided while preselected seeds are missing")

    def to_dict(self) -> dict[str, Any]:
        return {
            "replication_metric": self.replication_metric.to_dict(),
            "minimum_successful_seeds": self.minimum_successful_seeds,
            "require_all_preselected_seeds_terminal": self.require_all_preselected_seeds_terminal,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SuccessRule":
        return cls(
            replication_metric=MetricRule.from_dict(payload["replication_metric"]),
            minimum_successful_seeds=require_positive_int(
                payload["minimum_successful_seeds"], "minimum_successful_seeds"
            ),
            require_all_preselected_seeds_terminal=require_bool(
                payload["require_all_preselected_seeds_terminal"],
                "require_all_preselected_seeds_terminal",
            ),
        )


@dataclass(frozen=True)
class FailurePolicy:
    infrastructure_retry_classes: tuple[str, ...]
    max_infrastructure_retries: int
    candidate_failures_count_unsuccessful: bool = True
    scientific_failures_count_unsuccessful: bool = True
    retain_every_attempt: bool = True
    seed_replacement_policy: str = "forbidden"

    def __post_init__(self) -> None:
        require_bool(
            self.candidate_failures_count_unsuccessful,
            "candidate_failures_count_unsuccessful",
        )
        require_bool(
            self.scientific_failures_count_unsuccessful,
            "scientific_failures_count_unsuccessful",
        )
        require_bool(self.retain_every_attempt, "retain_every_attempt")
        object.__setattr__(
            self,
            "infrastructure_retry_classes",
            tuple(self.infrastructure_retry_classes),
        )
        for name in self.infrastructure_retry_classes:
            require_identifier(name, "infrastructure retry class")
        if len(set(self.infrastructure_retry_classes)) != len(
            self.infrastructure_retry_classes
        ):
            raise ValueError("infrastructure retry classes must be unique")
        require_nonnegative_int(
            self.max_infrastructure_retries, "max_infrastructure_retries"
        )
        if not self.candidate_failures_count_unsuccessful:
            raise ValueError("candidate failures must remain unsuccessful in ITT")
        if not self.scientific_failures_count_unsuccessful:
            raise ValueError("scientific failures must remain unsuccessful in ITT")
        if not self.retain_every_attempt:
            raise ValueError("replication must retain every attempt")
        if self.seed_replacement_policy != "forbidden":
            raise ValueError("replacement of failed replication seeds is forbidden")

    def to_dict(self) -> dict[str, Any]:
        return {
            "infrastructure_retry_classes": list(
                self.infrastructure_retry_classes
            ),
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "candidate_failures_count_unsuccessful": self.candidate_failures_count_unsuccessful,
            "scientific_failures_count_unsuccessful": self.scientific_failures_count_unsuccessful,
            "retain_every_attempt": self.retain_every_attempt,
            "seed_replacement_policy": self.seed_replacement_policy,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FailurePolicy":
        return cls(
            infrastructure_retry_classes=tuple(
                require_identifier(item, "infrastructure retry class")
                for item in payload["infrastructure_retry_classes"]
            ),
            max_infrastructure_retries=require_nonnegative_int(
                payload["max_infrastructure_retries"],
                "max_infrastructure_retries",
            ),
            candidate_failures_count_unsuccessful=require_bool(
                payload["candidate_failures_count_unsuccessful"],
                "candidate_failures_count_unsuccessful",
            ),
            scientific_failures_count_unsuccessful=require_bool(
                payload["scientific_failures_count_unsuccessful"],
                "scientific_failures_count_unsuccessful",
            ),
            retain_every_attempt=require_bool(
                payload["retain_every_attempt"], "retain_every_attempt"
            ),
            seed_replacement_policy=require_text(
                payload["seed_replacement_policy"], "seed_replacement_policy"
            ),
        )


@dataclass(frozen=True)
class ReplicationSeed:
    seed_id: str
    initialization_seed: int
    training_seed: int
    data_order_seed: int
    evaluation_seed: int

    def __post_init__(self) -> None:
        require_identifier(self.seed_id, "seed_id")
        for field_name in (
            "initialization_seed",
            "training_seed",
            "data_order_seed",
            "evaluation_seed",
        ):
            require_nonnegative_int(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_id": self.seed_id,
            "initialization_seed": self.initialization_seed,
            "training_seed": self.training_seed,
            "data_order_seed": self.data_order_seed,
            "evaluation_seed": self.evaluation_seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplicationSeed":
        return cls(
            seed_id=require_identifier(payload["seed_id"], "seed_id"),
            initialization_seed=require_nonnegative_int(
                payload["initialization_seed"], "initialization_seed"
            ),
            training_seed=require_nonnegative_int(
                payload["training_seed"], "training_seed"
            ),
            data_order_seed=require_nonnegative_int(
                payload["data_order_seed"], "data_order_seed"
            ),
            evaluation_seed=require_nonnegative_int(
                payload["evaluation_seed"], "evaluation_seed"
            ),
        )


@dataclass(frozen=True)
class ReplicationPolicy:
    policy_id: str
    study_id: str
    claim_id: str
    frozen_snapshot_id: str
    frozen_snapshot_sha256: str
    seeds: tuple[ReplicationSeed, ...]
    promotion_rule: PromotionRule
    success_rule: SuccessRule
    failure_policy: FailurePolicy
    training_budget: TrainingBudget
    clean_room_reimplementation_required: bool
    scientific: bool
    schema_name: str = field(default="ReplicationPolicy", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_bool(
            self.clean_room_reimplementation_required,
            "clean_room_reimplementation_required",
        )
        require_bool(self.scientific, "scientific")
        require_identifier(self.policy_id, "policy_id")
        require_identifier(self.study_id, "study_id")
        require_identifier(self.claim_id, "claim_id")
        require_identifier(self.frozen_snapshot_id, "frozen_snapshot_id")
        require_sha256(self.frozen_snapshot_sha256, "frozen_snapshot_sha256")
        object.__setattr__(self, "seeds", tuple(self.seeds))
        if not self.seeds:
            raise ValueError("replication policy needs preselected seeds")
        if len({seed.seed_id for seed in self.seeds}) != len(self.seeds):
            raise ValueError("replication seed IDs must be unique")
        seed_tuples = {
            (
                seed.initialization_seed,
                seed.training_seed,
                seed.data_order_seed,
                seed.evaluation_seed,
            )
            for seed in self.seeds
        }
        if len(seed_tuples) != len(self.seeds):
            raise ValueError("replication seed bundles must be unique")
        if self.success_rule.minimum_successful_seeds > len(self.seeds):
            raise ValueError("success rule requires more successes than seeds")
        if not self.clean_room_reimplementation_required:
            raise ValueError("replication must require a clean-room implementation")

    @property
    def policy_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "study_id": self.study_id,
            "claim_id": self.claim_id,
            "frozen_snapshot_id": self.frozen_snapshot_id,
            "frozen_snapshot_sha256": self.frozen_snapshot_sha256,
            "seeds": [seed.to_dict() for seed in self.seeds],
            "promotion_rule": self.promotion_rule.to_dict(),
            "success_rule": self.success_rule.to_dict(),
            "failure_policy": self.failure_policy.to_dict(),
            "training_budget": self.training_budget.to_dict(),
            "clean_room_reimplementation_required": self.clean_room_reimplementation_required,
            "scientific": self.scientific,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReplicationPolicy":
        if payload.get("schema_name") != "ReplicationPolicy":
            raise ValueError("not a ReplicationPolicy record")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported ReplicationPolicy schema version")
        return cls(
            policy_id=require_identifier(payload["policy_id"], "policy_id"),
            study_id=require_identifier(payload["study_id"], "study_id"),
            claim_id=require_identifier(payload["claim_id"], "claim_id"),
            frozen_snapshot_id=require_identifier(
                payload["frozen_snapshot_id"], "frozen_snapshot_id"
            ),
            frozen_snapshot_sha256=require_sha256(
                payload["frozen_snapshot_sha256"], "frozen_snapshot_sha256"
            ),
            seeds=tuple(ReplicationSeed.from_dict(item) for item in payload["seeds"]),
            promotion_rule=PromotionRule.from_dict(payload["promotion_rule"]),
            success_rule=SuccessRule.from_dict(payload["success_rule"]),
            failure_policy=FailurePolicy.from_dict(payload["failure_policy"]),
            training_budget=TrainingBudget.from_dict(payload["training_budget"]),
            clean_room_reimplementation_required=require_bool(
                payload["clean_room_reimplementation_required"],
                "clean_room_reimplementation_required",
            ),
            scientific=require_bool(payload["scientific"], "scientific"),
        )


@dataclass(frozen=True)
class FrozenReplicationPolicy:
    path: Path
    policy: ReplicationPolicy
    policy_hash: str
    frozen_payload_sha256: str

    def verify(self) -> None:
        payload = read_json(self.path)
        if payload.get("schema_name") != "FrozenReplicationPolicy":
            raise ValueError("frozen replication-policy envelope has the wrong schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported frozen replication-policy schema version")
        stored_policy = ReplicationPolicy.from_dict(payload["policy"])
        stored_hash = require_sha256(payload["policy_hash"], "policy_hash")
        if stored_hash != stored_policy.policy_hash or stored_hash != self.policy_hash:
            raise ValueError("frozen replication policy hash mismatch")
        if stored_policy.to_dict() != self.policy.to_dict():
            raise ValueError("frozen replication policy does not match the receipt")
        if content_hash(payload) != self.frozen_payload_sha256:
            raise ValueError("frozen replication-policy envelope was modified")


def freeze_replication_policy(
    policy: ReplicationPolicy, path: str | Path
) -> FrozenReplicationPolicy:
    destination = Path(path).resolve()
    payload = {
        "schema_name": "FrozenReplicationPolicy",
        "schema_version": "1.0",
        "policy_hash": policy.policy_hash,
        "policy": policy.to_dict(),
    }
    create_json_exclusive(destination, payload)
    return FrozenReplicationPolicy(
        path=destination,
        policy=policy,
        policy_hash=policy.policy_hash,
        frozen_payload_sha256=content_hash(payload),
    )


def load_frozen_replication_policy(path: str | Path) -> FrozenReplicationPolicy:
    destination = Path(path).resolve()
    payload = read_json(destination)
    policy = ReplicationPolicy.from_dict(payload["policy"])
    receipt = FrozenReplicationPolicy(
        path=destination,
        policy=policy,
        policy_hash=require_sha256(payload["policy_hash"], "policy_hash"),
        frozen_payload_sha256=content_hash(payload),
    )
    receipt.verify()
    return receipt
