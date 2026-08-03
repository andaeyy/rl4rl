"""Preregistered plans for causal mechanism experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from mechanism.claims import MechanismClaim
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


class ArmKind(StrEnum):
    ORIGINAL = "original"
    COMPONENT_REMOVED = "component_removed"
    COMPONENT_REPLACED = "component_replaced"
    CAPACITY_MATCHED = "capacity_matched"
    COMPUTE_MATCHED = "compute_matched"


class InterventionOperation(StrEnum):
    ZERO = "zero"
    BYPASS = "bypass"
    PERMUTE = "permute"
    CLAMP = "clamp"
    REPLACE = "replace"
    RESTORE = "restore"


@dataclass(frozen=True)
class TrainingBudget:
    dataset_id: str
    optimizer_id: str
    schedule_id: str
    max_steps: int
    max_examples: int
    wall_time_seconds: float

    def __post_init__(self) -> None:
        require_identifier(self.dataset_id, "dataset_id")
        require_identifier(self.optimizer_id, "optimizer_id")
        require_identifier(self.schedule_id, "schedule_id")
        require_positive_int(self.max_steps, "max_steps")
        require_positive_int(self.max_examples, "max_examples")
        if require_finite(self.wall_time_seconds, "wall_time_seconds") <= 0:
            raise ValueError("wall_time_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "optimizer_id": self.optimizer_id,
            "schedule_id": self.schedule_id,
            "max_steps": self.max_steps,
            "max_examples": self.max_examples,
            "wall_time_seconds": self.wall_time_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingBudget":
        return cls(
            dataset_id=require_identifier(payload["dataset_id"], "dataset_id"),
            optimizer_id=require_identifier(payload["optimizer_id"], "optimizer_id"),
            schedule_id=require_identifier(payload["schedule_id"], "schedule_id"),
            max_steps=require_positive_int(payload["max_steps"], "max_steps"),
            max_examples=require_positive_int(
                payload["max_examples"], "max_examples"
            ),
            wall_time_seconds=require_finite(
                payload["wall_time_seconds"], "wall_time_seconds"
            ),
        )


@dataclass(frozen=True)
class SeedBundle:
    bundle_id: str
    initialization_seed: int
    training_seed: int
    data_order_seed: int

    def __post_init__(self) -> None:
        require_identifier(self.bundle_id, "bundle_id")
        for field_name in ("initialization_seed", "training_seed", "data_order_seed"):
            require_nonnegative_int(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "initialization_seed": self.initialization_seed,
            "training_seed": self.training_seed,
            "data_order_seed": self.data_order_seed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SeedBundle":
        return cls(
            bundle_id=require_identifier(payload["bundle_id"], "bundle_id"),
            initialization_seed=require_nonnegative_int(
                payload["initialization_seed"], "initialization_seed"
            ),
            training_seed=require_nonnegative_int(
                payload["training_seed"], "training_seed"
            ),
            data_order_seed=require_nonnegative_int(
                payload["data_order_seed"], "data_order_seed"
            ),
        )


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    kind: ArmKind
    builder_variant_id: str
    transformation: str
    target_component: str
    initialization_policy: str = "from_scratch"
    initial_checkpoint_id: str | None = None
    target_parameter_count: int | None = None

    def __post_init__(self) -> None:
        require_identifier(self.arm_id, "arm_id")
        require_identifier(self.builder_variant_id, "builder_variant_id")
        require_text(self.transformation, "transformation")
        require_text(self.target_component, "target_component")
        if self.initialization_policy != "from_scratch":
            raise ValueError("every mechanism arm must initialize from scratch")
        if self.initial_checkpoint_id is not None:
            raise ValueError("mechanism arms cannot load a parent or candidate checkpoint")
        if self.target_parameter_count is not None:
            require_positive_int(self.target_parameter_count, "target_parameter_count")
        if self.kind is ArmKind.ORIGINAL and self.transformation != "none":
            raise ValueError("the original arm must declare transformation='none'")
        if self.kind is ArmKind.CAPACITY_MATCHED and self.target_parameter_count is None:
            raise ValueError("the capacity-matched arm needs a target parameter count")
        if self.kind is not ArmKind.CAPACITY_MATCHED and self.target_parameter_count is not None:
            raise ValueError(
                "target parameter count is only a capacity-control instruction"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "kind": self.kind.value,
            "builder_variant_id": self.builder_variant_id,
            "transformation": self.transformation,
            "target_component": self.target_component,
            "initialization_policy": self.initialization_policy,
            "initial_checkpoint_id": self.initial_checkpoint_id,
            "target_parameter_count": self.target_parameter_count,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ArmSpec":
        return cls(
            arm_id=require_identifier(payload["arm_id"], "arm_id"),
            kind=ArmKind(payload["kind"]),
            builder_variant_id=require_identifier(
                payload["builder_variant_id"], "builder_variant_id"
            ),
            transformation=require_text(payload["transformation"], "transformation"),
            target_component=require_text(
                payload["target_component"], "target_component"
            ),
            initialization_policy=require_text(
                payload["initialization_policy"], "initialization_policy"
            ),
            initial_checkpoint_id=payload["initial_checkpoint_id"],
            target_parameter_count=(
                None
                if payload["target_parameter_count"] is None
                else require_positive_int(
                    payload["target_parameter_count"], "target_parameter_count"
                )
            ),
        )


@dataclass(frozen=True)
class InterventionSpec:
    intervention_id: str
    hook_id: str
    target_component: str
    operation: InterventionOperation
    value: float | None
    predicted_effect: str
    falsification_condition: str

    def __post_init__(self) -> None:
        require_identifier(self.intervention_id, "intervention_id")
        require_identifier(self.hook_id, "hook_id")
        require_text(self.target_component, "target_component")
        if self.value is not None:
            require_finite(self.value, "intervention value")
        require_text(self.predicted_effect, "predicted_effect")
        require_text(self.falsification_condition, "falsification_condition")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "hook_id": self.hook_id,
            "target_component": self.target_component,
            "operation": self.operation.value,
            "value": self.value,
            "predicted_effect": self.predicted_effect,
            "falsification_condition": self.falsification_condition,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterventionSpec":
        return cls(
            intervention_id=require_identifier(
                payload["intervention_id"], "intervention_id"
            ),
            hook_id=require_identifier(payload["hook_id"], "hook_id"),
            target_component=require_text(
                payload["target_component"], "target_component"
            ),
            operation=InterventionOperation(payload["operation"]),
            value=(
                None
                if payload["value"] is None
                else require_finite(payload["value"], "intervention value")
            ),
            predicted_effect=require_text(
                payload["predicted_effect"], "predicted_effect"
            ),
            falsification_condition=require_text(
                payload["falsification_condition"], "falsification_condition"
            ),
        )


@dataclass(frozen=True)
class RescueSpec:
    rescue_id: str
    disabling_intervention_id: str
    restoring_intervention_id: str
    predicted_recovery: str

    def __post_init__(self) -> None:
        require_identifier(self.rescue_id, "rescue_id")
        require_identifier(
            self.disabling_intervention_id, "disabling_intervention_id"
        )
        require_identifier(
            self.restoring_intervention_id, "restoring_intervention_id"
        )
        if self.disabling_intervention_id == self.restoring_intervention_id:
            raise ValueError("rescue must use distinct disable and restore interventions")
        require_text(self.predicted_recovery, "predicted_recovery")

    def to_dict(self) -> dict[str, str]:
        return {
            "rescue_id": self.rescue_id,
            "disabling_intervention_id": self.disabling_intervention_id,
            "restoring_intervention_id": self.restoring_intervention_id,
            "predicted_recovery": self.predicted_recovery,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RescueSpec":
        return cls(
            rescue_id=require_identifier(payload["rescue_id"], "rescue_id"),
            disabling_intervention_id=require_identifier(
                payload["disabling_intervention_id"],
                "disabling_intervention_id",
            ),
            restoring_intervention_id=require_identifier(
                payload["restoring_intervention_id"],
                "restoring_intervention_id",
            ),
            predicted_recovery=require_text(
                payload["predicted_recovery"], "predicted_recovery"
            ),
        )


def _positive_unique(values: tuple[int, ...], field_name: str) -> tuple[int, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    for value in normalized:
        require_positive_int(value, field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


def _nonempty_unique(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{field_name} cannot be empty")
    for value in normalized:
        require_identifier(value, field_name)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must not contain duplicates")
    return normalized


@dataclass(frozen=True)
class CounterfactualGrid:
    carry_depths: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    numeric_bases: tuple[int, ...]
    symbol_mappings: tuple[str, ...]
    representation_shifts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "carry_depths", _positive_unique(self.carry_depths, "carry_depths")
        )
        object.__setattr__(
            self,
            "sequence_lengths",
            _positive_unique(self.sequence_lengths, "sequence_lengths"),
        )
        bases = _positive_unique(self.numeric_bases, "numeric_bases")
        if any(base < 2 for base in bases):
            raise ValueError("numeric bases must be at least two")
        object.__setattr__(self, "numeric_bases", bases)
        object.__setattr__(
            self,
            "symbol_mappings",
            _nonempty_unique(self.symbol_mappings, "symbol_mappings"),
        )
        object.__setattr__(
            self,
            "representation_shifts",
            _nonempty_unique(self.representation_shifts, "representation_shifts"),
        )

    @property
    def case_count(self) -> int:
        return (
            len(self.carry_depths)
            * len(self.sequence_lengths)
            * len(self.numeric_bases)
            * len(self.symbol_mappings)
            * len(self.representation_shifts)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "carry_depths": list(self.carry_depths),
            "sequence_lengths": list(self.sequence_lengths),
            "numeric_bases": list(self.numeric_bases),
            "symbol_mappings": list(self.symbol_mappings),
            "representation_shifts": list(self.representation_shifts),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CounterfactualGrid":
        return cls(
            carry_depths=tuple(
                require_positive_int(item, "carry_depths")
                for item in payload["carry_depths"]
            ),
            sequence_lengths=tuple(
                require_positive_int(item, "sequence_lengths")
                for item in payload["sequence_lengths"]
            ),
            numeric_bases=tuple(
                require_positive_int(item, "numeric_bases")
                for item in payload["numeric_bases"]
            ),
            symbol_mappings=tuple(
                require_identifier(item, "symbol_mappings")
                for item in payload["symbol_mappings"]
            ),
            representation_shifts=tuple(
                require_identifier(item, "representation_shifts")
                for item in payload["representation_shifts"]
            ),
        )


@dataclass(frozen=True)
class ScalingManifest:
    training_compute_steps: tuple[int, ...]
    widths: tuple[int, ...]
    depths: tuple[int, ...]
    data_examples: tuple[int, ...]
    parameter_count_role: str = field(
        default="metadata_and_capacity_control_only", init=False
    )

    def __post_init__(self) -> None:
        for field_name in (
            "training_compute_steps",
            "widths",
            "depths",
            "data_examples",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_unique(getattr(self, field_name), field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_compute_steps": list(self.training_compute_steps),
            "widths": list(self.widths),
            "depths": list(self.depths),
            "data_examples": list(self.data_examples),
            "parameter_count_role": self.parameter_count_role,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScalingManifest":
        if payload.get("parameter_count_role") != "metadata_and_capacity_control_only":
            raise ValueError("parameter count cannot become a scaling objective")
        return cls(
            training_compute_steps=tuple(
                require_positive_int(item, "training_compute_steps")
                for item in payload["training_compute_steps"]
            ),
            widths=tuple(
                require_positive_int(item, "widths") for item in payload["widths"]
            ),
            depths=tuple(
                require_positive_int(item, "depths") for item in payload["depths"]
            ),
            data_examples=tuple(
                require_positive_int(item, "data_examples")
                for item in payload["data_examples"]
            ),
        )


@dataclass(frozen=True)
class MechanismExperimentPlan:
    plan_id: str
    study_id: str
    run_id: str
    claim: MechanismClaim
    frozen_snapshot_id: str
    frozen_snapshot_sha256: str
    arms: tuple[ArmSpec, ...]
    seed_bundles: tuple[SeedBundle, ...]
    training_budget: TrainingBudget
    interventions: tuple[InterventionSpec, ...]
    rescues: tuple[RescueSpec, ...]
    counterfactual_grid: CounterfactualGrid
    scaling_manifest: ScalingManifest
    scientific: bool
    schema_name: str = field(default="MechanismExperimentPlan", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_bool(self.scientific, "scientific")
        require_identifier(self.plan_id, "plan_id")
        require_identifier(self.study_id, "study_id")
        require_identifier(self.run_id, "run_id")
        require_identifier(self.frozen_snapshot_id, "frozen_snapshot_id")
        require_sha256(self.frozen_snapshot_sha256, "frozen_snapshot_sha256")
        if self.claim.candidate_snapshot_id != self.frozen_snapshot_id:
            raise ValueError("claim and experiment plan refer to different snapshots")
        if self.claim.candidate_snapshot_sha256 != self.frozen_snapshot_sha256:
            raise ValueError("claim and experiment plan snapshot hashes differ")
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "seed_bundles", tuple(self.seed_bundles))
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(self, "rescues", tuple(self.rescues))
        kinds = [arm.kind for arm in self.arms]
        if len(self.arms) != len(ArmKind) or set(kinds) != set(ArmKind):
            raise ValueError("plan must contain exactly one arm of every required kind")
        if len({arm.arm_id for arm in self.arms}) != len(self.arms):
            raise ValueError("arm IDs must be unique")
        if not self.seed_bundles:
            raise ValueError("plan needs at least one paired seed bundle")
        if len({item.bundle_id for item in self.seed_bundles}) != len(
            self.seed_bundles
        ):
            raise ValueError("seed bundle IDs must be unique")
        intervention_ids = {item.intervention_id for item in self.interventions}
        if len(intervention_ids) != len(self.interventions):
            raise ValueError("intervention IDs must be unique")
        for rescue in self.rescues:
            required = {
                rescue.disabling_intervention_id,
                rescue.restoring_intervention_id,
            }
            if not required.issubset(intervention_ids):
                raise ValueError("rescue references an undeclared intervention")
        if self.scientific and not self.interventions:
            raise ValueError("a scientific mechanism plan needs interventions")
        if self.scientific and not self.rescues:
            raise ValueError("a scientific mechanism plan needs a rescue experiment")

    @property
    def plan_hash(self) -> str:
        return content_hash(self.to_dict())

    @property
    def parameter_count_role(self) -> str:
        return self.scaling_manifest.parameter_count_role

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "study_id": self.study_id,
            "run_id": self.run_id,
            "claim": self.claim.to_dict(),
            "frozen_snapshot_id": self.frozen_snapshot_id,
            "frozen_snapshot_sha256": self.frozen_snapshot_sha256,
            "arms": [arm.to_dict() for arm in self.arms],
            "seed_bundles": [item.to_dict() for item in self.seed_bundles],
            "training_budget": self.training_budget.to_dict(),
            "interventions": [item.to_dict() for item in self.interventions],
            "rescues": [item.to_dict() for item in self.rescues],
            "counterfactual_grid": self.counterfactual_grid.to_dict(),
            "scaling_manifest": self.scaling_manifest.to_dict(),
            "scientific": self.scientific,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MechanismExperimentPlan":
        if payload.get("schema_name") != "MechanismExperimentPlan":
            raise ValueError("not a MechanismExperimentPlan record")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported MechanismExperimentPlan schema version")
        return cls(
            plan_id=require_identifier(payload["plan_id"], "plan_id"),
            study_id=require_identifier(payload["study_id"], "study_id"),
            run_id=require_identifier(payload["run_id"], "run_id"),
            claim=MechanismClaim.from_dict(payload["claim"]),
            frozen_snapshot_id=require_identifier(
                payload["frozen_snapshot_id"], "frozen_snapshot_id"
            ),
            frozen_snapshot_sha256=require_sha256(
                payload["frozen_snapshot_sha256"], "frozen_snapshot_sha256"
            ),
            arms=tuple(ArmSpec.from_dict(item) for item in payload["arms"]),
            seed_bundles=tuple(
                SeedBundle.from_dict(item) for item in payload["seed_bundles"]
            ),
            training_budget=TrainingBudget.from_dict(payload["training_budget"]),
            interventions=tuple(
                InterventionSpec.from_dict(item) for item in payload["interventions"]
            ),
            rescues=tuple(RescueSpec.from_dict(item) for item in payload["rescues"]),
            counterfactual_grid=CounterfactualGrid.from_dict(
                payload["counterfactual_grid"]
            ),
            scaling_manifest=ScalingManifest.from_dict(payload["scaling_manifest"]),
            scientific=require_bool(payload["scientific"], "scientific"),
        )


@dataclass(frozen=True)
class FrozenMechanismPlan:
    path: Path
    plan: MechanismExperimentPlan
    plan_hash: str
    frozen_payload_sha256: str

    def verify(self) -> None:
        payload = read_json(self.path)
        if payload.get("schema_name") != "FrozenMechanismPlan":
            raise ValueError("frozen mechanism-plan envelope has the wrong schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported frozen mechanism-plan schema version")
        stored_plan = MechanismExperimentPlan.from_dict(payload["plan"])
        stored_hash = require_sha256(payload["plan_hash"], "plan_hash")
        if stored_hash != stored_plan.plan_hash or stored_hash != self.plan_hash:
            raise ValueError("frozen mechanism plan hash mismatch")
        if stored_plan.to_dict() != self.plan.to_dict():
            raise ValueError("frozen mechanism plan does not match the receipt")
        if content_hash(payload) != self.frozen_payload_sha256:
            raise ValueError("frozen mechanism-plan envelope was modified")


def freeze_mechanism_plan(
    plan: MechanismExperimentPlan, path: str | Path
) -> FrozenMechanismPlan:
    destination = Path(path).resolve()
    payload = {
        "schema_name": "FrozenMechanismPlan",
        "schema_version": "1.0",
        "plan_hash": plan.plan_hash,
        "plan": plan.to_dict(),
    }
    create_json_exclusive(destination, payload)
    return FrozenMechanismPlan(
        path=destination,
        plan=plan,
        plan_hash=plan.plan_hash,
        frozen_payload_sha256=content_hash(payload),
    )


def load_frozen_mechanism_plan(path: str | Path) -> FrozenMechanismPlan:
    destination = Path(path).resolve()
    payload = read_json(destination)
    plan = MechanismExperimentPlan.from_dict(payload["plan"])
    receipt = FrozenMechanismPlan(
        path=destination,
        plan=plan,
        plan_hash=require_sha256(payload["plan_hash"], "plan_hash"),
        frozen_payload_sha256=content_hash(payload),
    )
    receipt.verify()
    return receipt
