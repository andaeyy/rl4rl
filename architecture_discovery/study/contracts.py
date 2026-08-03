"""Typed records for the project-owned two-by-two causal experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from study.budget import BudgetSpec
from study.serialization import (
    content_hash,
    json_value,
    require_bool,
    require_int,
    require_str,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ParentPolicy(StrEnum):
    SINGLE = "single"
    PORTFOLIO = "portfolio"


class ProposalPolicy(StrEnum):
    ORDINARY = "ordinary"
    SCHEDULED_TRANSITION = "scheduled_transition"


class ConditionId(StrEnum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


_PRIMARY_TREATMENTS = {
    ConditionId.C0: (ParentPolicy.SINGLE, ProposalPolicy.ORDINARY),
    ConditionId.C1: (ParentPolicy.SINGLE, ProposalPolicy.SCHEDULED_TRANSITION),
    ConditionId.C2: (ParentPolicy.PORTFOLIO, ProposalPolicy.ORDINARY),
    ConditionId.C3: (ParentPolicy.PORTFOLIO, ProposalPolicy.SCHEDULED_TRANSITION),
}


@dataclass(frozen=True)
class ConditionSpec:
    """One cell of the design. Only the two policy fields are treatments."""

    condition_id: ConditionId
    parent_policy: ParentPolicy
    proposal_policy: ProposalPolicy

    SCHEMA_NAME: ClassVar[str] = "ConditionSpec"
    SCHEMA_VERSION: ClassVar[str] = "1.0"
    TREATMENT_FIELDS: ClassVar[tuple[str, str]] = (
        "parent_policy",
        "proposal_policy",
    )

    def __post_init__(self) -> None:
        expected = _PRIMARY_TREATMENTS[self.condition_id]
        actual = (self.parent_policy, self.proposal_policy)
        if actual != expected:
            raise ValueError(
                f"{self.condition_id.value} has treatment {actual}, expected {expected}"
            )

    @classmethod
    def primary(cls) -> tuple[ConditionSpec, ...]:
        return tuple(
            cls(condition_id, parent_policy, proposal_policy)
            for condition_id, (parent_policy, proposal_policy) in _PRIMARY_TREATMENTS.items()
        )

    @classmethod
    def for_id(cls, condition_id: ConditionId | str) -> ConditionSpec:
        resolved = ConditionId(condition_id)
        parent_policy, proposal_policy = _PRIMARY_TREATMENTS[resolved]
        return cls(resolved, parent_policy, proposal_policy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.SCHEMA_NAME,
            "schema_version": self.SCHEMA_VERSION,
            "condition_id": self.condition_id.value,
            "parent_policy": self.parent_policy.value,
            "proposal_policy": self.proposal_policy.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConditionSpec:
        if payload.get("schema_name") != cls.SCHEMA_NAME:
            raise ValueError("expected ConditionSpec schema")
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported ConditionSpec schema version")
        return cls(
            condition_id=ConditionId(payload["condition_id"]),
            parent_policy=ParentPolicy(payload["parent_policy"]),
            proposal_policy=ProposalPolicy(payload["proposal_policy"]),
        )


@dataclass(frozen=True)
class StudySpec:
    """Common configuration shared byte-for-byte across all four treatments."""

    study_id: str
    study_seed: int
    block_count: int
    budget: BudgetSpec
    portfolio_size: int
    transition_opportunities: tuple[int, ...]
    initial_candidate_id: str
    common_config_hash: str
    code_hash: str
    environment_hash: str
    scientific: bool
    schema_name: str = field(default="StudySpec", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_bool(self.scientific, "scientific")
        require_str(self.study_id, "study_id")
        require_int(self.study_seed, "study_seed")
        require_int(self.block_count, "block_count")
        require_int(self.portfolio_size, "portfolio_size")
        for index, opportunity in enumerate(self.transition_opportunities):
            require_int(opportunity, f"transition_opportunities[{index}]")
        if not self.study_id or any(character.isspace() for character in self.study_id):
            raise ValueError("study_id must be non-empty and contain no whitespace")
        if self.block_count < 1:
            raise ValueError("block_count must be positive")
        if self.portfolio_size < 2:
            raise ValueError("portfolio_size must be at least two")
        schedule = self.transition_opportunities
        if tuple(sorted(set(schedule))) != schedule:
            raise ValueError("transition opportunities must be sorted and unique")
        if any(
            opportunity < 1
            or opportunity > self.budget.proposal_opportunities
            for opportunity in schedule
        ):
            raise ValueError("transition opportunity outside the proposal budget")
        if self.budget.proposal_opportunities and not schedule:
            raise ValueError("scheduled-transition conditions require a non-empty schedule")
        for field_name in (
            "initial_candidate_id",
            "common_config_hash",
            "code_hash",
            "environment_hash",
        ):
            require_str(getattr(self, field_name), field_name)
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} cannot be empty")
        for field_name in ("common_config_hash", "code_hash", "environment_hash"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")

    @property
    def conditions(self) -> tuple[ConditionSpec, ...]:
        return ConditionSpec.primary()

    @property
    def spec_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_seed": self.study_seed,
            "block_count": self.block_count,
            "budget": self.budget.to_dict(),
            "portfolio_size": self.portfolio_size,
            "transition_opportunities": list(self.transition_opportunities),
            "initial_candidate_id": self.initial_candidate_id,
            "common_config_hash": self.common_config_hash,
            "code_hash": self.code_hash,
            "environment_hash": self.environment_hash,
            "scientific": self.scientific,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> StudySpec:
        if payload.get("schema_name") != "StudySpec":
            raise ValueError("expected StudySpec schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported StudySpec schema version")
        return cls(
            study_id=require_str(payload["study_id"], "study_id"),
            study_seed=require_int(payload["study_seed"], "study_seed"),
            block_count=require_int(payload["block_count"], "block_count"),
            budget=BudgetSpec.from_dict(payload["budget"]),
            portfolio_size=require_int(payload["portfolio_size"], "portfolio_size"),
            transition_opportunities=tuple(
                require_int(value, "transition opportunity")
                for value in payload["transition_opportunities"]
            ),
            initial_candidate_id=require_str(
                payload["initial_candidate_id"], "initial_candidate_id"
            ),
            common_config_hash=require_str(
                payload["common_config_hash"], "common_config_hash"
            ),
            code_hash=require_str(payload["code_hash"], "code_hash"),
            environment_hash=require_str(
                payload["environment_hash"], "environment_hash"
            ),
            scientific=require_bool(payload["scientific"], "scientific"),
        )

    @classmethod
    def toy(
        cls,
        *,
        study_id: str = "offline-toy-study",
        study_seed: int = 7,
        block_count: int = 1,
        proposal_opportunities: int = 3,
    ) -> StudySpec:
        """Explicitly non-scientific fixture for tests and plumbing checks."""

        schedule = (
            (max(1, (proposal_opportunities + 1) // 2),)
            if proposal_opportunities
            else ()
        )
        return cls(
            study_id=study_id,
            study_seed=study_seed,
            block_count=block_count,
            budget=BudgetSpec.toy(proposal_opportunities),
            portfolio_size=2,
            transition_opportunities=schedule,
            initial_candidate_id="offline-initial-candidate",
            common_config_hash=content_hash({"fixture": "offline-toy-config"}),
            code_hash=content_hash({"fixture": "offline-toy-code"}),
            environment_hash=content_hash(
                {"fixture": "offline-toy-environment"}
            ),
            scientific=False,
        )


@dataclass(frozen=True)
class RunSpec:
    study_id: str
    block_id: str
    run_id: str
    condition: ConditionSpec
    order_index: int
    run_seed: int
    run_directory: str
    assignment_hash: str
    schema_name: str = field(default="RunSpec", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "study_id",
            "block_id",
            "run_id",
            "run_directory",
            "assignment_hash",
        ):
            require_str(getattr(self, field_name), field_name)
        require_int(self.order_index, "order_index")
        require_int(self.run_seed, "run_seed")
        if self.order_index < 0:
            raise ValueError("order_index cannot be negative")
        if not Path(self.run_directory).is_absolute():
            raise ValueError("run_directory must be absolute")
        if not self.assignment_hash:
            raise ValueError("assignment_hash cannot be empty")

    def assignment_payload(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "block_id": self.block_id,
            "run_id": self.run_id,
            "condition": self.condition.to_dict(),
            "order_index": self.order_index,
            "run_seed": self.run_seed,
            "run_directory": self.run_directory,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            **self.assignment_payload(),
            "assignment_hash": self.assignment_hash,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunSpec:
        if payload.get("schema_name") != "RunSpec":
            raise ValueError("expected RunSpec schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported RunSpec schema version")
        return cls(
            study_id=require_str(payload["study_id"], "study_id"),
            block_id=require_str(payload["block_id"], "block_id"),
            run_id=require_str(payload["run_id"], "run_id"),
            condition=ConditionSpec.from_dict(payload["condition"]),
            order_index=require_int(payload["order_index"], "order_index"),
            run_seed=require_int(payload["run_seed"], "run_seed"),
            run_directory=require_str(payload["run_directory"], "run_directory"),
            assignment_hash=require_str(
                payload["assignment_hash"], "assignment_hash"
            ),
        )


@dataclass(frozen=True)
class BlockSpec:
    study_id: str
    block_id: str
    block_index: int
    randomization_seed: int
    runs: tuple[RunSpec, ...]
    schema_name: str = field(default="BlockSpec", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_str(self.study_id, "study_id")
        require_str(self.block_id, "block_id")
        require_int(self.block_index, "block_index")
        require_int(self.randomization_seed, "randomization_seed")
        if self.block_index < 0:
            raise ValueError("block_index cannot be negative")
        if len(self.runs) != len(ConditionId):
            raise ValueError("each block must contain exactly four primary conditions")
        if {run.condition.condition_id for run in self.runs} != set(ConditionId):
            raise ValueError("each block must contain C0, C1, C2, and C3 once")
        if [run.order_index for run in self.runs] != list(range(len(self.runs))):
            raise ValueError("run order indices must be contiguous within a block")
        if any(run.block_id != self.block_id for run in self.runs):
            raise ValueError("run refers to a different block")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "block_id": self.block_id,
            "block_index": self.block_index,
            "randomization_seed": self.randomization_seed,
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlockSpec:
        if payload.get("schema_name") != "BlockSpec":
            raise ValueError("expected BlockSpec schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported BlockSpec schema version")
        return cls(
            study_id=require_str(payload["study_id"], "study_id"),
            block_id=require_str(payload["block_id"], "block_id"),
            block_index=require_int(payload["block_index"], "block_index"),
            randomization_seed=require_int(
                payload["randomization_seed"], "randomization_seed"
            ),
            runs=tuple(RunSpec.from_dict(run) for run in payload["runs"]),
        )


@dataclass
class RunState:
    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    assignment_hash: str
    status: str
    initial_candidate_id: str
    incumbent_id: str
    portfolio_ids: list[str]
    seed_evaluation: dict[str, Any] | None
    next_opportunity: int
    active_opportunity: dict[str, Any] | None
    terminal_opportunities: list[dict[str, Any]]
    ledger: dict[str, Any]
    state_revision: int = 0
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    schema_name: str = field(default="RunState", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "study_id",
            "block_id",
            "run_id",
            "condition_id",
            "assignment_hash",
            "status",
            "initial_candidate_id",
            "incumbent_id",
            "created_at",
            "updated_at",
        ):
            value = require_str(getattr(self, field_name), field_name)
            if not value:
                raise ValueError(f"{field_name} cannot be empty")
        ConditionId(self.condition_id)
        require_int(self.next_opportunity, "next_opportunity")
        require_int(self.state_revision, "state_revision")
        if self.next_opportunity < 1 or self.state_revision < 0:
            raise ValueError("run-state counters are outside their valid range")
        if not isinstance(self.portfolio_ids, list) or any(
            not isinstance(value, str) or not value for value in self.portfolio_ids
        ):
            raise ValueError("portfolio_ids must be a list of non-empty strings")
        if self.seed_evaluation is not None and not isinstance(
            self.seed_evaluation, dict
        ):
            raise ValueError("seed_evaluation must be an object or null")
        if self.active_opportunity is not None and not isinstance(
            self.active_opportunity, dict
        ):
            raise ValueError("active_opportunity must be an object or null")
        if not isinstance(self.terminal_opportunities, list) or any(
            not isinstance(value, dict) for value in self.terminal_opportunities
        ):
            raise ValueError("terminal_opportunities must be a list of objects")
        if not isinstance(self.ledger, dict):
            raise ValueError("ledger must be an object")

    def to_dict(self) -> dict[str, Any]:
        return json_value(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunState:
        if payload.get("schema_name") != "RunState":
            raise ValueError("expected RunState schema")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported RunState schema version")
        init_names = {item.name for item in dataclass_fields(cls) if item.init}
        expected_names = init_names | {"schema_name", "schema_version"}
        if set(payload) != expected_names:
            missing = sorted(expected_names - set(payload))
            extra = sorted(set(payload) - expected_names)
            raise ValueError(
                f"invalid RunState fields; missing={missing}, extra={extra}"
            )
        return cls(**{name: payload[name] for name in init_names})
