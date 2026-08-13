"""One-row-per-assigned-run outcome and pilot-data schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from study.serialization import content_hash, require_int, require_str


class RunTerminalStatus(StrEnum):
    COMPLETED = "completed"
    SCIENTIFIC_FAILURE = "scientific_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


@dataclass(frozen=True)
class RunOutcome:
    """Primary statistical row.

    There is deliberately no candidate identifier.  The independent unit is one
    assigned, terminal research run.  Failures remain as rows and contribute zero
    qualifying clusters to the primary intent-to-treat estimand.
    """

    study_id: str
    block_id: str
    run_id: str
    condition_id: str
    run_seed: int
    terminal_status: RunTerminalStatus
    qualifying_cluster_count: int | None
    proposal_exposure: int
    token_exposure: int
    failure_class: str = ""
    assignment_hash: str = ""
    run_artifact_hash: str = ""
    assigned: bool = field(default=True, init=False)
    schema_name: str = field(default="RunOutcome", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        for name in ("study_id", "block_id", "run_id", "condition_id"):
            value = require_str(getattr(self, name), name)
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")
        require_int(self.run_seed, "run_seed")
        require_int(self.proposal_exposure, "proposal_exposure")
        require_int(self.token_exposure, "token_exposure")
        if self.proposal_exposure < 0 or self.token_exposure < 0:
            raise ValueError("run exposure cannot be negative")
        if self.qualifying_cluster_count is not None:
            if (
                not isinstance(self.qualifying_cluster_count, int)
                or isinstance(self.qualifying_cluster_count, bool)
                or self.qualifying_cluster_count < 0
            ):
                raise ValueError("qualifying_cluster_count must be non-negative")
        if self.terminal_status is RunTerminalStatus.COMPLETED:
            if self.qualifying_cluster_count is None:
                raise ValueError("completed runs require an observed cluster count")
            if self.failure_class:
                raise ValueError("completed runs cannot have a failure class")
        elif not self.failure_class:
            raise ValueError("failed runs require a predeclared failure class")

    @property
    def itt_cluster_count(self) -> int:
        """Conservative assigned-run outcome used by the primary ITT analysis."""

        if self.terminal_status is not RunTerminalStatus.COMPLETED:
            return 0
        assert self.qualifying_cluster_count is not None
        return self.qualifying_cluster_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "block_id": self.block_id,
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "run_seed": self.run_seed,
            "terminal_status": self.terminal_status.value,
            "qualifying_cluster_count": self.qualifying_cluster_count,
            "proposal_exposure": self.proposal_exposure,
            "token_exposure": self.token_exposure,
            "failure_class": self.failure_class,
            "assignment_hash": self.assignment_hash,
            "run_artifact_hash": self.run_artifact_hash,
            "assigned": True,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunOutcome:
        if payload.get("schema_name") != "RunOutcome":
            raise ValueError(
                "analysis accepts only RunOutcome records; candidate rows are not replicates"
            )
        if payload.get("assigned") is not True:
            raise ValueError("only assigned runs belong in the outcome table")
        if payload.get("schema_version") != "1.0":
            raise ValueError("unsupported RunOutcome schema version")
        return cls(
            study_id=require_str(payload["study_id"], "study_id"),
            block_id=require_str(payload["block_id"], "block_id"),
            run_id=require_str(payload["run_id"], "run_id"),
            condition_id=require_str(payload["condition_id"], "condition_id"),
            run_seed=require_int(payload["run_seed"], "run_seed"),
            terminal_status=RunTerminalStatus(
                require_str(payload["terminal_status"], "terminal_status")
            ),
            qualifying_cluster_count=(
                None
                if payload.get("qualifying_cluster_count") is None
                else require_int(
                    payload["qualifying_cluster_count"],
                    "qualifying_cluster_count",
                )
            ),
            proposal_exposure=require_int(
                payload["proposal_exposure"], "proposal_exposure"
            ),
            token_exposure=require_int(payload["token_exposure"], "token_exposure"),
            failure_class=require_str(payload.get("failure_class", ""), "failure_class"),
            assignment_hash=require_str(
                payload.get("assignment_hash", ""), "assignment_hash"
            ),
            run_artifact_hash=require_str(
                payload.get("run_artifact_hash", ""), "run_artifact_hash"
            ),
        )


@dataclass(frozen=True)
class RunOutcomeTable:
    """Validated table with exactly one terminal row per assigned run ID."""

    rows: tuple[RunOutcome, ...]
    assigned_run_ids: tuple[str, ...]
    schema_name: str = field(default="RunOutcomeTable", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        if not self.rows:
            raise ValueError("run outcome table cannot be empty")
        if any(type(row) is not RunOutcome for row in self.rows):
            raise TypeError(
                "each independent observation must be an exact RunOutcome row"
            )
        studies = {row.study_id for row in self.rows}
        if len(studies) != 1:
            raise ValueError("one outcome table cannot mix studies")
        keys = [(row.study_id, row.run_id) for row in self.rows]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate run rows would create pseudoreplication")
        if (
            not self.assigned_run_ids
            or len(self.assigned_run_ids) != len(set(self.assigned_run_ids))
            or any(not run_id.strip() for run_id in self.assigned_run_ids)
        ):
            raise ValueError("assigned_run_ids must be a unique non-empty frozen roster")
        observed = {row.run_id for row in self.rows}
        assigned = set(self.assigned_run_ids)
        if observed != assigned:
            missing = sorted(assigned - observed)
            unexpected = sorted(observed - assigned)
            raise ValueError(
                "outcome rows do not match the frozen assignment roster: "
                f"missing={missing}, unexpected={unexpected}"
            )

    @property
    def study_id(self) -> str:
        return self.rows[0].study_id

    @property
    def table_hash(self) -> str:
        return content_hash(self.to_dict())

    def for_condition(self, condition_id: str) -> tuple[RunOutcome, ...]:
        return tuple(row for row in self.rows if row.condition_id == condition_id)

    def for_block(self, block_id: str) -> tuple[RunOutcome, ...]:
        return tuple(row for row in self.rows if row.block_id == block_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "assigned_run_ids": list(self.assigned_run_ids),
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        assigned_run_ids: Iterable[str],
    ) -> RunOutcomeTable:
        return cls(
            tuple(RunOutcome.from_dict(record) for record in records),
            tuple(assigned_run_ids),
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RunOutcomeTable:
        if payload.get("schema_name") != "RunOutcomeTable":
            raise ValueError("expected RunOutcomeTable schema")
        table = cls.from_records(
            payload["rows"], assigned_run_ids=payload["assigned_run_ids"]
        )
        if payload.get("study_id") != table.study_id:
            raise ValueError("table study_id does not match its rows")
        return table


@dataclass(frozen=True)
class PilotDataset:
    """Frozen pilot extract used only for dispersion and power planning."""

    pilot_id: str
    outcomes: RunOutcomeTable
    extraction_rule_hash: str
    schema_name: str = field(default="PilotDataset", init=False)
    schema_version: str = field(default="1.0", init=False)

    def __post_init__(self) -> None:
        require_str(self.pilot_id, "pilot_id")
        require_str(self.extraction_rule_hash, "extraction_rule_hash")
        if not self.pilot_id or not self.extraction_rule_hash:
            raise ValueError("pilot_id and extraction_rule_hash cannot be empty")

    @property
    def dataset_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "pilot_id": self.pilot_id,
            "extraction_rule_hash": self.extraction_rule_hash,
            "outcomes": self.outcomes.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PilotDataset:
        if payload.get("schema_name") != "PilotDataset":
            raise ValueError("expected PilotDataset schema")
        return cls(
            pilot_id=require_str(payload["pilot_id"], "pilot_id"),
            extraction_rule_hash=require_str(
                payload["extraction_rule_hash"], "extraction_rule_hash"
            ),
            outcomes=RunOutcomeTable.from_dict(payload["outcomes"]),
        )
