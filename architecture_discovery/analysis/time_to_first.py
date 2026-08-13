"""Time-to-first qualifying mechanism in proposal and token exposure units."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

from analysis.outcomes import RunOutcomeTable, RunTerminalStatus
from study.serialization import require_bool, require_int, require_str


class ExposureUnit(StrEnum):
    PROPOSALS = "proposals"
    TOKENS = "tokens"


@dataclass(frozen=True)
class DiscoveryRecord:
    run_id: str
    mechanism_cluster_id: str
    opportunity_index: int
    cumulative_generator_tokens: int
    qualifies: bool

    def __post_init__(self) -> None:
        require_str(self.run_id, "run_id")
        require_str(self.mechanism_cluster_id, "mechanism_cluster_id")
        require_int(self.opportunity_index, "opportunity_index")
        require_int(
            self.cumulative_generator_tokens, "cumulative_generator_tokens"
        )
        require_bool(self.qualifies, "qualifies")
        if not self.run_id or not self.mechanism_cluster_id:
            raise ValueError("run and cluster identifiers cannot be empty")
        if self.opportunity_index < 1:
            raise ValueError("discovery opportunity must be positive")
        if self.cumulative_generator_tokens < 1:
            raise ValueError("qualifying discovery requires positive token exposure")


@dataclass(frozen=True)
class TimeToFirstRecord:
    run_id: str
    condition_id: str
    event: bool
    proposal_time: int
    token_time: int

    def __post_init__(self) -> None:
        require_str(self.run_id, "run_id")
        require_str(self.condition_id, "condition_id")
        require_bool(self.event, "event")
        require_int(self.proposal_time, "proposal_time")
        require_int(self.token_time, "token_time")
        if self.proposal_time < 0 or self.token_time < 0:
            raise ValueError("time-to-first exposure cannot be negative")


@dataclass(frozen=True)
class SurvivalPoint:
    exposure: int
    at_risk: int
    events: int
    censored: int
    survival_probability: float


def derive_time_to_first(
    outcomes: RunOutcomeTable,
    discoveries: Iterable[DiscoveryRecord],
) -> tuple[TimeToFirstRecord, ...]:
    """Create one event-or-censoring record for every assigned run."""

    by_run: dict[str, list[DiscoveryRecord]] = {
        row.run_id: [] for row in outcomes.rows
    }
    for discovery in discoveries:
        if discovery.run_id not in by_run:
            raise ValueError(f"discovery refers to unknown run {discovery.run_id}")
        if discovery.qualifies:
            by_run[discovery.run_id].append(discovery)

    records: list[TimeToFirstRecord] = []
    for row in outcomes.rows:
        qualifying = sorted(
            by_run[row.run_id],
            key=lambda item: (
                item.opportunity_index,
                item.cumulative_generator_tokens,
                item.mechanism_cluster_id,
            ),
        )
        if qualifying:
            if row.terminal_status is not RunTerminalStatus.COMPLETED:
                raise ValueError("failed runs cannot contain qualifying discoveries")
            first = qualifying[0]
            if first.opportunity_index > row.proposal_exposure:
                raise ValueError("discovery occurs after the run's proposal exposure")
            if first.cumulative_generator_tokens > row.token_exposure:
                raise ValueError("discovery occurs after the run's token exposure")
            records.append(
                TimeToFirstRecord(
                    run_id=row.run_id,
                    condition_id=row.condition_id,
                    event=True,
                    proposal_time=first.opportunity_index,
                    token_time=first.cumulative_generator_tokens,
                )
            )
        else:
            records.append(
                TimeToFirstRecord(
                    run_id=row.run_id,
                    condition_id=row.condition_id,
                    event=False,
                    proposal_time=row.proposal_exposure,
                    token_time=row.token_exposure,
                )
            )
    return tuple(records)


def kaplan_meier(
    records: Iterable[TimeToFirstRecord],
    *,
    exposure_unit: ExposureUnit | str,
) -> tuple[SurvivalPoint, ...]:
    """Dependency-free Kaplan-Meier curve for one declared exposure scale."""

    rows = tuple(records)
    if not rows:
        raise ValueError("at least one time-to-first record is required")
    if len({row.run_id for row in rows}) != len(rows):
        raise ValueError("time-to-first input must contain one row per run")
    unit = ExposureUnit(exposure_unit)
    field = "proposal_time" if unit is ExposureUnit.PROPOSALS else "token_time"
    if any(getattr(row, field) < 0 for row in rows):
        raise ValueError("exposure times cannot be negative")
    if any(row.event and getattr(row, field) == 0 for row in rows):
        raise ValueError("an event cannot occur at zero exposure")

    at_risk = len(rows)
    survival = 1.0
    points: list[SurvivalPoint] = []
    for exposure in sorted({getattr(row, field) for row in rows}):
        at_time = [row for row in rows if getattr(row, field) == exposure]
        events = sum(row.event for row in at_time)
        censored = len(at_time) - events
        if events:
            survival *= 1.0 - events / at_risk
        points.append(
            SurvivalPoint(
                exposure=exposure,
                at_risk=at_risk,
                events=events,
                censored=censored,
                survival_probability=survival,
            )
        )
        at_risk -= len(at_time)
    return tuple(points)
