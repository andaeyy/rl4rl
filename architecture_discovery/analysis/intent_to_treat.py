"""Intent-to-treat summaries and explicit failure-class sensitivities."""

from __future__ import annotations

from dataclasses import dataclass

from analysis.outcomes import RunOutcomeTable, RunTerminalStatus


@dataclass(frozen=True)
class ConditionITTSummary:
    condition_id: str
    assigned_runs: int
    completed_runs: int
    scientific_failures: int
    infrastructure_failures: int
    qualifying_clusters: int
    mean_clusters_per_assigned_run: float


@dataclass(frozen=True)
class FailureClassCount:
    condition_id: str
    failure_class: str
    terminal_status: RunTerminalStatus
    count: int


@dataclass(frozen=True)
class InfrastructureSensitivity:
    """Secondary exclusion result that keeps excluded IDs auditable."""

    summaries: tuple[ConditionITTSummary, ...]
    excluded_infrastructure_run_ids: tuple[str, ...]


def intent_to_treat_summary(
    outcomes: RunOutcomeTable,
) -> tuple[ConditionITTSummary, ...]:
    summaries: list[ConditionITTSummary] = []
    for condition_id in sorted({row.condition_id for row in outcomes.rows}):
        rows = outcomes.for_condition(condition_id)
        assigned = len(rows)
        completed = sum(
            row.terminal_status is RunTerminalStatus.COMPLETED for row in rows
        )
        scientific = sum(
            row.terminal_status is RunTerminalStatus.SCIENTIFIC_FAILURE
            for row in rows
        )
        infrastructure = sum(
            row.terminal_status is RunTerminalStatus.INFRASTRUCTURE_FAILURE
            for row in rows
        )
        clusters = sum(row.itt_cluster_count for row in rows)
        summaries.append(
            ConditionITTSummary(
                condition_id=condition_id,
                assigned_runs=assigned,
                completed_runs=completed,
                scientific_failures=scientific,
                infrastructure_failures=infrastructure,
                qualifying_clusters=clusters,
                mean_clusters_per_assigned_run=clusters / assigned,
            )
        )
    return tuple(summaries)


def failure_class_counts(
    outcomes: RunOutcomeTable,
) -> tuple[FailureClassCount, ...]:
    counts: dict[tuple[str, str, RunTerminalStatus], int] = {}
    for row in outcomes.rows:
        if row.terminal_status is RunTerminalStatus.COMPLETED:
            continue
        key = (row.condition_id, row.failure_class, row.terminal_status)
        counts[key] = counts.get(key, 0) + 1
    return tuple(
        FailureClassCount(
            condition_id=condition_id,
            failure_class=failure_class,
            terminal_status=status,
            count=count,
        )
        for (condition_id, failure_class, status), count in sorted(
            counts.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    )


def infrastructure_failure_sensitivity(
    outcomes: RunOutcomeTable,
) -> InfrastructureSensitivity:
    """Secondary complete-case view; scientific failures are never excluded."""

    excluded = tuple(
        sorted(
            row.run_id
            for row in outcomes.rows
            if row.terminal_status is RunTerminalStatus.INFRASTRUCTURE_FAILURE
        )
    )
    retained = tuple(
        row
        for row in outcomes.rows
        if row.terminal_status is not RunTerminalStatus.INFRASTRUCTURE_FAILURE
    )
    if not retained:
        raise ValueError("cannot exclude every assigned run")
    return InfrastructureSensitivity(
        summaries=intent_to_treat_summary(
            RunOutcomeTable(
                retained,
                tuple(row.run_id for row in retained),
            )
        ),
        excluded_infrastructure_run_ids=excluded,
    )
