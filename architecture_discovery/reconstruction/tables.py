"""Analysis-ready one-row-per-assigned-run reconstruction."""

from __future__ import annotations

from collections.abc import Iterable

from analysis.outcomes import RunOutcomeTable
from reconstruction.models import ReconstructedRun


def build_analysis_table(
    runs: Iterable[ReconstructedRun], *, assigned_run_ids: Iterable[str]
) -> RunOutcomeTable:
    """Build an ITT table only when rows match the external frozen assignment roster."""

    reconstructed = tuple(runs)
    frozen_roster = tuple(assigned_run_ids)
    incomplete = [run.context.run_id for run in reconstructed if run.outcome is None]
    if incomplete:
        raise ValueError(
            f"analysis table requires terminal assigned runs, incomplete: {incomplete}"
        )
    return RunOutcomeTable(
        tuple(run.outcome for run in reconstructed if run.outcome),
        frozen_roster,
    )
