"""Validity-first replacement policy shared by both OpenEvolve conditions."""

from __future__ import annotations

from typing import Any


def _quality(metrics: dict[str, Any]) -> tuple[float, float]:
    """Rank only by frozen validity/accuracy fields; metadata is intentionally absent."""
    return (
        float(metrics.get("eligible_for_parent", 0.0)),
        float(metrics.get("search_score", 0.0)),
    )


def install_validity_first_policy() -> None:
    """Patch cell/archive comparisons without modifying the pinned vendor checkout."""
    from openevolve.database import ProgramDatabase

    if getattr(ProgramDatabase, "_discovery_quality_policy_installed", False):
        return

    def validity_first(self, program1, program2):
        quality1 = _quality(program1.metrics)
        quality2 = _quality(program2.metrics)
        if quality1 != quality2:
            return quality1 > quality2

        # Earlier completion wins an exact tie. The ID makes the rule total and
        # deterministic when imported traces have equal timestamps.
        order1 = (program1.iteration_found, program1.timestamp, program1.id)
        order2 = (program2.iteration_found, program2.timestamp, program2.id)
        return order1 < order2

    ProgramDatabase._is_better = validity_first
    ProgramDatabase._discovery_quality_policy_installed = True
