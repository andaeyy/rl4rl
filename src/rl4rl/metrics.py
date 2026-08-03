"""Descriptive metrics for autonomous discovery trajectories."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from rl4rl.lineage import summarize_lineage
from rl4rl.schema import EventStatus, TrajectoryEvent


@dataclass(frozen=True, slots=True)
class MetricSummary:
    total_events: int
    accepted_events: int
    rejected_events: int
    invalid_events: int
    rollback_events: int
    mutation_acceptance_rate: float | None
    rollback_rate: float
    valid_rate: float | None
    boundary_crossing_events: int
    boundary_crossing_rate: float | None
    accepted_boundary_crossing_rate: float | None
    edit_entropy_bits: float
    normalized_edit_entropy: float
    architecture_observations: int
    unique_architecture_regions: int
    architecture_diversity_ratio: float | None
    architecture_revisit_rate: float | None
    verified_reward_hacks: int
    suspected_reward_hacks: int
    best_qualifying_parameters: int | None
    external_frontier_parameters: int | None
    frontier_gap_ratio: float | None
    lineage_roots: int
    lineage_edges: int
    lineage_max_depth: int
    missing_parent_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_metrics(
    events: list[TrajectoryEvent], *, external_frontier: int | None = None
) -> MetricSummary:
    accepted = [event for event in events if _is_accepted(event)]
    rejected = [event for event in events if event.status == EventStatus.REJECTED]
    invalid = [event for event in events if event.status == EventStatus.INVALID]
    rolled_back = [event for event in events if event.status == EventStatus.ROLLED_BACK]
    decision_events = accepted + rejected
    validity_known = [event for event in events if event.valid is not None]
    edits = [edit for event in events for edit in event.edits]
    event_with_edits = [event for event in events if event.edits]
    boundary_events = [event for event in event_with_edits if event.crosses_boundary]
    accepted_with_edits = [event for event in accepted if event.edits]
    accepted_boundary = [
        event for event in accepted_with_edits if event.crosses_boundary
    ]

    edit_counts = Counter(
        f"{edit.component}:{edit.operation}:{edit.boundary_label}" for edit in edits
    )
    entropy = shannon_entropy(edit_counts.values())
    normalized_entropy = (
        entropy / math.log2(len(edit_counts)) if len(edit_counts) > 1 else 0.0
    )

    design_keys = [
        key for event in events if (key := event.architecture.design_key()) is not None
    ]
    unique_designs = len(set(design_keys))
    qualifying_params = [
        event.architecture.parameters
        for event in events
        if event.architecture.parameters is not None
        and event.architecture.qualifies is True
        and event.valid is not False
    ]
    best_parameters = min(qualifying_params) if qualifying_params else None
    if external_frontier is not None and external_frontier <= 0:
        raise ValueError("external_frontier must be positive")
    frontier_gap = (
        best_parameters / external_frontier
        if best_parameters is not None and external_frontier is not None
        else None
    )
    lineage = summarize_lineage(events)

    return MetricSummary(
        total_events=len(events),
        accepted_events=len(accepted),
        rejected_events=len(rejected),
        invalid_events=len(invalid),
        rollback_events=len(rolled_back),
        mutation_acceptance_rate=_ratio(len(accepted), len(decision_events)),
        rollback_rate=_ratio(len(rolled_back), len(events)) or 0.0,
        valid_rate=_ratio(
            sum(event.valid is True for event in validity_known), len(validity_known)
        ),
        boundary_crossing_events=len(boundary_events),
        boundary_crossing_rate=_ratio(len(boundary_events), len(event_with_edits)),
        accepted_boundary_crossing_rate=_ratio(
            len(accepted_boundary), len(accepted_with_edits)
        ),
        edit_entropy_bits=entropy,
        normalized_edit_entropy=normalized_entropy,
        architecture_observations=len(design_keys),
        unique_architecture_regions=unique_designs,
        architecture_diversity_ratio=_ratio(unique_designs, len(design_keys)),
        architecture_revisit_rate=(
            1 - unique_designs / len(design_keys) if design_keys else None
        ),
        verified_reward_hacks=sum(event.reward_hack.verified for event in events),
        suspected_reward_hacks=sum(event.reward_hack.suspected for event in events),
        best_qualifying_parameters=best_parameters,
        external_frontier_parameters=external_frontier,
        frontier_gap_ratio=frontier_gap,
        lineage_roots=len(lineage.roots),
        lineage_edges=lineage.edge_count,
        lineage_max_depth=lineage.max_depth,
        missing_parent_count=len(lineage.missing_parents),
    )


def frontier_progression(events: list[TrajectoryEvent]) -> list[tuple[int, int]]:
    """Return (step, best qualifying parameter count) updates."""
    best: int | None = None
    progression: list[tuple[int, int]] = []
    for event in sorted(events, key=lambda item: (item.step, item.event_id)):
        parameters = event.architecture.parameters
        if (
            parameters is not None
            and event.architecture.qualifies is True
            and event.valid is not False
            and (best is None or parameters < best)
        ):
            best = parameters
            progression.append((event.step, best))
    return progression


def rolling_diversity(
    events: list[TrajectoryEvent], *, window: int = 20
) -> list[tuple[int, float]]:
    if window <= 0:
        raise ValueError("window must be positive")
    ordered = sorted(events, key=lambda item: (item.step, item.event_id))
    values: list[tuple[int, float]] = []
    for index, event in enumerate(ordered):
        chunk = ordered[max(0, index - window + 1) : index + 1]
        keys = [
            key
            for candidate in chunk
            if (key := candidate.architecture.design_key()) is not None
        ]
        values.append((event.step, len(set(keys)) / len(keys) if keys else 0.0))
    return values


def shannon_entropy(counts: Any) -> float:
    values = [int(count) for count in counts if count > 0]
    total = sum(values)
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in values)


def _is_accepted(event: TrajectoryEvent) -> bool:
    return event.accepted is True or event.status == EventStatus.ACCEPTED


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
