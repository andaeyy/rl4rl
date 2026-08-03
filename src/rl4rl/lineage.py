"""Lineage reconstruction and validation without a graph dependency."""

from __future__ import annotations

from dataclasses import dataclass

from rl4rl.schema import TrajectoryEvent


@dataclass(frozen=True, slots=True)
class LineageSummary:
    roots: tuple[str, ...]
    missing_parents: tuple[str, ...]
    max_depth: int
    edge_count: int


def summarize_lineage(events: list[TrajectoryEvent]) -> LineageSummary:
    by_id = {event.event_id: event for event in events}
    if len(by_id) != len(events):
        raise ValueError("event IDs must be unique")

    missing = sorted(
        {
            parent
            for event in events
            for parent in event.parent_ids
            if parent not in by_id
        }
    )
    roots = tuple(sorted(event.event_id for event in events if not event.parent_ids))
    state: dict[str, int] = {}
    depths: dict[str, int] = {}

    def depth(event_id: str) -> int:
        marker = state.get(event_id, 0)
        if marker == 1:
            raise ValueError(f"cycle detected at event {event_id!r}")
        if marker == 2:
            return depths[event_id]
        state[event_id] = 1
        known_parents = [p for p in by_id[event_id].parent_ids if p in by_id]
        value = 0 if not known_parents else 1 + max(depth(p) for p in known_parents)
        depths[event_id] = value
        state[event_id] = 2
        return value

    max_depth = max((depth(event_id) for event_id in by_id), default=0)
    return LineageSummary(
        roots=roots,
        missing_parents=tuple(missing),
        max_depth=max_depth,
        edge_count=sum(len(event.parent_ids) for event in events),
    )
