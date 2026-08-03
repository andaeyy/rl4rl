"""JSONL input/output for canonical trajectory events."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from rl4rl.schema import TrajectoryEvent


class EventFileError(ValueError):
    """An event file contains an invalid row."""


def load_events(path: str | Path) -> list[TrajectoryEvent]:
    source = Path(path)
    events: list[TrajectoryEvent] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("row must be a JSON object")
                event = TrajectoryEvent.from_dict(value)
            except (ValueError, TypeError, KeyError) as exc:
                raise EventFileError(f"{source}:{line_number}: {exc}") from exc
            if event.event_id in seen:
                raise EventFileError(
                    f"{source}:{line_number}: duplicate event_id {event.event_id!r}"
                )
            seen.add(event.event_id)
            events.append(event)
    return events


def write_events(path: str | Path, events: Iterable[TrajectoryEvent]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for event in events:
            json.dump(event.to_dict(), handle, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
