"""Normalize the TSV log format used by autoresearch-style runs."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from rl4rl.schema import (
    ArchitectureSnapshot,
    EventStatus,
    Paradigm,
    TrajectoryEvent,
)

ALIASES = {
    "commit": ("commit", "hash", "commit_hash", "sha"),
    "accuracy": ("accuracy", "acc", "verified_accuracy", "val_accuracy"),
    "parameters": ("parameters", "params", "parameter_count", "n_params"),
    "status": ("status", "result", "decision"),
    "description": ("description", "notes", "summary", "idea"),
    "parent": ("parent", "parent_hash", "parent_commit"),
}


def parse_autoresearch_tsv(path: str | Path, *, run_id: str) -> list[TrajectoryEvent]:
    source = Path(path)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"{source}: missing TSV header")
        columns = {_normalize(name): name for name in reader.fieldnames}
        resolved = {
            field: next((columns[alias] for alias in aliases if alias in columns), None)
            for field, aliases in ALIASES.items()
        }
        required = [
            field
            for field in ("commit", "accuracy", "parameters")
            if not resolved[field]
        ]
        if required:
            raise ValueError(
                f"{source}: missing columns for {', '.join(required)}; "
                f"found {', '.join(reader.fieldnames)}"
            )

        events: list[TrajectoryEvent] = []
        for step, row in enumerate(reader):
            commit = _cell(row, resolved["commit"]) or f"row-{step:05d}"
            raw_accuracy = _parse_float(_cell(row, resolved["accuracy"]))
            accuracy = raw_accuracy / 100 if raw_accuracy > 1 else raw_accuracy
            parameters = _parse_int(_cell(row, resolved["parameters"]))
            raw_status = _cell(row, resolved["status"])
            status, accepted, valid = _status(raw_status, accuracy)
            parent = _cell(row, resolved["parent"])
            event_id = f"{run_id}:{commit}"
            events.append(
                TrajectoryEvent(
                    event_id=event_id,
                    run_id=run_id,
                    paradigm=Paradigm.AUTORESEARCH,
                    step=step,
                    status=status,
                    parent_ids=[f"{run_id}:{parent}"] if parent else [],
                    accepted=accepted,
                    valid=valid,
                    proposal=_cell(row, resolved["description"]),
                    architecture=ArchitectureSnapshot(
                        parameters=parameters,
                        accuracy=accuracy,
                        qualifies=accuracy >= 0.99,
                    ),
                    artifact_refs=[f"{source}:{step + 2}"],
                    provenance={
                        "adapter": "autoresearch-tsv-v1",
                        "raw_status": raw_status,
                        "taxonomy_pending": True,
                    },
                )
            )
    return events


def _status(raw: str | None, accuracy: float) -> tuple[EventStatus, bool, bool]:
    normalized = _normalize(raw or "")
    if any(token in normalized for token in ("keep", "accept", "success", "pass")):
        return EventStatus.ACCEPTED, True, True
    if "rollback" in normalized or "revert" in normalized:
        return EventStatus.ROLLED_BACK, False, True
    if any(token in normalized for token in ("invalid", "cheat", "hack")):
        return EventStatus.INVALID, False, False
    if any(token in normalized for token in ("error", "crash", "timeout")):
        return EventStatus.ERROR, False, False
    if any(token in normalized for token in ("discard", "reject", "fail")):
        return EventStatus.REJECTED, False, True
    if accuracy >= 0.99:
        return EventStatus.EVALUATED, None, True
    return EventStatus.EVALUATED, None, True


def _cell(row: dict[str, str], column: str | None) -> str | None:
    if column is None:
        return None
    value = (row.get(column) or "").strip()
    return value or None


def _parse_float(value: str | None) -> float:
    if value is None:
        raise ValueError("accuracy cell is empty")
    return float(value.rstrip("%"))


def _parse_int(value: str | None) -> int:
    if value is None:
        raise ValueError("parameter cell is empty")
    return int(re.sub(r"[,_ ]", "", value))


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
