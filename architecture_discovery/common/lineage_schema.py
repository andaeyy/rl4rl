"""Append-only lineage records for all research controllers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class CandidateRecord:
    run_id: str
    condition: str
    seed: int
    candidate_id: str
    parent_id: str | None
    inspiration_ids: list[str] = field(default_factory=list)
    proposal_text: str = ""
    mechanism_hypothesis: str = ""
    prompt_hash: str = ""
    response_hash: str = ""
    code_hash: str = ""
    diff: str = ""
    proposal_timestamp: str = ""
    completion_timestamp: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    retention_decision: str = ""
    archive_cells: list[str] = field(default_factory=list)
    rollback_target: str | None = None
    future_parent_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def append_record(path: str | Path, record: CandidateRecord) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    evaluation = payload.pop("evaluation")
    payload.update(evaluation)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
