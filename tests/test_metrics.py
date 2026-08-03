from pathlib import Path

import pytest

from rl4rl.io import load_events
from rl4rl.metrics import frontier_progression, summarize_metrics

ROOT = Path(__file__).parents[1]


def test_example_metrics() -> None:
    events = load_events(ROOT / "data" / "examples" / "synthetic_trajectory.jsonl")
    summary = summarize_metrics(events, external_frontier=36)
    assert summary.total_events == 5
    assert summary.accepted_events == 4
    assert summary.invalid_events == 1
    assert summary.boundary_crossing_events == 1
    assert summary.best_qualifying_parameters == 36
    assert summary.frontier_gap_ratio == pytest.approx(1.0)
    assert summary.verified_reward_hacks == 1


def test_frontier_progression_only_improves() -> None:
    events = load_events(ROOT / "data" / "examples" / "synthetic_trajectory.jsonl")
    assert frontier_progression(events) == [
        (0, 6080),
        (1, 3000),
        (3, 1694),
        (4, 36),
    ]
