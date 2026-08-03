import pytest

from rl4rl.lineage import summarize_lineage
from rl4rl.schema import EventStatus, Paradigm, TrajectoryEvent


def _event(event_id: str, parents: list[str]) -> TrajectoryEvent:
    return TrajectoryEvent(
        event_id=event_id,
        run_id="run",
        paradigm=Paradigm.OTHER,
        step=0,
        status=EventStatus.PROPOSED,
        parent_ids=parents,
    )


def test_lineage_depth() -> None:
    summary = summarize_lineage(
        [_event("a", []), _event("b", ["a"]), _event("c", ["b"])]
    )
    assert summary.roots == ("a",)
    assert summary.max_depth == 2
    assert summary.edge_count == 2


def test_lineage_cycle_is_rejected() -> None:
    with pytest.raises(ValueError, match="cycle"):
        summarize_lineage([_event("a", ["b"]), _event("b", ["a"])])
