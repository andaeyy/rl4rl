from __future__ import annotations

import hashlib

from novelty.clustering import (
    CandidateMechanism,
    cluster_candidates,
    unique_cluster_counts_by_run,
)
from test_novelty_signatures import build_graph, signature


def candidate(
    candidate_id: str,
    run_id: str,
    *,
    width: int = 16,
    intervention: str = "large_effect",
) -> CandidateMechanism:
    return CandidateMechanism(
        study_id="synthetic-study",
        candidate_id=candidate_id,
        run_id=run_id,
        snapshot_sha256=hashlib.sha256(candidate_id.encode()).hexdigest(),
        qualification_record_id=f"qualification-{candidate_id}",
        signature=signature(
            build_graph(prefix=candidate_id.replace("-", ""), width=width),
            intervention=intervention,
        ),
    )


def test_clustering_is_stable_and_counts_once_per_run() -> None:
    members = [
        candidate("candidate-a", "run-one", width=16),
        candidate("candidate-b", "run-one", width=64),
        candidate("candidate-c", "run-two", width=32),
        candidate("candidate-d", "run-one", intervention="no_effect"),
    ]

    forward = cluster_candidates(members)
    reverse = cluster_candidates(reversed(members))

    assert [item.to_dict() for item in forward] == [item.to_dict() for item in reverse]
    assert len(forward) == 2
    shared = next(item for item in forward if len(item.candidate_ids) == 3)
    assert shared.run_ids == ("run-one", "run-two")
    assert shared.representative_by_run == (
        ("run-one", "candidate-a"),
        ("run-two", "candidate-c"),
    )
    assert unique_cluster_counts_by_run(forward) == {"run-one": 2, "run-two": 1}
