from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

from agents.openevolve_semantic import semantic_archive
from common.descriptor_schema import SEMANTIC_METRIC_NAMES
from common.openevolve_policy import (
    canonical_combined_score,
    install_validity_first_policy,
)


def _metrics(**overrides):
    metrics = {
        name: 1.0 for name in SEMANTIC_METRIC_NAMES.values()
    }
    metrics.update(
        {
            "eligible_for_parent": 1.0,
            "search_score": 1.0,
        }
    )
    metrics.update(overrides)
    metrics["combined_score"] = canonical_combined_score(metrics)
    return metrics


def test_semantic_category_codes_map_to_stable_cells():
    semantic_archive.install_semantic_archive()
    dimensions = [
        "semantic_token_representation",
        "semantic_positional_integration",
    ]
    config = DatabaseConfig(
        feature_dimensions=dimensions,
        feature_bins={
            "semantic_token_representation": 5,
            "semantic_positional_integration": 6,
        },
        num_islands=1,
    )
    database = ProgramDatabase(config)
    program = Program(
        id="candidate",
        code="pass",
        metrics={
            "combined_score": 1.0,
            "semantic_token_representation": 3.0,
            "semantic_positional_integration": 4.0,
        },
    )
    assert database._calculate_feature_coords(program) == [3, 4]


def test_failed_candidate_has_unknown_signature_without_entering_parent_pool():
    install_validity_first_policy()
    semantic_archive.install_semantic_archive()
    dimensions = list(SEMANTIC_METRIC_NAMES.values())[:4]
    database = ProgramDatabase(
        DatabaseConfig(
            feature_dimensions=dimensions,
            feature_bins={name: 8 for name in dimensions},
            num_islands=1,
        )
    )
    # This is the minimal fallback emitted by OpenEvolve itself when an
    # evaluator raises before the project adapter can produce typed metrics.
    failed = Program(id="failed", code="pass", metrics={"error": 0.0})

    database.add(failed, iteration=0, target_island=0)

    assert failed.metadata["semantic_signature"] == [0] * len(
        SEMANTIC_METRIC_NAMES
    )
    assert failed.metadata["semantic_coverage_eligible"] is False
    assert failed.id in database.semantic_coverage.ineligible_programs
    assert failed.id not in database.semantic_coverage.signatures.get(
        tuple([0] * len(SEMANTIC_METRIC_NAMES)),
        set(),
    )
    assert failed.id not in database.islands[0]


def test_actual_island_sampling_uses_rare_semantic_families(monkeypatch):
    install_validity_first_policy()
    semantic_archive.install_semantic_archive()
    dimensions = [
        SEMANTIC_METRIC_NAMES["token_representation"],
        SEMANTIC_METRIC_NAMES["positional_integration"],
        SEMANTIC_METRIC_NAMES["attention_organization"],
        SEMANTIC_METRIC_NAMES["depth_topology"],
    ]
    database = ProgramDatabase(
        DatabaseConfig(
            feature_dimensions=dimensions,
            feature_bins={name: 8 for name in dimensions},
            num_islands=1,
            exploration_ratio=1.0,
            exploitation_ratio=0.0,
        )
    )
    common_a = Program(id="common-a", code="a", metrics=_metrics())
    common_b = Program(id="common-b", code="b", metrics=_metrics())
    rare = Program(
        id="rare",
        code="rare",
        metrics=_metrics(semantic_token_representation=2.0),
    )
    for program in (common_a, common_b, rare):
        database.add(program, target_island=0)

    captured = {}

    def choose(population, *, weights, k):
        captured["weights"] = dict(
            zip((program.id for program in population), weights)
        )
        return [population[weights.index(max(weights))]]

    monkeypatch.setattr(semantic_archive.random, "random", lambda: 0.0)
    monkeypatch.setattr(semantic_archive.random, "choices", choose)

    parent, _ = database.sample_from_island(0, num_inspirations=1)

    assert parent.id == rare.id
    assert captured["weights"]["rare"] > captured["weights"]["common-a"]
