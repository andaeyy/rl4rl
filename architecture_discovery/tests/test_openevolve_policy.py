from __future__ import annotations

import pytest
from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

from common.openevolve_policy import (
    canonical_combined_score,
    install_validity_first_policy,
)


def _program(identifier: str, *, eligible: float, score: float):
    return Program(
        id=identifier,
        code="pass",
        metrics={
            "eligible_for_parent": eligible,
            "search_score": score,
        },
        iteration_found=1,
        timestamp=1.0,
    )


def test_validity_precedes_scalar_score_and_never_reads_parameter_metadata():
    install_validity_first_policy()
    database = ProgramDatabase(DatabaseConfig())
    valid = _program("valid", eligible=1.0, score=0.99)
    invalid = _program("invalid", eligible=0.0, score=1.0)
    invalid.metrics["parameter_count_metadata"] = 1
    valid.metrics["parameter_count_metadata"] = 10**9

    assert database._is_better(valid, invalid)
    assert not database._is_better(invalid, valid)


def test_canonical_combined_score_ignores_descriptors_and_parameter_metadata():
    reference = {
        "eligible_for_parent": 1.0,
        "search_score": 0.75,
    }
    polluted = {
        **reference,
        "semantic_attention_organization": 99_999.0,
        "parameter_count_metadata": -10**12,
        "public_accuracy": 0.0,
    }

    assert canonical_combined_score(reference) == 2.75
    assert canonical_combined_score(polluted) == 2.75
    assert canonical_combined_score(
        {"eligible_for_parent": 0.0, "search_score": 1.0}
    ) < canonical_combined_score(
        {"eligible_for_parent": 1.0, "search_score": 0.0}
    )


def test_ineligible_programs_are_records_but_never_parents_or_inspirations():
    install_validity_first_policy()
    database = ProgramDatabase(DatabaseConfig(num_islands=1))
    valid = _program("valid-parent", eligible=1.0, score=0.99)
    invalid = _program("failed-child", eligible=0.0, score=1.0)

    database.add(valid, iteration=0, target_island=0)
    database.add(invalid, iteration=1, target_island=0)

    assert invalid.id in database.programs
    assert invalid.id not in database.islands[0]
    assert invalid.id not in database.archive

    # Simulate an older database that persisted an invalid program in active
    # pools. Sampling must sanitize it before selecting a parent.
    database.islands[0].add(invalid.id)
    database.archive.add(invalid.id)
    parent, inspirations = database.sample_from_island(0, num_inspirations=3)

    assert parent.id == valid.id
    assert database._sample_random_parent().id == valid.id
    assert all(program.id != invalid.id for program in inspirations)
    assert invalid.id not in database.islands[0]
    assert invalid.id not in database.archive


def test_sampling_fails_closed_when_no_candidate_is_parent_eligible():
    install_validity_first_policy()
    database = ProgramDatabase(DatabaseConfig(num_islands=1))
    database.add(_program("failed", eligible=0.0, score=0.0), iteration=0)

    with pytest.raises(RuntimeError, match="no eligible parent"):
        database.sample_from_island(0)


def test_ineligible_only_database_has_no_best_or_top_artifact(tmp_path):
    install_validity_first_policy()
    database = ProgramDatabase(DatabaseConfig(num_islands=1))
    failed = _program("failed", eligible=0.0, score=1.0)
    database.add(failed, iteration=0)

    best = database.get_best_program()
    artifact = tmp_path / "best_program.json"
    if best is not None:
        artifact.write_text(best.code)

    assert failed.id in database.programs
    assert best is None
    assert database.best_program_id is None
    assert database.get_top_programs() == []
    assert not artifact.exists()


def test_best_and_top_programs_keep_validity_first_accuracy_ranking():
    install_validity_first_policy()
    database = ProgramDatabase(DatabaseConfig(num_islands=1))
    lower = _program("lower", eligible=1.0, score=0.4)
    higher = _program("higher", eligible=1.0, score=0.8)
    failed = _program("failed", eligible=0.0, score=1.0)
    lower.metrics["semantic_attention_organization"] = 10**9
    higher.metrics["semantic_attention_organization"] = -10**9

    # Model an imported database so this regression also covers stale invalid
    # best pointers that bypassed the gated add path in an older run.
    database.programs.update(
        {program.id: program for program in (lower, higher, failed)}
    )
    database.best_program_id = failed.id

    assert database.get_best_program().id == higher.id
    assert [program.id for program in database.get_top_programs()] == [
        higher.id,
        lower.id,
    ]
    assert database.best_program_id == higher.id
