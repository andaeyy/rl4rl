from __future__ import annotations

from openevolve.config import DatabaseConfig
from openevolve.database import Program, ProgramDatabase

from common.openevolve_policy import install_validity_first_policy


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
