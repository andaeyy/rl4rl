from openevolve.config import Config, DatabaseConfig
from openevolve.database import Program, ProgramDatabase

from agents.openevolve_semantic.semantic_archive import install_semantic_archive


def test_semantic_category_codes_map_to_stable_cells():
    install_semantic_archive()
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

