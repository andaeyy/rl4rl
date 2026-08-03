from pathlib import Path

from rl4rl.schema import BoundaryLabel
from rl4rl.taxonomy import Taxonomy

ROOT = Path(__file__).parents[1]


def test_additive_position_substitution_is_preserving() -> None:
    taxonomy = Taxonomy.load(ROOT / "configs" / "taxonomy.toml")
    edit = taxonomy.suggest(
        component="positional_encoding",
        operation="substitute",
        before="learned_absolute",
        after="sinusoidal",
    )
    assert edit.boundary_label == BoundaryLabel.PRESERVING
    assert edit.needs_review


def test_additive_to_rope_is_changing() -> None:
    taxonomy = Taxonomy.load(ROOT / "configs" / "taxonomy.toml")
    edit = taxonomy.suggest(
        component="positional_encoding",
        operation="substitute",
        before="sinusoidal",
        after="rope",
    )
    assert edit.boundary_label == BoundaryLabel.CHANGING


def test_unknown_technique_is_ambiguous() -> None:
    taxonomy = Taxonomy.load(ROOT / "configs" / "taxonomy.toml")
    edit = taxonomy.suggest(
        component="token_embedding",
        operation="substitute",
        before="learned_table",
        after="new_unknown_thing",
    )
    assert edit.boundary_label == BoundaryLabel.AMBIGUOUS
