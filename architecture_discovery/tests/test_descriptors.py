from pathlib import Path

from common.descriptor_extractor import extract_descriptors
from common.evaluator import load_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_starting_architecture_descriptors():
    module = load_candidate(ROOT / "common" / "initial_candidate.py")
    model, _ = module.build_untrained_model(1)
    result = extract_descriptors(module, model)
    assert result.categories["token_representation"] == "learned_lookup"
    assert result.categories["positional_integration"] == "learned_additive"
    assert result.categories["attention_organization"] == "standard_multihead"
    assert result.categories["depth_topology"] == "sequential_blocks"
