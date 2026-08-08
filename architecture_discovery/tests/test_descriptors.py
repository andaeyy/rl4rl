from pathlib import Path

from common.descriptor_extractor import extract_descriptors, extract_ir_descriptors
from common.evaluator import load_candidate
from tests.test_architecture_ir_graph import valid_graph


ROOT = Path(__file__).resolve().parents[1]


def test_starting_architecture_descriptors():
    module = load_candidate(ROOT / "common" / "initial_candidate.py")
    model, _ = module.build_untrained_model(1)
    result = extract_descriptors(module, model)
    assert result.categories["token_representation"] == "learned_lookup"
    assert result.categories["positional_integration"] == "learned_additive"
    assert result.categories["attention_organization"] == "standard_multihead"
    assert result.categories["depth_topology"] == "sequential_blocks"


def test_ir_descriptors_use_typed_attributes_not_candidate_prose():
    graph = valid_graph(
        metadata={
            "tokenization": "digit_reversed_output",
            "description": "claims a revolutionary tiny model",
        }
    )
    result = extract_ir_descriptors(graph)
    assert result.categories["token_representation"] == "learned_lookup"
    assert result.categories["positional_integration"] == "learned_additive"
    assert result.categories["attention_organization"] == "standard_multihead"
    assert result.categories["tokenization"] == "digit_reversed_output"


def test_ir_descriptor_vector_is_invariant_to_all_candidate_metadata():
    baseline = extract_ir_descriptors(valid_graph(metadata={}))
    metadata_only_mutation = extract_ir_descriptors(
        valid_graph(
            metadata={
                "tokenization": "digit_pair",
                "description": "pretend this changes every architecture axis",
                "parameter_count": 1,
                "nested_claims": ("rotary", ("multiquery", "rmsnorm")),
            }
        )
    )

    assert metadata_only_mutation.categories == baseline.categories
    assert metadata_only_mutation.codes == baseline.codes
    assert metadata_only_mutation.confidence == baseline.confidence
    assert baseline.categories["tokenization"] == "digit_reversed_output"
