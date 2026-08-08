import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from architecture_ir.runtime_evidence import (
    RuntimeBindings,
    probe_fresh_build,
    probe_runtime_validity,
)
from common.initial_candidate import build_untrained_model


FIXTURES = Path(__file__).parent / "fixtures" / "adversarial_candidates"
GRAPH_HASH = "a" * 64
TOKENS = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


def _load_fixture(name: str):
    path = FIXTURES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"adversarial_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trusted_attention_bindings_collect_runtime_causal_and_intervention_evidence():
    model, _ = build_untrained_model(17)
    model.graph_hash = GRAPH_HASH
    evidence = probe_runtime_validity(
        model,
        bindings=RuntimeBindings(
            graph_hash=GRAPH_HASH,
            attention_modules={
                "attention.0": "blocks.0.attn",
                "attention.1": "blocks.1.attn",
            },
        ),
        token_ids=TOKENS,
        expected_device="cpu",
    )
    assert evidence.passed, evidence.to_dict()
    assert all(count >= 3 for count in evidence.attention_calls.values())
    assert evidence.checks["causal_prefix_invariance"]
    assert evidence.checks["attention_influences_output"]
    assert set(evidence.causal_mask_buffers_observed) == {
        "attention.0",
        "attention.1",
    }


def test_attention_name_and_execution_are_insufficient_when_output_is_ignored():
    fixture = _load_fixture("dummy_attention")
    model, _ = fixture.build_untrained_model(11)
    evidence = probe_runtime_validity(
        model,
        bindings=RuntimeBindings(
            graph_hash=GRAPH_HASH,
            attention_modules={"attention": "attention"},
        ),
        token_ids=TOKENS,
        expected_device="cpu",
    )
    assert evidence.checks["attention_executed"]
    assert not evidence.checks["attention_influences_output"]
    assert not evidence.passed


def test_noncausal_attention_fails_future_token_metamorphic_probe():
    fixture = _load_fixture("noncausal_attention")
    model, _ = fixture.build_untrained_model(11)
    evidence = probe_runtime_validity(
        model,
        bindings=RuntimeBindings(
            graph_hash=GRAPH_HASH,
            attention_modules={"attention": "attention"},
        ),
        token_ids=TOKENS,
        expected_device="cpu",
    )
    assert evidence.checks["attention_executed"]
    assert not evidence.checks["causal_prefix_invariance"]
    assert not evidence.passed


def test_heuristic_or_missing_module_bindings_cannot_create_scientific_evidence():
    model, _ = build_untrained_model(5)
    heuristic = probe_runtime_validity(
        model,
        bindings=RuntimeBindings(
            graph_hash=GRAPH_HASH,
            attention_modules={"claimed": "blocks.0.attn"},
            provenance="class_name_heuristic",
        ),
        token_ids=TOKENS,
        expected_device="cpu",
    )
    assert not heuristic.checks["binding_valid"]
    assert not heuristic.passed

    missing = probe_runtime_validity(
        model,
        bindings=RuntimeBindings(
            graph_hash=GRAPH_HASH,
            attention_modules={"claimed": "does.not.exist"},
        ),
        token_ids=TOKENS,
        expected_device="cpu",
    )
    assert not missing.passed
    assert any("binding" in error for error in missing.errors)


def test_fresh_build_probe_requires_seed_reproducibility_and_seed_sensitivity():
    torch.manual_seed(404)
    rng_before = torch.get_rng_state().clone()
    evidence = probe_fresh_build(build_untrained_model, seed=101)
    assert evidence.passed, evidence.to_dict()
    assert torch.equal(torch.get_rng_state(), rng_before)

    def ignores_seed(_seed: int):
        torch.manual_seed(99)
        return nn.Linear(2, 2)

    stale = probe_fresh_build(ignores_seed, seed=101)
    assert stale.same_seed_state_equal
    assert not stale.different_seed_state_differs
    assert not stale.passed


def test_runtime_evidence_records_device_mismatch_as_failure():
    model, _ = build_untrained_model(3)
    evidence = probe_runtime_validity(
        model,
        bindings=RuntimeBindings(
            graph_hash=GRAPH_HASH,
            attention_modules={"attention": "blocks.0.attn"},
        ),
        token_ids=TOKENS,
        expected_device="cpu",
    )
    assert evidence.observed_parameter_devices == ("cpu",)
    assert evidence.output_device == "cpu"
    assert evidence.checks["device_placement"]
