from pathlib import Path

import torch

from common.evaluator import load_candidate


ROOT = Path(__file__).resolve().parents[1]


def _state(seed: int):
    module = load_candidate(ROOT / "common" / "initial_candidate.py")
    model, _ = module.build_untrained_model(seed)
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def test_same_initialization_seed_produces_identical_cpu_state():
    first = _state(123)
    second = _state(123)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert all(tensor.device.type == "cpu" for tensor in first.values())


def test_different_initialization_seed_changes_parameters():
    first = _state(123)
    second = _state(124)
    assert any(not torch.equal(first[name], second[name]) for name in first)


def test_builder_never_loads_the_vendor_checkpoint(monkeypatch):
    def forbidden_load(*_args, **_kwargs):
        raise AssertionError("candidate builder attempted checkpoint loading")

    monkeypatch.setattr(torch, "load", forbidden_load)
    module = load_candidate(ROOT / "common" / "initial_candidate.py")
    model, metadata = module.build_untrained_model(5)
    assert isinstance(model, torch.nn.Module)
    assert metadata["initial_device"] == "cpu"
    source = (ROOT / "common" / "initial_candidate.py").read_text()
    assert "best.pt" not in source
    assert "torch.load" not in source
