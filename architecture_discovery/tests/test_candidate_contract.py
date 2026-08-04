from pathlib import Path

from common.candidate_contract import inspect_candidate_source, validate_candidate
from common.evaluator import load_candidate


ROOT = Path(__file__).resolve().parents[1]


def test_initial_candidate_satisfies_contract():
    module = load_candidate(ROOT / "common" / "initial_candidate.py")
    model, _ = module.build_untrained_model(1)
    result = validate_candidate(module, model)
    assert result.valid, result.reasons


def test_static_contract_rejects_checkpoint_optimizer_and_file_control():
    source = """
import os
import torch

def build_untrained_model(seed):
    model = torch.load("best.pt")
    optimizer = torch.optim.AdamW(model.parameters())
    open("leak.txt", "w")
    return model, {}
"""
    result = inspect_candidate_source(source)
    assert not result.valid
    joined = " ".join(result.reasons)
    assert "forbidden candidate import" in joined
    assert "candidate-controlled optimizer" in joined
    assert "forbidden candidate call" in joined


def test_static_contract_rejects_determinism_control_mutation():
    source = """
import torch

def build_untrained_model(seed):
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    return torch.nn.Linear(1, 1), {}
"""
    result = inspect_candidate_source(source)
    assert not result.valid
    assert "candidate-controlled deterministic runtime" in " ".join(result.reasons)
