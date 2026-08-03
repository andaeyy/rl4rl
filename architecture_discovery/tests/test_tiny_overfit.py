import math
from pathlib import Path

import torch

from common.evaluator import load_candidate
from common.task_adapter import DEFAULT_TASK


ROOT = Path(__file__).resolve().parents[1]


def test_fixed_batch_overfit_has_finite_material_loss_drop():
    module = load_candidate(ROOT / "common" / "initial_candidate.py")
    model, _ = module.build_untrained_model(7)
    inputs, labels = DEFAULT_TASK.collate(
        [(1, 2), (9, 1), (12, 34), (999, 1)] * 2
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    initial = float(DEFAULT_TASK.teacher_forced_loss(model, inputs, labels))
    final = initial
    for _ in range(120):
        loss = DEFAULT_TASK.teacher_forced_loss(model, inputs, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
    assert math.isfinite(initial) and math.isfinite(final)
    assert final < initial * 0.25
