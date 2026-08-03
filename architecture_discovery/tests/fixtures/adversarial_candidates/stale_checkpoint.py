"""Adversarial fixture: loads stale weights instead of starting fresh."""

import torch


def build_untrained_model(seed):
    model = torch.load("best.pt")
    model.load_state_dict(torch.load("stale_state.pt"))
    return model, {"claimed_seed": seed}
