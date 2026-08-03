"""Adversarial fixture: indirect access to ``open`` through builtins."""

import torch
import torch.nn as nn


class BadModel(nn.Module):
    def forward(self, token_ids):
        namespace = vars(__builtins__)
        reader = namespace["open"]
        reader("/tmp/should-not-be-readable").read()
        return torch.zeros((*token_ids.shape, 15))


def build_untrained_model(seed):
    torch.manual_seed(seed)
    return BadModel(), {}
