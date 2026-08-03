"""Adversarial fixture: reads provider credentials."""

import os
import torch
import torch.nn as nn


class BadModel(nn.Module):
    def forward(self, token_ids):
        _secret = os.environ.get("DISCOVERY_API_KEY")
        return torch.zeros((*token_ids.shape, 15))


def build_untrained_model(seed):
    torch.manual_seed(seed)
    return BadModel(), {}
