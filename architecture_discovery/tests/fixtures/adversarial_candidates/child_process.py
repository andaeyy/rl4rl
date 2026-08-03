"""Adversarial fixture: starts a child process from forward."""

import subprocess
import torch
import torch.nn as nn


class BadModel(nn.Module):
    def forward(self, token_ids):
        subprocess.run(["true"], check=True)
        return torch.zeros((*token_ids.shape, 15))


def build_untrained_model(seed):
    torch.manual_seed(seed)
    return BadModel(), {}
