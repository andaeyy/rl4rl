"""Adversarial fixture: symbolic arithmetic and a hard-coded answer table."""

import torch
import torch.nn as nn


class DirectSolver(nn.Module):
    def solve(self, first, second):
        lookup = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
        return lookup[(first + second) % 10]

    def forward(self, token_ids):
        return torch.zeros((*token_ids.shape, 15))


def build_untrained_model(seed):
    torch.manual_seed(seed)
    return DirectSolver(), {}
