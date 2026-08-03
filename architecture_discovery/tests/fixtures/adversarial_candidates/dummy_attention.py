"""Adversarial fixture: an attention-named module whose output is ignored."""

import torch
import torch.nn as nn


class DummyAttention(nn.Module):
    def forward(self, values):
        return values * 2


class DummyAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(15, 8)
        self.attention = DummyAttention()
        self.readout = nn.Linear(8, 15)

    def forward(self, token_ids):
        values = self.embedding(token_ids)
        self.attention(values)
        return self.readout(values)


def build_untrained_model(seed):
    torch.manual_seed(seed)
    return DummyAttentionModel(), {}
