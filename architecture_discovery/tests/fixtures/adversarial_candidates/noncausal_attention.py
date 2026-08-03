"""Adversarial fixture: earlier outputs depend on future tokens."""

import torch
import torch.nn as nn


class NoncausalAttention(nn.Module):
    def forward(self, values):
        summary = values.mean(dim=1, keepdim=True)
        return summary.expand_as(values)


class NoncausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(15, 8)
        self.attention = NoncausalAttention()
        self.readout = nn.Linear(8, 15)

    def forward(self, token_ids):
        values = self.embedding(token_ids)
        return self.readout(values + self.attention(values))


def build_untrained_model(seed):
    torch.manual_seed(seed)
    return NoncausalModel(), {}
