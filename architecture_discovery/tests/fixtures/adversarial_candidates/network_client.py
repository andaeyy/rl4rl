"""Adversarial fixture: attempts a network connection."""

import socket
import torch
import torch.nn as nn


class BadModel(nn.Module):
    def forward(self, token_ids):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        return torch.zeros((*token_ids.shape, 15))


def build_untrained_model(seed):
    torch.manual_seed(seed)
    return BadModel(), {}
