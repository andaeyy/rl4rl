"""Conventional untrained starting architecture for discovery runs.

The candidate owns architecture only.  The trusted evaluator owns data,
optimization, checkpointing, decoding, and task semantics.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 15
FIXED_SEQ_LEN = 35


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must divide evenly across heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        qkv = self.qkv(x).reshape(batch, steps, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        values = F.softmax(scores, dim=-1) @ v
        values = values.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(values)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, max_seq_len: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim, bias=False),
            nn.GELU(),
            nn.Linear(ff_dim, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class AdditionTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 16,
        n_heads: int = 2,
        n_layers: int = 2,
        ff_dim: int = 48,
        max_seq_len: int = FIXED_SEQ_LEN,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ff_dim, max_seq_len)
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, steps = token_ids.shape
        positions = torch.arange(steps, device=token_ids.device)
        x = self.token_emb(token_ids) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

def build_untrained_model(seed: int) -> tuple[nn.Module, dict]:
    """Build a freshly initialized CPU model from a supplied evaluator seed."""

    torch.manual_seed(seed)
    model = AdditionTransformer(
        d_model=16,
        n_heads=2,
        n_layers=2,
        ff_dim=48,
    )
    metadata = {
        "name": "Conventional AdderBoard Starting Architecture",
        "author": "anadim, experiment wrapper",
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": "2-layer decoder, learned token and position embeddings",
        "tricks": ["reversed output digits", "autoregressive carry propagation"],
        "initialization_seed": seed,
        "initial_device": "cpu",
    }
    return model, metadata
