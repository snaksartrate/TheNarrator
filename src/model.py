"""Shared decoder-only Transformer for TheNarrator.

One architecture is used for all six experiments. Only vocab_size changes
(it comes from the tokenizer).

Layout (pre-norm, GPT style):
    token emb + position emb -> dropout
    -> N x [ x + Attn(LN(x)) ; x + FFN(LN(x)) ]
    -> final LayerNorm -> linear head -> logits
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Architecture settings. Defaults are one small baseline, shared by all runs."""
    vocab_size: int                 # injected from the tokenizer
    block_size: int = 256           # maximum context length in tokens
    n_embd: int = 128               # embedding dimension
    n_head: int = 4                 # attention heads
    n_layer: int = 4                # Transformer blocks
    ff_mult: int = 4                # feed-forward hidden size = ff_mult * n_embd
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size <= 0 or self.block_size <= 0:
            raise ValueError("vocab_size and block_size must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t can only see positions <= t."""

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # q, k, v in one matmul
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)      # merge heads back
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        # lower-triangular mask: 1 = allowed, 0 = future (blocked).
        # A buffer moves with the model to the right device automatically.
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # [B, T, C] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, h, T, T]
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))  # hide the future
        weights = self.attn_drop(F.softmax(scores, dim=-1))

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)  # merge heads
        return self.resid_drop(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP with GELU (fixed for all experiments)."""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, cfg.ff_mult * cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.ff_mult * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm Transformer block with residual connections."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        """idx, targets: LongTensor [B, T].

        Returns (logits [B, T, vocab_size], loss). loss is None when targets
        is None. Sequences longer than block_size raise ValueError (the
        generator crops the context before calling the model).
        """
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.cfg.block_size}"
            )
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if targets is not None:
            # flatten [B, T, V] -> [B*T, V] and [B, T] -> [B*T]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss