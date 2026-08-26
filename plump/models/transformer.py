# plump/models/transformer.py — v3 skeleton, decoupled from DQN.
# Inspired by FableDan/DanLM: Llama-style causal encoder + Q-head + NTP/belief.
# Stub interfaces stable; full impl is opt-in so vibe coding can't break DQN path.

from __future__ import annotations

import torch
import torch.nn as nn


class PlumpTransformerConfig:
    """Small, swappable — mirrors FableDan ModelConfig."""

    def __init__(
        self,
        vocab: int = 64,
        d_model: int = 128,
        n_blocks: int = 4,
        n_heads: int = 4,
        max_seq: int = 128,
        hand_dim: int = 52,
    ):
        self.vocab = vocab
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.max_seq = max_seq
        self.hand_dim = hand_dim


class PlumpTransformer(nn.Module):
    """Stub — interface compatible with future training.

    Forward: tokens [B,T] + hand_feats [B, hand_dim] -> Q [B, 52] + aux.
    Current impl is identity MLP placeholder; replace block-by-block without
    touching env/encode.
    """

    def __init__(self, cfg: PlumpTransformerConfig | None = None, action_size: int = 52):
        super().__init__()
        self.cfg = cfg or PlumpTransformerConfig()
        self.emb = nn.Embedding(self.cfg.vocab, self.cfg.d_model)
        # Placeholder single block — expand to 4 when ready.
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.cfg.d_model, nhead=self.cfg.n_heads, batch_first=True
            ),
            num_layers=1,
        )
        self.q_head = nn.Sequential(
            nn.Linear(self.cfg.d_model + self.cfg.hand_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, tokens: torch.Tensor, hand_feats: torch.Tensor) -> torch.Tensor:
        # tokens: [B,T] -> emb -> encoder -> last token
        x = self.emb(tokens)
        x = self.encoder(x)
        ctx = x[:, -1, :]  # causal last
        return self.q_head(torch.cat([ctx, hand_feats], dim=-1))
