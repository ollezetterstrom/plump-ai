"""Causal RoPE Transformer with distributional, belief and trick heads."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from plump.config import PlumpTransformerConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


class RoPE(nn.Module):
    """Rotary position embeddings over adjacent dimension pairs."""

    def __init__(self, dim: int, max_seq: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        positions = torch.arange(max_seq, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        t = x.shape[-2]
        dtype = x.dtype
        cos = self.cos_cached[:t].to(dtype=torch.float32)
        sin = self.sin_cached[:t].to(dtype=torch.float32)
        cos = cos.repeat_interleave(2, dim=-1)[None, None, :, :]
        sin = sin.repeat_interleave(2, dim=-1)[None, None, :, :]

        xf = x.float()
        even = xf[..., ::2]
        odd = xf[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return (xf * cos + rotated * sin).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: PlumpTransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qk_dim = cfg.qk_dim
        self.v_dim = cfg.v_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.qk_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.qk_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.v_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.v_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.qk_dim)
        self.k_norm = RMSNorm(cfg.qk_dim)
        self.rope = RoPE(cfg.qk_dim, cfg.max_seq)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads, self.qk_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_heads, self.qk_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_heads, self.v_dim).transpose(1, 2)

        q = self.rope(self.q_norm(q))
        k = self.rope(self.k_norm(k))

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).reshape(b, t, self.n_heads * self.v_dim)
        return self.o_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: PlumpTransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DuelingHead(nn.Module):
    """value + advantage - mean(advantage) for a discrete action set."""

    def __init__(self, in_dim: int, hidden: int, n_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        value = self.value(h)
        advantage = self.advantage(h)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class HandTrunk(nn.Module):
    def __init__(self, feat_dim: int, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class PlumpTransformer(nn.Module):
    """Causal RoPE policy/return model for Plump.

    The model consumes the engine's 96-token event stream and the 296-float
    state vector, then produces:
      * dueling Q-values for bids 0..10 and cards 0..51,
      * per-opponent belief logits for all 52 cards,
      * a categorical distribution over final tricks 0..10.
    """

    def __init__(self, cfg: Optional[PlumpTransformerConfig] = None):
        super().__init__()
        self.cfg = cfg or PlumpTransformerConfig()
        c = self.cfg

        self.token_emb = nn.Embedding(c.vocab, c.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(c) for _ in range(c.n_blocks)]
        )
        self.final_norm = RMSNorm(c.d_model)
        self.hand_trunk = HandTrunk(c.feat_dim, c.hand_hidden)

        joint_dim = c.d_model + c.hand_hidden
        self.bid_head = DuelingHead(joint_dim, c.q_hidden, c.n_bid_actions)
        self.play_head = DuelingHead(joint_dim, c.q_hidden, c.n_play_actions)
        self.belief_head = nn.Linear(joint_dim, 3 * 52)
        self.tricks_dist_head = nn.Linear(joint_dim, c.n_tricks)

    def encode_seq(
        self, tokens: torch.Tensor, hand_feats: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if hand_feats.dim() == 1:
            hand_feats = hand_feats.unsqueeze(0)

        lengths = (tokens != 0).sum(dim=1)
        max_t = max(2, int(lengths.max().item()))
        max_t = min(max_t, tokens.shape[1])
        tokens = tokens[:, :max_t]

        x = self.token_emb(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        idx = (lengths - 1).clamp(min=0, max=max_t - 1)
        ctx = x[torch.arange(x.shape[0], device=x.device), idx]
        hand_emb = self.hand_trunk(hand_feats)
        joint = torch.cat([ctx, hand_emb], dim=-1)
        return ctx, hand_emb, joint

    def forward_with_aux(
        self,
        tokens: torch.Tensor,
        hand_feats: torch.Tensor,
        phase: str = "play",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, joint = self.encode_seq(tokens, hand_feats)
        if phase.lower() in ("bid", "bidding"):
            q = self.bid_head(joint)
        else:
            q = self.play_head(joint)
        belief_logits = self.belief_head(joint).view(-1, 3, 52)
        tricks_logits = self.tricks_dist_head(joint)
        return q, belief_logits, tricks_logits

    def forward(
        self,
        tokens: torch.Tensor,
        hand_feats: torch.Tensor,
        phase: str = "play",
    ) -> torch.Tensor:
        _, _, joint = self.encode_seq(tokens, hand_feats)
        if phase.lower() in ("bid", "bidding"):
            return self.bid_head(joint)
        return self.play_head(joint)

    def predict_belief_probs(
        self,
        tokens: Union[torch.Tensor, np.ndarray],
        hand_feats: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        """Return sigmoid belief probabilities for relative opponents +1,+2,+3.

        A single (unbatched) observation returns shape ``[3, 52]``; a batch
        returns ``[B, 3, 52]``.
        """
        squeezed = isinstance(tokens, np.ndarray) and tokens.ndim == 1
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        if isinstance(tokens, np.ndarray):
            tokens_t = torch.from_numpy(tokens).long()
        else:
            tokens_t = tokens.long()
        if isinstance(hand_feats, np.ndarray):
            feats_t = torch.from_numpy(hand_feats).float()
        else:
            feats_t = hand_feats.float()

        if tokens_t.dim() == 1:
            tokens_t = tokens_t.unsqueeze(0)
            squeezed = True
        if feats_t.dim() == 1:
            feats_t = feats_t.unsqueeze(0)

        tokens_t = tokens_t.to(device)
        feats_t = feats_t.to(device=device, dtype=dtype)

        with torch.no_grad():
            _, belief_logits, _ = self.forward_with_aux(
                tokens_t, feats_t, phase="play"
            )
            probs = torch.sigmoid(belief_logits.float()).cpu().numpy()
        if squeezed:
            return probs[0]
        return probs
