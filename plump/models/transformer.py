# plump/models/transformer.py — FableDan-style causal transformer for Plump.
# Decoupled: no env, no encode mutation. Restart-from-scratch ready, league includes old DQN.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encode.tokenizer import VOCAB, MAX_SEQ, PAD_TOK, FEAT_DIM


class PlumpTransformerConfig:
    def __init__(
        self,
        vocab: int = VOCAB,
        d_model: int = 128,
        n_blocks: int = 4,
        n_heads: int = 4,
        qk_dim: int = 32,
        v_dim: int = 32,
        ffn_hidden: int = 512,
        hand_hidden: int = 256,
        n_hand_layers: int = 2,
        q_hidden: int = 512,
        n_q_layers: int = 2,
        max_seq: int = MAX_SEQ,
        ntp_weight: float = 0.02,
        belief_weight: float = 0.05,
        feat_dim: int = FEAT_DIM,
    ):
        self.vocab = vocab
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.ffn_hidden = ffn_hidden
        self.hand_hidden = hand_hidden
        self.n_hand_layers = n_hand_layers
        self.q_hidden = q_hidden
        self.n_q_layers = n_q_layers
        self.max_seq = max_seq
        self.ntp_weight = ntp_weight
        self.belief_weight = belief_weight
        self.feat_dim = feat_dim

    def to_dict(self):
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d):
        c = cls()
        for k, v in d.items():
            if hasattr(c, k):
                setattr(c, k, v)
        return c


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps).to(x.dtype)
        return x * self.weight


def build_rope(max_seq, dim, theta=10000.0):
    half = dim // 2
    if half == 0:
        return torch.zeros(max_seq, 1), torch.zeros(max_seq, 1)
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    t = torch.arange(max_seq).float()
    ang = torch.outer(t, freqs)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    if d == 0:
        return x
    x1, x2 = x[..., :d], x[..., d:]
    c = cos[None, None, : x.shape[2], :]
    s = sin[None, None, : x.shape[2], :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: PlumpTransformerConfig):
        super().__init__()
        h, qk, v = cfg.n_heads, cfg.qk_dim, cfg.v_dim
        self.h, self.qk, self.v = h, qk, v
        self.q_proj = nn.Linear(cfg.d_model, h * qk, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, h * qk, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, h * v, bias=False)
        self.out_proj = nn.Linear(h * v, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(qk)
        self.k_norm = RMSNorm(qk)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.h, self.qk).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.h, self.qk).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.h, self.v).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, self.h * self.v)
        return self.out_proj(o)


class FFN(nn.Module):
    def __init__(self, cfg: PlumpTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: PlumpTransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


def _mlp(in_dim, hidden, n_hidden, out_dim):
    layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class PlumpTransformer(nn.Module):
    """Transformer for Plump with Q-head + NTP + belief (training only).

    Forward for DMC: tokens [B,T], hand_feats [B, F], legal mask optional -> Q [B, A].
    Action space is unified 52 play + 11 bid, but we mask.
    For simplicity we have separate Q heads for bid/play but shared encoder.
    """

    def __init__(self, cfg: PlumpTransformerConfig | None = None, bid_actions: int = 11, play_actions: int = 52):
        super().__init__()
        self.cfg = cfg or PlumpTransformerConfig()
        cfg = self.cfg
        self.token_emb = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_blocks))
        self.final_norm = RMSNorm(cfg.d_model)
        cos, sin = build_rope(cfg.max_seq, cfg.qk_dim)
        self.register_buffer("rope_cos", cos, persistent=True)
        self.register_buffer("rope_sin", sin, persistent=True)

        self.hand_mlp = _mlp(cfg.feat_dim, cfg.hand_hidden, cfg.n_hand_layers, cfg.hand_hidden)
        # Q heads: bid 11, play 52 — share context + hand
        self.q_bid = _mlp(cfg.d_model + cfg.hand_hidden, cfg.q_hidden, cfg.n_q_layers, bid_actions)
        self.q_play = _mlp(cfg.d_model + cfg.hand_hidden, cfg.q_hidden, cfg.n_q_layers, play_actions)
        # Aux heads (training only)
        self.ntp_head = nn.Linear(cfg.d_model, cfg.vocab)  # next token
        self.belief_head = nn.Linear(cfg.d_model, 52)  # predict opp hand one-hot 52

    def encode_seq(self, tokens: torch.Tensor, lengths: torch.Tensor | None = None):
        # tokens: [B,T]
        B, T = tokens.shape
        x = self.token_emb(tokens)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.final_norm(x)
        # last non-pad token per sequence
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0)
            ctx = x[torch.arange(B, device=x.device), idx]
        else:
            ctx = x[:, -1, :]
        return ctx, x  # ctx [B,d], full [B,T,d]

    def forward(self, tokens: torch.Tensor, hand_feats: torch.Tensor, phase: str = "play"):
        # tokens [B,T], hand_feats [B, F]
        lengths = (tokens != PAD_TOK).sum(dim=1)
        ctx, full = self.encode_seq(tokens, lengths)
        hand_emb = self.hand_mlp(hand_feats)
        cat = torch.cat([ctx, hand_emb], dim=-1)
        if phase == "bid":
            return self.q_bid(cat)
        return self.q_play(cat)

    def forward_with_aux(self, tokens, hand_feats, phase="play"):
        q = self.forward(tokens, hand_feats, phase)
        # aux: ntp on full sequence
        _, full = self.encode_seq(tokens)
        ntp_logits = self.ntp_head(full)  # [B,T,V]
        belief = self.belief_head(full[:, -1, :])  # [B,52]
        return q, ntp_logits, belief
