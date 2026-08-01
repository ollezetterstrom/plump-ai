# Network architecture (§5.5).
#
# EmbeddingBag over a fixed-width index tensor with a constant offsets tensor
# and a padding index: static shapes everywhere, which is what makes CUDA-graph
# capture possible. Disjoint bid/play heads (§5.1) plus a categorical value
# head (§4.2), a trick-count head, and a belief head over 52*8 classes (§5.3).

from typing import Optional

import torch
import torch.nn as nn

from . import config


class PlumpNet(nn.Module):
    def __init__(self, net: Optional[config.NetConfig] = None,
                 scoring: Optional[config.ScoringConfig] = None):
        super().__init__()
        nc = net or config.NetConfig()
        sc = scoring or config.ScoringConfig()
        self.n_features = nc.n_features
        self.n_belief_classes = nc.n_belief_classes

        # Categorical value support (§4.2): enumerate achievable scores from the
        # scoring config, pad to min_atoms with a sentinel far outside reach.
        self.score_atoms = _score_atoms(sc, nc.min_atoms)
        self.register_buffer("atoms", torch.tensor(self.score_atoms, dtype=torch.float32))
        # score_table[bid][tricks] for the factored post-bid value (§4.2).
        tbl = torch.full((sc.max_bid + 1, sc.max_bid + 1), 0, dtype=torch.float32)
        for bid in range(sc.min_bid, sc.max_bid + 1):
            for tricks in range(sc.max_bid + 1):
                tbl[bid, tricks] = _score(sc, bid, tricks)
        self.register_buffer("score_table", tbl)

        # index n_features is a no-op pad (padding_idx zeroes it and drops its
        # gradient). offsets is constant: arange(B + 1) * K_MAX.
        self.embed = nn.EmbeddingBag(
            nc.n_features + 1, nc.hidden, mode="sum", include_last_offset=True,
            padding_idx=nc.n_features,
        )
        self.b0 = nn.Parameter(torch.zeros(nc.hidden))
        self.body = nn.Sequential(
            nn.LayerNorm(nc.hidden), nn.ReLU(), nn.Linear(nc.hidden, nc.hidden),
            nn.LayerNorm(nc.hidden), nn.ReLU(), nn.Linear(nc.hidden, nc.hidden),
        )
        self.bid_head = nn.Linear(nc.hidden, 11)
        self.play_head = nn.Linear(nc.hidden, 52)
        self.value_head = nn.Linear(nc.hidden, len(self.score_atoms))
        self.trick_head = nn.Linear(nc.hidden, 11)
        self.belief_head = nn.Linear(nc.hidden, 52 * nc.n_belief_classes)

    def offsets(self, batch: int, device, dtype=torch.int64) -> torch.Tensor:
        return torch.arange(batch + 1, device=device, dtype=dtype) * config.K_MAX

    def features(self, obs_idx: torch.Tensor) -> torch.Tensor:
        # obs_idx: [B, K_MAX] int (fixed width), pad slots == n_features.
        # EmbeddingBag over the flat view with a constant offsets tensor
        # (§5.5): static shapes, CUDA-graph capturable.
        b, _ = obs_idx.shape
        h = self.embed(obs_idx.reshape(-1), self.offsets(b, obs_idx.device))
        return self.body(h + self.b0)

    def forward(self, obs_idx: torch.Tensor):
        h = self.features(obs_idx)
        return {
            "bid": self.bid_head(h),
            "play": self.play_head(h),
            "value": self.value_head(h),
            "tricks": self.trick_head(h),
            "belief": self.belief_head(h),
        }

    def value(self, value_logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(value_logits, -1) @ self.atoms

    def log_prob(self, logits: torch.Tensor, action: torch.Tensor,
                 legal: Optional[torch.Tensor] = None) -> torch.Tensor:
        if legal is not None:
            logits = logits.masked_fill(~legal, -1e30)
        return torch.log_softmax(logits, -1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

    def entropy(self, logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        logits = logits.masked_fill(~legal, -1e30)
        p = torch.softmax(logits, -1)
        return -(p * torch.log_softmax(logits, -1)).sum(-1)


def _score(sc: config.ScoringConfig, bid: int, tricks: int) -> int:
    if bid == tricks:
        if bid == 0:
            return sc.zero_bid_bonus  # special case: the "05" rule
        return sc.make_bonus + bid
    return -sc.miss_penalty * abs(bid - tricks)


def _score_atoms(sc: config.ScoringConfig, min_atoms: int) -> list[int]:
    atoms = sorted({
        _score(sc, bid, tricks)
        for bid in range(sc.min_bid, sc.max_bid + 1)
        for tricks in range(0, sc.max_bid + 1)
    })
    while len(atoms) < min_atoms:
        atoms.append(atoms[-1] + 1_000)  # unreachable sentinel pads
    return atoms
