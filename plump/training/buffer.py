"""Pre-allocated NumPy ring buffers for fast replay sampling."""

from __future__ import annotations

from typing import Dict

import numpy as np


class RingBuffer:
    """Contiguous ring buffer with 24x suit augmentation on sample()."""

    def __init__(
        self,
        capacity: int,
        toks_dim: int = 96,
        feat_dim: int = 296,
        opp_dim: int = 156,
    ):
        self.capacity = int(capacity)
        self.toks = np.zeros((self.capacity, toks_dim), dtype=np.int16)
        self.feats = np.zeros((self.capacity, feat_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.returns = np.zeros(self.capacity, dtype=np.float32)
        self.opp_hands = np.zeros((self.capacity, opp_dim), dtype=np.float32)
        self.final_tricks = np.zeros(self.capacity, dtype=np.int64)
        self.is_play = np.zeros(self.capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add_batch(
        self,
        toks: np.ndarray,
        feats: np.ndarray,
        actions: np.ndarray,
        returns: np.ndarray,
        opp_hands: np.ndarray,
        final_tricks: np.ndarray,
        is_play: np.ndarray,
    ) -> None:
        m = len(actions)
        if m == 0:
            return
        if m > self.capacity:
            toks = toks[-self.capacity :]
            feats = feats[-self.capacity :]
            actions = actions[-self.capacity :]
            returns = returns[-self.capacity :]
            opp_hands = opp_hands[-self.capacity :]
            final_tricks = final_tricks[-self.capacity :]
            is_play = is_play[-self.capacity :]
            m = self.capacity

        idx = (self.pos + np.arange(m)) % self.capacity
        self.toks[idx] = toks.astype(np.int16)
        self.feats[idx] = feats.astype(np.float32)
        self.actions[idx] = actions.astype(np.int64)
        self.returns[idx] = returns.astype(np.float32)
        self.opp_hands[idx] = opp_hands.astype(np.float32)
        self.final_tricks[idx] = final_tricks.astype(np.int64)
        self.is_play[idx] = is_play.astype(np.bool_)

        self.pos = (self.pos + m) % self.capacity
        self.size = min(self.capacity, self.size + m)

    def sample(
        self,
        batch_size: int,
        augment: bool = True,
        rng: np.random.Generator | None = None,
    ) -> Dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        toks = self.toks[idx].astype(np.int64)
        feats = self.feats[idx].copy()
        actions = self.actions[idx].astype(np.int64)
        returns = self.returns[idx].copy()
        opp_hands = self.opp_hands[idx].copy()
        final_tricks = self.final_tricks[idx].astype(np.int64)
        is_play = self.is_play[idx].astype(np.bool_)

        if augment:
            import plump_engine

            toks, feats, opp_hands, actions = plump_engine.augment_batch_numpy(
                toks, feats, opp_hands, actions, is_play
            )

        return {
            "toks": toks,
            "feats": feats,
            "actions": actions,
            "returns": returns,
            "opp_hands": opp_hands,
            "final_tricks": final_tricks,
            "is_play": is_play,
        }
