# Lockstep rollout driver (§7).
#
# Wraps the Rust `Rollout` and owns the fixed-width buffers Python reads each
# step: observations, legal masks, actor seats, and phase flags. Buffers are
# allocated once (optionally pinned for H2D) and reused; Rust writes into them
# in place, so there is no per-step host allocation or copy.
#
# Double buffering (§3.6, §7.2): the engine encodes the *next* observation
# into the alternate buffer on a background thread (the Rust call releases the
# GIL) while the caller consumes the current buffer — overlapping encode with
# the network forward/H2D of the current step.

from __future__ import annotations

import threading
from typing import Tuple

import numpy as np

from . import _engine, config

# per-step snapshot arrays (plain numpy; legal masks are bitmasks, §5.2)
_DTYPE_BID = np.uint16
_DTYPE_PLAY = np.uint64
_DTYPE_SEAT = np.uint8


class RolloutDriver:
    """Drives `batch` games in lockstep and serves the per-step tensors."""

    def __init__(self, n_players: int, n_cards: int, batch: int, seed: int = 0,
                 pin: bool = False):
        if n_players * n_cards > 52:
            raise ValueError("deck too small for n_players * n_cards")
        self.engine = _engine.Rollout(n_players, n_cards, batch, seed)
        self.n_players = n_players
        self.n_cards = n_cards
        self.batch = batch
        self.k_max = config.K_MAX
        self.n_features = config.N_FEATURES

        # two buffer pairs (double buffering)
        self.idx = [np.zeros((batch, self.k_max), dtype=np.int16),
                    np.zeros((batch, self.k_max), dtype=np.int16)]
        self.len = [np.zeros(batch, dtype=np.uint16),
                    np.zeros(batch, dtype=np.uint16)]
        self.legal_bid = np.zeros(batch, dtype=_DTYPE_BID)
        self.legal_play = np.zeros(batch, dtype=_DTYPE_PLAY)
        self.actor = np.zeros(batch, dtype=_DTYPE_SEAT)
        self.is_bid = np.zeros(batch, dtype=_DTYPE_SEAT)

        self.cur = 0
        self._thread: threading.Thread | None = None
        self._err: BaseException | None = None

        # initial snapshot
        self.engine.encode(self.idx[0], self.len[0])
        self.snapshot()

    # -- buffering -----------------------------------------------------------

    def _join(self):
        t, self._thread = self._thread, None
        if t is not None:
            t.join()
        if self._err is not None:
            err, self._err = self._err, None
            raise err

    def _encode_bg(self, buf: int):
        try:
            self.engine.encode(self.idx[buf], self.len[buf])
        except BaseException as e:  # surfaced on the next obs()/step()
            self._err = e

    def obs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Waits for the background encode and returns (idx, len) for the
        current decision point."""
        self._join()
        return self.idx[self.cur], self.len[self.cur]

    def step(self, actions):
        """Applies `actions` (length B), advancing the games by one decision,
        and starts encoding the next snapshot into the alternate buffer."""
        self._join()
        a = np.ascontiguousarray(actions, dtype=_DTYPE_SEAT)
        if a.shape != (self.batch,):
            raise ValueError(f"actions must be ({self.batch},), got {a.shape}")
        self.engine.step(a)
        self._err = None
        nxt = 1 - self.cur
        self.cur = nxt
        self._thread = threading.Thread(target=self._encode_bg, args=(nxt,), daemon=True)
        self._thread.start()

    # -- snapshots -----------------------------------------------------------

    def snapshot(self):
        """Recomputes legal masks / actor / phase flags for the current state.
        Cheap; called synchronously after `step`."""
        self.engine.snapshot(self.legal_bid, self.legal_play, self.actor, self.is_bid)

    def masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(legal_bid u16, legal_play u64, is_bid u8) for the current state."""
        return self.legal_bid, self.legal_play, self.is_bid

    def legal_bool(self) -> np.ndarray:
        """`[B, 52]` bool legal-action mask (§3.4): bid bits in 0..=C, else the
        52-bit play mask."""
        bid, play, is_bid = self.masks()
        bid_bits = self._unpack_bits(bid, self.n_cards + 1)
        play_bits = self._unpack_bits(play, 52)
        legal = play_bits.copy()
        legal[:, : self.n_cards + 1] = np.where(
            (is_bid != 0)[:, None], bid_bits, legal[:, : self.n_cards + 1]
        )
        return legal

    def belief_targets(self) -> np.ndarray:
        """`[B, 52]` u8 ground-truth belief targets (§5.3), actor-relative."""
        out = np.zeros(self.batch * 52, dtype=np.uint8)
        self.engine.belief_targets(out)
        return out.reshape(self.batch, 52)

    def reset(self, seed: int):
        self._join()
        self.engine.reset(seed)
        self.engine.encode(self.idx[0], self.len[0])
        self.cur = 0
        self.snapshot()

    # -- round bookkeeping ---------------------------------------------------

    @property
    def round_over(self) -> bool:
        return self.engine.round_over

    @property
    def step_index(self) -> int:
        return self.engine.step_index

    @property
    def decisions_per_round(self) -> int:
        return self.engine.decisions_per_round

    def round_scores(self, g: int):
        return self.engine.round_scores(g)

    def tricks(self, g: int):
        return self.engine.tricks(g)

    def play_random(self, rng: np.random.Generator | None = None) -> None:
        """Samples uniformly over legal actions for every game and steps once."""
        rng = rng or np.random.default_rng()
        actions = self.sample_actions(rng)
        self.step(actions)
        self.snapshot()

    def sample_actions(self, rng: np.random.Generator) -> np.ndarray:
        """Vectorized uniform sampling over legal actions (§3.4): bit-unpack the
        per-row masks into `[B, 11]`/`[B, 52]` bool tensors, then pick one legal
        index per row with a cumsum trick — no Python loop over games."""
        bid, play, is_bid = self.masks()
        bid_bits = self._unpack_bits(bid, self.n_cards + 1)
        play_bits = self._unpack_bits(play, 52)
        legal = play_bits.copy()
        legal[:, : self.n_cards + 1] = np.where(
            (is_bid != 0)[:, None], bid_bits, legal[:, : self.n_cards + 1]
        )
        csum = np.cumsum(legal, axis=1)
        counts = csum[:, -1].astype(np.float64)
        r = rng.random(self.batch) * counts
        return np.argmax(csum >= r[:, None], axis=1).astype(np.uint8)

    @staticmethod
    def _unpack_bits(masks: np.ndarray, width: int) -> np.ndarray:
        """`(B,)` uintN -> `(B, width)` bool, little-endian bit order."""
        nbytes = masks.dtype.itemsize
        bits = np.unpackbits(
            masks.view(np.uint8).reshape(-1, nbytes), axis=1, bitorder="little"
        )
        return bits[:, :width].astype(bool)
