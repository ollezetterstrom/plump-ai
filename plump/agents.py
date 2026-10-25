"""Small shared policy helpers for training, evaluation and play.py."""

from __future__ import annotations

import numpy as np


def masked_argmax(q: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Argmax over legal actions. `q` is [M, A], `legal` is [M, A] bool."""
    masked = np.where(legal, q, -1e9)
    return masked.argmax(axis=1)


def epsilon_greedy(
    q: np.ndarray,
    legal: np.ndarray,
    epsilons: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedy action per row, replacing a fraction `epsilons[i]` by random
    legal actions."""
    actions = masked_argmax(q, legal).astype(np.uint8)
    explore = rng.random(len(q)) < epsilons
    for i in np.flatnonzero(explore):
        options = np.flatnonzero(legal[i])
        if len(options) == 0:
            continue
        actions[i] = options[rng.integers(0, len(options))]
    return actions


def random_actions(
    legal: np.ndarray, indices: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    actions = np.zeros(len(indices), dtype=np.uint8)
    for j, i in enumerate(indices):
        options = np.flatnonzero(legal[i])
        if len(options):
            actions[j] = options[rng.integers(0, len(options))]
    return actions


def heuristic_bid(legal_row: np.ndarray, round_cards: int) -> int:
    """Bid near the expected per-seat trick share (round_cards / 4)."""
    options = np.flatnonzero(legal_row[:11])
    if len(options) == 0:
        return 0
    target = max(0, min(round_cards, round_cards // 4))
    return int(options[np.argmin(np.abs(options - target))])


def heuristic_play(legal_row: np.ndarray, bid: int, tricks: int) -> int:
    """Highest legal card when tricks are still needed, otherwise the lowest.

    The legal mask has already enforced follow-suit, so this is a reasonable
    baseline and a cheap rollout policy.
    """
    options = np.flatnonzero(legal_row[:52])
    if len(options) == 0:
        return 0
    ranks = options % 13
    if tricks < bid:
        return int(options[np.argmax(ranks)])
    return int(options[np.argmin(ranks)])
