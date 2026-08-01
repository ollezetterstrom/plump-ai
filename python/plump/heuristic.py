# Hand-coded baseline for §9.1 duplicate-deal evaluation.
#
# Deterministic and fast, so it can run over tens of thousands of games on the
# CPU while the trained policy is evaluated on the exact same deals.

import numpy as np

from .rollout import RolloutDriver


def play_heuristic(n_players: int, n_cards: int, n_games: int, seed: int) -> float:
    """Bid the lowest legal bid (i.e. 0 whenever possible) and always play the
    lowest legal card (lowest of the led suit, or lowest overall when void).
    Returns the mean per-seat round score."""
    drv = RolloutDriver(n_players, n_cards, n_games, seed=seed)
    for _ in range(drv.decisions_per_round):
        legal = drv.legal_bool()
        is_bid = drv.is_bid != 0
        actions = np.zeros(n_games, dtype=np.uint8)
        for g in range(n_games):
            row = legal[g]
            if is_bid[g]:
                actions[g] = int(np.nonzero(row[: n_cards + 1])[0][0])
            else:
                actions[g] = int(np.nonzero(row)[0][0])
        drv.step(actions)
        drv.snapshot()
    return np.mean([drv.round_scores(g) for g in range(n_games)])
