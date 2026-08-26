# plump/training/rewards.py — pure functions, decoupled.
from ..config.settings import terminal_reward, display_points, HOUSE


def shaped_reward(is_winner: bool, tricks_now: int, bid: int) -> float:
    if is_winner:
        return 1.0 if tricks_now <= bid else -1.5
    else:
        return -1.0 if tricks_now < bid else 0.5


__all__ = ["shaped_reward", "terminal_reward", "display_points", "HOUSE"]
