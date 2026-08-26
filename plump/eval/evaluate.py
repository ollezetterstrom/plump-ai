# plump/eval/evaluate.py — decoupled eval, no training deps.
from ..env.engine import PlumpEnv
from ..encode.legacy import encode_state_278, encode_state_294, mask_actions
from ..env.cards import INDEX_TO_CARD
from ..config.settings import terminal_reward
import torch, random


def evaluate_vs_random(bid_model, play_model, games=500):
    wins = 0
    for _ in range(games):
        env = PlumpEnv()
        env.new_round(random.randint(1, 10))
        # placeholder — real eval uses runner
        pass
    return wins / games
