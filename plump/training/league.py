# plump/training/league.py — opponent pool, decoupled.
# Restart training includes old DQN as frozen sparring partner.

from __future__ import annotations

import random
import torch
from ..env import PlumpEnv
from ..encode import encode_state_278, encode_state_294, mask_actions
from ..env.cards import INDEX_TO_CARD
from ..models import DuelingQNetwork


class DQNOpponent:
    """Wrapper for frozen DQN champion to act as league member."""

    def __init__(self, bid_path: str, play_path: str, device="cpu"):
        self.device = torch.device(device)
        # Bid: 278->11
        sd = torch.load(bid_path, map_location=self.device)
        dim = sd["features.0.weight"].shape[1]
        self.bid = DuelingQNetwork(dim, 11).to(self.device)
        self.bid.load_state_dict(sd)
        self.bid.eval()
        # Play: may be 278 or 294
        sd = torch.load(play_path, map_location=self.device)
        dim = sd["features.0.weight"].shape[1]
        self.play = DuelingQNetwork(dim, 52).to(self.device)
        self.play.load_state_dict(sd)
        self.play.eval()
        # detect dims for encode choice
        self.play_dim = dim

    def act_bid(self, env: PlumpEnv, player: int) -> int:
        if self.bid is None:
            raise ValueError
        # DQN always uses 278
        from ..encode.legacy import encode_state_278

        s = encode_state_278(env, player)
        legal = mask_actions(env, player, "bid")
        lt = torch.tensor(legal, dtype=torch.bool, device=self.device)
        st = torch.tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.bid(st)
        q[~lt] = -1e9
        return int(q.argmax().item())

    def act_play(self, env: PlumpEnv, player: int):
        # choose encoder matching dim
        if self.play_dim == 294:
            from ..encode.legacy import encode_state_294

            s = encode_state_294(env, player)
        else:
            from ..encode.legacy import encode_state_278

            s = encode_state_278(env, player)
        legal = mask_actions(env, player, "play")
        lt = torch.tensor(legal, dtype=torch.bool, device=self.device)
        st = torch.tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q = self.play(st)
        q[~lt] = -1e9
        return INDEX_TO_CARD[int(q.argmax().item())]


class RandomOpponent:
    def act_bid(self, env, player):
        legal = mask_actions(env, player, "bid")
        cands = [i for i, v in enumerate(legal) if v]
        return random.choice(cands)

    def act_play(self, env, player):
        legal = env.legal_cards(player)
        return random.choice(legal)


class League:
    """Simple pool: sample opponent type per seat per game."""

    def __init__(self, dqn_paths: list[tuple[str, str]] | None = None):
        self.members: list[tuple[str, object]] = []
        # always have random
        self.members.append(("random", RandomOpponent()))
        if dqn_paths:
            for bid_p, play_p in dqn_paths:
                try:
                    opp = DQNOpponent(bid_p, play_p)
                    self.members.append((f"dqn:{bid_p}", opp))
                except Exception as e:
                    print(f"[league] skip {bid_p}: {e}")

    def sample(self) -> tuple[str, object]:
        # weight random 30%, rest uniform
        if random.random() < 0.3:
            return self.members[0]
        return random.choice(self.members[1:]) if len(self.members) > 1 else self.members[0]

    def sample_for_game(self) -> list[tuple[str, object]]:
        # 3 opponents for seats 1,2,3 (seat 0 is learner)
        return [self.sample() for _ in range(3)]
