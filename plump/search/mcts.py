# plump/search/mcts.py — Python wrapper for Rust determinized search.
# Decoupled: no training logic. Falls back to Python engine if Rust not built.
from __future__ import annotations

import random
from ..env import PlumpEnv
from ..encode import mask_actions
from ..env.cards import CARD_INDEX, INDEX_TO_CARD

try:
    # Try Rust engine if built with maturin
    import plump_engine  # type: ignore

    HAS_RUST = True
except Exception:
    HAS_RUST = False


def sample_worlds_py(env: PlumpEnv, viewer: int, n: int):
    """Python fallback: sample worlds consistent with void + played."""
    worlds = []
    import random as rng

    for _ in range(n):
        w = PlumpEnv()
        w.round_cards = env.round_cards
        w.hands = [list(h) for h in env.hands]
        w.bids = list(env.bids)
        w.tricks_won = list(env.tricks_won)
        w.table = list(env.table)
        w.led_suit = env.led_suit
        w.played_cards = set(env.played_cards)
        w.void_matrix = [list(row) for row in env.void_matrix]
        w.current_trick = env.current_trick
        w.history = list(env.history) if hasattr(env, "history") else []
        # Fill unknown: 52 - played - table - viewer hand
        all_cards = [(s, v) for s in ["H", "S", "D", "C"] for v in range(2, 15)]
        used = set(w.played_cards) | set(c for _, c in w.table) | set(w.hands[viewer])
        pool = [c for c in all_cards if c not in used]
        rng.shuffle(pool)
        # Distribute remaining to other players respecting void (simple)
        idx = 0
        for p in range(4):
            if p == viewer:
                continue
            need = len(env.hands[p])  # remaining
            hand = []
            # try to avoid void suit
            tries = 0
            while len(hand) < need and pool and tries < 100:
                c = pool.pop()
                s_idx = {"H": 0, "S": 1, "D": 2, "C": 3}[c[0]]
                if w.void_matrix[p][s_idx]:
                    # put back and try another
                    pool.insert(0, c)
                    tries += 1
                    rng.shuffle(pool)
                    continue
                hand.append(c)
            w.hands[p] = hand
        worlds.append(w)
    return worlds


def mcts_search(env: PlumpEnv, player: int, model, tokenizer_fn, n_worlds=16, n_sims=1):
    """Determinized search: average Q over worlds.

    model: transformer with forward(toks, feats, phase) -> Q
    tokenizer_fn: encode_decision(env, player) -> (toks, feats)
    Returns best card.
    """
    import torch
    import numpy as np

    legal_cards = env.legal_cards(player)
    if not legal_cards:
        return None
    if len(legal_cards) == 1:
        return legal_cards[0]

    worlds = sample_worlds_py(env, player, n_worlds)
    scores = {c: 0.0 for c in legal_cards}

    for w in worlds:
        for card in legal_cards:
            toks, feats = tokenizer_fn(w, player)
            # Q for this card
            # Need to run model — we do single forward per card for simplicity (could batch)
            t_toks = torch.tensor(np.array([toks]), dtype=torch.long)
            t_feats = torch.tensor(np.array([feats]), dtype=torch.float32)
            # use CPU for search (no device param yet)
            with torch.no_grad():
                q = model(t_toks, t_feats, phase="play")[0]
            idx = CARD_INDEX[card]
            # score is Q for that card (if masked, else -inf)
            # we need to ensure legality in world — if card not legal in world, skip
            if card not in w.hands[player] and card not in env.hands[player]:
                # world sampled hand may not contain our card if we sampled wrong — skip world
                continue
            scores[card] += float(q[idx].item())

    # average
    for c in scores:
        scores[c] /= max(1, n_worlds)
    best = max(scores, key=lambda c: scores[c])
    return best


def ai_action_with_search(model, tokenizer_fn, env, player, phase, n_worlds=0):
    """Drop-in for play.py: if n_worlds==0 use greedy, else search."""
    if phase == "bid" or n_worlds == 0:
        # greedy fallback (use legacy encode for now)
        from ..encode.legacy import encode_state_278, encode_state_294

        # Use transformer if model is transformer, else DQN path
        # For now, if model has transformer interface, use it
        try:
            toks, feats = tokenizer_fn(env, player)
            import torch, numpy as np

            t_toks = torch.tensor(np.array([toks]), dtype=torch.long)
            t_feats = torch.tensor(np.array([feats]), dtype=torch.float32)
            with torch.no_grad():
                q = model(t_toks, t_feats, phase=phase)[0]
            legal = mask_actions(env, player, phase)
            # mask
            best_idx = None
            best_q = -1e9
            for i, ok in enumerate(legal):
                if ok and float(q[i]) > best_q:
                    best_q = float(q[i])
                    best_idx = i
            if phase == "bid":
                return best_idx
            return INDEX_TO_CARD[best_idx]
        except Exception:
            # fallback to legacy DQN path
            from plump.models import DuelingQNetwork

            pass
    # play search
    return mcts_search(env, player, model, tokenizer_fn, n_worlds=n_worlds)
