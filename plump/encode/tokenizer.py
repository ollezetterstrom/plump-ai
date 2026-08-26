# plump/encode/tokenizer.py — v3 history tokenizer, decoupled.
# Inspired by FableDan/DanLM: raw play-by-play tokenization, no hand-crafted danger.
# Pure python, no torch, stable interface for transformer.

from __future__ import annotations

from ..env.cards import SUIT_INDEX, CARD_INDEX

# Vocab layout — small, extensible, mirrors FableDan 48 but for Plump.
# 0 PAD, 1 BOS, 2..12 BID 0..10, 13..16 PLAYER 0..3 (relative), 17..68 CARD (52), 69 TRICK_END
PAD_TOK = 0
BOS_TOK = 1
BID_BASE = 2          # 2..12 = bid 0..10
PLAYER_BASE = 13      # 13..16 = relative player 0..3
CARD_BASE = 17        # 17..68 = card idx 0..51
TRICK_TOK = 69
VOCAB = 70
MAX_SEQ = 256

# Hand/action feature dims — lightweight, per-decision, not history.
# For each decision we give hand one-hot (52) + a few scalars.
FEAT_DIM = 52 + 4 + 4 + 1 + 4  # hand 52 + bids 4 + tricks 4 + round 1 + led 4 = 65
# hand 52 one-hot, bids relative 4/10, tricks 4/10, round 1/10, led suit one-hot 4


def card_token(card: tuple[str, int]) -> int:
    return CARD_BASE + CARD_INDEX[card]


def bid_token(bid: int) -> int:
    return BID_BASE + bid


def player_token(rel: int) -> int:
    return PLAYER_BASE + rel


def tokenize_history(env, viewer: int) -> list[int]:
    """Build causal token stream for viewer relative.

    Order: BOS, round_cards, then history events in order.
    Uses env.history if present (bid/play), else fallback to bids/table.
    """
    toks: list[int] = [BOS_TOK, BID_BASE + env.round_cards if 0 <= env.round_cards <= 10 else PAD_TOK]

    # Prefer ordered history (new engine)
    if hasattr(env, "history") and env.history:
        for kind, p, payload in env.history:
            rel = (p - viewer) % 4
            toks.append(player_token(rel))
            if kind == "bid":
                toks.append(bid_token(int(payload)))  # type: ignore
            elif kind == "play":
                toks.append(card_token(payload))  # type: ignore
        # current unfinished trick's table is already in history via play events,
        # but if history lags (should not), add table again
        # No TRICK_TOK needed yet — FableDan uses it for trick boundaries, Plump can add later
    else:
        # Fallback for old checkpoints / tests without history
        for abs_p in range(4):
            rel = (abs_p - viewer) % 4
            b = env.bids[abs_p]
            if b == -1:
                continue
            toks.append(player_token(rel))
            toks.append(bid_token(b))
        for p, card in env.table:
            rel = (p - viewer) % 4
            toks.append(player_token(rel))
            toks.append(card_token(card))
        for card in sorted(env.played_cards):
            toks.append(card_token(card))

    # pad to MAX_SEQ like FableDan
    if len(toks) > MAX_SEQ:
        toks = toks[:2] + toks[-(MAX_SEQ - 2):]
    else:
        toks = toks + [PAD_TOK] * (MAX_SEQ - len(toks))
    return toks


def hand_features(env, player: int):
    """Simple per-decision float features, decoupled from history."""
    import numpy as np

    f = np.zeros(FEAT_DIM, dtype=np.float32)
    idx = 0
    # hand one-hot 52
    for card in env.hands[player]:
        f[CARD_INDEX[card]] = 1.0
    idx += 52
    # bids relative 4
    for i in range(4):
        b = env.bids[(player + i) % 4]
        f[idx + i] = (b / 10.0) if b != -1 else -0.1
    idx += 4
    # tricks relative 4
    for i in range(4):
        f[idx + i] = env.tricks_won[(player + i) % 4] / 10.0
    idx += 4
    f[idx] = env.round_cards / 10.0
    idx += 1
    # led suit one-hot 4
    if env.led_suit != "":
        f[idx + SUIT_INDEX[env.led_suit]] = 1.0
    return f


def encode_decision(env, player: int):
    """Interface for transformer: (tokens, feats) for current decision."""
    toks = tokenize_history(env, player)
    feats = hand_features(env, player)
    return toks, feats
