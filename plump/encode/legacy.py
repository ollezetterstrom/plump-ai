# plump/encode/legacy.py — legacy encoders, isolated from engine mutation.
# Do not add new features here; use tokenizer.py for v3.
from ..env.cards import CARD_INDEX, SUIT_INDEX
from ..env.engine import PlumpEnv


def encode_state_old(env: PlumpEnv, player: int):
    s = [0.0] * 222
    idx = 0
    if env.round_cards == 1:
        idx += 52
        for p in range(4):
            if p != player:
                for card in env.hands[p]:
                    s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
        idx += 52
    else:
        for card in env.hands[player]:
            s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
        idx += 104
    for p, card in env.table:
        s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
    idx += 52
    for card in env.played_cards:
        s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
    idx += 52
    for i in range(4):
        b = env.bids[(player + i) % 4]
        s[idx] = (b / 10.0) if b != -1 else -0.1
        idx += 1
    s[idx] = sum(b for b in env.bids if b != -1) / 40.0
    idx += 1
    for i in range(4):
        s[idx] = env.tricks_won[(player + i) % 4] / 10.0
        idx += 1
    s[idx] = env.current_trick / 10.0
    idx += 1
    if env.led_suit != "":
        s[idx + SUIT_INDEX[env.led_suit]] = 1.0
    idx += 4
    return s


def encode_state_226(env: PlumpEnv, player: int):
    s = [0.0] * 226
    idx = 0
    if env.round_cards == 1:
        idx += 52
        for p in range(4):
            if p != player:
                for card in env.hands[p]:
                    s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
        idx += 52
    else:
        for card in env.hands[player]:
            s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
        idx += 104
    for p, card in env.table:
        s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
    idx += 52
    for card in env.played_cards:
        s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
    idx += 52
    for i in range(4):
        b = env.bids[(player + i) % 4]
        s[idx] = (b / 10.0) if b != -1 else -0.1
        idx += 1
    s[idx] = sum(b for b in env.bids if b != -1) / 40.0
    idx += 1
    for i in range(4):
        s[idx] = env.tricks_won[(player + i) % 4] / 10.0
        idx += 1
    s[idx] = env.current_trick / 10.0
    idx += 1
    if env.led_suit != "":
        s[idx + SUIT_INDEX[env.led_suit]] = 1.0
    idx += 4
    position = len(env.table)
    for i in range(4):
        s[idx + i] = 1.0 if i == position else 0.0
    return s


def encode_state_278(env: PlumpEnv, player: int):
    s = [0.0] * 278
    idx = 0
    if env.round_cards == 1:
        idx += 52
        for p in range(4):
            if p != player:
                for card in env.hands[p]:
                    s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
        idx += 52
    else:
        for card in env.hands[player]:
            s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
        idx += 104
    for p, card in env.table:
        s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
    idx += 52
    for card in env.played_cards:
        s[idx + SUIT_INDEX[card[0]] * 13 + (card[1] - 2)] = 1.0
    idx += 52
    for i in range(4):
        b = env.bids[(player + i) % 4]
        s[idx] = (b / 10.0) if b != -1 else -0.1
        idx += 1
    s[idx] = sum(b for b in env.bids if b != -1) / 40.0
    idx += 1
    for i in range(4):
        s[idx] = env.tricks_won[(player + i) % 4] / 10.0
        idx += 1
    s[idx] = env.current_trick / 10.0
    idx += 1
    if env.led_suit != "":
        s[idx + SUIT_INDEX[env.led_suit]] = 1.0
    idx += 4
    position = len(env.table)
    for i in range(4):
        s[idx + i] = 1.0 if i == position else 0.0
    idx += 4
    table_cards = [c for _, c in env.table]
    for card in env.hands[player]:
        suit, val = card
        higher = sum(
            1 for tv in range(val + 1, 15)
            if (suit, tv) not in env.played_cards
            and (suit, tv) not in table_cards
            and (suit, tv) not in env.hands[player]
        )
        s[idx + SUIT_INDEX[suit] * 13 + (val - 2)] = higher / 12.0
    return s


def encode_state_294(env: PlumpEnv, player: int):
    """278 + 16 void bits = 294. Ordered relative to player."""
    s = encode_state_278(env, player)
    for i in range(4):
        target_p = (player + i) % 4
        for suit_idx in range(4):
            s.append(1.0 if env.void_matrix[target_p][suit_idx] else 0.0)
    return s


def mask_actions_old(env: PlumpEnv, player: int, phase: str):
    mask = [False] * 63
    if phase == "bid":
        for b in env.legal_bids(player):
            if 0 <= b <= 10:
                mask[52 + b] = True
    else:
        for card in env.legal_cards(player):
            mask[CARD_INDEX[card]] = True
    return mask


def mask_actions(env: PlumpEnv, player: int, phase: str):
    if phase == "bid":
        mask = [False] * 11
        for b in env.legal_bids(player):
            if 0 <= b <= 10:
                mask[b] = True
        return mask
    else:
        mask = [False] * 52
        for card in env.legal_cards(player):
            mask[CARD_INDEX[card]] = True
        return mask
