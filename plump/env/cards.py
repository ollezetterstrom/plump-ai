# plump/env/cards.py — pure constants, no deps
# Decoupled: any change here must not touch engine/encode.

SUIT_INDEX = {"H": 0, "S": 1, "D": 2, "C": 3}

CARD_INDEX: dict[tuple[str, int], int] = {}
for _si, _s in enumerate(["H", "S", "D", "C"]):
    for _v in range(2, 15):
        CARD_INDEX[(_s, _v)] = _si * 13 + (_v - 2)

INDEX_TO_CARD: dict[int, tuple[str, int]] = {idx: card for card, idx in CARD_INDEX.items()}

SUITS = ["H", "S", "D", "C"]
RANKS = list(range(2, 15))
