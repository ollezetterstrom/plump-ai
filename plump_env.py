# plump_env.py — shim, single source is plump/env + plump/encode
# Vibe-proof: edits go to plump/env/engine.py and plump/encode/legacy.py
from plump.env.cards import CARD_INDEX, INDEX_TO_CARD, SUIT_INDEX, SUITS
from plump.env.engine import PlumpEnv
from plump.encode.legacy import (
    encode_state_old,
    encode_state_226,
    encode_state_278,
    encode_state_294,
    mask_actions,
    mask_actions_old,
)

__all__ = [
    "PlumpEnv",
    "CARD_INDEX",
    "INDEX_TO_CARD",
    "SUIT_INDEX",
    "SUITS",
    "encode_state_old",
    "encode_state_226",
    "encode_state_278",
    "encode_state_294",
    "mask_actions",
    "mask_actions_old",
]
