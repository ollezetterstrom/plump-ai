"""plump.encode — state → vector.

Legacy encoders (222/226/278/294) isolated here, no training logic.
Future tokenizer stub lives in tokenizer.py, decoupled from legacy.
"""

from .legacy import (
    encode_state_old,
    encode_state_226,
    encode_state_278,
    encode_state_294,
    mask_actions,
    mask_actions_old,
)

# New path (v3) — token stream for transformer. Stub now, full later.
try:
    from .tokenizer import Tokenizer, PlumpTokenizer  # noqa: F401
except Exception:
    pass

__all__ = [
    "encode_state_old",
    "encode_state_226",
    "encode_state_278",
    "encode_state_294",
    "mask_actions",
    "mask_actions_old",
]
