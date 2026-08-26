# plump/encode/tokenizer.py — v3 stub for future transformer.
# Decoupled: pure tokenization, no torch required for encode.
# Inspired by FableDan/DanLM: history tokenization + hand features.
# Keep this file free of training logic. Full transformer will read tokens.

from ..env.cards import SUIT_INDEX


# Vocabulary: 52 cards + 4 players + bid tokens + special
VOCAB_SIZE = 64  # placeholder: 52 cards + 12 specials
PAD_TOK = 0
# Real ids: card idx 1..52, bids 53..63 etc. Stub mapping for now.

class PlumpTokenizer:
    """Minimal v3 tokenizer — decoupled from legacy.

    Future: tokenizes `who bid/played what` as causal sequence.
    Currently returns stub ids; interface is stable for transformer.
    """

    def __init__(self, max_len: int = 128):
        self.max_len = max_len

    def encode_history(self, env, player: int) -> list[int]:
        """Tokenize table + played + bids relative to player. Stub."""
        # TODO: full token stream like FableDan encode.py: history tokens 48 vocab
        # For now return card ids for compatibility checks.
        toks: list[int] = []
        # bids
        for i in range(4):
            b = env.bids[(player + i) % 4]
            toks.append(53 + b if b != -1 else PAD_TOK)
        # table
        for _, card in env.table:
            toks.append(1 + SUIT_INDEX[card[0]] * 13 + (card[1] - 2))
        # truncate/pad
        return (toks + [PAD_TOK] * self.max_len)[: self.max_len]

    def decode(self, ids: list[int]) -> list[str]:
        return [str(i) for i in ids]


# Alias for import stability
Tokenizer = PlumpTokenizer
