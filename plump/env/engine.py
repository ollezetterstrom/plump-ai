# plump/env/engine.py — pure game engine, no torch, no encode.
# All house rules live here. No training logic.
from __future__ import annotations

import random
from .cards import SUIT_INDEX, SUITS


class PlumpEnv:
    """4-player trick-taking engine.

    House v2 rules:
    - 0-bid success = 5 pts (not 10) — scored outside, but noted here.
    - Highest bidder (earliest on tie) leads first trick.
    - No trump. Must follow suit. Void tracking.
    """

    SUITS = SUITS

    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.hands: list[list[tuple[str, int]]] = []
        self.bids: list[int] = []
        self.tricks_won: list[int] = []
        self.table: list[tuple[int, tuple[str, int]]] = []
        self.led_suit: str = ""
        self.round_cards: int = 0
        self.current_trick: int = 0
        self.played_cards: set[tuple[str, int]] = set()
        # void_matrix[player][suit_idx] = True once proven void
        self.void_matrix: list[list[bool]] = [[False] * 4 for _ in range(self.num_players)]
        # history for tokenizer — ordered events (kind, player, payload)
        # kind: "bid" -> payload int, "play" -> payload card tuple
        self.history: list[tuple[str, int, object]] = []

    # -- setup --
    def new_round(self, round_cards: int = 10) -> None:
        self.round_cards = round_cards
        self.current_trick = 0
        self.table = []
        self.led_suit = ""
        self.played_cards = set()
        self.hands = [[] for _ in range(self.num_players)]
        self.bids = [-1] * self.num_players
        self.tricks_won = [0] * self.num_players
        self.void_matrix = [[False] * 4 for _ in range(self.num_players)]
        self.history = []

        deck = [(s, v) for s in self.SUITS for v in range(2, 15)]
        random.shuffle(deck)
        for i in range(round_cards * self.num_players):
            self.hands[i % self.num_players].append(deck[i])

    # -- queries --
    def can_follow_suit(self, player: int) -> bool:
        if self.led_suit == "":
            return False
        return any(c[0] == self.led_suit for c in self.hands[player])

    def is_legal(self, player: int, card: tuple[str, int]) -> bool:
        if card not in self.hands[player]:
            return False
        if self.led_suit == "" or card[0] == self.led_suit:
            return True
        return not self.can_follow_suit(player)

    def legal_cards(self, player: int) -> list[tuple[str, int]]:
        return [c for c in self.hands[player] if self.is_legal(player, c)]

    def legal_bids(self, player: int) -> list[int]:
        bids_made = sum(1 for b in self.bids if b != -1)
        is_last = bids_made == self.num_players - 1
        bid_sum = sum(b for b in self.bids if b != -1)
        forbidden = self.round_cards - bid_sum
        result: list[int] = []
        for b in range(0, self.round_cards + 1):
            if is_last and b == forbidden:
                continue
            result.append(b)
        return result

    def get_leader(self) -> int:
        """Earliest highest bidder leads. Fallback 0 if bids incomplete."""
        if any(b == -1 for b in self.bids):
            return 0
        max_bid = max(self.bids)
        for i, b in enumerate(self.bids):
            if b == max_bid:
                return i
        return 0

    # -- helpers to record history (decoupled from logic) --
    def record_bid(self, player: int, bid: int) -> None:
        self.history.append(("bid", player, bid))

    # -- mutations --
    def play_card(self, player: int, card: tuple[str, int]) -> None:
        if self.led_suit == "":
            self.led_suit = card[0]
        elif card[0] != self.led_suit:
            self.void_matrix[player][SUIT_INDEX[self.led_suit]] = True
        self.hands[player].remove(card)
        self.table.append((player, card))
        self.history.append(("play", player, card))

    def resolve_trick(self) -> tuple[int, list[tuple[int, tuple[str, int]]]]:
        if not self.table:
            raise ValueError("resolve_trick called with empty table")
        led_suit = self.table[0][1][0]
        winner = self.table[0][0]
        best_val = self.table[0][1][1]
        for p, c in self.table[1:]:
            if c[0] == led_suit and c[1] > best_val:
                best_val = c[1]
                winner = p
        self.tricks_won[winner] += 1
        result = (winner, list(self.table))
        for _, card in self.table:
            self.played_cards.add(card)
        self.table = []
        self.led_suit = ""
        self.current_trick += 1
        return result
