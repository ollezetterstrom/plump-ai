# Plump — rules as implemented

Plump is a trick-taking card game for 3–6 players (default 5). A **match** is a ladder of
**rounds**; each round is deal → bid → play → score. This document describes the rules exactly as
the engine implements them (defaults in parentheses). The authoritative definitions live in the
Rust engine: `crates/plump-engine/src/scoring.rs`, `legal.rs`, and the Python `config.py`.

## Deal

- A standard 52-card deck, canonical index `suit*13 + rank`, no jokers.
- Each of the `P` players is dealt `C` cards (default `P=5`, `C=10` → 50 cards dealt).
- The remaining `52 - P*C` cards are **dead stock**: not shown to any player, never played.
- All cards are dealt at once; the deal is the only source of hidden information.

## Bidding

- Players bid, in a fixed order, how many tricks they will win. The **dealer bids last**.
- Legal bids are `0..=C`.
- **Dealer constraint (only for the last bidder):** the dealer may not bid the value that would
  make the sum of all bids equal to `C`. Concretely, the forbidden bid is
  `C - (sum of the other players' bids)`; if that value falls outside `0..=C`, no bid is
  forbidden. This guarantees the total of all bids never equals the number of tricks available,
  so at least one player must miss their bid.
- The engine exposes this as an 11-bit mask per row (`legal_bids` in `legal.rs`); the computation
  is guarded against `u8` underflow.

## Play

- **No trump.** The winner of each trick is the highest card of the **led suit**.
- The first player of a trick is the winner of the previous trick; the player who leads first in
  a round is seat-determined.
- Every player must **follow suit**; a player with no card of the led suit (void) may play any
  card. The engine's `legal_plays` implements this exactly.
- Each player plays exactly `C` cards; a round has exactly `C` tricks.

## Scoring (default variant)

Given a bid `b` and a final trick count `t`:

| Condition | Score |
|---|---|
| `t == b` and `b > 0` (made the bid) | `+10 + b` |
| `t == b == 0` (made a 0-bid) | `+5` (special case, the "05" rule) |
| otherwise (miss) | `0` — a "plump", no negatives |

The made 0-bid is deliberately worth less than a made positive bid: a 0-bid is the safest
possible bid (you can always dump high cards), so awarding it the full `10 + b` would make it
too attractive. All scoring parameters are configurable (`ScoringConfig`): `make_bonus`
(default 10), `miss_penalty` (default 0; set to 1 for `-|t - b|` per trick off), `zero_bid_bonus`
(default 5, or `None` to fall back to `make_bonus + 0`), and `min_bid`/`max_bid` (default 0/10).
Because Plump's score is a step function of `|t - b|`, the training value function is
categorical over the finite set of achievable scores (`{0, 5, 11..20}` by default) rather than
a regression target.

## Round end & match

- A round ends after all `C` tricks; every seat is scored independently on `(bid, tricks)`.
- A match is a ladder of rounds (e.g. a descending schedule). The final standings are the
  sum of round scores (or a configured payout by rank) — see `ROADMAP.md` Phase 9 for the
  match-level driver.

## Configurable vs fixed

| Aspect | Default | Configurable |
|---|---|---|
| Players / cards | `P=5`, `C=10` | yes (`EnvConfig`) |
| Make bonus | `10 + bid` | yes (`make_bonus`) |
| Made 0-bid | `+5` (the "05" rule) | yes (`zero_bid_bonus`) |
| Miss | `0` (plump) | yes (`miss_penalty`; 1 = `-|diff|`) |
| Bid range | `0..=C` | yes (`min_bid`/`max_bid`) |
| Dealer forbidden bid | on | not yet (engine behavior) |
| Trump | none | fixed |
| Dead-card swap | never | fixed |
