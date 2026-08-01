# Differences from other Plump rule variations

Plump (Swedish "Plump") is a Nordic trick-taking game with many house rules. This project
implements a config-driven subset whose **defaults follow the most common Swedish form**. The
table below lists common variations and how this project handles them.

| Variation found in the wild | Common rule | This project |
|---|---|---|
| Making bonus | Some rules award `12 + bid` or more for making the bid | Default `10 + bid`, configurable via `make_bonus` |
| Made 0-bid | The most common rule gives a made 0-bid only **5 points** (the protocol writes "05"), because a 0-bid is the safest; some groups give the full 10 | Default `+5` (the "05" rule), configurable via `zero_bid_bonus` |
| Miss | Common rules score a miss as **0** ("a plump in the protocol"); some variants penalize, e.g. `-1` or `-3` per trick off | Default `0`, configurable via `miss_penalty` (set to 1 for `-|diff|`) |
| Minimum bid | Some rules require a minimum bid (e.g. 1) in all or some seats | Default `0`, configurable via `min_bid` |
| Total-bid constraint | The rule "sum of all bids may not equal the number of cards" is enforced differently across rulesets: dealer-only restriction, any-last-bidder, or not at all | Enforced **only on the dealer** (last bidder); forbidden bid `C - sum(others)`. Not yet configurable off |
| Dead cards ("skit") | Some variants reveal dead cards, or let players swap one of their cards with a dead card after the auction | Dead cards are never shown and never swapped; they exist only as hidden information for the belief model |
| Trump | A few regional variants add a trump suit | Never; standard Plump has no trump |
| Players / cards | Often `5 × 10` with 2 dead cards; also `4 × 13` (no dead stock) | Any `3..=6` players with any `C` such that `P*C <= 52` (default `5 × 10`) |
| Ladder schedule | Ladders may ascend (`1 → 10`) or descend (`10 → 1`) | Round-size schedule is modeled in the match features; both directions planned in the match simulator (Phase 9) |
| Scoring a bid of 0 | Most common: a made 0-bid scores 5 ("05"); some score 0 or the full 10 | Default 5; configurable via `zero_bid_bonus` |

The key design consequence: the project optimizes whatever variant it is configured for. The
engine's `ScoringConfig` drives the scoring atoms and the value head, so changing a house rule
only changes configuration, not the training pipeline.
