# House Rules v2

- Players: 4
- Deck: 52, ranks 2–A
- Rounds: random 1–10 in training, 10→1 in `play.py`
- Bidding: 0..round_cards, last bidder cannot make sum == round_cards (`plump/env/engine.py:legal_bids`)
- Starter: **earliest highest bidder** `PlumpEnv.get_leader()` (not dealer rotation)
- Play: must follow suit `is_legal`, void tracking `void_matrix`, no trump
- Scoring: **0 bid made = 5 pts** `HOUSE.zero_bid_points`, otherwise `10+bid`; miss = 0 display, -2*diff train `terminal_reward`
- Verify: `python -c "from plump.config import terminal_reward; print(terminal_reward(0,0))"` → 5.0
