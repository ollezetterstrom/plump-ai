# Architecture

See `plump/`:

- `plump/env/engine.py` — pure Plump rules, no torch. `PlumpEnv`, `get_leader()` (earliest highest), `void_matrix`, `legal_bids` (no-sum). Edit here for rule changes.
- `plump/env/cards.py` — `CARD_INDEX`, `SUIT_INDEX`, deck constants.
- `plump/encode/legacy.py` — `encode_state_278/294` (222/226 legacy kept), `mask_actions`. Do not extend — use `tokenizer.py`.
- `plump/encode/tokenizer.py` — stub for FableDan-style raw history tokens, future transformer path.
- `plump/models/dqn.py` — `DuelingQNetwork` compat for legacy .pt.
- `plump/models/transformer.py` — `PlumpTransformer` stub (Llama 4-block RoPE), ready to swap in training.
- `plump/config/settings.py` — single source `HOUSE` (0=5), `TRAIN`. No scattered magic.
- `plump/training/` — `buffers.py`, `rewards.py`, `runner.py` (decoupled game runner).
- `plump/eval/` / `plump/play/` — evaluation / rich CLI.

Shims: `plump_env.py`, `train.py` re-export from `plump/*` for backward compat. Edit `plump/` only.

Long-term: replace DQN with DMC + NTP/belief + league + batched actors (see FableDan/DanLM). Rust optional for MCTS scale.
