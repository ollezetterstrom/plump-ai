# Plump V2 — House Rules, Clean Architecture

**House v2:** `0-bid = 5pts` (not 10), `highest bidder (earliest tie) leads`.
Tested on `RX 9060 XT` — `35 eps/s → ~630k /5h`.

## Structure (vibe-proof, decoupled)

```
plump/
  env/         pure rules, no torch  → engine.py (PlumpEnv), cards.py
  encode/      legacy 278/294 + tokenizer.py (future transformer)
  models/      dqn.py (compat) + transformer.py (stub, FableDan-style)
  config/      settings.py + settings.yaml (single source)
  training/    buffers.py / rewards.py / runner.py
  eval/        evaluate.py
  play/        cli (rich)
plump_env.py / train.py — thin shims re-exporting plump/* for compat
tests/test_house.py — house rules
```

**Rule:** Edit only `plump/` modules. Shims never contain logic.

## Quick start (tomorrow 5h)

```powershell
python -u train.py          # house rules, resumes latest_v2.pt, ~630k in 5h
# Ctrl+C saves interrupted_v2.pt
python play.py              # human vs AI, uses new 5pt/leader
```

Legacy checkpoints (`best_v2.pt` 278/294) still load — warm-start.

## Long-term (from research: DouZero DMC + FableDan Transformer + ScrofaZero MCTS)

- **Current:** `Dueling DQN` DMC would be next — wait round end, `MSE(Q, return)`, no bootstrap.
- **Next:** `plump/models/transformer.py` is stub `128d 4-block RoPE` ready to swap; `tokenizer.py` raw history `vocab 64`. Add `NTP + belief` aux heads (`FableDan` +0.02/0.05).
- **Scale:** `plump/training/runner.py` is single-thread. For `M` hours continuous improvement add `train_fast.py` style `24 actors + batched GPU infer + league` (see `C:\Users\ozett\AppData\Local\Temp\opencode\FableDan`). No Rust needed until MCTS `10k sims` — then port `plump/env/engine.py` to Rust `pyo3` for `10-50x`.

## Config

Edit `plump/config/settings.yaml` or `settings.py:TRAIN/HOUSE`. No scattered magic numbers.

## Verify

```powershell
python -c "from plump.config import terminal_reward; print(terminal_reward(0,0))" # 5.0
python -c "from plump.env import PlumpEnv; e=PlumpEnv(); e.bids=[1,3,3,2]; print(e.get_leader())" # 1
```
