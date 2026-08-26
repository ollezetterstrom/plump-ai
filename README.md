# Plump V2 — House Rules, Clean Architecture (Restart)

**House v2:** `0-bid = 5pts`, `highest bidder (earliest tie) leads`.

## Two training paths

- `python -u train.py` — **DQN compat** (old 278/294). Resumes `plump_*_v2.pt`, ~630k/5h. Keeps your saves. Fine-tune for house rules, but will plateau.
- `python -u train_transformer.py` — **Restart, correct long-term**: `Transformer + DMC + league` (FableDan/DanLM style). Starts from scratch, `league` includes `champion/best_v2.pt` as frozen sparring partners + random. ~`200k` budget, then keep running. This is the one that keeps getting smarter for weeks.

Both use `plump/env/engine.py:105` house rules and `plump/` decoupled structure.

## Structure (vibe-proof)

```
plump/
  env/         pure rules → engine.py, cards.py
  encode/      tokenizer.py (raw history) + legacy.py (278/294 compat)
  models/      transformer.py (FableDan 4-block) + dqn.py
  config/      settings.py/yaml single source
  training/    train_dmc.py (DMC+league), buffers.py, runner.py
plump_env.py / train.py / train_transformer.py — thin shims
docs/          ARCHITECTURE/HOUSE_RULES/TRAINING
```

**I chose restart Transformer for you:** old DQN stays in league so new brain learns to beat it, not from it. Run `train_transformer.py` tomorrow to begin long-term curve.

## Verify

```powershell
python -c "from plump.config import terminal_reward; print(terminal_reward(0,0))" # 5.0
python -c "from plump.env import PlumpEnv; e=PlumpEnv(); e.bids=[1,3,3,2]; print(e.get_leader())" # 1
python -c "from plump.models.transformer import PlumpTransformer; print(\"transformer ok\")"
```
