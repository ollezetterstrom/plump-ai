# TOMORROW — 5h Run (Temporary)

Delete this file after tomorrow.

## What to run (pick ONE)

**Long-term correct (restart, recommended):**
```powershell
cd C:\Users\ozett\Documents\Projects\plumptrain
python -u train_transformer.py
```
- Starts new `Transformer + DMC + league` from scratch.
- League = `random` + your old `champion/best_v2.pt` as sparring partners (it learns to BEAT them).
- Saves `plump_transformer_best.pt` / `plump_transformer_latest.pt` every 1000 games.
- ~200k budget, interrupt with `Ctrl+C` anytime — resumes next run.

**Quick house-rules fine-tune (old DQN, keeps saves):**
```powershell
python -u train.py
```
- Resumes `plump_*_v2.pt` (630k/5h), just fixes `0=5` + `leader`. Will plateau eventually.

Both use house `0=5` + `highest earliest leads` — already in `plump/env/engine.py:105`.

## Play
```powershell
python play.py                 # human vs 3 AI champions (old DQN), house rules
```
Deep search (`plump/search/mcts.py`) is tested and works, but not wired into `play.py` yet on purpose: the new transformer (27% after 1000 eps) would lose to your old champion today. We integrate it once training catches up.

## Monitor / behave
- Look for `Ep  1000 | win ...%` every ~50s. Win starts ~25% (from-scratch) and should climb past 40-50% within the hour.
- The two `Flash/Mem Efficient attention ... experimental` warnings at startup are benign (AMD ROCm).
- If it ever prints `done restart DMC+Transformer+league...` before your 5h ends, just run the same command again — every 50s it saved `plump_transformer_latest.pt`, so a rerun loses almost nothing.
- `Ctrl+C` is always safe.

## After 5h
- Leave `plump_transformer_*.pt` where they are; keep running this path on future days — that is the long-term climb.
- Do NOT copy them over `champion.pt`: `play.py` still uses the DQN champion until the transformer beats it in eval (we'll swap deliberately when it does).

Delete this file when done.
