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

## Play — now with deep search (Rust optional)
```powershell
python play.py                 # instant greedy
python play.py --search 16     # expert: 16 worlds, ~0.3s/move, Rust 10x if cargo build --release
python play.py --search 32     # stronger, ~0.6s
```
Search wrapper `plump/search/mcts.py:1` averages `Transformer` Q over worlds. Rust `crates/plump-engine/src/search.rs:1` is fallback to Python if not built.

`cargo check` already passes — run `cargo build --release` to enable Rust speed.

## Monitor
Look for `Ep 1000 | win 58%` lines. `Ctrl+C` saves `interrupted_v2.pt` / `plump_transformer_latest.pt`.

## After 5h
- `train_transformer.py` → keep running days/weeks, it keeps improving.
- `train.py` → copy `best_v2.pt` → `champion.pt` to make play.py use it.

Delete this file when done.
