# TOMORROW — 5h Run (Temporary)

Delete this file after tomorrow.

## Before you leave: REBOOT once (recommended)

Some half-killed GPU test processes from my testing are stuck "terminating" and only a
reboot reliably clears them. 30 seconds now = clean GPU for the next 5 hours.

## Then run (ONE command)

```powershell
cd C:\Users\ozett\Documents\Projects\plumptrain
.\start_overnight.ps1
```

That starts training detached (survives closing the terminal) and prints the log file name.
Close the lid / walk away. It saves progress every ~50 seconds.

## What it does

`train_transformer.py` — new brain (`Transformer + DMC + league`) trained against your old
champion bots + randomness. **Fully resumable**: weights, replay buffers, and episode
counter all persist — Ctrl+C, crash, or stop → rerun the same script, it continues exactly.

- Checkpoints: `plump_transformer_latest.pt` (every eval), `plump_transformer_best.pt`
  (whenever average score improves)
- Progress log lines look like:
  `Ep   6000 | win  34.5% avg +1.84 | buf ...`
- Two `AMD ROCm attention experimental` warnings at startup are normal noise.

## Playing while/after training

```powershell
python play.py               # old champion bots, instant
python play.py --search 12   # same bots + deep search (imagines 12 hidden-hands per move) — still instant (~10ms/move), usually stronger
```

House rules are active in both paths: `0-bid made = 5 pts`, highest bidder (earliest tie) leads.

## Honest status — done vs not-done

DONE & TESTED TODAY
- Both training paths run for hours, checkpoint + resume correctly
- House rules in engine + scoring, verified by tests
- play.py `--search N` deep search wired and tested (10ms at 12 worlds)
- Rust engine compiles & unit tests pass (not yet bridged to Python — see below)

NOT DONE YET (deliberate, next sessions)
- Rust-to-Python bridge (maturin): search currently uses the Python world-sampler;
  at 10ms/move that is irrelevant for playing, matters only for heavy research later
- NTP/belief auxiliary heads exist in the model but aren't in the loss yet (+few % when added)
- Training is single-process; FableDan-style multi-actor batching would ~10x throughput
- No Elo/auto-promotion between checkpoints yet

None of these block tonight. The run collects experience regardless.

Delete this file when done.
