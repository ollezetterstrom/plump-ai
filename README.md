# Plump AI

A strong Plump-playing agent trained on a single consumer GPU: **Rust game engine + PyTorch**.

Plump is a trick-taking card game. Each player bids how many tricks they will win; you score
`10 + bid` for making the bid exactly (a made 0-bid scores 5, the "05" rule) and a miss scores 0.
A ladder of rounds decides the match. The full master plan (motivations, architecture, and
failure modes) lives in [`original plan/plan.md`](original%20plan/plan.md).

## Design at a glance

- **Rust engine** — canonical card encoding (`suit*13+rank`), legal bid/play masks (dealer-only forbidden bid,
  no underflow), public-knowledge state, a compile-time-checked feature layout registry, and a structural
  suit-permutation table.
- **PyTorch net** — sparse `EmbeddingBag` features over fixed-width rows, disjoint bid/play heads, a categorical
  (distributional) value head, plus trick-count and belief auxiliary heads.
- **Training** — per-seat PPO with terminal Monte-Carlo returns, a frozen opponent pool, and Expert-Iteration
  search distilled back into the network.
- **Static shapes everywhere** — fixed-width batches, pre-allocated buffers, CUDA-graph-friendly.

## Layout

```
crates/plump-engine/   Rust game engine (cards, scoring, legal, layout, knowledge, encoder, rollout)
crates/plump-py/       PyO3 bridge -> plump._engine
python/plump/          Python package (config, net, rollout, train, heuristic)
python/tests/          Python tests (engine bridge + network + trainer)
original plan/         The master plan
```

## Setup & testing

Requires Rust, Python 3.10+, and PyTorch.

```
make setup      # create .venv (reuses system torch), install maturin + pytest
make py-build   # build the PyO3 bridge
make test       # run Rust + Python tests
```

`cargo test` runs the Rust suite without any Python; `make test-py` runs the Python suite.

## Training

The Phase 4 trainer rolls out `batch` games in lockstep through the Rust engine,
then applies a per-seat PPO update over the fixed-shape trajectory buffers:

```
.venv/bin/python -m plump.train --iters 100 --batch 4096 --players 4 --cards 5 --eval-games 1024
```

Every iteration plays a full round and prints rollout/update timing and the
approx-KL; after training it reports duplicate-deal evals against random and the
`heuristic.py` baseline. Set `--device mps` (or `cuda`) to train on the GPU; the
default is the best available backend.

## Status

Phase 0–3 (engine, encoder, rollout driver) are complete. Phase 4 — the network and
per-seat PPO trainer — is implemented and training; the exit criterion is to beat
random by > 8 pts/round and a hand-coded heuristic on `P=4, C=5`. See
[ROADMAP.md](ROADMAP.md).
