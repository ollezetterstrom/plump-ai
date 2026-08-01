# Plump AI

A strong Plump-playing agent trained on a single consumer GPU: **Rust game engine + PyTorch**.

Plump is a trick-taking card game. Each player bids how many tricks they will win; you score
`10 + bid` for making the bid exactly and `-|tricks - bid|` otherwise. A ladder of rounds decides
the match. The full master plan (motivations, architecture, and failure modes) lives in
[`original plan/plan.md`](original%20plan/plan.md).

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
crates/plump-engine/   Rust game engine (cards, scoring, legal, layout, knowledge, encoder)
crates/plump-py/       PyO3 bridge -> plump._engine
python/plump/          Python package (config, net, train)
python/tests/          Python tests (engine bridge + network)
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

## Status

Phase 0 (scaffolding) is complete; the round engine is next. See [ROADMAP.md](ROADMAP.md).
