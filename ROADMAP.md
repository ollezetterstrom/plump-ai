# Roadmap

Milestones track the phases in the [master plan](original%20plan/plan.md) (§11). Status: `done`, `next`, or `planned`.

| Phase | Milestone | Days | Exit criterion | Status |
|---|---|---|---|---|
| 0 | Scaffolding, card types, scoring config | 0 | `cargo test` green | done |
| 1 | Single-game engine, `PublicKnowledge`, legal masks | 1–2 | Property tests green; brute-force `P=3,C=2` tree enumerable | done |
| 2 | Layout registry, encoder, permutation table, SoA batch | 3–4 | Parity + invariance tests green; encode >= 2M states/s on 16 cores | done |
| 3 | PyO3 bridge, pinned buffers, double buffering | 5 | Full rollout of 16k games at >= 40 steps/s | done |
| 4 | Network, per-seat PPO, distributional value, aux heads | 6–8 | Beats random by > 8 pts/round; beats heuristic on `P=4,C=5` | next |
| 5 | All configs, per-deal suit relabeling, opponent pool (K=2), duplicate-deal baselining | 9–11 | Trick-head calibration ECE < 0.03; best-response gap measured | planned |
| 6 | Multi-lineage population, cross-play, exploitability probe | 12–13 | Cross-lineage score within 1 pt of self-play score | planned |
| 7 | Constrained world sampler, depth-limited PIMC, DD endgame, bid-by-policy-rollout | 14–16 | Search beats net by >= 2 sigma on duplicate deals, or is disabled | planned |
| 8 | Expert Iteration distillation | 17–18 | Distilled net >= search-blended net at 1/100 the inference cost | planned |
| 9 | Match simulator, `V_match`, Stage-B fine-tune, final eval | 19–21 | Measurable standings-dependent bid variance; match win rate up | planned |

## Current focus

**Phase 4 — network + PPO trainer.** Build the training loop on top of the Phase 3 rollout:
per-seat terminal returns (§4.1), the categorical value head with the trick-count
factorization (§4.2), the disjoint bid/play policy heads (§5.1), the belief head (§5.3),
and a minibatch update over the `[D, B, K_MAX]` trajectory tensors. Exit: beats random by
> 8 pts/round and a hand-coded heuristic on `P=4, C=5`.
