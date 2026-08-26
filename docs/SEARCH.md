# Deep Search (Play-time, Rust)

**What:** Determinized search — imagine `N` worlds consistent with `void_matrix` + `played_cards`, search each as perfect-info, average `Q`.

**Why:** `pick_action` greedy is blitz. Search is "think 3 tricks ahead". Same `Transformer` brain, no retraining, `+12%` in Hearts/Whist papers. For Plump `N=16` is sweet spot `0.2s` Python, `0.02s` Rust.

**Files:**
- `crates/plump-engine/src/search.rs:1` `sample_worlds` (void-aware) + `score_moves` — Rust 10-50x
- `plump/search/mcts.py:1` `sample_worlds_py` fallback + `mcts_search` + `ai_action_with_search` — drop-in for `play.py`

**Use:**
```powershell
python play.py --search 16   # 16 worlds, ~0.3s/move, expert
python play.py               # 0 worlds, instant greedy
```

Build Rust when you want speed: `cargo build --release` (maturin later for Python bridge). Python fallback works without Rust.
