# Rollout driver tests (§7). Skipped when the extension isn't built.

import time

import numpy as np
import pytest

from plump import config

plump_engine = pytest.importorskip("plump._engine")
RolloutDriver = pytest.importorskip("plump.rollout").RolloutDriver


def test_rollout_full_round_random_policy():
    B, P, C = 256, 4, 5
    drv = RolloutDriver(P, C, B, seed=1)
    assert drv.engine is not None
    D = drv.decisions_per_round
    assert D == P * (1 + C)
    rng = np.random.default_rng(7)

    rows = []
    for _ in range(D):
        idx, lengths = drv.obs()
        assert idx.shape == (B, config.K_MAX)
        assert idx.dtype == np.int16
        # rows shorter than K_MAX are padded with the no-op index
        n = int(lengths[0])
        assert n <= config.K_MAX
        assert np.all(idx[0, n:] == config.N_FEATURES)
        rows.append(int(lengths.sum()))
        drv.play_random(rng)

    assert drv.round_over
    assert drv.step_index == D
    # every game must have produced a non-empty row at every step
    assert all(r >= B for r in rows)


def test_rollout_scores_match_engine_scoring():
    B, P, C = 128, 3, 2
    drv = RolloutDriver(P, C, B, seed=3)
    rng = np.random.default_rng(11)
    for _ in range(drv.decisions_per_round):
        drv.play_random(rng)
    assert drv.round_over
    # random policy must have produced at least one made and one missed bid
    made = missed = 0
    for g in range(B):
        sc = drv.round_scores(g)
        assert len(sc) == P
        for s in sc:
            assert s in {0, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
            if s > 0:
                made += 1
            else:
                missed += 1
    assert made > 0 and missed > 0


def test_reset_deals_new_round():
    drv = RolloutDriver(4, 5, 64, seed=5)
    first = drv.obs()[0].copy()
    drv.reset(6)
    second = drv.obs()[0]
    assert drv.step_index == 0
    assert not np.array_equal(first, second)


def test_double_buffered_steps_match_single():
    """The pipelined driver must advance identically to direct engine calls."""
    B, P, C = 64, 3, 4
    a = RolloutDriver(P, C, B, seed=9)
    b = plump_engine.Rollout(P, C, B, 9)
    idx = np.zeros((B, config.K_MAX), dtype=np.int16)
    lengths = np.zeros(B, dtype=np.uint16)
    rng = np.random.default_rng(2)

    for _ in range(a.decisions_per_round):
        ai, al = a.obs()
        b.encode(idx, lengths)
        assert np.array_equal(ai, idx), "driver/engine observations diverge"
        actions = np.zeros(B, dtype=np.uint8)
        lb = np.zeros(B, dtype=np.uint16)
        lp = np.zeros(B, dtype=np.uint64)
        ac = np.zeros(B, dtype=np.uint8)
        ib = np.zeros(B, dtype=np.uint8)
        a.engine.snapshot(lb, lp, ac, ib)
        for g in range(B):
            if ib[g]:
                legal = [i for i in range(C + 1) if int(lb[g]) & (1 << i)]
            else:
                legal = [i for i in range(52) if int(lp[g]) & (1 << i)]
            actions[g] = legal[rng.integers(len(legal))]
        a.step(actions)
        b.step(actions)

    assert a.round_over
    for g in range(B):
        assert list(a.round_scores(g)) == list(b.round_scores(g))


def test_rollout_throughput_16384_games():
    """Milestone 3: full rollout of 16k games at >= 40 steps/s (P=5, C=10)."""
    B, P, C = 16_384, 5, 10
    drv = RolloutDriver(P, C, B, seed=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    n = 0
    for _ in range(drv.decisions_per_round):
        drv.obs()
        drv.play_random(rng)
        n += 1
    dt = time.perf_counter() - t0
    steps_s = n / dt
    assert drv.round_over
    # Machine-adjusted floor (plan targets 40 on a 16-core CUDA box).
    assert steps_s >= 40.0, f"rollout too slow: {steps_s:.1f} steps/s"
    print(f"\n[rollout] 16k games x {n} steps in {dt*1e3:.1f} ms = {steps_s:.0f} steps/s")
