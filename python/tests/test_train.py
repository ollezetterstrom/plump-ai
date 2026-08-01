# Phase 4 trainer smoke tests (§4). Small config, CPU device, quick.

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from plump import config  # noqa: E402
from plump.train import PPOTrainer, compute_policy_loss, _select_logits  # noqa: E402


def _trainer(players=4, cards=5, batch=8):
    from plump.net import PlumpNet

    env = config.EnvConfig(
        n_players=players, n_cards=cards, batch=batch, seed=0, duplicate_deals=1
    )
    ppo = config.PpoConfig(epochs=1, minibatches=1)
    return PPOTrainer(PlumpNet(), env, ppo, "cpu")


def test_rollout_fills_buffers():
    tr = _trainer()
    tr.rollout()
    D, B = tr.rollout_len, tr.env_cfg.batch
    assert tr.obs_idx.shape == (D, B, config.K_MAX)
    assert tr.obs_idx.dtype == torch.int16
    assert tr.legal.shape == (D, B, 52)
    assert tr.ret.shape == (D, B)
    # every sampled action is legal and in range
    for t in range(D):
        for g in range(B):
            assert tr.legal[t, g, tr.action[t, g].item()] == 1
            if tr.is_bid[t, g].item():
                assert tr.action[t, g].item() <= tr.env_cfg.n_cards
    # per-seat terminal returns are achievable scores; atom index is valid
    assert (tr.atom_idx >= 0).all()
    assert (tr.atom_idx < len(tr.net.score_atoms)).all()


def test_belief_targets_all_cards_assigned():
    tr = _trainer()
    tr.rollout()
    tgt = tr.belief_tgt.reshape(-1, 52).numpy()
    assert set(np.unique(tgt)).issubset({0, 1, 2, 3, 6, 7})  # P=4 seats + played/undealt


def test_update_runs_and_kl_finite():
    tr = _trainer()
    tr.rollout()
    kl = tr.update()
    assert np.isfinite(kl)
    assert kl >= 0


def test_select_logits_disjoint_heads():
    from plump.net import PlumpNet

    net = PlumpNet()
    b = 4
    out = net(torch.full((b, config.K_MAX), config.N_FEATURES, dtype=torch.int64))
    is_bid = torch.tensor([True, False, True, False])
    logits = _select_logits(out, is_bid)
    assert logits.shape == (b, 52)
    # bid rows share the bid head's leading 11 slots
    assert torch.allclose(logits[0, :11], out["bid"][0])
    assert torch.allclose(logits[1], out["play"][1])


def test_evaluate_vs_baselines():
    tr = _trainer(batch=16)
    ev = tr.evaluate(n_games=32, seed=1, baseline="random")
    assert np.isfinite(ev["net"]) and np.isfinite(ev["baseline"])
    ev = tr.evaluate(n_games=32, seed=1, baseline="heuristic")
    assert np.isfinite(ev["net"]) and np.isfinite(ev["baseline"])


def test_compute_policy_loss_components():
    from plump.net import PlumpNet

    net = PlumpNet()
    cfg = config.PpoConfig()
    b, K = 16, config.K_MAX
    obs = torch.randint(0, config.N_FEATURES, (b, K), dtype=torch.int64)
    is_bid = torch.zeros(b, dtype=torch.bool)
    is_bid[::2] = True
    action = torch.randint(0, 52, (b,), dtype=torch.long)
    action[is_bid] = 3  # bids must index the (11, 11) score table
    legal = torch.ones(b, 52, dtype=torch.bool)
    old_logp = torch.full((b,), -1.0)
    ret = torch.full((b,), 13.0)
    atom_idx = torch.zeros(b, dtype=torch.long)
    trick_tgt = torch.randint(0, 11, (b,), dtype=torch.long)
    belief_tgt = torch.randint(0, net.n_belief_classes, (b, 52), dtype=torch.long)
    post_bid = is_bid.clone()
    out = net(obs)
    loss = compute_policy_loss(
        net, out, is_bid, action, legal, old_logp, ret, atom_idx,
        trick_tgt, belief_tgt, post_bid, action, cfg, cfg.entropy_coef,
    )
    assert torch.isfinite(loss.total)
    assert loss.value_consistency.item() >= 0.0
