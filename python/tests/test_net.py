# Network sanity tests (§5.5). Runs on CPU/MPS; no engine bridge required.

import pytest

torch = pytest.importorskip("torch")
from plump import PlumpNet, config  # noqa: E402


def test_forward_shapes():
    net = PlumpNet()
    B, K = 16, config.K_MAX
    obs = torch.full((B, K), config.N_FEATURES, dtype=torch.int64)  # all pad
    obs[0, 0] = 5
    obs[0, 1] = 700
    out = net(obs)
    assert out["bid"].shape == (B, 11)
    assert out["play"].shape == (B, 52)
    assert out["value"].shape == (B, len(net.score_atoms))
    assert out["tricks"].shape == (B, 11)
    assert out["belief"].shape == (B, 52 * net.n_belief_classes)


def test_embeddingbag_offsets():
    net = PlumpNet()
    off = net.offsets(4, "cpu")
    assert off.tolist() == [0, 128, 256, 384, 512]


def test_value_is_weighted_mean_of_atoms():
    net = PlumpNet()
    logits = torch.randn(3, len(net.score_atoms))
    v = net.value(logits)
    expected = torch.softmax(logits, -1) @ net.atoms
    assert torch.allclose(v, expected)


def test_score_table_buffer():
    net = PlumpNet()
    assert net.score_table[3, 3].item() == 13
    assert net.score_table[3, 2].item() == 0  # miss: no negatives by default
    assert net.score_table[0, 0].item() == 5  # made 0-bid: the "05" rule
    assert net.score_table[10, 10].item() == 20


def test_log_prob_masking():
    net = PlumpNet()
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    legal = torch.tensor([[True, False, False]])
    lp = net.log_prob(logits, torch.tensor([0]), legal)
    assert torch.isfinite(lp).all()


def test_config_defaults():
    assert config.EnvConfig().n_cards == 10
    assert config.EnvConfig().n_players == 5
    assert config.N_FEATURES == 1225
