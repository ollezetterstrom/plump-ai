import numpy as np
import pytest
import torch

from plump.config import PlumpTransformerConfig
from plump.models.transformer import PlumpTransformer

MAX_SEQ = 96
FEAT_DIM = 296


def make_batch(batch_size=4, lengths=(2, 6, 10, 24)):
    tokens = torch.zeros(batch_size, MAX_SEQ, dtype=torch.long)
    feats = torch.zeros(batch_size, FEAT_DIM, dtype=torch.float32)
    for row in range(batch_size):
        length = lengths[row % len(lengths)]
        tokens[row, 0] = 1
        tokens[row, 1] = 2 + 5
        for i in range(2, length):
            tokens[row, i] = 17 + (i * 7 + row) % 52
        feats[row, row % 52] = 1.0
        feats[row, 208 + (row * 5) % 52] = 1.0
    return tokens, feats


def test_config_defaults():
    cfg = PlumpTransformerConfig()
    assert cfg.vocab == 70
    assert cfg.max_seq == 96
    assert cfg.feat_dim == 296
    assert cfg.d_model == 128
    assert cfg.n_heads * cfg.qk_dim == cfg.d_model
    assert cfg.hand_hidden == 256
    assert cfg.q_hidden == 512


@pytest.mark.parametrize("phase,n_actions", [("bid", 11), ("play", 52)])
def test_forward_shapes(phase, n_actions):
    model = PlumpTransformer().eval()
    tokens, feats = make_batch()
    with torch.no_grad():
        q = model(tokens, feats, phase=phase)
    assert q.shape == (4, n_actions)
    assert torch.isfinite(q).all()


def test_forward_with_aux_shapes():
    model = PlumpTransformer().eval()
    tokens, feats = make_batch(batch_size=3, lengths=(2, 30, 90))
    with torch.no_grad():
        q_bid, belief, tricks = model.forward_with_aux(tokens, feats, "bid")
        q_play, belief_play, _ = model.forward_with_aux(tokens, feats, "play")
    assert q_bid.shape == (3, 11)
    assert q_play.shape == (3, 52)
    assert belief.shape == (3, 3, 52)
    assert belief_play.shape == (3, 3, 52)
    assert tricks.shape == (3, 11)
    assert torch.isfinite(belief).all()


def test_dynamic_sequence_slicing_and_single_observation():
    model = PlumpTransformer().eval()
    tokens, feats = make_batch(batch_size=2, lengths=(2, 5))
    with torch.no_grad():
        q = model(tokens, feats, phase="bid")
    assert q.shape == (2, 11)

    with torch.no_grad():
        q_single = model(tokens[0], feats[0], phase="play")
    assert q_single.shape == (1, 52)


def test_bfloat16_autocast():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PlumpTransformer().to(device).eval()
    tokens, feats = make_batch()
    tokens = tokens.to(device)
    feats = feats.to(device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        with torch.no_grad():
            q, belief, tricks = model.forward_with_aux(tokens, feats, "play")
    assert q.shape == (4, 52)
    assert belief.shape == (4, 3, 52)
    assert tricks.shape == (4, 11)
    assert torch.isfinite(q.float()).all()


def test_predict_belief_probs_shapes():
    model = PlumpTransformer().eval()
    tokens, feats = make_batch(batch_size=3)
    probs = model.predict_belief_probs(tokens, feats)
    assert probs.shape == (3, 3, 52)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    single = model.predict_belief_probs(tokens[0].numpy(), feats[0].numpy())
    assert single.shape == (3, 52)


def test_integration_with_rust_vec_env():
    plump_engine = pytest.importorskip("plump_engine")
    model = PlumpTransformer().eval()
    vec = plump_engine.PyRustVecEnv(4, seed=11)
    plump_engine.PyRustVecEnv  # keep the module referenced

    toks, feats, legal, is_bid, players = vec.observe_batch()
    assert toks.shape == (4, MAX_SEQ)
    assert feats.shape == (4, FEAT_DIM)
    assert legal.shape == (4, 52)
    assert is_bid.all()

    tokens = torch.from_numpy(toks).long()
    features = torch.from_numpy(feats).float()
    with torch.no_grad():
        q, belief, tricks = model.forward_with_aux(tokens, features, "bid")
    assert q.shape == (4, 11)
    assert belief.shape == (4, 3, 52)
    assert tricks.shape == (4, 11)

    actions = [int(np.flatnonzero(legal[i]).min()) for i in range(4)]
    out = vec.step_batch(actions)
    assert "toks" in out
    assert out["toks"].shape[1] == MAX_SEQ
    assert out["feats"].shape[1] == FEAT_DIM
    assert out["opp_hands"].shape[1] == 156


def test_augment_batch_numpy_roundtrip_values():
    plump_engine = pytest.importorskip("plump_engine")
    n = 16
    rng = np.random.default_rng(3)
    toks = np.zeros((n, MAX_SEQ), dtype=np.int64)
    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)
    opp = np.zeros((n, 156), dtype=np.float32)
    actions = rng.integers(0, 52, size=n, dtype=np.int64)
    is_play = np.ones(n, dtype=np.bool_)
    for row in range(n):
        toks[row, 0] = 1
        toks[row, 1] = 2 + 4
        toks[row, 2] = 13
        toks[row, 3] = 17 + int(actions[row])
        feats[row, 0] = 1.0
        opp[row, 52 + int(actions[row])] = 1.0
    a_toks, a_feats, a_opp, a_actions = plump_engine.augment_batch_numpy(
        toks, feats, opp, actions, is_play
    )
    assert a_toks.shape == toks.shape
    assert a_feats.shape == feats.shape
    assert a_opp.shape == opp.shape
    assert a_actions.shape == actions.shape
    assert np.all((a_actions >= 0) & (a_actions < 52))
    assert np.all(a_feats.sum(axis=1) == 1.0)
    assert np.all(a_opp.sum(axis=1) == 1.0)
    assert (a_toks[:, 1] == toks[:, 1]).all()
