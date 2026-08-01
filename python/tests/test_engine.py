# Engine bridge tests (§10). Skipped when the extension isn't built
# (run `make py-build` first).

import pytest

from plump import config

plump_engine = pytest.importorskip("plump._engine")


def test_layout_matches_python_contract():
    rows = plump_engine.layout()
    names = [r[0] for r in rows]
    assert names == [
        "my_hand", "played_by_rel", "trick_cards", "led_suit", "my_voids",
        "other_voids", "unseen_by_suit", "bids_rel", "tricks_rel", "need_rel",
        "role_flags", "round_ctx", "match_ctx",
    ]
    # Contiguous offsets summing to N_FEATURES.
    off = 0
    for _, o, size in rows:
        assert o == off
        off += size
    assert off == plump_engine.n_features() == 1225


def test_permutation_table_shape_and_bijectivity():
    t = plump_engine.permutation_table()
    n = plump_engine.n_features()
    perms = plump_engine.n_permutations()
    assert perms == 24
    assert len(t) == perms * n
    for pi in range(perms):
        row = t[pi * n:(pi + 1) * n]
        assert sorted(row) == list(range(n))


def test_score_and_atoms():
    assert plump_engine.score(3, 3) == 13
    assert plump_engine.score(10, 10) == 20
    assert plump_engine.score(0, 0) == 5  # special case: the "05" rule
    assert plump_engine.score(3, 2) == 0  # miss: no negatives by default
    atoms = plump_engine.score_atoms()
    assert atoms == [0, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]


def test_atoms_match_python_net():
    # Parity: Rust atoms must equal the Python net's atom support, so the
    # value head can never drift from the engine's scoring (§5.2).
    from plump.net import _score_atoms  # noqa: PLC0415
    assert plump_engine.score_atoms() == _score_atoms(
        config.ScoringConfig(), min_atoms=0)


def test_legal_masks():
    hand = (1 << 12) | (1 << 13)  # A of suit 0, 2 of suit 1
    # Follow suit: only the A of suit 0.
    assert plump_engine.legal_plays(hand, 0) == (1 << 12)
    # Void in suit 3: whole hand.
    assert plump_engine.legal_plays(hand, 3) == hand
    assert plump_engine.legal_plays(hand, None) == hand
    # No underflow, dealer-only forbidden bid.
    assert plump_engine.legal_bids(10, 12, True) == 0x7FF
    assert plump_engine.legal_bids(10, 7, False) == 0x7FF
    assert plump_engine.legal_bids(10, 7, True) == 0x7FF & ~(1 << 3)
