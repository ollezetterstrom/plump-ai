import numpy as np
import pytest
import torch

from plump.config import TrainConfig
from plump.models.transformer import PlumpTransformer, PlumpTransformerConfig
from plump.training.trainer import PlumpTrainer
from plump_engine import PyFastEnv
import play


def test_table_card_feature_encoding_normal_rounds():
    """Verify feats[52..208] encodes opponent cards on table for round_cards >= 2."""
    env = PyFastEnv(seed=42)
    env.new_round(3, 0)
    # Bidding phase: everyone bids 0
    for _ in range(4):
        env.step_bid(0)

    assert env.phase == "Playing"
    leader = int(env.current_player)
    assert leader == 0

    # Before any card is played, table is empty: feats[52..208] must be all zeros
    _, feats0, _ = env.encode(0)
    assert np.all(feats0[52:208] == 0.0)

    # Player 0 leads a legal card
    mask0 = int(env.legal_cards_mask(0))
    card0 = (mask0 & -mask0).bit_length() - 1
    env.step_play(card0)

    # Now it is Player 1's turn
    assert int(env.current_player) == 1
    # From Player 1's perspective:
    # abs_p 0 is relative opponent: (0 + 4 - 1) % 4 - 1 = 2
    # So feats[52 + 2 * 52 + card0] == 1.0
    _, feats1, _ = env.encode(1)
    assert feats1[52 + 2 * 52 + card0] == 1.0
    assert np.sum(feats1[52:208]) == 1.0

    # Player 1 plays
    mask1 = int(env.legal_cards_mask(1))
    card1 = (mask1 & -mask1).bit_length() - 1
    env.step_play(card1)

    # Now it is Player 2's turn
    assert int(env.current_player) == 2
    # From Player 2's perspective:
    # abs_p 0: rel_opp = (0 + 4 - 2) % 4 - 1 = 1 -> feats[52 + 1 * 52 + card0]
    # abs_p 1: rel_opp = (1 + 4 - 2) % 4 - 1 = 2 -> feats[52 + 2 * 52 + card1]
    _, feats2, _ = env.encode(2)
    assert feats2[52 + 1 * 52 + card0] == 1.0
    assert feats2[52 + 2 * 52 + card1] == 1.0
    assert np.sum(feats2[52:208]) == 2.0


def test_forced_card_avoids_model_and_search():
    """When only 1 legal card remains, choose_card returns immediately."""
    env = PyFastEnv(seed=123)
    env.new_round(1, 0)
    for _ in range(4):
        env.step_bid(0)

    # In round 1, each player has exactly 1 card in hand!
    legal = play.mask_to_bool(int(env.legal_cards_mask(int(env.current_player))), 52)
    assert int(legal.sum()) == 1
    expected_card = int(np.flatnonzero(legal)[0])

    settings = play.PlaySettings(autoplay=False, use_search=True, worlds=16, model_available=True)
    # Even if model is None or dummy object without forward, it should return instantly
    card, decision = play.choose_card(env, model=None, settings=settings, device=torch.device("cpu"))
    assert card == expected_card
    assert "only legal card" in decision.method


def test_head_to_head_margin_evaluation_and_league_promotion(tmp_path):
    """Test margin calculation and promotion logic when league champions exist."""
    cfg = TrainConfig(
        num_envs=4,
        eval_envs=4,
        eval_batch_rounds=4,
        checkpoint_dir=tmp_path,
        device="cpu",
    )
    trainer = PlumpTrainer(cfg)

    # Evaluate returns margin = avg_score - opp_avg_score
    metrics = trainer.evaluate()
    assert "avg_score" in metrics
    assert "opp_avg_score" in metrics
    assert "margin" in metrics
    assert abs(metrics["margin"] - (metrics["avg_score"] - metrics["opp_avg_score"])) < 1e-5

    # Test initial promotion (no champion yet)
    assert len(trainer.league) == 0
    metrics_win = {
        "avg_score": 10.0,
        "opp_avg_score": 5.0,
        "margin": 5.0,
        "made_rate": 0.5,
        "avg_tricks": 2.0,
    }
    trainer.evaluate = lambda: metrics_win
    trainer._evaluate_and_checkpoint()
    assert len(trainer.league) == 1
    assert trainer.best_score == 10.0

    # With champion: margin <= 0.05 does NOT promote even if avg_score is high
    metrics_tie = {
        "avg_score": 12.0,
        "opp_avg_score": 12.0,
        "margin": 0.0,
        "made_rate": 0.6,
        "avg_tricks": 2.0,
    }
    trainer.evaluate = lambda: metrics_tie
    trainer._evaluate_and_checkpoint()
    assert len(trainer.league) == 1  # Not promoted!

    # With champion: margin > 0.05 promotes and adds champion!
    metrics_beat_champ = {
        "avg_score": 8.0,
        "opp_avg_score": 7.0,
        "margin": 1.0,
        "made_rate": 0.5,
        "avg_tricks": 2.0,
    }
    # Slight weight perturbation so hash is different
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.add_(torch.randn_like(p) * 0.01)
    trainer.evaluate = lambda: metrics_beat_champ
    trainer._evaluate_and_checkpoint()
    assert len(trainer.league) == 2
    assert trainer.best_score == 8.0
