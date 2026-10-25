"""Configuration for the Plump Transformer and the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Token vocabulary (must match crates/plump-engine/src/engine.rs).
VOCAB = 70
MAX_SEQ = 96
FEAT_DIM = 296
PAD_TOK = 0
BOS_TOK = 1
BID_BASE = 2
PLAYER_BASE = 13
CARD_BASE = 17

NUM_PLAYERS = 4
NUM_CARDS = 52
ROUND_SCHEDULE = (10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LATEST_CHECKPOINT = PROJECT_ROOT / "plump_transformer_latest.pt"
BEST_CHECKPOINT = PROJECT_ROOT / "plump_transformer_best.pt"
SESSION_FILE = PROJECT_ROOT / "plump_transformer_session.txt"


@dataclass
class PlumpTransformerConfig:
    vocab: int = VOCAB
    max_seq: int = MAX_SEQ
    feat_dim: int = FEAT_DIM
    d_model: int = 128
    n_blocks: int = 4
    n_heads: int = 4
    qk_dim: int = 32
    v_dim: int = 32
    ffn_hidden: int = 512
    hand_hidden: int = 256
    q_hidden: int = 512
    n_bid_actions: int = 11
    n_play_actions: int = 52
    n_tricks: int = 11
    dropout: float = 0.0

    def as_dict(self) -> dict:
        return {
            "vocab": self.vocab,
            "max_seq": self.max_seq,
            "feat_dim": self.feat_dim,
            "d_model": self.d_model,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "qk_dim": self.qk_dim,
            "v_dim": self.v_dim,
            "ffn_hidden": self.ffn_hidden,
            "hand_hidden": self.hand_hidden,
            "q_hidden": self.q_hidden,
            "n_bid_actions": self.n_bid_actions,
            "n_play_actions": self.n_play_actions,
            "n_tricks": self.n_tricks,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, values: dict) -> "PlumpTransformerConfig":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in values.items() if k in allowed})


@dataclass
class TrainConfig:
    num_envs: int = 256
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 1e-4
    replay_play: int = 250_000
    replay_bid: int = 100_000
    eval_every: int = 5_000
    eval_rounds: int = 400
    eval_envs: int = 20
    eval_batch_rounds: int = 100
    league_fraction: float = 0.5
    max_champions: int = 5
    epsilon_min: float = 0.02
    epsilon_max: float = 0.25
    belief_loss_weight: float = 0.15
    tricks_loss_weight: float = 0.10
    shaping_weight: float = 0.20
    grad_clip: float = 1.0
    checkpoint_every_evals: int = 1
    use_amp: bool = True
    device: str = "auto"
    seed: int = 0
    checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT)

    @property
    def latest_path(self) -> Path:
        return self.checkpoint_dir / "plump_transformer_latest.pt"

    @property
    def best_path(self) -> Path:
        return self.checkpoint_dir / "plump_transformer_best.pt"

    @property
    def session_path(self) -> Path:
        return self.checkpoint_dir / "plump_transformer_session.txt"
