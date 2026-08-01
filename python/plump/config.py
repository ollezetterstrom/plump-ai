# Plump training configuration.
#
# Defaults follow the Master Plan: B=16384, P=5, C=10, H=768, 1225 features.

from dataclasses import dataclass

# Fixed widths (contract with the Rust encoder, §3.6 / §5.2).
N_FEATURES = 1225
K_MAX = 128

# Belief head classes: relative seat 0..5, PLAYED, UNDEALT (§5.3).
N_BELIEF_CLASSES = 8


@dataclass
class ScoringConfig:
    make_bonus: int = 10
    miss_penalty: int = 0  # 0 = a miss scores 0 points (common rule)
    zero_bid_bonus: int = 5  # special case: a made 0-bid (the "05" rule)
    min_bid: int = 0
    max_bid: int = 10


@dataclass
class EnvConfig:
    n_players: int = 5
    n_cards: int = 10
    batch: int = 16384
    seed: int = 0
    duplicate_deals: int = 4  # replica deals for variance reduction (§4.1)


@dataclass
class NetConfig:
    n_features: int = N_FEATURES
    hidden: int = 768
    # Categorical value support (§4.2), derived from the scoring config at
    # construction; padded to at least this many atoms.
    min_atoms: int = 25
    n_belief_classes: int = N_BELIEF_CLASSES


@dataclass
class PpoConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    target_kl: float = 0.02
    epochs: int = 2
    minibatches: int = 8
    gamma: float = 1.0
    gae_lambda: float = 1.0  # unused (per-seat terminal MC, §4.1); kept for symmetry
    value_coef: float = 0.5
    trick_coef: float = 0.1
    belief_coef: float = 0.1
    consistency_coef: float = 0.1  # ||V - V_factored||² (§4.2)
    entropy_coef: float = 0.01
    entropy_floor: float = 5e-3
    grad_clip: float = 1.0


@dataclass
class OpponentPoolConfig:
    pool_size: int = 8
    heuristic_prob: float = 0.1
    noise_eps: float = 0.05
    temperature_lo: float = 0.8
    temperature_hi: float = 1.3
    self_play_prob: float = 0.5


@dataclass
class SearchConfig:
    n_worlds: int = 200
    swap_mcmc_steps: int = 200
    rollout_endgame_tricks: int = 3
    exit_interval_iters: int = 1000
    exit_n_states: int = 50_000


def default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
