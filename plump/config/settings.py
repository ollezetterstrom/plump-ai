# plump/config/settings.py — single source of truth, decoupled.
from __future__ import annotations

from dataclasses import dataclass, asdict
import yaml
from pathlib import Path


@dataclass(frozen=True)
class HouseRules:
    """House v2 — change here, not scattered."""

    zero_bid_points: int = 5  # 0==0 gives 5, not 10
    leader_earliest_highest: bool = True
    forbidden_no_sum: bool = True
    num_players: int = 4


@dataclass
class TrainConfig:
    episodes: int = 1_000_000
    batch_size: int = 512
    gamma: float = 0.99
    tau: float = 0.005
    learn_every: int = 8
    cpu_sync_every: int = 4
    eval_every: int = 1_000
    eval_games: int = 500
    lr: float = 2.5e-4
    epsilon_start: float = 0.35
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.99985
    epsilon_restart_every: int = 25_000
    epsilon_restart_value: float = 0.20
    train_random_chance: float = 0.07
    bid_warmup: int = 5_000
    bid_buffer: int = 100_000
    play_buffer: int = 200_000


HOUSE = HouseRules()
TRAIN = TrainConfig()


def terminal_reward(tricks: int, bid: int, house: HouseRules = HOUSE) -> float:
    if tricks == bid:
        if bid == 0:
            return float(house.zero_bid_points)
        return float(10 + tricks)
    return -abs(tricks - bid) * 2.0


def display_points(tricks: int, bid: int, house: HouseRules = HOUSE) -> int:
    if tricks == bid:
        if bid == 0:
            return house.zero_bid_points
        return 10 + tricks
    return 0


def load_config(path: str | Path) -> TrainConfig:
    data = yaml.safe_load(Path(path).read_text())
    return TrainConfig(**data)


def save_config(cfg: TrainConfig, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(asdict(cfg)))
