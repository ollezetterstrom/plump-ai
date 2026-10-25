"""Entry point for Plump Transformer self-play training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from plump.config import TrainConfig
from plump.training.trainer import PlumpTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Plump Transformer")
    parser.add_argument("--rounds", type=int, default=800_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--league-fraction", type=float, default=0.5)
    parser.add_argument("--no-league", action="store_true")
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Smoke test: 2000 rounds, 10 gradient updates, one eval, save.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_every=args.eval_every,
        device=args.device,
        seed=args.seed,
        league_fraction=args.league_fraction,
    )
    if args.no_league:
        cfg.league_fraction = 0.0
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = PlumpTrainer(cfg, resume=args.resume)
    parameter_count = sum(p.numel() for p in trainer.model.parameters())
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = "none"
    print(
        f"[plump] device={trainer.device} gpu={gpu} amp={trainer.amp_enabled} "
        f"params={parameter_count:,} envs={cfg.num_envs} "
        f"league={cfg.league_fraction:.2f}",
        flush=True,
    )

    if args.test_run:
        trainer.run_test_run(rounds=2_000, updates=10)
    else:
        trainer.run(max_rounds=args.rounds)


if __name__ == "__main__":
    main()
