"""Checkpoint helpers shared by training and play.py."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from .config import PlumpTransformerConfig
from .models.transformer import PlumpTransformer


def save_checkpoint(
    path: Path,
    model: PlumpTransformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[PlumpTransformerConfig] = None,
    **metadata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or model.cfg
    payload = {
        "model": model.state_dict(),
        "config": cfg.as_dict(),
        **metadata,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_model(
    path: Path, device: torch.device | str = "cpu"
) -> Tuple[PlumpTransformer, dict]:
    """Load either a full training checkpoint or a raw state_dict."""
    payload = load_checkpoint(path, device)
    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
        cfg = PlumpTransformerConfig.from_dict(payload.get("config", {}))
    else:
        state = payload
        cfg = PlumpTransformerConfig()
    model = PlumpTransformer(cfg)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, payload if isinstance(payload, dict) else {}
