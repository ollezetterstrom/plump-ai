"""Plump: hybrid Rust engine + PyTorch self-play AI."""

from .config import PlumpTransformerConfig, TrainConfig
from .models.transformer import PlumpTransformer

__all__ = ["PlumpTransformer", "PlumpTransformerConfig", "TrainConfig"]
