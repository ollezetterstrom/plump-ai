# Plump: a strong Plump AI on a consumer GPU (Rust engine + PyTorch).

from . import config
from .net import PlumpNet
from .rollout import RolloutDriver

__all__ = ["config", "PlumpNet", "RolloutDriver", "__version__"]

__version__ = "0.1.0"
