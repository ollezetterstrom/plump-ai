# Plump: a strong Plump AI on a consumer GPU (Rust engine + PyTorch).

from . import config
from .net import PlumpNet

__all__ = ["config", "PlumpNet", "__version__"]

__version__ = "0.1.0"
