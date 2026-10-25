"""Frozen champion pool for dynamic self-play league training."""

from __future__ import annotations

import copy
import hashlib
import random
from typing import List, Optional

import torch

from ..models.transformer import PlumpTransformer

# 0 = hand-written heuristic, 1 = uniform random, >= 2 = frozen champion.
HEURISTIC = 0
RANDOM = 1
CHAMPION_BASE = 2


class League:
    def __init__(
        self,
        device: torch.device | str,
        max_champions: int = 5,
        seed: int = 0,
    ):
        self.device = device
        self.max_champions = max_champions
        self.champions: List[PlumpTransformer] = []
        self.scores: List[float] = []
        self._hashes: List[str] = []
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.champions)

    @staticmethod
    def _hash_model(model: PlumpTransformer) -> str:
        digest = hashlib.sha1()
        for key, value in model.state_dict().items():
            digest.update(key.encode())
            digest.update(value.detach().float().cpu().numpy().tobytes())
        return digest.hexdigest()

    def add_champion(
        self, model: PlumpTransformer, score: Optional[float] = None
    ) -> bool:
        """Freeze a snapshot, unless identical to an existing champion."""
        fingerprint = self._hash_model(model)
        if fingerprint in self._hashes:
            return False
        clone = copy.deepcopy(model).to(self.device)
        clone.eval()
        for parameter in clone.parameters():
            parameter.requires_grad_(False)
        self.champions.append(clone)
        self.scores.append(float(score) if score is not None else 0.0)
        self._hashes.append(fingerprint)
        while len(self.champions) > self.max_champions:
            self.champions.pop(0)
            self.scores.pop(0)
            self._hashes.pop(0)
        return True

    def load_champions(
        self, states: List[dict], scores: Optional[List[float]] = None
    ) -> None:
        from ..config import PlumpTransformerConfig

        for i, state in enumerate(states):
            cfg = PlumpTransformerConfig()
            model = PlumpTransformer(cfg)
            try:
                model.load_state_dict(state)
            except RuntimeError:
                continue
            self.add_champion(
                model, scores[i] if scores and i < len(scores) else 0.0
            )

    def sample_policy(self) -> int:
        """Sample one of heuristic/random/champion for a league round."""
        if not self.champions:
            return HEURISTIC if self.rng.random() < 0.8 else RANDOM
        roll = self.rng.random()
        if roll < 0.70:
            return CHAMPION_BASE + self.rng.randrange(len(self.champions))
        if roll < 0.90:
            return HEURISTIC
        return RANDOM

    def model_for(self, policy: int) -> Optional[PlumpTransformer]:
        if policy >= CHAMPION_BASE:
            index = policy - CHAMPION_BASE
            if 0 <= index < len(self.champions):
                return self.champions[index]
        return None

    def champion(self) -> Optional[PlumpTransformer]:
        return self.champions[-1] if self.champions else None

    def state_dicts(self) -> List[dict]:
        return [c.state_dict() for c in self.champions]
