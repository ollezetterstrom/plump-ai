"""Frozen champion pool for dynamic self-play league training."""

from __future__ import annotations

import copy
import hashlib
import random
from typing import List, Optional

import torch

from ..models.transformer import PlumpTransformer

# Bot policy IDs (0..=3 match plump_engine bot levels) and frozen champions (>= 4)
ROOKIE = 0
REGULAR = 1
TACTICIAN = 2
FINAL_BOSS = 3
CHAMPION_BASE = 4

# Aliases for backward compatibility
HEURISTIC = TACTICIAN
RANDOM = ROOKIE


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
        """Sample one of Rookie/Regular/Tactician/FinalBoss/Champion for a league round."""
        roll = self.rng.random()
        if self.champions:
            if roll < 0.10:
                return ROOKIE
            if roll < 0.30:
                return REGULAR
            if roll < 0.60:
                return TACTICIAN
            if roll < 0.70:
                return TACTICIAN
            return CHAMPION_BASE + self.rng.randrange(len(self.champions))
        # No champions yet: redistribute the 30% champion share to Tactician (+20%) and Regular (+10%)
        if roll < 0.10:
            return ROOKIE
        if roll < 0.40:
            return REGULAR
        if roll < 0.90:
            return TACTICIAN
        return TACTICIAN

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
