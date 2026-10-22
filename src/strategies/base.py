"""Strategy interface. Strategies see candidate descriptions/metadata and the e_hull of candidates they chose;
they never see hit/known/novel labels."""
from __future__ import annotations

import random


class Strategy:
    name = "base"

    def __init__(self, seed: int, evaluator, batch_size: int = 10):
        self.seed = seed
        self.ev = evaluator
        self.batch_size = batch_size
        self.rng = random.Random(f"{self.name}-{seed}")
        self.reasons: dict[str, str] = {}  # cid -> reason (LLM agent fills this)
        self.round = 0

    def propose(self, pool_ids: list[str], history: list[tuple[str, float | None]], k: int) -> list[str]:
        raise NotImplementedError
