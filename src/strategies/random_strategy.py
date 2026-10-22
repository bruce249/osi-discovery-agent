"""Uniform random sampling without replacement."""
from __future__ import annotations

from .base import Strategy


class RandomStrategy(Strategy):
    name = "random"

    def propose(self, pool_ids, history, k):
        return self.rng.sample(sorted(pool_ids), min(k, len(pool_ids)))
