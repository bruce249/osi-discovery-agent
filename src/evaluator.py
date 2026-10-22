"""Pluggable evaluator. Campaigns replay precomputed MLIP results (standard BO-benchmark practice):
every strategy sees exactly the same numbers, runs instantly and reproducibly."""
from __future__ import annotations

from typing import Protocol


class Evaluator(Protocol):
    def candidate_ids(self) -> list[str]: ...
    def describe(self, cid: str) -> str: ...
    def evaluate(self, cid: str) -> dict: ...


class MLIPEvaluator:
    """Reads results/evaluations_<model>.json (lowest-energy ordering, e_hull vs same-model reference hull)."""

    def __init__(self, model: str = "chgnet"):
        from .candidates import load_candidates
        from .hull import load_evaluations

        self.model = model
        self.cands = {c["id"]: c for c in load_candidates()}
        self.evals = load_evaluations(model)
        if not self.evals:
            raise RuntimeError(f"no evaluations for {model}: run `python -m src.hull` first")
        # candidates whose every ordering failed are dropped from the pool (reported in the README)
        self._ids = [cid for cid in self.cands if self.evals.get(cid, {}).get("ok")]
        self.n_evaluations = 0

    def candidate_ids(self) -> list[str]:
        return list(self._ids)

    def describe(self, cid: str) -> str:
        return self.cands[cid]["description"]

    def candidate(self, cid: str) -> dict:
        return self.cands[cid]

    def evaluate(self, cid: str) -> dict:
        self.n_evaluations += 1
        r = self.evals[cid]
        return {"e_hull": r["e_hull"], "ok": r["ok"], "decomposition": r.get("decomposition", [])}

    def decomposition(self, cid: str) -> list[str]:
        """Competing phases from the hull (simulation output). Strategies only ask for candidates they evaluated."""
        return self.evals[cid].get("decomposition", [])
