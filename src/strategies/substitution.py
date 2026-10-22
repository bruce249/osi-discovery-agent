"""Data-mined substitution baseline (Hautier et al., pymatgen SubstitutionProbability), plus an ionic-radius heuristic.

Score of a candidate = product over substituted sublattices (M, X) of the site-averaged conditional probability
P(new species | parent species) mined from the ICSD. Unchanged sites count as probability 1. Candidates are
ranked once and evaluated in that order; ties are broken randomly per seed.
"""
from __future__ import annotations

from functools import lru_cache

from pymatgen.core import Species

from .base import Strategy


@lru_cache(maxsize=None)
def _sp():
    from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionProbability

    return SubstitutionProbability()


@lru_cache(maxsize=None)
def cond_prob(new: str, parent: str) -> float:
    """P(parent site is taken by `new`), from the ICSD-mined lambda table."""
    if new == parent:
        return 1.0
    sp = _sp()
    a, b = Species.from_str(new), Species.from_str(parent)
    try:
        return float(sp.cond_prob(a, b))
    except AttributeError:  # older pymatgen without cond_prob
        return float(sp.prob(a, b))


def substitution_score(cand: dict) -> float:
    score = 1.0
    for sub in cand["substitutions"]:
        score *= sum(frac * cond_prob(new, sub["parent"]) for new, frac in sub["new"].items())
    return score


def radius_mismatch(cand: dict) -> float:
    total = 0.0
    for sub in cand["substitutions"]:
        rp = Species.from_str(sub["parent"]).ionic_radius
        for new, frac in sub["new"].items():
            rn = Species.from_str(new).ionic_radius
            if rp and rn:
                total += frac * abs(rn - rp) / rp
            else:
                total += frac * 0.5
    return total


class _RankedStrategy(Strategy):
    def __init__(self, seed, evaluator, batch_size=10):
        super().__init__(seed, evaluator, batch_size)
        ids = evaluator.candidate_ids()
        keyed = [(self._key(evaluator.candidate(i)), self.rng.random(), i) for i in sorted(ids)]
        self.order = [i for _, _, i in sorted(keyed)]
        self.scores = {i: k for k, _, i in keyed}

    def _key(self, cand):
        raise NotImplementedError

    def propose(self, pool_ids, history, k):
        pool = set(pool_ids)
        return [i for i in self.order if i in pool][:k]


class SubstitutionStrategy(_RankedStrategy):
    name = "substitution"

    def _key(self, cand):
        return -substitution_score(cand)


class RadiusStrategy(_RankedStrategy):
    name = "radius"

    def _key(self, cand):
        return radius_mismatch(cand)
