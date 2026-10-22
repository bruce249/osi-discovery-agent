"""Backend for the notebook playground: re-run campaigns with your own settings (instant replay over the cached
CHGNet results) and step an LLM copilot round by round.

Nothing here relaxes structures; every evaluation is a lookup in results/evaluations_chgnet.json.
"""
from __future__ import annotations

import math
import os
import random
from functools import lru_cache

import numpy as np

from .utils import CACHE, get_logger, load_config

try:
    import fcntl  # Linux (Modal); the counter lock is skipped on Windows
except ImportError:  # pragma: no cover
    fcntl = None

log = get_logger("playground")
LLM_CALL_CAP = int(os.environ.get("PLAYGROUND_LLM_CALL_CAP", "300"))  # per container, all visitors together
_COUNTER = os.path.join(os.environ.get("TMPDIR", "/tmp"), "osi_playground_llm_calls")


@lru_cache(maxsize=1)
def data():
    """Evaluator plus per-candidate facts used for scoring (never shown to strategies)."""
    from .evaluator import MLIPEvaluator
    from .mp_fetch import load_mp_index
    from .novelty import load_novelty

    ev = MLIPEvaluator("chgnet")
    nov = load_novelty()
    mp_index = load_mp_index()
    facts = {}
    for cid in ev.candidate_ids():
        c = ev.candidate(cid)
        f = c["reduced_formula"]
        n = nov.get(f)
        facts[cid] = {
            "formula": f, "display": c["display_formula"], "e_hull": ev.evals[cid]["e_hull"],
            "known": any(not x["theoretical"] for x in mp_index.get(f, [])) or bool(n and n["lit"].get("exact_reported")),
            # novelty was checked for every composition with e_hull <= 0.05 eV/atom
            "novel": bool(n and n["db"]["absent"] and n["lit"]["lit_status"] == "none found"),
        }
    return ev, facts


def make_strategy(name: str, seed: int, batch: int, budget: int, thr: float, **llm):
    from .strategies.gp_bo import GPBOStrategy
    from .strategies.llm_agent import LLMAgentStrategy, LLMBasicStrategy
    from .strategies.random_strategy import RandomStrategy
    from .strategies.substitution import RadiusStrategy, SubstitutionStrategy

    ev, _ = data()
    if name.startswith("llm"):
        cls = LLMBasicStrategy if llm.get("prompt") == "basic" else LLMAgentStrategy
        s = cls(seed, ev, batch, budget=budget)
        if llm.get("model"):
            s.model = llm["model"]
    else:
        cls = {"random": RandomStrategy, "substitution": SubstitutionStrategy, "radius": RadiusStrategy,
               "gp_bo": GPBOStrategy}[name]
        s = cls(seed, ev, batch)
    s.thr = thr  # GP-BO acquisition and the LLM prompt use the playground threshold
    return s


class Campaign:
    """One strategy, one seed. step() spends one batch; the hidden facts only score the result."""

    def __init__(self, strategy, budget: int, batch: int, thr: float, skip_found: bool = True):
        self.s, self.budget, self.batch, self.thr, self.skip = strategy, budget, batch, thr, skip_found
        self.ev, self.facts = data()
        self.ids = self.ev.candidate_ids()
        self.history: list[tuple[str, float]] = []
        self.done: set[str] = set()
        self.confirmed: set[str] = set()
        self.found = {"hit": set(), "known": set(), "novel": set()}
        self.curves = {m: [] for m in self.found}
        self.picks: list[dict] = []
        self.fill_rng = random.Random(f"fill-{strategy.name}-{strategy.seed}")
        self.round = 0

    @property
    def finished(self) -> bool:
        return len(self.history) >= self.budget or not self._pool()

    def _pool(self):
        return [c for c in self.ids if c not in self.done
                and not (self.skip and self.facts[c]["formula"] in self.confirmed)]

    def step(self) -> list[dict]:
        pool = self._pool()
        k = min(self.batch, self.budget - len(self.history), len(pool))
        if k <= 0:
            return []
        proposed = self.s.propose(pool, list(self.history), k)
        pool_set, picks = set(pool), []
        for c in proposed:
            if c in pool_set and c not in picks:
                picks.append(c)
        picks = picks[:k]
        filled = set()
        if len(picks) < k:
            filled = set(self.fill_rng.sample(sorted(pool_set - set(picks)), k - len(picks)))
            picks += sorted(filled)
        self.round += 1
        out = []
        for cid in picks:
            f = self.facts[cid]
            e = f["e_hull"]
            self.history.append((cid, e))
            self.done.add(cid)
            hit = e <= self.thr + 1e-9
            if hit:
                self.confirmed.add(f["formula"])
                self.found["hit"].add(f["formula"])
                if f["known"]:
                    self.found["known"].add(f["formula"])
                if f["novel"]:
                    self.found["novel"].add(f["formula"])
            for m in self.found:
                self.curves[m].append(len(self.found[m]))
            row = {"round": self.round, "id": cid, "candidate": self.ev.describe(cid), "e_hull": round(e, 4),
                   "stable": hit, "novel": hit and f["novel"], "known": hit and f["known"],
                   "reason": "(filled randomly: invalid pick)" if cid in filled else self.s.reasons.get(cid, "")}
            out.append(row)
            self.picks.append(row)
        return out

    def run(self):
        while not self.finished:
            self.step()
        return self


def totals(thr: float) -> dict:
    _, facts = data()
    hits = {f["formula"]: f for f in facts.values() if f["e_hull"] <= thr + 1e-9}
    return {"hit": len(hits), "known": sum(f["known"] for f in hits.values()),
            "novel": sum(f["novel"] for f in hits.values())}


def n_to_reach(curve, target):
    if target <= 0:
        return 0.0
    for i, v in enumerate(curve):
        if v >= target:
            return float(i + 1)
    return math.nan


def compare(strategies: list[str], n_seeds: int, budget: int, batch: int, thr: float, skip_found: bool) -> dict:
    """Run the chosen strategies; random always runs over the whole pool as the speed-up reference."""
    ev, _ = data()
    n_pool = len(ev.candidate_ids())
    cap = {"gp_bo": min(n_seeds, 5)}
    runs = {}
    for name in dict.fromkeys(["random", *strategies]):
        b = n_pool if name == "random" else budget
        runs[name] = [Campaign(make_strategy(name, seed, batch, b, thr), b, batch, thr, skip_found).run()
                      for seed in range(cap.get(name, n_seeds))]
    rows = []
    for name, rs in runs.items():
        if name == "random" and "random" not in strategies:
            continue
        row = {"strategy": name, "seeds": len(rs)}
        for m in ("hit", "novel", "known"):
            at_b = [c.curves[m][min(budget, len(c.curves[m])) - 1] for c in rs]
            mean = float(np.mean(at_b))
            n_match = [n_to_reach(r.curves[m], math.ceil(mean - 1e-9)) for r in runs["random"]]
            sp = float(np.nanmean(n_match)) / budget if mean > 0 and not all(math.isnan(x) for x in n_match) else None
            row[f"{m} @budget"] = f"{mean:.1f} ± {np.std(at_b):.1f}"
            row[f"{m} speed-up"] = None if sp is None else round(sp, 2)
        rows.append(row)
    return {"runs": runs, "rows": rows, "totals": totals(thr), "budget": budget}


def reserve_llm_call() -> bool:
    """Container-wide cap on live LLM calls (file lock so concurrent sessions share one counter)."""
    with open(_COUNTER, "a+") as f:
        if fcntl:
            fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        n = int(f.read().strip() or 0)
        if n >= LLM_CALL_CAP:
            return False
        f.seek(0)
        f.truncate()
        f.write(str(n + 1))
        return True


def llm_calls_used() -> int:
    try:
        with open(_COUNTER) as f:
            return int(f.read().strip() or 0)
    except FileNotFoundError:
        return 0


def random_expectation(thr: float, n: int, seeds: int = 20) -> dict:
    """Mean random curve up to n evaluations (reference line for the copilot)."""
    rs = [Campaign(make_strategy("random", s, 10, n, thr), n, 10, thr, True).run() for s in range(seeds)]
    return {m: np.mean([r.curves[m][:n] for r in rs], axis=0) for m in ("hit", "novel", "known")}


def llm_cache_dir():
    return CACHE / "llm"


def config():
    return load_config()


if __name__ == "__main__":  # quick self-test: python -m src.playground
    import json
    import time

    t0 = time.time()
    res = compare(["random", "substitution", "radius", "gp_bo"], 2, 90, 10, 0.0, True)
    print(json.dumps(res["rows"], indent=1), res["totals"], f"{time.time() - t0:.1f}s")
    c = Campaign(make_strategy("llm", 0, 10, 20, 0.0, model="gpt-4.1-mini"), 20, 10, 0.0, True)
    print(c.step()[:2], len(c.history), "llm calls used", llm_calls_used())
