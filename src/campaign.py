"""Run every strategy under the same budget on the replayed MLIP evaluator and score with hidden labels.

Counting: a composition is "found" once any candidate with that reduced formula has been evaluated and is a hit,
so evaluating the same composition in a second prototype earns nothing extra. Campaign rule (all strategies alike):
once a strategy's OWN history shows a composition at e_hull <= threshold, that composition's other candidates leave
its pool (nobody re-simulates a composition already confirmed stable). Only visible e_hull values are used.
Acceleration factors:
  AF50     = (evaluations random needs to find 50% of all hit compositions in the pool) / (same for strategy)
  AF@budget = (evaluations random needs to match the strategy's count at the budget) / budget
Random (and the cheap ranking baselines) are run over the whole pool so these are measured, not extrapolated.
"""
from __future__ import annotations

import argparse
import math
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .evaluator import MLIPEvaluator
from .labels import load_labels, thr_key
from .strategies.gp_bo import (GPBOStrategy, GPLatentAblRandStrategy, GPLatentINLPStrategy,
                               GPLatentRandomStrategy, GPLatentStrategy)
from .strategies.llm_agent import (LLMAgentStrategy, LLMBasicStrategy, LLMGPHybridStrategy,
                                       LLMStrongStrategy)
from .strategies.random_strategy import RandomStrategy
from .strategies.substitution import RadiusStrategy, SubstitutionStrategy
from .utils import RESULTS, dump_json, get_logger, load_config, load_json, optional_env

log = get_logger("campaign")
STRATEGIES = {"random": RandomStrategy, "substitution": SubstitutionStrategy, "radius": RadiusStrategy,
              "gp_bo": GPBOStrategy, "gp_latent": GPLatentStrategy,
              "gp_latent_rand": GPLatentRandomStrategy,
              "gp_latent_inlp": GPLatentINLPStrategy, "gp_latent_ablrand": GPLatentAblRandStrategy, "llm": LLMAgentStrategy, "llm_basic": LLMBasicStrategy,
              "llm_strong": LLMStrongStrategy, "llm_gp": LLMGPHybridStrategy}
FULL_POOL = {"random", "substitution", "radius"}  # cheap: run to exhaustion for exact AF reference
METRICS = ("hit", "known", "novel")


def budget_for(n_pool: int, cfg: dict) -> int:
    c = cfg["campaign"]
    if n_pool * c["pool_fraction_if_small"] >= c["budget"]:
        return c["budget"]
    return max(c["batch_size"], int(round(n_pool * c["pool_fraction_if_small"] / c["batch_size"])) * c["batch_size"])


def run_one(name: str, seed: int, ev: MLIPEvaluator, labels: dict, budget: int, batch: int, thresholds) -> dict:
    cls = STRATEGIES[name]
    strat = cls(seed, ev, batch, budget=budget) if name.startswith("llm") else cls(seed, ev, batch)
    all_ids = ev.candidate_ids()
    fill_rng = random.Random(f"fill-{name}-{seed}")
    evaluated, done, history = [], set(), []
    found = {t: {m: set() for m in METRICS} for t in thresholds}
    curves = {t: {m: [] for m in METRICS} for t in thresholds}
    rounds, picks_log = [], []
    rnd = 0
    cfg = load_config()
    head = cfg["labels"]["hit_threshold"]
    skip_found = cfg["campaign"].get("skip_found_compositions", True)
    formula = {c: ev.candidate(c)["reduced_formula"] for c in all_ids}
    confirmed: set[str] = set()  # compositions this strategy has itself seen at e_hull <= threshold
    while len(evaluated) < budget:
        pool = [c for c in all_ids if c not in done and not (skip_found and formula[c] in confirmed)]
        if not pool:
            break
        k = min(batch, budget - len(evaluated), len(pool))
        proposed = strat.propose(pool, list(history), k)
        pool_set, picks = set(pool), []
        for c in proposed:
            if c in pool_set and c not in picks:
                picks.append(c)
        picks = picks[:k]
        n_filled = k - len(picks)
        if n_filled:
            rest = sorted(pool_set - set(picks))
            fill = fill_rng.sample(rest, n_filled)
            picks += fill
        else:
            fill = []
        for cid in picks:
            res = ev.evaluate(cid)
            history.append((cid, res["e_hull"] if res["ok"] else None))
            if res["ok"] and res["e_hull"] <= head + 1e-9:
                confirmed.add(formula[cid])
            evaluated.append(cid)
            done.add(cid)
            lab = labels[cid]
            for t in thresholds:
                tk = thr_key(t)
                if lab[f"hit_{tk}"]:
                    found[t]["hit"].add(lab["formula"])
                if lab[f"known_hit_{tk}"]:
                    found[t]["known"].add(lab["formula"])
                if lab[f"novel_{tk}"]:
                    found[t]["novel"].add(lab["formula"])
                for m in METRICS:
                    curves[t][m].append(len(found[t][m]))
            picks_log.append({"strategy": name, "seed": seed, "round": rnd + 1, "id": cid,
                              "description": ev.describe(cid), "e_hull": res["e_hull"],
                              "filled_randomly": cid in fill, "reason": strat.reasons.get(cid, "")})
        row = {"strategy": name, "seed": seed, "round": rnd + 1, "evaluations": len(evaluated), "n_filled": n_filled}
        for t in thresholds:
            for m in METRICS:
                row[f"{m}_{thr_key(t)}"] = len(found[t][m])
        rounds.append(row)
        rnd += 1
    meta = {"llm_calls": getattr(strat, "calls", None)} if name.startswith("llm") else {}
    return {"strategy": name, "seed": seed, "rounds": rounds, "picks": picks_log,
            "curves": {thr_key(t): curves[t] for t in thresholds}, **meta}


def n_to_reach(curve: list[int], target: int) -> float:
    if target <= 0:
        return 0.0
    for i, v in enumerate(curve):
        if v >= target:
            return float(i + 1)
    return math.nan


def summarise(runs: list[dict], labels: dict, ev: MLIPEvaluator, budget: int, thresholds, cfg) -> dict:
    pool = set(ev.candidate_ids())
    totals = {}
    for t in thresholds:
        tk = thr_key(t)
        totals[tk] = {m: len({labels[c]["formula"] for c in pool if labels[c][f"{'known_hit' if m == 'known' else m}_{tk}"]})
                      for m in METRICS}
    by = {}
    for r in runs:
        by.setdefault(r["strategy"], []).append(r)
    rand = by.get("random", [])
    out = {"pool_size": len(pool), "unique_formulas_in_pool": len({labels[c]["formula"] for c in pool}),
           "budget": budget, "batch_size": cfg["campaign"]["batch_size"], "headline_threshold": cfg["labels"]["hit_threshold"],
           "totals_in_pool": totals, "seeds": {k: [r["seed"] for r in v] for k, v in by.items()}, "strategies": {}}
    for name, rs in by.items():
        s = {}
        for t in thresholds:
            tk = thr_key(t)
            st = {}
            for m in METRICS:
                curves = [r["curves"][tk][m] for r in rs]
                at_b = [c[min(budget, len(c)) - 1] if c else 0 for c in curves]
                target = math.ceil(0.5 * totals[tk][m])
                n50 = [n_to_reach(c[:budget], target) for c in curves]
                n50_rand = [n_to_reach(r["curves"][tk][m], target) for r in rand]
                n50_r = float(np.nanmean(n50_rand)) if rand and not all(math.isnan(x) for x in n50_rand) else math.nan
                n50_s = float(np.mean(n50)) if not any(math.isnan(x) for x in n50) else math.nan
                # random evaluations needed to match this strategy's mean count at budget
                mean_at_b = float(np.mean(at_b))
                n_match = [n_to_reach(r["curves"][tk][m], math.ceil(mean_at_b - 1e-9)) for r in rand]
                n_match_r = float(np.nanmean(n_match)) if rand and not all(math.isnan(x) for x in n_match) else math.nan
                st[m] = {"at_budget_mean": mean_at_b, "at_budget_std": float(np.std(at_b)),
                         "at_budget_per_seed": at_b, "target_50pct": target,
                         "n50_mean": n50_s, "n50_per_seed": n50, "n50_random": n50_r,
                         "AF50": (n50_r / n50_s) if (n50_s and not math.isnan(n50_s) and not math.isnan(n50_r)) else None,
                         "AF_at_budget": (n_match_r / budget) if (mean_at_b > 0 and not math.isnan(n_match_r)) else None}
            s[tk] = st
        if name.startswith("llm"):
            calls = [c for r in rs for c in (r.get("llm_calls") or [])]
            s["llm_calls"] = {"n": len(calls), "cached": sum(c["cached"] for c in calls),
                              "picks_filled_randomly": int(sum(row["n_filled"] for r in rs for row in r["rounds"]))}
        out["strategies"][name] = s
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="random,substitution,radius,gp_bo,llm,llm_basic,llm_strong,llm_gp")
    args = ap.parse_args(argv)
    cfg = load_config()
    ev = MLIPEvaluator("chgnet")
    labels = load_labels()
    thresholds = sorted({cfg["labels"]["hit_threshold"], *cfg["labels"]["extra_thresholds"]})
    n_pool = len(ev.candidate_ids())
    budget = budget_for(n_pool, cfg)
    batch = cfg["campaign"]["batch_size"]
    log.info("pool %d candidates, budget %d, batch %d", n_pool, budget, batch)
    names = [s for s in args.strategies.split(",") if s]
    if any(n.startswith("llm") for n in names) and not optional_env("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY missing: ask the human to set it in .env")

    prev = load_json(RESULTS / "campaign_runs.json", [])
    runs = [r for r in prev if r["strategy"] not in names]  # keep strategies we are not re-running
    for name in names:
        seeds = cfg["seeds"]["llm" if name.startswith("llm") else name]
        b = n_pool if name in FULL_POOL else budget
        if name.startswith("llm"):
            with ThreadPoolExecutor(max_workers=len(seeds)) as pool:
                res = list(pool.map(lambda s: run_one(name, s, ev, labels, b, batch, thresholds), seeds))
        else:
            res = [run_one(name, s, ev, labels, b, batch, thresholds) for s in seeds]
        runs += res
        log.info("%s done (%d seeds, %d evals each)", name, len(seeds), b)
    dump_json(runs, RESULTS / "campaign_runs.json")

    # per-round log truncated to the common budget
    rows = [row for r in runs for row in r["rounds"] if row["evaluations"] <= budget]
    pd.DataFrame(rows).to_csv(RESULTS / "campaign_log.csv", index=False)
    picks = [p for r in runs for p in r["picks"] if p["round"] * batch <= budget]
    pd.DataFrame(picks).to_csv(RESULTS / "campaign_picks.csv", index=False)
    summary = summarise(runs, labels, ev, budget, thresholds, cfg)
    summary["llm_model"] = optional_env("LLM_MODEL", cfg["llm"]["default_model"])
    summary["llm_strong_model"] = cfg["llm"]["strong_model"]
    summary["llm_base_url"] = optional_env("LLM_BASE_URL", cfg["llm"]["default_base_url"])
    dump_json(summary, RESULTS / "summary.json")
    tk = thr_key(cfg["labels"]["hit_threshold"])
    for name, s in summary["strategies"].items():
        log.info("%-12s hits %.1f  known %.1f  novel %.1f | AF50 hit=%s novel=%s | AF@B hit=%s novel=%s", name,
                 s[tk]["hit"]["at_budget_mean"], s[tk]["known"]["at_budget_mean"], s[tk]["novel"]["at_budget_mean"],
                 s[tk]["hit"]["AF50"], s[tk]["novel"]["AF50"], s[tk]["hit"]["AF_at_budget"], s[tk]["novel"]["AF_at_budget"])
    return summary


if __name__ == "__main__":
    main()
