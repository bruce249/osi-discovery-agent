"""Same pattern in two domains: the LLM's head start comes from what it already knows.

Reactions: results of our team's reaction-optimisation benchmark (github.com/bruce249/osihackiitd, copied to
external/osihackiitd/): Buchwald-Hartwig yield screens (Ahneman et al. 2018), 5 substrates, budget 20.
The LLM agent there was also run with chemical names hidden ("anonymised"), which removes its prior knowledge.
Materials: this repository's campaigns (results/summary.json).

    python -m src.cross_domain      # results/cross_domain.json + results/fig5_cross_domain.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import RESULTS, dump_json, get_logger, load_json

log = get_logger("cross_domain")
EXT = Path(__file__).resolve().parent.parent / "external" / "osihackiitd"
RXN = {"random": ("#7f7f7f", "Random"), "gp_ei": ("#9467bd", "GP-BO"),
       "llm": ("#d62728", "LLM agent, chemical names shown"),
       "llm_anon": ("#ff9896", "LLM agent, names hidden"), "hybrid": ("#ff7f0e", "LLM + surrogate hybrid")}
MAT = {"random": "Random", "gp_bo": "GP-BO", "gp_latent": "GP on MACE latent", "llm": "LLM (gpt-4.1-mini)",
       "llm_strong": "LLM (gpt-5-mini)", "llm_gp": "LLM + GP hybrid"}


def reaction_curves(dataset: str = "BH") -> dict:
    df = pd.read_csv(EXT / "runs.csv")
    df = df[df.dataset == dataset]
    out = {}
    for s in RXN:
        d = df[df.strategy == s].groupby("n")["norm_best"]
        n_runs = int(df[df.strategy == s].groupby(["task_id", "seed"]).ngroups)
        m, se = d.mean(), d.std() / np.sqrt(max(n_runs, 1))
        out[s] = {"n": m.index.tolist(), "mean": m.tolist(), "se": se.fillna(0).tolist(), "runs": n_runs}
    return out


def materials_bars() -> dict:
    summ = load_json(RESULTS / "summary.json")
    tk = next(iter(summ["strategies"]["random"]))  # headline threshold key comes first
    tk = [k for k in summ["strategies"]["random"] if k != "llm_calls"][0]
    out = {}
    for s in MAT:
        if s in summ["strategies"]:
            st = summ["strategies"][s][tk]
            out[s] = {m: (st[m]["at_budget_mean"], st[m]["at_budget_std"]) for m in ("known", "novel")}
    return {"threshold_key": tk, "budget": summ["budget"], "bars": out}


def main():
    import matplotlib.pyplot as plt

    rx, mat = reaction_curves(), materials_bars()
    dump_json({"reactions_BH": rx, "materials": mat}, RESULTS / "cross_domain.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for s, (col, lab) in RXN.items():
        c = rx[s]
        n, m, se = np.array(c["n"]), np.array(c["mean"]), np.array(c["se"])
        a.plot(n, m, color=col, lw=2.2 if s.startswith("llm") else 1.6, label=f"{lab} ({c['runs']} runs)")
        a.fill_between(n, m - se, m + se, color=col, alpha=0.12)
    a.axvline(10, color="k", ls=":", lw=0.8)
    a.set_xlabel("experiments run")
    a.set_ylabel("best yield so far / best possible")
    a.set_title("Reactions (real yield screens): the LLM's early lead\ndisappears when chemical names are hidden",
                fontsize=10)
    a.legend(fontsize=8, loc="lower right")
    names = list(mat["bars"])
    x = np.arange(len(names))
    for off, metric, col, lab in ((-0.2, "known", "#1f77b4", "known compounds rediscovered"),
                                  (0.2, "novel", "#2ca02c", "new stable compositions found")):
        vals = [mat["bars"][s][metric] for s in names]
        b.bar(x + off, [v[0] for v in vals], 0.38, yerr=[v[1] for v in vals], color=col, label=lab, capsize=3)
    b.set_xticks(x, [MAT[s] for s in names], rotation=20, ha="right", fontsize=8)
    b.set_ylabel(f"count after {mat['budget']} simulations")
    b.set_title("Materials (Li-halide electrolytes): LLMs rediscover the known,\nstatistical surrogates find the new",
                fontsize=10)
    b.legend(fontsize=8)
    fig.savefig(RESULTS / "fig5_cross_domain.png", dpi=160)
    plt.close(fig)
    at10 = {s: rx[s]["mean"][9] for s in rx}
    log.info("BH norm_best at n=10: %s", {k: round(v, 3) for k, v in at10.items()})


if __name__ == "__main__":
    main()
