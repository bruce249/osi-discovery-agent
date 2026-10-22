"""Figures and the final candidate table.

  results/fig1_discovery.png   novel hits (headline) + rediscovery of known compounds vs evaluations
  results/fig1b_all_hits.png   all stable compositions vs evaluations
  results/top_candidates.csv   section 7.4 table
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import load_labels, thr_key
from .utils import CACHE, RESULTS, get_logger, load_config, load_json

log = get_logger("report")
STYLE = {"llm": ("#d62728", "LLM agent (feedback)"), "llm_basic": ("#ff9896", "LLM, basic prompt (ablation)"),
         "llm_strong": ("#8b0000", "LLM agent (reasoning model)"),
         "llm_gp": ("#ff7f0e", "LLM + GP hybrid"), "gp_bo": ("#9467bd", "GP-BO"),
         "substitution": ("#1f77b4", "Data-mined substitution"), "radius": ("#8c564b", "Ionic-radius heuristic"),
         "random": ("#7f7f7f", "Random")}
ORDER = ["llm_gp", "llm_strong", "llm", "llm_basic", "gp_bo", "substitution", "radius", "random"]


def _curves(runs, name, tk, metric, budget):
    cs = [r["curves"][tk][metric][:budget] for r in runs if r["strategy"] == name]
    if not cs:
        return None
    n = min(len(c) for c in cs)
    a = np.array([c[:n] for c in cs], dtype=float)
    return np.arange(1, n + 1), a.mean(0), a.std(0), len(cs)


def plot_metric(ax, runs, tk, metric, budget, total, title, ylabel):
    for name in ORDER:
        res = _curves(runs, name, tk, metric, budget)
        if res is None:
            continue
        x, m, s, n = res
        color, label = STYLE[name]
        ax.plot(x, m, color=color, lw=2.4 if name in ("llm_gp", "llm_strong") else 1.5, ls="--" if name == "llm_basic" else "-", label=f"{label} [{n} seeds]")
        if n > 1:
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.15, lw=0)
    ax.axhline(total, color="k", ls=":", lw=1)
    ax.text(budget * 0.98, total, f"all in pool: {total}", va="bottom", ha="right", fontsize=9)
    ax.set_xlim(0, budget)
    ax.set_ylim(0, max(total * 1.12, 1))
    ax.set_xlabel("Simulations used (evaluations)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.3)


def figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config()
    tk = thr_key(cfg["labels"]["hit_threshold"])
    runs = load_json(RESULTS / "campaign_runs.json")
    summ = load_json(RESULTS / "summary.json")
    budget, tot = summ["budget"], summ["totals_in_pool"][tk]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    plot_metric(axes[0], runs, tk, "novel", budget, tot["novel"],
                f"Novel stable compositions found (e_hull ≤ {cfg['labels']['hit_threshold']} eV/atom,\n"
                "not in checked databases or literature)", "Cumulative novel stable compositions")
    plot_metric(axes[1], runs, tk, "known", budget, tot["known"],
                "Rediscovery of experimentally known compounds", "Cumulative known compounds rediscovered")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(RESULTS / "fig1_discovery.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    plot_metric(ax, runs, tk, "hit", budget, tot["hit"], "All stable compositions found",
                "Cumulative stable compositions")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1b_all_hits.png", dpi=200)
    plt.close(fig)
    log.info("figures written")


def top_table() -> pd.DataFrame:
    from .candidates import load_candidates

    cands = {c["id"]: c for c in load_candidates()}
    labels = load_labels()
    sec = load_json(RESULTS / "second_model.json", {})
    sel = load_json(CACHE / "mace_selection.json", [])
    picks = pd.read_csv(RESULTS / "campaign_picks.csv") if (RESULTS / "campaign_picks.csv").exists() else pd.DataFrame()
    llm = picks[(picks.strategy == "llm") & picks.reason.notna() & (picks.reason != "")] if len(picks) else picks
    tk = thr_key(load_config()["labels"]["hit_threshold"])
    rows = []
    for x in sel:
        cid = x["id"]
        c, lab, s = cands[cid], labels[cid], sec.get(cid, {})
        reason = ""
        if len(llm):
            same = llm[llm.id == cid]
            if not len(same):  # the LLM may have reached the composition through the other prototype
                same_f = [i for i in llm.id if cands[i]["reduced_formula"] == c["reduced_formula"]]
                same = llm[llm.id.isin(same_f)]
            if len(same):
                reason = str(same.iloc[0].reason)
        rows.append({"id": cid, "formula": c["display_formula"], "reduced_formula": c["reduced_formula"],
                     "prototype": c["prototype_label"], "description": c["description"],
                     "chgnet_e_hull": round(lab["e_hull"], 4),
                     "mace_e_hull": None if s.get("mace_e_hull") is None else round(s["mace_e_hull"], 4),
                     "agreed": s.get("agreed"), "novel": lab[f"novel_{tk}"], "database_status": lab["db_status"],
                     "lit_status": lab["lit_status"], "llm_reason": reason})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "top_candidates.csv", index=False)
    log.info("top_candidates.csv: %d rows", len(df))
    return df


if __name__ == "__main__":
    figures()
    print(top_table().to_string())
