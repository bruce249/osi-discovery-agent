import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", app_title="Budgeted Discovery Agent: Li-halide solid electrolytes")


@app.cell
def _():
    import json
    import os
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    try:
        HERE = Path(mo.notebook_dir())
    except Exception:
        HERE = Path(__file__).resolve().parent
    import sys

    if str(HERE) not in sys.path:  # so the playground can import src.*
        sys.path.insert(0, str(HERE))
    ROOT = Path(os.environ.get("OSI_DATA_ROOT", HERE))
    RES = ROOT / "results"
    CACHE = ROOT / "data" / "cache"

    def load(p, default=None):
        p = Path(p)
        if not p.exists():
            return default
        with open(p) as f:
            return json.load(f)

    return CACHE, Path, RES, load, mo, pd


@app.cell
def _(RES, load, mo):
    summary = load(RES / "summary.json", {})
    validation = load(RES / "validation_chgnet.json", {})
    smoke = load(RES / "smoke.json", {})
    mo.md(
        """
        # Budgeted Discovery Agent for Li-halide solid electrolytes

        ## 1. Problem

        Solid electrolytes are the enabling material for safer, denser all-solid-state lithium batteries. Lithium
        halides of the **Li3MX6** (Li3YCl6, Li3InCl6) and **Li2MX6** (Li2ZrCl6) families combine high Li-ion
        conductivity with good oxidative stability, and the space of possible metal / halide / ordering
        combinations is large. Every candidate costs a simulation, so the real question is **which candidates are
        worth simulating** under a fixed budget.

        We compare an **LLM agent that reasons over results as they come in** with **random search**, a
        **data-mined substitution baseline** (Hautier et al., ICSD substitution probabilities), an ionic-radius
        heuristic and **GP Bayesian optimisation**. All strategies spend the same budget on the same pool; each
        evaluation is a CHGNet relaxation scored by its energy above a CHGNet convex hull.
        """
    )
    return summary, validation


@app.cell
def _(CACHE, load, mo, pd):
    candidates = load(CACHE / "candidates.json", [])
    protos = load(CACHE / "prototypes.json", {})
    fam_names = {"A": "Li3MX6, full M/X substitution", "B": "Li2MX6 (tetravalent M)",
                 "C": "Li3M(Cl,Br)6 ordered mixed anion", "D": "Li3(M,M')Cl6 ordered mixed cation"}
    fam_df = pd.DataFrame(
        [{"family": f, "description": fam_names[f],
          "candidates": sum(c["family"] == f for c in candidates),
          "orderings relaxed": sum(len(c["orderings"]) for c in candidates if c["family"] == f),
          "max atoms/cell": max([o["natoms"] for c in candidates if c["family"] == f for o in c["orderings"]] or [0])}
         for f in "ABCD"]
    )
    proto_df = pd.DataFrame(
        [{"prototype": k, "MP id": v["material_id"], "space group": v["spacegroup"], "atoms": v["nsites"],
          "MP e_hull (eV/atom)": v["e_hull"]} for k, v in protos.items()]
    )
    mo.vstack([
        mo.md(f"""## 2. Candidate space

A candidate is a (composition, prototype) pair made by substituting sites of a Materials Project parent
(looked up by formula at run time). Partially substituted candidates get up to two symmetry-distinct
orderings; a candidate's value is its lowest-energy ordering. **{len(candidates)} candidates**,
**{len({c['reduced_formula'] for c in candidates})} distinct compositions**."""),
        mo.ui.table(proto_df, selection=None, label="Prototypes"),
        mo.ui.table(fam_df, selection=None, label="Candidate families"),
    ])
    return (candidates,)


@app.cell
def _(RES, mo, summary, validation):
    _mae = validation.get("mae_compounds")
    _txt = (f"MAE = **{_mae*1000:.0f} meV/atom** over {validation.get('n_compounds')} compounds, "
            f"Spearman rho = {validation.get('spearman_compounds', 0):.2f}; "
            f"{validation.get('stable_ternaries_above_0.1', '?')} DFT-stable ternaries sit above 0.1 eV/atom."
            if _mae is not None else "validation not available")
    mo.vstack([
        mo.md(f"""## 3. Evaluator and validation

Each evaluation = CHGNet relaxation (FIRE, cell + positions, fmax 0.1 eV/Å, ≤300 steps) → energy above a
convex hull built **only from Materials Project reference phases re-relaxed with the same potential** (never
mixing ML and DFT energies). Campaigns replay these precomputed evaluations, the standard way to benchmark
Bayesian-optimisation style strategies: every strategy sees identical numbers.

**Figure 2.** CHGNet vs DFT (MP, GGA/GGA+U) energy above hull for all reference phases. {_txt}
Pool used by the campaigns: **{summary.get('pool_size', '?')} candidates**, budget
**{summary.get('budget', '?')} evaluations** in rounds of {summary.get('batch_size', '?')}."""),
        mo.image(src=str(RES / "fig2_parity.png"), width=520) if (RES / "fig2_parity.png").exists() else mo.md("_figure missing_"),
    ])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 4. Strategies

        | strategy | how it picks the next 10 | seeds |
        |---|---|---|
        | **LLM agent** (gpt-4.1-mini) | prompt with every unevaluated candidate (`id | description`, seed-shuffled), the full history (`id | description | e_hull | hit | competing phases`) and the scoring rule; returns 10 ids + a one-line chemical reason as strict JSON | 3 |
        | **LLM agent, reasoning model** (gpt-5-mini) | same prompt and feedback | 3 |
        | LLM, basic prompt (ablation) | the brief's original prompt, e_hull-only feedback | 3 |
        | GP-BO | Matern GP on Magpie composition features + prototype/family one-hot; picks the 10 highest posterior P(e_hull ≤ threshold) | 5 |
        | Data-mined substitution | rank once by ICSD-mined P(new ion \\| parent ion) (pymatgen `SubstitutionProbability`) | 10 |
        | Ionic-radius heuristic | rank once by Shannon-radius mismatch to the parent ions | 10 |
        | Random | uniform without replacement | 10 |

        Strategies only see the e_hull of candidates they chose (plus, for the LLM agents, the competing phases the
        hull reports for those same candidates). Ground-truth labels (known / novel) are hidden and used only for
        scoring. A composition counts once, however many prototypes of it were evaluated, and every strategy drops
        compositions its own history already shows as stable.

        **Hit threshold.** The brief proposed e_hull ≤ 0.03 eV/atom, but Li3MX6 halides sit within a few meV/atom of
        LiX + MX3: 92% of the pool is within 0.03 eV/atom, so every strategy "wins". We use **e_hull ≤ 0**
        (on or below the same-model hull: 84 of 123 compositions) as the headline and report 0.03 and 0.05 in
        `results/summary.json`.
        """
    )
    return


@app.cell
def _(RES, mo, pd, summary):
    _tk = f"{summary.get('headline_threshold', 0.03):.2f}"
    _rows = []
    for _name, _s in summary.get("strategies", {}).items():
        _st = _s.get(_tk, {})
        def _f(m, k):
            v = _st.get(m, {}).get(k)
            return None if v is None else round(v, 2)
        _rows.append({"strategy": _name,
                      "stable found @budget": f"{_st['hit']['at_budget_mean']:.1f} ± {_st['hit']['at_budget_std']:.1f}",
                      "novel found @budget": f"{_st['novel']['at_budget_mean']:.1f} ± {_st['novel']['at_budget_std']:.1f}",
                      "known rediscovered @budget": f"{_st['known']['at_budget_mean']:.1f} ± {_st['known']['at_budget_std']:.1f}",
                      "AF50 (stable)": _f("hit", "AF50"), "AF50 (novel)": _f("novel", "AF50"),
                      "AF@budget (stable)": _f("hit", "AF_at_budget"), "AF@budget (novel)": _f("novel", "AF_at_budget")})
    _tot = summary.get("totals_in_pool", {}).get(_tk, {})
    mo.vstack([
        mo.md(f"""## 5. Results

**Figure 1.** Left: new stable candidates (e_hull ≤ {summary.get('headline_threshold')} eV/atom, absent from
databases and literature) found vs simulations used: the fair test, since no strategy can have memorised them.
Right: rediscovery of experimentally known compounds (the LLM may benefit from memory here). Mean ± std over
seeds. The pool holds {_tot.get('hit', '?')} stable compositions, {_tot.get('novel', '?')} of them novel and
{_tot.get('known', '?')} experimentally known."""),
        mo.image(src=str(RES / "fig1_discovery.png")) if (RES / "fig1_discovery.png").exists() else mo.md("_figure missing_"),
        mo.md("""**Acceleration factors.** AF50 = evaluations random needs to find 50% of the pool's hits ÷ evaluations
the strategy needs (blank if not reached within the budget). AF@budget = evaluations random needs to match the
strategy's count at the budget ÷ budget."""),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ])
    return


@app.cell
def _(RES, mo, pd):
    picks = pd.read_csv(RES / "campaign_picks.csv") if (RES / "campaign_picks.csv").exists() else pd.DataFrame(
        columns=["strategy", "seed", "round", "id", "description", "e_hull", "filled_randomly", "reason"])
    _opts = sorted({f"{s} / seed {d}" for s, d in zip(picks.strategy, picks.seed)})
    run_pick = mo.ui.dropdown(options=_opts, value=next((o for o in _opts if o.startswith("llm")), _opts[0] if _opts else None),
                              label="Strategy / seed")
    mo.vstack([mo.md("### Round-by-round picks (with the LLM's reasons)"), run_pick])
    return picks, run_pick


@app.cell
def _(mo, picks, run_pick):
    if run_pick.value:
        _s, _seed = run_pick.value.split(" / seed ")
        _df = picks[(picks.strategy == _s) & (picks.seed == int(_seed))][
            ["round", "id", "description", "e_hull", "reason", "filled_randomly"]].copy()
        _df["e_hull"] = _df.e_hull.round(3)
        _out = mo.ui.table(_df, selection=None, page_size=20)
    else:
        _out = mo.md("_no campaign results yet_")
    _out
    return


@app.cell
def _(RES, mo, pd):
    top = pd.read_csv(RES / "top_candidates.csv") if (RES / "top_candidates.csv").exists() else pd.DataFrame()
    second = (RES / "fig2_parity_mace.png")
    mo.vstack([
        mo.md("""## 6. New candidates

Top hits by CHGNet e_hull, re-relaxed with a second universal potential (**MACE-MP-0**, own MACE reference
hull). *agreed* = both potentials put the candidate within 0.05 eV/atom of the hull. Database status from
Materials Project + OPTIMADE (Alexandria, OQMD, JARVIS); literature status from OpenAlex, arXiv and Semantic
Scholar searches by formula and by description."""),
        mo.ui.table(top, selection=None) if len(top) else mo.md("_top_candidates.csv missing_"),
        mo.image(src=str(second), width=420) if second.exists() else mo.md(""),
    ])
    return (top,)


@app.cell
def _(mo, top):
    cand_pick = mo.ui.dropdown(options=list(top["id"]) if len(top) else [], label="Structure viewer: candidate",
                               value=(list(top["id"])[0] if len(top) else None))
    cand_pick
    return (cand_pick,)


@app.cell
def _(CACHE, RES, cand_pick, load, mo):
    _out = mo.md("")
    if cand_pick.value:
        _ev = load(RES / "evaluations_chgnet.json", {}).get(cand_pick.value, {})
        _res = load(CACHE / "mlip" / "chgnet" / f"{_ev.get('best_ordering')}.json", {})
        if _res.get("final_structure"):
            try:
                from pymatgen.core import Structure
                from pymatgen.io.cif import CifWriter
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

                _s = Structure.from_dict(_res["final_structure"])
                _sg = SpacegroupAnalyzer(_s, symprec=0.1).get_space_group_symbol()
                _cif = str(CifWriter(_s))
                _lat = _s.lattice
                _out = mo.vstack([
                    mo.md(f"**{_s.composition.reduced_formula}**, {len(_s)} atoms, space group ≈ {_sg} (symprec 0.1), "
                          f"a={_lat.a:.3f} b={_lat.b:.3f} c={_lat.c:.3f} Å, α={_lat.alpha:.1f} β={_lat.beta:.1f} "
                          f"γ={_lat.gamma:.1f}°, V={_lat.volume:.1f} Å³, e_hull={_ev['e_hull']:.3f} eV/atom; "
                          f"decomposes to {', '.join(_ev.get('decomposition', []))}"),
                    mo.accordion({"CHGNet-relaxed CIF": mo.md(f"```\n{_cif}\n```")}),
                ])
            except ImportError:
                _out = mo.md("_install pymatgen to view structures_")
    _out
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 7. Limitations

        * ML potentials approximate DFT; every candidate here needs DFT and experimental confirmation.
        * Many real halide electrolytes are partly disordered; we evaluate ordered approximants (≤2 orderings,
          ≤80 atoms) and keep the lowest-energy one, so mixing energies are upper bounds.
        * Thermodynamic stability (0 K, no entropy) is necessary but not sufficient; ionic conductivity and the
          electrochemical window were not computed.
        * Campaigns replay precomputed evaluations (standard benchmarking practice), so strategy cost is counted in
          simulations, not wall-clock.
        * The LLM has read the literature: its rediscovery advantage may be memory. The novel-hit curve is the fair
          comparison.
        * Novelty means "not found" in MP, Alexandria, OQMD, JARVIS and our literature searches, not proof that
          nobody has made it; solid-solution reports count as "possibly reported".
        """
    )
    return


@app.cell
def _(mo):
    pg_thr = mo.ui.slider(-0.03, 0.05, step=0.01, value=0.0, label="Hit threshold (eV/atom)", show_value=True)
    pg_budget = mo.ui.slider(20, 200, step=10, value=90, label="Budget (simulations)", show_value=True)
    pg_batch = mo.ui.dropdown(["5", "10", "20"], value="10", label="Batch size")
    pg_strats = mo.ui.multiselect(["random", "substitution", "radius", "gp_bo"],
                                  value=["random", "substitution", "radius", "gp_bo"], label="Strategies")
    pg_seeds = mo.ui.slider(1, 10, value=3, label="Seeds per strategy (GP-BO max 5)", show_value=True)
    pg_skip = mo.ui.checkbox(value=True, label="Drop compositions a strategy has already confirmed stable")
    pg_run = mo.ui.run_button(label="Run campaigns")
    mo.vstack([
        mo.md("""## 8. Playground: re-run the race with your own settings

Every evaluation is a lookup in the cached CHGNet results, so campaigns replay in seconds. Random always runs
over the whole pool as the speed-up reference (speed-up = simulations random needs to match the strategy's
count at the budget, divided by the budget). Novelty was checked for every composition with
e_hull ≤ 0.05 eV/atom, which is why the threshold stops there."""),
        mo.hstack([pg_thr, pg_budget, pg_batch], justify="start", gap=2),
        mo.hstack([pg_strats, pg_seeds], justify="start", gap=2),
        pg_skip,
        pg_run,
    ])
    return pg_batch, pg_budget, pg_run, pg_seeds, pg_skip, pg_strats, pg_thr


@app.cell
def _(mo, pd, pg_batch, pg_budget, pg_run, pg_seeds, pg_skip, pg_strats, pg_thr):
    mo.stop(not pg_run.value, mo.md("_Set the knobs above and press **Run campaigns**._"))
    import matplotlib.pyplot as _plt
    import numpy as _np

    from src.playground import compare as _compare

    with mo.status.spinner(title="Replaying campaigns..."):
        _res = _compare(list(pg_strats.value), pg_seeds.value, pg_budget.value, int(pg_batch.value),
                        pg_thr.value, pg_skip.value)
    _colors = {"random": "#7f7f7f", "substitution": "#1f77b4", "radius": "#8c564b", "gp_bo": "#9467bd"}
    _fig, _axes = _plt.subplots(1, 3, figsize=(13, 3.8))
    _B = _res["budget"]
    for _ax, (_m, _title) in zip(_axes, [("hit", "All stable compositions"), ("novel", "Novel stable compositions"),
                                         ("known", "Known compounds rediscovered")]):
        for _name, _rs in _res["runs"].items():
            if _name == "random" and "random" not in pg_strats.value:
                continue
            _a = _np.array([_r.curves[_m][:_B] for _r in _rs], dtype=float)
            _x = _np.arange(1, _a.shape[1] + 1)
            _ax.plot(_x, _a.mean(0), color=_colors[_name], label=_name, lw=1.8)
            _ax.fill_between(_x, _a.mean(0) - _a.std(0), _a.mean(0) + _a.std(0), color=_colors[_name], alpha=0.15, lw=0)
        _ax.axhline(_res["totals"][_m], color="k", ls=":", lw=1)
        _ax.set_title(f"{_title} (pool: {_res['totals'][_m]})", fontsize=10)
        _ax.set_xlabel("Simulations used")
        _ax.set_xlim(0, _B)
        _ax.grid(alpha=0.3)
    _axes[0].legend(fontsize=8, frameon=False)
    _fig.tight_layout()
    mo.vstack([mo.as_html(_fig), mo.ui.table(pd.DataFrame(_res["rows"]), selection=None)])
    return


@app.cell
def _(mo):
    cp_model = mo.ui.dropdown(["gpt-4.1-mini", "gpt-5-mini"], value="gpt-4.1-mini", label="Model")
    cp_prompt = mo.ui.dropdown({"main (scoring rule + competing phases)": "main", "basic (brief's prompt)": "basic"},
                               value="main (scoring rule + competing phases)", label="Prompt")
    cp_seed = mo.ui.number(start=0, stop=999, value=0, label="Seed (shuffles the candidate list)")
    cp_k = mo.ui.slider(5, 10, value=10, label="Picks per round", show_value=True)
    cp_budget = mo.ui.slider(10, 150, step=10, value=60, label="Budget", show_value=True)
    cp_thr = mo.ui.slider(-0.03, 0.05, step=0.01, value=0.0, label="Hit threshold (eV/atom)", show_value=True)
    cp_reset = mo.ui.run_button(label="Start a new copilot run")
    cp_next = mo.ui.run_button(label="Suggest next batch")
    mo.vstack([
        mo.md("""## 9. LLM copilot, live

The LLM sees every unevaluated candidate and the results of its own earlier picks, chooses the next batch and
explains each choice; the CHGNet result is then revealed. gpt-5-mini is a reasoning model and takes 30-90 s per
round. Calls are cached (identical prompts are free) and capped per server."""),
        mo.hstack([cp_model, cp_prompt, cp_seed], justify="start", gap=2),
        mo.hstack([cp_k, cp_budget, cp_thr], justify="start", gap=2),
        mo.hstack([cp_reset, cp_next], justify="start", gap=1),
    ])
    return cp_budget, cp_k, cp_model, cp_next, cp_prompt, cp_reset, cp_seed, cp_thr


@app.cell
def _(cp_budget, cp_k, cp_model, cp_prompt, cp_reset, cp_seed, cp_thr):
    from src.playground import Campaign as _Campaign, make_strategy as _make_strategy, random_expectation as _rexp

    cp_reset  # a new run starts whenever this button or any setting changes
    copilot = _Campaign(_make_strategy("llm", int(cp_seed.value), cp_k.value, cp_budget.value, cp_thr.value,
                                       model=cp_model.value, prompt=cp_prompt.value),
                        cp_budget.value, cp_k.value, cp_thr.value, True)
    copilot_random = _rexp(cp_thr.value, cp_budget.value)
    return copilot, copilot_random


@app.cell
def _(copilot, copilot_random, cp_next, mo, pd):
    from src.playground import LLM_CALL_CAP as _CAP, llm_calls_used as _used, reserve_llm_call as _reserve

    _msg = None
    if cp_next.value:
        if copilot.finished:
            _msg = mo.callout(mo.md("Budget spent. Change a setting or press **Start a new copilot run**."), kind="info")
        elif not _reserve():
            _msg = mo.callout(mo.md(f"The live-call cap for this server ({_CAP}) is reached."), kind="warn")
        else:
            try:
                with mo.status.spinner(title=f"{copilot.s.model} is choosing round {copilot.round + 1}..."):
                    copilot.step()
            except Exception as _e:  # noqa: BLE001 - e.g. no API key when run locally
                _msg = mo.callout(mo.md(f"LLM call failed: `{str(_e)[:300]}`"), kind="danger")
    _n = len(copilot.history)
    if _n == 0:
        _body = mo.md("_Press **Suggest next batch** to let the copilot pick._")
    else:
        _hist = pd.DataFrame(copilot.picks)
        _last = _hist[_hist["round"] == copilot.round]
        _r = {m: float(copilot_random[m][_n - 1]) for m in ("hit", "novel", "known")}
        _stats = mo.hstack([
            mo.stat(value=str(len(copilot.found["hit"])), label="stable compositions",
                    caption=f"random ≈ {_r['hit']:.1f} after {_n} sims"),
            mo.stat(value=str(len(copilot.found["novel"])), label="novel stable",
                    caption=f"random ≈ {_r['novel']:.1f}"),
            mo.stat(value=str(len(copilot.found["known"])), label="known rediscovered",
                    caption=f"random ≈ {_r['known']:.1f}"),
            mo.stat(value=f"{_n}/{copilot.budget}", label="simulations used"),
        ], justify="start", gap=2)
        _body = mo.vstack([
            _stats,
            mo.md(f"**Round {copilot.round}: picks, reasons, and revealed e_hull**"),
            mo.ui.table(_last[["id", "candidate", "reason", "e_hull", "stable", "novel"]], selection=None,
                        page_size=10),
            mo.accordion({"Full history": mo.ui.table(_hist, selection=None, page_size=15)}),
        ])
    mo.vstack([x for x in (_msg, _body, mo.md(f"<small>live LLM calls on this server: {_used()}/{_CAP}</small>")) if x is not None])
    return


if __name__ == "__main__":
    app.run()
