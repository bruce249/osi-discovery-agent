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


if __name__ == "__main__":
    app.run()
