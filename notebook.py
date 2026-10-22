import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium", app_title="Which battery materials are worth simulating?",
                 css_file="notebook.css")


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd

    try:
        HERE = Path(mo.notebook_dir())
    except Exception:
        HERE = Path(__file__).resolve().parent
    if str(HERE) not in sys.path:  # so the playground can import src.*
        sys.path.insert(0, str(HERE))
    ROOT = Path(os.environ.get("OSI_DATA_ROOT", HERE))
    RES = ROOT / "results"
    CACHE = ROOT / "data" / "cache"
    REPO = "https://github.com/bruce249/osi-discovery-agent"

    def load(p, default=None):
        p = Path(p)
        if not p.exists():
            return default
        with open(p) as f:
            return json.load(f)

    return CACHE, REPO, RES, load, mo, pd


@app.cell
def _(RES, load):
    summary = load(RES / "summary.json", {})
    validation = load(RES / "validation_chgnet.json", {})
    second = load(RES / "second_model.json", {})
    NAMES = {"llm_gp": "LLM + GP hybrid", "llm_strong": "LLM agent (gpt-5-mini)", "llm": "LLM agent (gpt-4.1-mini)",
             "llm_basic": "LLM, basic prompt", "gp_bo": "GP-BO (Bayesian optimisation)",
             "substitution": "Data-mined substitution", "radius": "Ionic-radius heuristic", "random": "Random"}
    return NAMES, second, summary, validation


@app.cell
def _(REPO, mo, second, summary):
    _tk = f"{summary.get('headline_threshold', 0.0):.2f}"
    _st = summary.get("strategies", {})

    def _g(name, metric, key):
        try:
            return _st[name][_tk][metric][key]
        except KeyError:
            return None

    _gp = _g("gp_bo", "novel", "AF_at_budget") or 0
    _llm_known = max((_g(n, "known", "at_budget_mean") or 0) for n in _st if n.startswith("llm")) if _st else 0
    _rand_known = _g("random", "known", "at_budget_mean") or 0
    _agree = sum(1 for v in second.values() if v.get("agreed"))
    mo.Html(f"""
<div class="hero">
  <div class="eyebrow">OSI Hackathon 2026 · AI-guided materials discovery</div>
  <h1>Which battery materials are worth simulating?</h1>
  <p class="lead">Every candidate material costs a computer simulation. We gave several decision-makers
  (an <b>LLM agent</b>, a <b>Bayesian optimiser</b>, <b>chemistry rules of thumb</b> and <b>pure chance</b>)
  the same budget of <b>{summary.get('budget', 90)} simulations</b> and asked each to find stable
  <b>solid electrolytes for lithium batteries</b> among {summary.get('pool_size', 236)} candidates.</p>
  <div class="cards">
    <div class="card"><div class="num">{_gp:.1f}×</div>
      <div class="cap">faster than random at finding <b>new</b> stable materials: the Bayesian optimiser won</div></div>
    <div class="card"><div class="num">{_llm_known:.0f} vs {_rand_known:.0f}</div>
      <div class="cap">known compounds rediscovered by the LLM vs random: it remembers the literature, which
      also pulls it away from new chemistry</div></div>
    <div class="card"><div class="num">{_agree} / {len(second)}</div>
      <div class="cap">top new candidates confirmed by a second, independent AI model</div></div>
  </div>
  <div class="links"><a href="{REPO}" target="_blank">Code on GitHub</a> ·
    <a href="{REPO}/blob/main/writeup/writeup.pdf" target="_blank">One-page write-up</a> ·
    <a href="#try-it">Try the AI copilot ↓</a></div>
</div>
""")
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## How it works"),
        mo.mermaid("""
graph LR
  A["236 candidate materials<br/>made by swapping elements<br/>in known crystals"] --> B{"A strategy picks<br/>the next 10"}
  B --> C["Simulate each one<br/>(CHGNet AI model<br/>relaxes the crystal)"]
  C --> D["Stability score<br/>(energy above hull)"]
  D -->|"results feed back"| B
"""),
        mo.hstack([
            mo.callout(mo.md("**Stable**: the material would not break apart into simpler compounds "
                             "(energy above hull ≤ 0)."), kind="success"),
            mo.callout(mo.md("**Novel**: stable *and* not found in any database (Materials Project, GNoME, "
                             "JARVIS, OQMD) or paper we searched."), kind="info"),
            mo.callout(mo.md("**Known**: already reported by experimentalists, a rediscovery rather than a "
                             "discovery."), kind="neutral"),
        ], widths="equal", gap=1),
        mo.md("""Each strategy works in rounds: it picks 10 candidates, sees their simulated stability, and picks
again until the budget is spent. Strategies never see which materials are novel or known; that is only used
to score them afterwards. All simulations were computed once on [Modal](https://modal.com) and replayed, so
every strategy faces exactly the same numbers."""),
    ])
    return


@app.cell
def _(NAMES, RES, mo, pd, summary):
    _tk = f"{summary.get('headline_threshold', 0.0):.2f}"
    _rows = []
    for _name, _s in sorted(summary.get("strategies", {}).items(),
                            key=lambda kv: -kv[1][_tk]["novel"]["at_budget_mean"]):
        _x = _s[_tk]
        _rows.append({
            "strategy": NAMES.get(_name, _name),
            "stable found": f"{_x['hit']['at_budget_mean']:.1f}",
            "novel found": f"{_x['novel']['at_budget_mean']:.1f}",
            "known rediscovered": f"{_x['known']['at_budget_mean']:.1f}",
            "speed-up vs random (novel)": f"{_x['novel']['AF_at_budget']:.2f}×" if _x["novel"]["AF_at_budget"] else "n/a",
        })
    _tot = summary.get("totals_in_pool", {}).get(_tk, {})
    mo.vstack([
        mo.md(f"""## The race: results

**Left:** new stable materials found as the simulation budget is spent (higher and earlier is better).
**Right:** rediscovery of already-known compounds. Lines are averages over repeated runs; shading is the
spread. The pool hides {_tot.get('hit', '?')} stable compositions: {_tot.get('novel', '?')} novel and
{_tot.get('known', '?')} known."""),
        mo.image(src=str(RES / "fig1_discovery.png")) if (RES / "fig1_discovery.png").exists() else mo.md("_figure missing_"),
        mo.md(f"**After {summary.get('budget', 90)} simulations** (speed-up = how many times more simulations "
              "random search needs to match the strategy):"),
        mo.ui.table(pd.DataFrame(_rows), selection=None, show_column_summaries=False),
        mo.callout(mo.md("""**What we learned.** A statistical surrogate (GP-BO) that learns *which compositions
tend to be stable* was the most efficient at finding new materials. LLM agents were the best at finding compounds
already in the literature: their prior knowledge helps rediscovery but pulls them toward known chemistry. A
stronger reasoning model (gpt-5-mini) learns from feedback faster than a small one, and pairing the LLM with the
surrogate gives the strongest LLM arm."""), kind="info"),
    ])
    return


@app.cell
def _(mo):
    cp_model = mo.ui.dropdown(["gpt-4.1-mini", "gpt-5-mini"], value="gpt-4.1-mini", label="Model")
    cp_prompt = mo.ui.dropdown({"main (scoring rule + competing phases)": "main", "basic (brief's prompt)": "basic"},
                               value="main (scoring rule + competing phases)", label="Prompt")
    cp_seed = mo.ui.number(start=0, stop=999, value=0, label="Seed")
    cp_k = mo.ui.slider(5, 10, value=10, label="Picks per round", show_value=True)
    cp_budget = mo.ui.slider(10, 150, step=10, value=60, label="Budget", show_value=True)
    cp_thr = mo.ui.slider(-0.03, 0.05, step=0.01, value=0.0, label="Stability threshold (eV/atom)", show_value=True)
    return cp_budget, cp_k, cp_model, cp_prompt, cp_seed, cp_thr


@app.cell
def _(mo):
    # "New run" bumps a shuffle offset, so the candidate list is reordered and the LLM makes fresh choices
    get_fresh, set_fresh = mo.state(0)
    copilot_runs, set_copilot_runs = mo.state({})
    return copilot_runs, get_fresh, set_copilot_runs, set_fresh


@app.cell
def _(mo, set_fresh):
    cp_new = mo.ui.button(label="↻ New run (fresh shuffle)", kind="neutral", on_click=lambda _v: set_fresh(lambda x: x + 1))
    cp_restart = mo.ui.run_button(label="⟲ Restart (same settings)", kind="neutral")
    cp_next = mo.ui.run_button(label="▶ Suggest next batch", kind="success")
    return cp_new, cp_next, cp_restart


@app.cell
def _(cp_budget, cp_k, cp_model, cp_new, cp_next, cp_prompt, cp_restart, cp_seed, cp_thr, mo):
    mo.vstack([
        mo.Html('<a id="try-it"></a>'),
        mo.md("""## Try it: the AI copilot, live

Press **▶ Suggest next batch**. The LLM reads all unevaluated candidates plus the results of its own earlier
picks, chooses the next batch and explains each choice. Then the simulated result is revealed: watch whether it
beats random search and how its reasoning changes as evidence comes in.

* **↻ New run** starts over with a freshly shuffled candidate list: a new live run.
* **⟲ Restart** starts over with the same settings: it replays the cached run instantly, free.
* gpt-5-mini is a reasoning model: smarter, but 30-90 s per round."""),
        mo.hstack([cp_model, cp_prompt, cp_seed], justify="start", gap=2),
        mo.hstack([cp_k, cp_budget, cp_thr], justify="start", gap=2),
        mo.hstack([cp_next, cp_new, cp_restart], justify="start", gap=1),
    ])
    return


@app.cell
def _(cp_budget, cp_k, cp_model, cp_prompt, cp_restart, cp_seed, cp_thr, get_fresh):
    import time as _time

    from src.playground import Campaign as _Campaign, make_strategy as _make_strategy, random_expectation as _rexp

    cp_restart  # restart (same settings) and any setting change start a new run; "New run" also bumps the shuffle
    _shuffle = int(cp_seed.value) + 1000 * get_fresh()
    copilot = _Campaign(_make_strategy("llm", _shuffle, cp_k.value, cp_budget.value, cp_thr.value,
                                       model=cp_model.value, prompt=cp_prompt.value),
                        cp_budget.value, cp_k.value, cp_thr.value, True)
    copilot.run_id = _time.time_ns()
    copilot.settings = {"model": cp_model.value, "prompt": cp_prompt.value, "shuffle": _shuffle,
                        "picks/round": cp_k.value, "threshold": cp_thr.value}
    copilot_random = _rexp(cp_thr.value, cp_budget.value)
    return copilot, copilot_random


@app.cell
def _(copilot, copilot_random, cp_next, mo, pd, set_copilot_runs):
    from src.playground import LLM_CALL_CAP as _CAP, llm_calls_used as _used, reserve_llm_call as _reserve

    _msg = None
    if cp_next.value:
        if copilot.finished:
            _msg = mo.callout(mo.md("Budget spent. Press **↻ New run** or change a setting."), kind="info")
        elif not _reserve():
            _msg = mo.callout(mo.md(f"The live-call cap for this server ({_CAP}) is reached."), kind="warn")
        else:
            try:
                with mo.status.spinner(title=f"{copilot.s.model} is choosing round {copilot.round + 1}..."):
                    copilot.step()
                _k = len(copilot.history)
                _rec = {**copilot.settings, "sims": f"{_k}/{copilot.budget}",
                        **{_lab: f"{len(copilot.found[_m])} (random ≈ {float(copilot_random[_m][_k - 1]):.1f})"
                           for _m, _lab in (("hit", "stable"), ("novel", "novel"), ("known", "known"))}}
                set_copilot_runs(lambda _d: {**_d, copilot.run_id: _rec})
            except Exception as _e:  # noqa: BLE001 - e.g. no API key when run locally
                _msg = mo.callout(mo.md(f"LLM call failed: `{str(_e)[:300]}`"), kind="danger")
    _s = copilot.settings
    _banner = mo.md(f"<small>Current run: **{_s['model']}**, {_s['prompt']} prompt, shuffle {_s['shuffle']}, "
                    f"{_s['picks/round']} picks/round, budget {copilot.budget}, threshold {_s['threshold']}</small>")
    _n = len(copilot.history)
    if _n == 0:
        _body = mo.callout(mo.md("Ready. Press **▶ Suggest next batch** to let the copilot pick."), kind="neutral")
    else:
        _hist = pd.DataFrame(copilot.picks)
        _last = _hist[_hist["round"] == copilot.round].reset_index(drop=True)
        _r = {m: float(copilot_random[m][_n - 1]) for m in ("hit", "novel", "known")}

        def _delta(m):
            d = len(copilot.found[m]) - _r[m]
            return f"{'+' if d >= 0 else ''}{d:.1f} vs random ({_r[m]:.1f})"

        _stats = mo.hstack([
            mo.stat(value=str(len(copilot.found["hit"])), label="stable found", caption=_delta("hit"),
                    direction="increase" if len(copilot.found["hit"]) >= _r["hit"] else "decrease", bordered=True),
            mo.stat(value=str(len(copilot.found["novel"])), label="novel found", caption=_delta("novel"),
                    direction="increase" if len(copilot.found["novel"]) >= _r["novel"] else "decrease", bordered=True),
            mo.stat(value=str(len(copilot.found["known"])), label="known rediscovered", caption=_delta("known"),
                    bordered=True),
            mo.stat(value=f"{_n}/{copilot.budget}", label="simulations used", caption=f"round {copilot.round}",
                    bordered=True),
        ], widths="equal", gap=1)
        _show = _last[["id", "candidate", "reason", "e_hull", "stable", "novel"]].copy()
        _show["result"] = ["✅ novel" if (s and nv) else ("✅ stable" if s else "✗ unstable")
                           for s, nv in zip(_show["stable"], _show["novel"])]
        _body = mo.vstack([
            _stats,
            mo.md(f"#### Round {copilot.round}: what the LLM picked and why"),
            mo.ui.table(_show[["id", "candidate", "reason", "e_hull", "result"]], selection=None, page_size=10,
                        show_column_summaries=False, wrapped_columns=["reason", "candidate"]),
            mo.accordion({"Full history of this run": mo.ui.table(_hist, selection=None, page_size=15,
                                                                   show_column_summaries=False)}),
        ])
    mo.vstack([x for x in (_banner, _msg, _body,
                           mo.md(f"<small>Live LLM calls on this server: {_used()}/{_CAP}</small>")) if x is not None])
    return


@app.cell
def _(copilot_runs, mo, pd):
    _runs = list(copilot_runs().values())
    _out = mo.md("")
    if _runs:
        _out = mo.vstack([
            mo.md("#### Your runs this session\nCompare runs side by side (kept until you reload the page)."),
            mo.ui.table(pd.DataFrame(_runs[::-1]), selection=None, show_column_summaries=False),
        ])
    _out
    return


@app.cell
def _(mo):
    pg_thr = mo.ui.slider(-0.03, 0.05, step=0.01, value=0.0, label="Stability threshold (eV/atom)", show_value=True)
    pg_budget = mo.ui.slider(20, 200, step=10, value=90, label="Budget (simulations)", show_value=True)
    pg_batch = mo.ui.dropdown(["5", "10", "20"], value="10", label="Batch size")
    pg_strats = mo.ui.multiselect(["random", "substitution", "radius", "gp_bo"],
                                  value=["random", "substitution", "radius", "gp_bo"], label="Strategies")
    pg_seeds = mo.ui.slider(1, 10, value=3, label="Repeats (GP-BO max 5)", show_value=True)
    pg_skip = mo.ui.checkbox(value=True, label="Skip compositions a strategy already confirmed stable")
    pg_run = mo.ui.run_button(label="▶ Run the race", kind="success")
    mo.vstack([
        mo.md("""## Playground: re-run the race with your own rules

Change the rules and rerun the non-LLM strategies in seconds (every simulation result is already computed).
*random* = pure chance; *substitution* = which element swaps happen most often in known crystals;
*radius* = swap in ions of similar size; *gp_bo* = a statistical model that learns from each round."""),
        mo.hstack([pg_thr, pg_budget, pg_batch], justify="start", gap=2),
        mo.hstack([pg_strats, pg_seeds], justify="start", gap=2),
        pg_skip,
        pg_run,
    ])
    return pg_batch, pg_budget, pg_run, pg_seeds, pg_skip, pg_strats, pg_thr


@app.cell
def _(mo, pd, pg_batch, pg_budget, pg_run, pg_seeds, pg_skip, pg_strats, pg_thr):
    mo.stop(not pg_run.value, mo.callout(mo.md("Set the rules above and press **▶ Run the race**."), kind="neutral"))
    import matplotlib.pyplot as _plt
    import numpy as _np

    from src.playground import compare as _compare

    with mo.status.spinner(title="Replaying campaigns..."):
        _res = _compare(list(pg_strats.value), pg_seeds.value, pg_budget.value, int(pg_batch.value),
                        pg_thr.value, pg_skip.value)
    _colors = {"random": "#7f7f7f", "substitution": "#1f77b4", "radius": "#8c564b", "gp_bo": "#9467bd"}
    _fig, _axes = _plt.subplots(1, 3, figsize=(13, 3.8))
    _B = _res["budget"]
    for _ax, (_m, _title) in zip(_axes, [("hit", "All stable"), ("novel", "Novel stable"),
                                         ("known", "Known rediscovered")]):
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
    mo.vstack([mo.as_html(_fig), mo.ui.table(pd.DataFrame(_res["rows"]), selection=None, show_column_summaries=False)])
    return


@app.cell
def _(RES, mo, pd):
    top = pd.read_csv(RES / "top_candidates.csv") if (RES / "top_candidates.csv").exists() else pd.DataFrame()
    _cols = [c for c in ["formula", "prototype", "chgnet_e_hull", "mace_e_hull", "agreed", "database_status",
                         "lit_status", "llm_reason"] if c in top.columns]
    mo.vstack([
        mo.md("""## New candidate materials

The most stable new compositions found, double-checked with a second, independent AI model (**MACE-MP-0**).
Negative energy = more stable than the competing compounds. *agreed* = both models put it within 0.05 eV/atom of
stability. Database and literature columns show where we looked."""),
        mo.ui.table(top[_cols], selection=None, show_column_summaries=False, wrapped_columns=["llm_reason"])
        if len(top) else mo.md("_top_candidates.csv missing_"),
    ])
    return (top,)


@app.cell
def _(mo, top):
    cand_pick = mo.ui.dropdown(options=list(top["id"]) if len(top) else [], label="Look inside a candidate's crystal",
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
                _lat = _s.lattice
                _out = mo.vstack([
                    mo.md(f"**{_s.composition.reduced_formula}**: {len(_s)} atoms, space group ≈ {_sg}, "
                          f"a={_lat.a:.2f} b={_lat.b:.2f} c={_lat.c:.2f} Å, V={_lat.volume:.0f} Å³; energy above "
                          f"hull {_ev['e_hull']:+.3f} eV/atom; competing phases: {', '.join(_ev.get('decomposition', []))}"),
                    mo.accordion({"Relaxed crystal structure (CIF file)": mo.md(f"```\n{CifWriter(_s)}\n```")}),
                ])
            except ImportError:
                _out = mo.md("_install pymatgen to view structures_")
    _out
    return


@app.cell
def _(RES, mo, pd):
    picks = pd.read_csv(RES / "campaign_picks.csv") if (RES / "campaign_picks.csv").exists() else pd.DataFrame(
        columns=["strategy", "seed", "round", "id", "description", "e_hull", "filled_randomly", "reason"])
    _opts = sorted({f"{s} / seed {d}" for s, d in zip(picks.strategy, picks.seed)})
    run_pick = mo.ui.dropdown(options=_opts, value=next((o for o in _opts if o.startswith("llm_strong")), _opts[0] if _opts else None),
                              label="Strategy / run")
    mo.vstack([mo.md("## Replay any recorded run\nEvery pick of every benchmark run, with the LLM's reasoning."),
               run_pick])
    return picks, run_pick


@app.cell
def _(mo, picks, run_pick):
    if run_pick.value:
        _s, _seed = run_pick.value.split(" / seed ")
        _df = picks[(picks.strategy == _s) & (picks.seed == int(_seed))][
            ["round", "id", "description", "e_hull", "reason"]].reset_index(drop=True)
        _df["e_hull"] = _df.e_hull.round(3)
        _out = mo.ui.table(_df, selection=None, page_size=10, show_column_summaries=False, wrapped_columns=["reason"])
    else:
        _out = mo.md("_no campaign results yet_")
    _out
    return


@app.cell
def _(CACHE, RES, load, mo, pd, summary, validation):
    candidates = load(CACHE / "candidates.json", [])
    protos = load(CACHE / "prototypes.json", {})
    _fam = {"A": "Li3MX6: swap the metal (22 metals) and halogen (Cl/Br)", "B": "Li2MX6: tetravalent metals",
            "C": "Li3M(Cl,Br)6: mixed halogens", "D": "Li3(M,M')Cl6: two metals mixed 50/50"}
    _fam_df = pd.DataFrame([{"family": f, "what": _fam[f], "candidates": sum(c["family"] == f for c in candidates)}
                            for f in "ABCD"])
    _proto_df = pd.DataFrame([{"parent structure": k, "Materials Project id": v["material_id"],
                               "space group": v["spacegroup"], "atoms": v["nsites"],
                               "built from analogue": (v.get("analogue") or {}).get("formula", "")}
                              for k, v in protos.items()])
    _mae = validation.get("mae_compounds")
    mo.vstack([
        mo.md("## Technical details"),
        mo.accordion({
            "How trustworthy are the simulations? (validation)": mo.vstack([
                mo.md(f"""We recomputed {validation.get('n_compounds', '?')} known compounds from Materials Project
with the same AI model and compared to quantum-mechanical (DFT) values: mean error
**{(_mae or 0) * 1000:.0f} meV/atom**, and every DFT-stable ternary compound comes out within 0.03 eV/atom. Each
stability score uses a convex hull built only from reference phases relaxed with the same model (never mixing
AI and DFT energies)."""),
                mo.image(src=str(RES / "fig2_parity.png"), width=480) if (RES / "fig2_parity.png").exists() else mo.md(""),
            ]),
            "How were the candidates built?": mo.vstack([
                mo.md(f"""{len(candidates)} candidates ({len({c['reduced_formula'] for c in candidates})} distinct
compositions) made by substituting elements into three parent crystals looked up on Materials Project. Mixed
candidates get up to two atomic orderings; the lower-energy one counts."""),
                mo.ui.table(_proto_df, selection=None, show_column_summaries=False),
                mo.ui.table(_fam_df, selection=None, show_column_summaries=False),
            ]),
            "Why 'stable' means e_hull ≤ 0 (not 0.03)": mo.md("""These halides sit within a few meV/atom of
splitting into simpler salts: at 0.03 eV/atom, 92% of candidates would count as stable and no strategy could be
told apart from chance. Results at 0.03 and 0.05 are in `results/summary.json`."""),
            "Limitations": mo.md("""* AI interatomic models approximate quantum mechanics; stability margins here
are a few meV/atom, comparable to their error, so these are predictions that need DFT and lab confirmation.
* Yb compounds look most stable with CHGNet but much less so with MACE (Yb reference energies are delicate in
these models); Li2CeX6 needs Ce4+, which may not survive.
* Real halide electrolytes are often partly disordered; we evaluate ordered approximants.
* Stability is necessary but not sufficient: ionic conductivity was not computed.
* The LLM's rediscovery edge may be memorised literature; novel hits are the fair comparison.
* "Novel" = not found in the databases and papers we could search (the Alexandria server was down)."""),
            "Strategies in detail": mo.md("""| strategy | how it picks |
|---|---|
| LLM agent | reads all unevaluated candidates + its own results (with competing phases), returns 10 picks with reasons |
| LLM + GP hybrid | the GP shortlists 30 by probability of stability, the LLM picks 10 |
| GP-BO | Gaussian process on composition features; picks the 10 most likely to be stable |
| Data-mined substitution | ICSD statistics of which ion substitutions occur (Hautier et al.) |
| Ionic-radius heuristic | swap in ions of similar size |
| Random | pure chance |"""),
        }),
        mo.md(f"<small>Pool {summary.get('pool_size')} candidates · budget {summary.get('budget')} · models: "
              f"CHGNet, MACE-MP-0, {summary.get('llm_model')}, {summary.get('llm_strong_model')} · compute: Modal</small>"),
    ])
    return


if __name__ == "__main__":
    app.run()
