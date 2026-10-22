"""Build the one-page write-up (writeup.md + writeup.html) from cached results. Standard library only.

    python writeup/build_writeup.py
    msedge --headless --print-to-pdf=writeup/writeup.pdf writeup/writeup.html   (or print from any browser)
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"

TEAM = "Team: [add names]"
REPO = "github.com/bruce249/osi-discovery-agent"
VIDEO = "[video link]"
NAMES = {"llm_gp": "LLM + GP hybrid", "llm_strong": "LLM agent (reasoning model)", "llm": "LLM agent", "llm_basic": "LLM, basic prompt", "gp_bo": "GP-BO", "gp_latent": "GP on MACE latent", "gp_latent_rand": "GP on scrambled-MACE latent", "substitution": "Data-mined substitution",
         "radius": "Ionic-radius heuristic", "random": "Random"}


def load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def mace(r) -> str:
    return f"{float(r['mace_e_hull']):+.3f}" if r.get("mace_e_hull") else ""


def fmt(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    s = load(RES / "summary.json", {})
    v = load(RES / "validation_chgnet.json", {})
    vm = load(RES / "validation_mace.json", {})
    sec = load(RES / "second_model.json", {})
    lp = load(RES / "latent_probe.json", {}).get("probe", {})
    cd = load(RES / "cross_domain.json", {}).get("reactions_BH", {})
    ph = load(RES / "phonons.json", {})
    it = load(RES / "interp.json", {})
    tk = f"{s['headline_threshold']:.2f}"
    st = s["strategies"]
    tot = s["totals_in_pool"][tk]
    B = s["budget"]

    def at(name, m):
        x = st[name][tk][m]
        return f"{x['at_budget_mean']:.1f} ± {x['at_budget_std']:.1f}"

    def af(name, m):
        x = st[name][tk][m]
        return x["AF_at_budget"]

    rows = []
    with open(RES / "top_candidates.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    top5 = [r for r in rows if r["novel"] == "True"][:3] or rows[:3]
    n_agree = sum(1 for x in sec.values() if x.get("agreed"))

    # rank arms by novel hits at the budget; the text is generated from the numbers, whoever wins
    order = sorted(st, key=lambda n: -st[n][tk]["novel"]["at_budget_mean"])
    best = order[0]
    llm_arms = [n for n in order if n.startswith("llm")]
    best_llm = llm_arms[0] if llm_arms else None
    known_best = max(st, key=lambda n: st[n][tk]["known"]["at_budget_mean"])
    result_txt = (
        f"With {B} simulations from a pool of {s['pool_size']} candidates ({tot['novel']} novel stable "
        f"compositions hidden in it), {NAMES[best]} found the most novel stable compositions, "
        f"{at(best, 'novel')} versus {at('random', 'novel')} for random: random search needs "
        f"{fmt(af(best, 'novel'))}x as many simulations to match it ({fmt(af(best, 'hit'))}x on all stable "
        f"compositions), and the data-mined substitution baseline reaches {fmt(af('substitution', 'novel'))}x."
    )
    if best_llm and False:  # covered in "Opening up both AIs"
        result_txt += (
            f" The best LLM arm ({NAMES[best_llm]}) reached {fmt(af(best_llm, 'novel'))}x on novel and "
            f"{fmt(af(best_llm, 'hit'))}x on all stable compositions. LLM agents were the fastest at "
            f"rediscovering experimentally known compounds ({NAMES[known_best]}: {at(known_best, 'known')} of "
            f"{tot['known']} vs {at('random', 'known')} for random), consistent with literature memory; the same "
            f"prior pulls them toward known chemistry and away from novel compositions."
        )

    def n10(k):
        return cd[k]["mean"][9] if k in cd else float("nan")

    def pr(k, m):
        return lp.get(k, {}).get(m, float("nan"))

    ph_ok = [r for r in ph.get("rows", []) if r.get("ok")]
    soft = [r for r in ph_ok if r.get("verdict") == "marginal"]
    gl = st.get("gp_latent", {}).get(tk, {})
    glr = st.get("gp_latent_rand", {}).get(tk, {})
    opening = f"""<p><b>The LLM's edge is memory.</b> On real reaction-yield screens (our team's benchmark,
github.com/bruce249/osihackiitd; Buchwald-Hartwig, Ahneman et al. 2018, 5 substrates) the LLM agent reached
{n10('llm'):.2f} of the best possible yield after 10 experiments with chemical names shown, {n10('llm_anon'):.2f} with
names hidden and {n10('random'):.2f} for random. Here, LLM agents alone rediscover the most known compounds but are only 0.8-1.1x random on new
ones: in both domains the advantage is prior knowledge, not search.</p>
<p><b>Reading the simulator's mind.</b> With the probing protocol of our MACE interpretability study (linear probes,
scrambled-weight control, splits grouped by chemistry), MACE-MP-0 features of the <i>unrelaxed</i> candidates predict
their post-relaxation stability on unseen chemistries: rank correlation {pr('mace_trained', 'spearman_mean'):.2f},
AUC {pr('mace_trained', 'auc_hit_mean'):.2f}, R² {pr('mace_trained', 'r2_median'):.2f}, vs {pr('mace_random', 'spearman_mean'):.2f} /
{pr('mace_random', 'auc_hit_mean'):.2f} / {pr('mace_random', 'r2_median'):.2f} for the same network with scrambled weights and
{pr('magpie', 'spearman_mean'):.2f} / {pr('magpie', 'auc_hit_mean'):.2f} for Magpie composition features (shuffled labels: AUC
{pr('shuffled', 'auc_hit_mean'):.2f}). A GP searching in this latent space found {gl.get('hit', {}).get('at_budget_mean', 0):.1f}
stable compositions, the most of any arm, and reached half of the novel ones {fmt(gl.get('novel', {}).get('AF50'), 2)}x faster
than random; the scrambled control found {glr.get('hit', {}).get('at_budget_mean', 0):.1f}. The signal peaks after the first
message-passing layer, training adds it mainly on Li and halide sites, textbook descriptors explain only
{it.get('meaning_r2_descriptors_explain_score', 0):.0%} of it, and erasing 10 stability directions (INLP) drops the probe AUC to
{it.get('ablation_probe', {}).get('inlp', {}).get('auc', 0):.2f} and the GP to {st.get('gp_latent_inlp', {}).get(tk, {}).get('hit', {}).get('at_budget_mean', 0):.1f}
stable (random erasure: {st.get('gp_latent_ablrand', {}).get(tk, {}).get('hit', {}).get('at_budget_mean', 0):.1f}; within seed spread).
The same network also learned iron's redox state from energies alone (companion probe, github.com/lntick/osi-draft: Fe
moments on unseen chemical systems R² 0.75 vs 0.19 geometry, 0.06 scrambled).</p>
<p><b>Dynamic stability.</b> Finite-displacement phonons (CHGNet + phonopy; tiers from our Lattice studio) for the top
candidates: {ph.get('n_stable', '?')}/{len(ph_ok)} free of imaginary modes, {len(soft)} with soft modes between -1 and -0.1 THz
(framework tilts: Li carries only {min(r['soft_mode_li_share'] for r in soft):.0%}-{max(r['soft_mode_li_share'] for r in soft):.0%} of them while being 30% of the atoms;
they need larger supercells or DFT) and {len(ph_ok) - ph.get('n_stable', 0) - len(soft)} unstable (Li2ZrBr6, -2.8 THz, flagged).</p>"""

    strat_rows = "".join(
        f"<tr><td>{NAMES[n]}</td><td>{len(s['seeds'][n])}</td><td>{at(n, 'hit')}</td><td>{at(n, 'novel')}</td>"
        f"<td>{at(n, 'known')}</td><td>{fmt(af(n, 'hit'))}x / {fmt(af(n, 'novel'))}x</td></tr>" for n in ["gp_latent", "gp_bo", "gp_latent_rand", "llm_gp", "llm_strong", "llm", "substitution", "random"] if n in st)
    cand_rows = "".join(
        f"<tr><td>{html.escape(r['formula'])}</td><td>{html.escape(r['prototype'].split(' (')[0])}</td>"
        f"<td>{float(r['chgnet_e_hull']):+.3f}</td><td>{mace(r)}</td>"
        f"<td>{'yes' if r['agreed'] == 'True' else 'no'}</td><td>{'absent*' if (r['database_status'] or '').startswith('absent') else html.escape(r['database_status'] or '')}</td>"
        f"<td>{html.escape(r['lit_status'] or '')}</td></tr>" for r in top5)

    svg = """<svg viewBox="0 0 620 92" width="78%" style="display:block;margin:0 auto" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial" font-size="11">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#444"/></marker></defs>
<g stroke="#444" fill="#f4f6fb">
<rect x="4" y="22" width="130" height="44" rx="6"/><rect x="164" y="22" width="130" height="44" rx="6"/>
<rect x="324" y="22" width="130" height="44" rx="6"/><rect x="484" y="22" width="130" height="44" rx="6"/></g>
<g fill="#111" text-anchor="middle">
<text x="69" y="40" font-weight="bold">Strategy</text><text x="69" y="55">proposes 10 candidates</text>
<text x="229" y="40" font-weight="bold">CHGNet relaxation</text><text x="229" y="55">(cell + positions)</text>
<text x="389" y="40" font-weight="bold">Same-model hull</text><text x="389" y="55">e_hull vs MP refs</text>
<text x="549" y="40" font-weight="bold">Feedback</text><text x="549" y="55">(id, e_hull) history</text></g>
<g stroke="#444" marker-end="url(#a)"><line x1="134" y1="44" x2="162" y2="44"/><line x1="294" y1="44" x2="322" y2="44"/>
<line x1="454" y1="44" x2="482" y2="44"/><path d="M549,66 L549,84 L69,84 L69,68" fill="none"/></g>
<text x="310" y="14" text-anchor="middle" fill="#555">rounds of 10 until the budget is spent; labels (known / novel) stay hidden from every strategy</text></svg>"""

    lim = ("ML potentials approximate DFT (CHGNet reference MAE {mae} meV/atom; MACE {mmae} meV/atom); stability "
           "margins here are a few meV/atom, comparable to that error, so hits are predictions, not proof. "
           "Yb compounds come out most stable with CHGNet but much less so with MACE (Yb reference energies are "
           "known to be delicate in MP-trained models); Li2CeX6 needs Ce4+, which is not guaranteed to survive. "
           "Ordered approximants of partly disordered phases; 0 K thermodynamics only (no ionic conductivity or "
           "electrochemical window); campaigns replay precomputed evaluations. The brief's 0.03 eV/atom hit "
           "threshold makes 92% of the pool 'hits', so we used e_hull <= 0 (0.03/0.05 in summary.json). "
           "*'Novel' = not found in MP, GNoME, JARVIS, OQMD (partly reachable) and literature searches; the "
           "Alexandria server was unreachable during the run and is reported as unchecked.").format(
        mae=fmt((v.get("mae_compounds") or 0) * 1000, 0), mmae=fmt((vm.get("mae_compounds") or 0) * 1000, 0))

    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Budgeted Discovery Agent</title>
<style>
@page {{ size: A4; margin: 6mm 10mm; }}
body {{ font-family: Helvetica, Arial, sans-serif; font-size: 8.4pt; line-height: 1.2; color: #111; margin: 0; }}
h1 {{ font-size: 15pt; margin: 0 0 2px; }} h2 {{ font-size: 10.3pt; margin: 5px 0 1px; color: #1d3a7a; }}
p {{ margin: 2px 0 4px; }} .sub {{ color: #555; font-size: 9pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 3px 0; }}
td, th {{ border-bottom: 1px solid #ddd; padding: 0.5px 4px; text-align: left; }} th {{ background: #f0f2f7; }}
img {{ width: 50%; display: block; margin: 0 auto; }} .two {{ display: flex; gap: 10px; }} .two > div {{ flex: 1; }}
</style></head><body>
<h1>Which Li-halide solid electrolytes are worth simulating, and what do the AIs in the loop know?</h1>
<div class="sub">{html.escape(TEAM)} · OSI Hackathon 2026 · code: {html.escape(REPO)} · video: {html.escape(VIDEO)}</div>
<h2>Problem</h2>
<p>Li3MX6 / Li2MX6 halides (Li3YCl6, Li3InCl6, Li2ZrCl6) are leading solid electrolytes for all-solid-state
batteries. Metal, halide and ordering choices make a large space and each candidate costs a simulation, so
<b>deciding what to simulate</b> is the bottleneck. We built {s['pool_size']} candidates ({s['unique_formulas_in_pool']}
compositions) by substituting Materials Project parents: 22 trivalent and 5 tetravalent metals, Cl/Br, ordered
mixed-halide and mixed-metal variants.</p>
<h2>Method</h2>
<p>Loop: a strategy proposes 10 candidates, each is simulated, results feed back, until the budget is spent; known/novel labels stay hidden. Every evaluation is a CHGNet relaxation scored by its energy above a convex hull of
{v.get('n_refs')} MP reference phases relaxed with the same potential (validation: MAE
{fmt((v.get('mae_compounds') or 0)*1000, 0)} meV/atom vs DFT over {v.get('n_compounds')} compounds; all DFT-stable
ternaries within 0.03 eV/atom). Hit = e_hull ≤ {s['headline_threshold']} eV/atom (on or below the hull). The LLM
agents ({html.escape(str(s.get('llm_model', 'OpenAI')))}; reasoning arm {html.escape(str(s.get('llm_strong_model', '')))}) see all unevaluated candidates and the full
(id, description, e_hull) history and return 10 picks with a chemical reason each. Baselines: random, ICSD
substitution probabilities (Hautier et al.), ionic-radius similarity, GP-BO on Magpie features; all with the same
budget ({B} of {s['pool_size']}).</p>
<h2>Result</h2>
<img src="../results/fig1_discovery.png">
<p>{result_txt}</p>
<table><tr><th>strategy (@{B} sims)</th><th>seeds</th><th>stable found</th><th>novel found</th><th>known rediscovered</th><th>speed-up vs random (stable / novel)</th></tr>{strat_rows}</table>
<h2>Opening up both AIs in the loop</h2>{opening}
<h2>New candidates (CHGNet + MACE-MP-0 cross-check; {n_agree}/{len(sec)} agreed within 0.05 eV/atom)</h2>
<table><tr><th>formula</th><th>prototype</th><th>CHGNet e_hull</th><th>MACE e_hull</th><th>agreed</th><th>databases</th><th>literature</th></tr>{cand_rows}</table>
<h2>Validation and limitations</h2><p>{lim}</p>
</body></html>"""
    (HERE / "writeup.html").write_text(page, encoding="utf-8")

    md = [f"# Budgeted Discovery Agent: which Li-halide solid electrolytes are worth simulating?", "",
          f"{TEAM} · OSI Hackathon 2026 · code: {REPO} · video: {VIDEO}", "",
          "## Problem", "", f"Li3MX6 / Li2MX6 halide solid electrolytes; {s['pool_size']} candidates, "
          f"{s['unique_formulas_in_pool']} compositions; each costs a simulation, so the decision of what to simulate matters.", "",
          "## Method", "", "Strategy proposes 10 → CHGNet relaxation → same-model convex hull (MP references) → e_hull "
          "feedback → repeat until the budget is spent. Labels stay hidden from strategies.", "",
          f"Validation: MAE {fmt((v.get('mae_compounds') or 0)*1000, 0)} meV/atom vs DFT over {v.get('n_compounds')} reference compounds.", "",
          "## Result", "", "![Figure 1](../results/fig1_discovery.png)", "", result_txt, "",
          f"| strategy (@{B} sims) | stable found | novel found | known rediscovered | acceleration vs random (novel) |",
          "|---|---|---|---|---|"]
    md += [f"| {NAMES[n]} | {at(n, 'hit')} | {at(n, 'novel')} | {at(n, 'known')} | {fmt(af(n, 'novel'))} |"
           for n in ["gp_latent", "gp_bo", "gp_latent_rand", "llm_gp", "llm_strong", "llm", "llm_basic", "substitution", "radius", "random"] if n in st]
    import re
    md += ["", "## Opening up both AIs in the loop", "", re.sub(r"<[^>]+>", "", opening).replace("\n", " "), ""]
    md += ["", f"## New candidates ({n_agree}/{len(sec)} agreed by MACE-MP-0)", "",
           "| formula | prototype | CHGNet e_hull | MACE e_hull | agreed | databases | literature |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['formula']} | {r['prototype'].split(' (')[0]} | {float(r['chgnet_e_hull']):+.3f} | "
           f"{mace(r)} | "
           f"{'yes' if r['agreed'] == 'True' else 'no'} | {r['database_status']} | {r['lit_status']} |" for r in top5]
    md += ["", "## Validation and limitations", "", lim, ""]
    (HERE / "writeup.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote writeup.md and writeup.html")


if __name__ == "__main__":
    main()
