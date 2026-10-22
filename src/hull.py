"""MLIP-only phase diagrams and energies above hull.

Every phase diagram is built from reference phases relaxed with the SAME potential; candidates are never
added to the diagram. Negative e_hull means the candidate would be a new ground state.
"""
from __future__ import annotations

import math

import numpy as np
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Structure

from .mlip import all_results
from .mp_fetch import load_references
from .utils import RESULTS, dump_json, get_logger, load_config, load_json

log = get_logger("hull")


class MLIPHull:
    def __init__(self, model: str = "chgnet", ref_ids: set[str] | None = None):
        self.model = model
        self.results = all_results(model)
        self.refs = load_references()
        self.entries: dict[str, PDEntry] = {}
        for mid, r in self.refs.items():
            if ref_ids is not None and mid not in ref_ids:
                continue
            res = self.results.get(r["hash"])
            if res and res.get("ok"):
                comp = Composition(Structure.from_dict(r["structure"]).composition.element_composition)
                self.entries[mid] = PDEntry(comp, res["e_per_atom"] * comp.num_atoms, name=mid)
        self._pds: dict[tuple, PhaseDiagram | None] = {}

    def pd(self, elements) -> PhaseDiagram | None:
        key = tuple(sorted(set(elements)))
        if key not in self._pds:
            ents = [e for e in self.entries.values() if {el.symbol for el in e.composition.elements} <= set(key)]
            try:
                self._pds[key] = PhaseDiagram(ents)
            except Exception as ex:  # noqa: BLE001 - e.g. a missing elemental reference
                log.warning("phase diagram %s failed: %s", key, ex)
                self._pds[key] = None
        return self._pds[key]

    def e_above_hull(self, comp: Composition, e_per_atom: float) -> tuple[float | None, list[str]]:
        comp = Composition(comp).element_composition
        pd = self.pd([e.symbol for e in comp.elements])
        if pd is None:
            return None, []
        decomp, e = pd.get_decomp_and_e_above_hull(PDEntry(comp, e_per_atom * comp.num_atoms), allow_negative=True)
        return float(e), sorted(d.composition.reduced_formula for d in decomp)


def evaluate_candidates(model: str = "chgnet", hull: MLIPHull | None = None,
                        candidates: list[dict] | None = None) -> dict[str, dict]:
    """Candidate value = lowest-energy ordering; e_hull against the MLIP reference hull."""
    from .candidates import load_candidates

    hull = hull or MLIPHull(model)
    out = {}
    for c in candidates or load_candidates():
        ords = []
        for o in c["orderings"]:
            r = hull.results.get(o["hash"])
            if r and r.get("ok"):
                ords.append((r["e_per_atom"], o["hash"], r.get("converged", False)))
        rec = {"id": c["id"], "ok": False, "e_hull": None, "n_orderings_ok": len(ords),
               "n_orderings": len(c["orderings"])}
        if ords:
            e_pa, h, conv = min(ords)
            comp = Structure.from_dict(c["orderings"][0]["structure"]).composition
            e_hull, decomp = hull.e_above_hull(comp, e_pa)
            if e_hull is not None and math.isfinite(e_hull):
                rec.update(ok=True, e_hull=e_hull, e_per_atom=e_pa, best_ordering=h, converged=conv,
                           decomposition=decomp, ordering_energies=sorted(o[0] for o in ords))
        out[c["id"]] = rec
    return out


def validate(model: str = "chgnet", hull: MLIPHull | None = None) -> dict:
    """Parity of MLIP vs MP e_above_hull over all reference phases (Figure 2)."""
    cfg = load_config()
    hull = hull or MLIPHull(model)
    rows = []
    for mid, r in hull.refs.items():
        if mid not in hull.entries:
            continue
        ent = hull.entries[mid]
        pd = hull.pd(r["elements"])
        if pd is None:
            continue
        _, e = pd.get_decomp_and_e_above_hull(ent, allow_negative=True)
        mp_e = r["mp_e_hull_gga"] if r.get("mp_e_hull_gga") is not None else r["mp_e_hull"]
        if mp_e is None:  # MP has no thermo data for this entry (e.g. some Yb polymorphs)
            continue
        rows.append({"material_id": mid, "formula": r["formula"], "n_elements": len(r["elements"]),
                     "mp_e_hull": mp_e, "mp_e_hull_source": "GGA_GGA+U" if r.get("mp_e_hull_gga") is not None else "default",
                     "mlip_e_hull": float(e), "theoretical": r["theoretical"]})
    comp_rows = [x for x in rows if x["n_elements"] >= 2]
    err = np.array([abs(x["mlip_e_hull"] - x["mp_e_hull"]) for x in comp_rows])
    stable_ternary = [x for x in rows if x["n_elements"] >= 3 and x["mp_e_hull"] <= 1e-6]
    bad_ternary = [x for x in stable_ternary if x["mlip_e_hull"] > 0.1]
    rep = {
        "model": model, "n_refs": len(rows), "n_compounds": len(comp_rows),
        "n_refs_failed": len(hull.refs) - len(hull.entries),
        "mae_compounds": float(err.mean()) if len(err) else None,
        "mae_all": float(np.mean([abs(x["mlip_e_hull"] - x["mp_e_hull"]) for x in rows])) if rows else None,
        "spearman_compounds": _spearman([x["mp_e_hull"] for x in comp_rows], [x["mlip_e_hull"] for x in comp_rows]),
        "stable_ternaries": [{k: x[k] for k in ("formula", "material_id", "mlip_e_hull")} for x in stable_ternary],
        "stable_ternaries_above_0.1": len(bad_ternary),
        "go": bool(len(err) and err.mean() <= cfg["validation"]["mae_go_threshold"] and not bad_ternary),
        "rows": rows,
    }
    dump_json(rep, RESULTS / f"validation_{model}.json")
    plot_parity(rep, RESULTS / ("fig2_parity.png" if model == "chgnet" else f"fig2_parity_{model}.png"))
    return rep


def _spearman(a, b):
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def plot_parity(rep: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [x for x in rep["rows"] if x["n_elements"] >= 2]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    colors = {2: "#1f77b4", 3: "#d62728", 4: "#2ca02c"}
    names = {2: "binary", 3: "ternary", 4: "quaternary"}
    for n in (2, 3, 4):
        sel = [x for x in rows if x["n_elements"] == n]
        if sel:
            ax.scatter([x["mp_e_hull"] for x in sel], [x["mlip_e_hull"] for x in sel], s=18, alpha=0.75,
                       color=colors[n], label=f"{names[n]} (n={len(sel)})", edgecolors="none")
    hi = max([0.25] + [max(x["mp_e_hull"], x["mlip_e_hull"]) for x in rows])
    lo = min([0.0] + [x["mlip_e_hull"] for x in rows])
    lim = (min(lo - 0.02, -0.05), min(hi + 0.02, 0.6))
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("MP (DFT) energy above hull (eV/atom)")
    ax.set_ylabel(f"{rep['model'].upper()} energy above hull (eV/atom)")
    ax.set_title(f"Reference phases: {rep['model'].upper()} vs DFT\nMAE = {rep['mae_compounds']*1000:.0f} meV/atom "
                 f"over {rep['n_compounds']} compounds", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_evaluations(model: str = "chgnet") -> dict:
    hull = MLIPHull(model)
    ev = evaluate_candidates(model, hull)
    dump_json(ev, RESULTS / f"evaluations_{model}.json")
    rep = validate(model, hull)
    ok = [v for v in ev.values() if v["ok"]]
    log.info("candidates evaluated: %d/%d ok; hits(<=0.03): %d", len(ok), len(ev),
             sum(v["e_hull"] <= 0.03 for v in ok))
    log.info("validation: MAE %.3f eV/atom over %d compounds; stable ternaries >0.1: %d; GO=%s",
             rep["mae_compounds"] or -1, rep["n_compounds"], rep["stable_ternaries_above_0.1"], rep["go"])
    return {"evaluations": ev, "validation": rep}


def load_evaluations(model: str = "chgnet") -> dict:
    return load_json(RESULTS / f"evaluations_{model}.json", {})


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "chgnet"
    out = save_evaluations(model)
    v = out["validation"]
    print({k: v[k] for k in ("n_refs", "n_compounds", "n_refs_failed", "mae_compounds", "mae_all",
                             "spearman_compounds", "stable_ternaries_above_0.1", "go")})
    for x in v["stable_ternaries"]:
        print("  stable ternary", x)
