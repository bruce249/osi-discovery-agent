"""Build the candidate pool: (composition, prototype) pairs made by site substitution on MP parents.

Families
  A  Li3MX6, both Li3MX6 prototypes, all trivalent M, X in {Cl, Br}
  B  Li2MX6 in the Li2ZrCl6 prototype, tetravalent M, X in {Cl, Br}
  C  Li3M(Cl,Br)6 ordered, k of 6 halide sites Br, k in {2,3,4}
  D  Li3(M,M')Cl6 ordered, M sites split 50/50
Partial substitutions get up to N distinct orderings (StructureMatcher de-duplicated); the
candidate's value is later the lowest-energy ordering.
"""
from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Species, Structure

from .prototypes import load_prototypes, rescale_volume as _rescale_volume
from .utils import CACHE, chemsys_key, dump_json, get_logger, load_config, load_json, structure_hash

log = get_logger("candidates")
CAND_FILE = CACHE / "candidates.json"
SUPERCELLS = ([2, 1, 1], [1, 2, 1], [1, 1, 2])


def _fmt(x: float) -> str:
    if abs(x - round(x)) < 1e-6:
        return "" if round(x) == 1 else str(int(round(x)))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _num(x: float) -> str:
    return str(int(round(x))) if abs(x - round(x)) < 1e-6 else f"{x:.3f}".rstrip("0").rstrip(".")


def display_formula(comp: Composition, halides=("Cl", "Br")) -> str:
    """Formula normalised to 6 halides, Li first then metals then halides, e.g. Li3Y0.5Er0.5Cl6."""
    n_x = sum(comp[h] for h in halides if h in comp)
    f = 6.0 / n_x
    els = [e.symbol for e in comp.elements]
    metals = [e for e in els if e not in ("Li", *halides)]
    order = ["Li"] + metals + [h for h in halides if h in els]
    return "".join(f"{e}{_fmt(comp[e] * f)}" for e in order)


def _orderings(base: Structure, target: str, new_sp: Species, frac: Fraction, max_atoms: int,
               n_samples: int, keep: int, rng: random.Random) -> list[tuple[Structure, list[int]]]:
    """Replace `frac` of the `target`-element sites by `new_sp` in up to `keep` symmetry-distinct ways."""
    def count(s):
        return sum(1 for site in s if site.specie.symbol == target)

    cells = []
    if (frac * count(base)).denominator == 1:
        cells.append((base, [1, 1, 1]))
    sc_cells = []
    for sc in SUPERCELLS:
        if len(base) * 2 <= max_atoms and (frac * count(base) * 2).denominator == 1:
            s2 = base.copy()
            s2.make_supercell(sc)
            sc_cells.append((s2, sc))
    exact = bool(cells or sc_cells)
    if not exact:  # cannot split exactly within the atom cap: take the closest split in the parent cell
        cells.append((base, [1, 1, 1]))

    sm = StructureMatcher()
    reps: list[tuple[Structure, list[int]]] = []

    def _try(cell_list):
        for cell, sc in cell_list:
            idx = [i for i, site in enumerate(cell) if site.specie.symbol == target]
            n = int(round(float(frac) * len(idx))) if not exact else int(frac * len(idx))
            n = min(max(n, 1), len(idx) - 1)
            total = math.comb(len(idx), n)
            if total <= n_samples:
                subsets = list(itertools.combinations(idx, n))
                rng.shuffle(subsets)
            else:
                seen, subsets = set(), []
                while len(subsets) < n_samples:
                    sub = tuple(sorted(rng.sample(idx, n)))
                    if sub not in seen:
                        seen.add(sub)
                        subsets.append(sub)
            for sub in subsets:
                c = cell.copy()
                for i in sub:
                    c.replace(i, new_sp)
                if all(not sm.fit(c, r) for r, _ in reps):
                    reps.append((c, sc))
                if len(reps) >= keep:
                    return

    _try(cells)
    if len(reps) < keep and exact:
        _try(sc_cells)
    return reps


def _record(cid, family, proto, comp_struct_list, desc, subs, ox):
    comp = comp_struct_list[0][0].composition.element_composition
    orderings = []
    for s, sc in comp_struct_list:
        orderings.append({"structure": s.as_dict(), "hash": structure_hash(s), "natoms": len(s), "supercell": sc})
    return {
        "id": cid, "family": family, "prototype": proto["formula"],
        "prototype_label": f"{proto['formula']}-type ({proto['crystal_system']} {proto['spacegroup']})",
        "parent_mp_id": proto["material_id"],
        "reduced_formula": comp.reduced_formula, "display_formula": display_formula(comp),
        "elements": sorted(e.symbol for e in comp.elements), "chemsys": chemsys_key(e.symbol for e in comp.elements),
        "description": desc, "substitutions": subs, "oxidation_states": ox, "orderings": orderings,
    }


def build_pool(force: bool = False) -> list[dict]:
    if CAND_FILE.exists() and not force:
        return load_candidates()
    cfg = load_config()
    el = cfg["elements"]
    st = cfg["structures"]
    seed = cfg["seeds"]["candidates"]
    protos = load_prototypes()
    li3 = [p for p in protos.values() if p["type"] == "Li3MX6"]
    li2 = [p for p in protos.values() if p["type"] == "Li2MX6"]
    cands: list[dict] = []
    counters = {"A": 0, "B": 0, "C": 0, "D": 0}

    def next_id(fam):
        counters[fam] += 1
        return f"{fam}{counters[fam]:03d}"

    def full_sub(proto, mapping_el: dict, ox_m: int) -> Structure:
        parent = Structure.from_dict(proto["structure"])
        mp_, xp = proto["M"], proto["X"]
        mapping = {Species(mp_, ox_m): Species(mapping_el[mp_], ox_m), Species(xp, -1): Species(mapping_el[xp], -1)}
        mapping = {k: v for k, v in mapping.items() if k != v}
        s = parent.copy()
        if mapping:
            s.replace_species(mapping)
        return _rescale_volume(s, parent)

    def sub_text(proto, new_m, new_x):
        t = f"{proto['M']} -> {new_m}"
        if new_x != proto["X"]:
            t += f", {proto['X']} -> {new_x}"
        return t

    # ---- family A
    for proto in li3:
        for m in el["trivalent"]:
            for x in el["halides"]:
                s = full_sub(proto, {proto["M"]: m, proto["X"]: x}, 3)
                ox = {"Li": 1, m: 3, x: -1}
                comp = s.composition.element_composition
                desc = f"{proto['formula']}-type ({proto['crystal_system']}), {sub_text(proto, m, x)}; {display_formula(comp)}"
                subs = [{"site": "M", "parent": f"{proto['M']}3+", "new": {f"{m}3+": 1.0}},
                        {"site": "X", "parent": f"{proto['X']}-", "new": {f"{x}-": 1.0}}]
                cands.append(_record(next_id("A"), "A", proto, [(s, [1, 1, 1])], desc, subs, ox))
    # ---- family B
    for proto in li2:
        for m in el["tetravalent"]:
            for x in el["halides"]:
                s = full_sub(proto, {proto["M"]: m, proto["X"]: x}, 4)
                ox = {"Li": 1, m: 4, x: -1}
                comp = s.composition.element_composition
                desc = f"{proto['formula']}-type ({proto['crystal_system']}), {sub_text(proto, m, x)}; {display_formula(comp)}"
                subs = [{"site": "M", "parent": f"{proto['M']}4+", "new": {f"{m}4+": 1.0}},
                        {"site": "X", "parent": f"{proto['X']}-", "new": {f"{x}-": 1.0}}]
                cands.append(_record(next_id("B"), "B", proto, [(s, [1, 1, 1])], desc, subs, ox))
    # ---- family C: mixed anion
    for proto in li3:
        for m in el["family_C_metals"]:
            for k in el["family_C_k"]:
                cid = next_id("C")
                base = full_sub(proto, {proto["M"]: m, proto["X"]: proto["X"]}, 3)
                parent = Structure.from_dict(proto["structure"])
                rng = random.Random(f"{seed}-{cid}")
                reps = _orderings(base, proto["X"], Species("Br", -1), Fraction(k, 6), st["max_atoms"],
                                  st["ordering_samples"], st["orderings_per_candidate"], rng)
                reps = [(_rescale_volume(s, parent), sc) for s, sc in reps]
                if not reps:
                    log.warning("no ordering for %s", cid)
                    continue
                comp = reps[0][0].composition.element_composition
                frac_br = comp["Br"] / (comp["Br"] + comp["Cl"])
                ox = {"Li": 1, m: 3, "Cl": -1, "Br": -1}
                desc = (f"{proto['formula']}-type ({proto['crystal_system']}), {proto['M']} -> {m}, "
                        f"{_num(frac_br * 6)}/6 Cl -> Br (ordered); {display_formula(comp)}")
                subs = [{"site": "M", "parent": f"{proto['M']}3+", "new": {f"{m}3+": 1.0}},
                        {"site": "X", "parent": "Cl-", "new": {"Cl-": 1 - frac_br, "Br-": frac_br}}]
                cands.append(_record(cid, "C", proto, reps, desc, subs, ox))
    # ---- family D: mixed cation
    for proto in li3:
        for m1, m2 in itertools.combinations(el["family_D_metals"], 2):
            cid = next_id("D")
            base = full_sub(proto, {proto["M"]: m1, proto["X"]: "Cl"}, 3)
            parent = Structure.from_dict(proto["structure"])
            rng = random.Random(f"{seed}-{cid}")
            reps = _orderings(base, m1, Species(m2, 3), Fraction(1, 2), st["max_atoms"],
                              st["ordering_samples"], st["orderings_per_candidate"], rng)
            reps = [(_rescale_volume(s, parent), sc) for s, sc in reps]
            if not reps:
                log.warning("no ordering for %s", cid)
                continue
            comp = reps[0][0].composition.element_composition
            f2 = comp[m2] / (comp[m1] + comp[m2])
            ox = {"Li": 1, m1: 3, m2: 3, "Cl": -1}
            desc = (f"{proto['formula']}-type ({proto['crystal_system']}), {proto['M']} -> "
                    f"{_num((1 - f2) * 100)}% {m1} / {_num(f2 * 100)}% {m2} (ordered); {display_formula(comp)}")
            subs = [{"site": "M", "parent": f"{proto['M']}3+", "new": {f"{m1}3+": 1 - f2, f"{m2}3+": f2}},
                    {"site": "X", "parent": "Cl-", "new": {"Cl-": 1.0}}]
            cands.append(_record(cid, "D", proto, reps, desc, subs, ox))

    _sanity_check(protos, cands)
    dump_json(cands, CAND_FILE)
    return cands


def _sanity_check(protos: dict, cands: list[dict]) -> None:
    """Substituting the parent M with itself must reproduce the parent structure exactly."""
    sm = StructureMatcher()
    for proto in protos.values():
        same = [c for c in cands if c["family"] in "AB" and c["prototype"] == proto["formula"]
                and c["reduced_formula"] == Composition(proto["formula"]).reduced_formula]
        assert same, f"no identity candidate for {proto['formula']}"
        s = Structure.from_dict(same[0]["orderings"][0]["structure"])
        p = Structure.from_dict(proto["structure"])
        assert sm.fit(s, p) and abs(s.volume - p.volume) < 1e-6 * p.volume, f"identity check failed for {proto['formula']}"
        log.info("sanity check passed: %s -> %s reproduces parent", proto["formula"], same[0]["id"])


def load_candidates() -> list[dict]:
    c = load_json(CAND_FILE)
    if c is None:
        raise RuntimeError("candidates.json missing: run `python -m src.candidates` first")
    return c


def summary(cands: list[dict]) -> dict:
    fams = {}
    for c in cands:
        f = fams.setdefault(c["family"], {"candidates": 0, "orderings": 0, "max_atoms": 0})
        f["candidates"] += 1
        f["orderings"] += len(c["orderings"])
        f["max_atoms"] = max(f["max_atoms"], max(o["natoms"] for o in c["orderings"]))
    return fams


if __name__ == "__main__":
    import json
    import sys

    cands = build_pool(force="--force" in sys.argv)
    print(json.dumps(summary(cands), indent=1))
    print("total candidates", len(cands), "unique formulas", len({c["reduced_formula"] for c in cands}))
    for c in cands[:3] + cands[-3:]:
        print(c["id"], c["description"], [o["natoms"] for o in c["orderings"]])
