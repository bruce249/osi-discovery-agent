"""Find parent (prototype) structures by formula on Materials Project. No hardcoded MP ids."""
from __future__ import annotations

from pymatgen.core import Species, Structure

from .utils import CACHE, dump_json, get_logger, load_config, load_json, require_env

log = get_logger("prototypes")
PROTO_FILE = CACHE / "prototypes.json"

# Crystal system we want each prototype to represent (brief section 4.1). If the lowest-e_hull
# polymorph has a different crystal system we take the lowest one that matches, if any.
PREFERRED_SYSTEM = {"Li3YCl6": "trigonal", "Li3InCl6": "monoclinic", "Li2ZrCl6": "trigonal"}


def _radius(sp: Species) -> float:
    r = sp.ionic_radius
    if r is None:
        r = sp.element.atomic_radius or 1.5
    return float(r)


def rescale_volume(s: Structure, parent: Structure) -> Structure:
    """Scale the cell by the change in summed cubed ionic radii so relaxations start near the right volume."""
    num = sum(_radius(site.specie) ** 3 for site in s)
    den = sum(_radius(site.specie) ** 3 for site in parent)
    s = s.copy()
    s.scale_lattice(parent.volume * (len(s) / len(parent)) * num / den)
    return s


def fetch_prototypes(force: bool = False) -> dict:
    cached = load_json(PROTO_FILE)
    if cached and not force:
        return cached
    from mp_api.client import MPRester
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    cfg = load_config()
    max_atoms = cfg["structures"]["max_atoms"]
    out = {}
    with MPRester(require_env("MP_API_KEY")) as mpr:
        for formula, spec in cfg["prototypes"].items():
            fields = ["material_id", "formula_pretty", "structure", "energy_above_hull",
                      "theoretical", "symmetry", "nsites"]
            docs = mpr.materials.summary.search(formula=formula, fields=fields)
            analogue = None
            if not docs:
                # Parent absent from MP: take the lowest-e_hull isostructural analogue (same anonymous formula,
                # wildcard on M, preferred crystal system) and substitute M back. Logged, never hardcoded.
                wildcard = formula.replace(spec["M"], "*", 1)
                docs = [d for d in mpr.materials.summary.search(formula=wildcard, fields=fields)
                        if spec["X"] in {e.symbol for e in d.structure.composition.elements}]
                if not docs:
                    raise RuntimeError(f"no MP entry for prototype {formula} or analogue {wildcard}")
                log.warning("%s not in MP; using analogues of %s", formula, wildcard)
            docs = sorted(docs, key=lambda d: (d.energy_above_hull if d.energy_above_hull is not None else 9, d.nsites))
            all_found = [{"material_id": str(d.material_id), "formula": d.formula_pretty, "e_hull": d.energy_above_hull,
                          "spacegroup": d.symmetry.symbol, "crystal_system": str(d.symmetry.crystal_system).lower(),
                          "nsites": d.nsites, "theoretical": d.theoretical} for d in docs]
            want = PREFERRED_SYSTEM.get(formula)
            match = [d for d in docs if want and want in str(d.symmetry.crystal_system).lower()]
            pick = match[0] if match else docs[0]
            s = pick.structure.copy()
            if len(s) > max_atoms:
                s = SpacegroupAnalyzer(s).get_primitive_standard_structure()
            elif len(s) > 1:
                prim = s.get_primitive_structure()
                if len(prim) < len(s):
                    s = prim
            ox_m = 3 if spec["type"] == "Li3MX6" else 4
            ox = {"Li": 1, spec["M"]: ox_m, spec["X"]: -1}
            pick_m = [e.symbol for e in s.composition.elements if e.symbol not in ("Li", spec["X"])][0]
            if pick_m != spec["M"]:
                analogue = {"formula": pick.formula_pretty, "material_id": str(pick.material_id), "M": pick_m}
                s.add_oxidation_state_by_element({"Li": 1, pick_m: ox_m, spec["X"]: -1})
                parent_s = s.copy()
                s.replace_species({Species(pick_m, ox_m): Species(spec["M"], ox_m)})
                s = rescale_volume(s, parent_s)
            else:
                s.add_oxidation_state_by_element(ox)
            out[formula] = {
                "formula": formula, "type": spec["type"], "M": spec["M"], "X": spec["X"],
                "material_id": str(pick.material_id), "spacegroup": pick.symmetry.symbol,
                "spacegroup_number": pick.symmetry.number,
                "crystal_system": str(pick.symmetry.crystal_system).lower(),
                "e_hull": pick.energy_above_hull, "theoretical": pick.theoretical,
                "nsites": len(s), "oxidation_states": ox, "structure": s.as_dict(),
                "polymorphs_found": all_found, "analogue": analogue,
            }
            log.info("%s -> %s %s %s (%d atoms, e_hull %.3f, theoretical=%s, analogue=%s)", formula, pick.material_id,
                     pick.formula_pretty, pick.symmetry.symbol, len(s), pick.energy_above_hull, pick.theoretical, analogue)
    dump_json(out, PROTO_FILE)
    return out


def load_prototypes() -> dict:
    p = load_json(PROTO_FILE)
    if p is None:
        raise RuntimeError("prototypes.json missing: run `python -m src.prototypes` first")
    return p


if __name__ == "__main__":
    import sys

    res = fetch_prototypes(force="--force" in sys.argv)
    for k, v in res.items():
        print(k, v["material_id"], v["spacegroup"], v["nsites"], "atoms", "e_hull", v["e_hull"], "analogue", v["analogue"])
        for poly in v["polymorphs_found"]:
            print("    ", poly)
