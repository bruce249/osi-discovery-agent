"""Materials Project reference phases for every chemical system touched by a candidate.

One cached query per (top-level) chemical system covering all its subsystems. We keep:
  * every MP material (formula, theoretical flag, e_hull) -> used for "known" labels and database novelty
  * structures of phases with MP e_hull <= ref_ehull_max -> reference phases relaxed with the MLIP
"""
from __future__ import annotations

import itertools
import time

from pymatgen.core import Composition, Structure

from .utils import CACHE, chemsys_key, dump_json, get_logger, load_config, load_json, require_env, structure_hash

log = get_logger("mp_fetch")
SYS_DIR = CACHE / "mp_systems"
REF_FILE = CACHE / "references.json"
INDEX_FILE = CACHE / "mp_index.json"
SUMMARY_FIELDS = ["material_id", "formula_pretty", "structure", "energy_above_hull", "theoretical", "nsites"]


def subsystems(elements) -> list[str]:
    els = sorted(set(elements))
    return ["-".join(c) for r in range(1, len(els) + 1) for c in itertools.combinations(els, r)]


def _retry(fn, tries=3):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1:
                raise
            log.warning("MP call failed (%s), retrying", e)
            time.sleep(3 * (i + 1))


def fetch_system(elements, mpr, ref_max: float) -> list[dict]:
    key = chemsys_key(elements)
    path = SYS_DIR / f"{key}.json"
    cached = load_json(path)
    if cached is not None:
        return cached
    subs = subsystems(elements)
    try:
        docs = _retry(lambda: mpr.materials.summary.search(chemsys=subs, fields=SUMMARY_FIELDS))
    except Exception as e:  # noqa: BLE001 - fall back to one call per subsystem
        log.warning("list chemsys query failed for %s (%s); querying subsystems one by one", key, e)
        docs = []
        for s in subs:
            docs += _retry(lambda s=s: mpr.materials.summary.search(chemsys=s, fields=SUMMARY_FIELDS))
    gga = {}
    try:
        tdocs = _retry(lambda: mpr.materials.thermo.search(chemsys=subs, thermo_types=["GGA_GGA+U"],
                                                            fields=["material_id", "energy_above_hull"]))
        gga = {str(t.material_id): t.energy_above_hull for t in tdocs}
    except Exception as e:  # noqa: BLE001
        log.warning("GGA thermo query failed for %s: %s", key, e)
    rows = []
    for d in docs:
        mid = str(d.material_id)
        comp = Composition(d.formula_pretty)
        row = {"material_id": mid, "formula": d.formula_pretty, "reduced_formula": comp.reduced_formula,
               "elements": sorted(e.symbol for e in comp.elements), "e_hull": d.energy_above_hull,
               "e_hull_gga": gga.get(mid), "theoretical": bool(d.theoretical), "nsites": d.nsites}
        # elemental phases always carry a structure so the lowest polymorph can be kept whatever its e_hull
        if (d.energy_above_hull is not None and d.energy_above_hull <= ref_max) or len(comp.elements) == 1:
            row["structure"] = d.structure.as_dict()
        rows.append(row)
    dump_json(rows, path)
    log.info("%s: %d materials (%d with e_hull <= %.2f)", key, len(rows),
             sum("structure" in r for r in rows), ref_max)
    return rows


def candidate_systems() -> list[tuple[str, ...]]:
    from .candidates import load_candidates

    return sorted({tuple(c["elements"]) for c in load_candidates()})


def build_references(force: bool = False) -> dict:
    if REF_FILE.exists() and not force:
        return load_references()
    from mp_api.client import MPRester

    cfg = load_config()
    ref_max = cfg["structures"]["ref_ehull_max"]
    max_atoms = cfg["structures"]["max_atoms"]
    systems = candidate_systems()
    log.info("%d chemical systems", len(systems))
    all_rows: dict[str, dict] = {}
    with MPRester(require_env("MP_API_KEY")) as mpr:
        for i, els in enumerate(systems):
            for r in fetch_system(els, mpr, ref_max):
                all_rows.setdefault(r["material_id"], r)
            if i % 10 == 0:
                log.info("system %d/%d", i + 1, len(systems))

    index: dict[str, list] = {}
    for r in all_rows.values():
        index.setdefault(r["reduced_formula"], []).append(
            {"material_id": r["material_id"], "theoretical": r["theoretical"], "e_hull": r["e_hull"]})
    dump_json(index, INDEX_FILE)

    # lowest-e_hull polymorph of each element is always a reference ("always keep the elemental references")
    elem_best: dict[str, dict] = {}
    for r in all_rows.values():
        if len(r["elements"]) == 1 and "structure" in r:
            el = r["elements"][0]
            if el not in elem_best or (r["e_hull"] if r["e_hull"] is not None else 9) < (elem_best[el]["e_hull"] if elem_best[el]["e_hull"] is not None else 9):
                elem_best[el] = r
    keep_ids = {r["material_id"] for r in elem_best.values()}
    for el, r in elem_best.items():
        if r["e_hull"] is None or r["e_hull"] > ref_max:
            log.warning("element %s: lowest MP polymorph %s has e_hull %s > %.2f; kept as reference anyway",
                        el, r["material_id"], r["e_hull"], ref_max)

    refs, skipped = {}, []
    for r in all_rows.values():
        if "structure" not in r:
            continue
        if len(r["elements"]) == 1 and r["material_id"] not in keep_ids and (r["e_hull"] is None or r["e_hull"] > ref_max):
            continue
        s = Structure.from_dict(r["structure"])
        if len(s) > max_atoms:
            s = s.get_primitive_structure()
        if len(s) > max_atoms and len(r["elements"]) > 1:
            skipped.append((r["material_id"], r["formula"], len(s)))
            continue
        s.remove_oxidation_states()
        refs[r["material_id"]] = {
            "material_id": r["material_id"], "formula": r["formula"], "reduced_formula": r["reduced_formula"],
            "elements": r["elements"], "chemsys": chemsys_key(r["elements"]), "mp_e_hull": r["e_hull"],
            "mp_e_hull_gga": r["e_hull_gga"], "theoretical": r["theoretical"], "nsites": len(s),
            "structure": s.as_dict(), "hash": structure_hash(s),
        }
    elements_needed = {e for els in systems for e in els}
    have = {r["elements"][0] for r in refs.values() if len(r["elements"]) == 1}
    missing = elements_needed - have
    if missing:
        raise RuntimeError(f"no elemental reference for {missing}")
    dump_json(refs, REF_FILE)
    dump_json(skipped, CACHE / "references_skipped.json")
    log.info("%d reference phases kept, %d skipped for size > %d atoms", len(refs), len(skipped), max_atoms)
    return refs


def load_references() -> dict:
    r = load_json(REF_FILE)
    if r is None:
        raise RuntimeError("references.json missing: run `python -m src.mp_fetch` first")
    return r


def load_mp_index() -> dict:
    return load_json(INDEX_FILE, {})


if __name__ == "__main__":
    import sys
    from collections import Counter

    refs = build_references(force="--force" in sys.argv)
    sizes = [r["nsites"] for r in refs.values()]
    print("references:", len(refs), "atoms total:", sum(sizes), "max:", max(sizes))
    print("by n-elements:", Counter(len(r["elements"]) for r in refs.values()))
