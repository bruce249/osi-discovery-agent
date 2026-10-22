"""Cross-check the top novel hits with a second potential (MACE-MP-0 medium).

  python -m src.second_model prepare   -> data/cache/mace_tasks.json (candidate orderings + their systems' references)
  (modal) relax_batch --model mace --tasks-file data/cache/mace_tasks.json
  python -m src.second_model analyze   -> results/second_model.json, results/fig2_parity_mace.png

MACE relaxations start from the CHGNet-relaxed structures (faster, same minimum basin) and are keyed by the same
input hash, so the MACE hull is built exactly like the CHGNet one, from MACE-relaxed references only.
"""
from __future__ import annotations

import sys

from .utils import CACHE, RESULTS, dump_json, get_logger, load_config, load_json

log = get_logger("second_model")
TASKS = CACHE / "mace_tasks.json"
SELECTION = CACHE / "mace_selection.json"


def select_top(n: int) -> list[dict]:
    from .labels import load_labels, thr_key

    cfg = load_config()
    tk = thr_key(cfg["labels"]["hit_threshold"])
    labels = load_labels()
    best: dict[str, dict] = {}
    for lab in labels.values():
        if lab[f"hit_{tk}"]:
            f = lab["formula"]
            if f not in best or lab["e_hull"] < best[f]["e_hull"]:
                best[f] = lab
    novel = sorted([x for x in best.values() if x[f"novel_{tk}"]], key=lambda x: x["e_hull"])
    # diversity: at most `max_per_metal` picks containing any one metal, so a single element (whose reference
    # energies may be off, e.g. Yb) cannot fill the whole cross-check list
    cap = cfg["validation"].get("max_per_metal", 3)
    from pymatgen.core import Composition

    count: dict[str, int] = {}
    chosen = []
    for x in novel:
        metals = [e.symbol for e in Composition(x["formula"]).elements if e.symbol not in ("Li", "Cl", "Br")]
        if any(count.get(m, 0) >= cap for m in metals):
            continue
        chosen.append(x)
        for m in metals:
            count[m] = count.get(m, 0) + 1
        if len(chosen) == n:
            break
    if len(chosen) < n:  # not enough novel hits: also cross-check the best other hits (flagged as not novel)
        others = sorted([x for x in best.values() if not x[f"novel_{tk}"]], key=lambda x: x["e_hull"])
        chosen += others[: n - len(chosen)]
    return chosen


def prepare() -> list[dict]:
    from pymatgen.core import Structure

    from .candidates import load_candidates
    from .mlip import load_result
    from .mp_fetch import load_references

    cfg = load_config()
    m = cfg["mlip"]
    chosen = select_top(cfg["validation"]["top_n_second_model"])
    cands = {c["id"]: c for c in load_candidates()}
    refs = load_references()
    tasks, seen = [], set()

    def add(h, struct, kind, label):
        if h in seen:
            return
        seen.add(h)
        r = load_result("chgnet", h)
        s = r["final_structure"] if r and r.get("ok") else struct
        tasks.append({"hash": h, "kind": kind, "label": label, "structure": s,
                      "natoms": len(Structure.from_dict(s)), "model": "mace", "fmax": m["fmax_second"],
                      "steps": m["steps"], "timeout_s": m["timeout_s"], "mace_model": m["mace_model"]})

    for lab in chosen:
        c = cands[lab["id"]]
        for o in c["orderings"]:
            add(o["hash"], o["structure"], "candidate", c["id"])
        for mid, r in refs.items():
            if set(r["elements"]) <= set(c["elements"]):
                add(r["hash"], r["structure"], "reference", mid)
    dump_json(tasks, TASKS)
    dump_json(chosen, SELECTION)
    log.info("MACE: %d candidates selected, %d relaxation tasks", len(chosen), len(tasks))
    return tasks


def analyze() -> dict:
    from .candidates import load_candidates
    from .hull import MLIPHull, evaluate_candidates, load_evaluations, validate

    cfg = load_config()
    chosen = load_json(SELECTION)
    cands = {c["id"]: c for c in load_candidates()}
    sel = [cands[x["id"]] for x in chosen]
    hull = MLIPHull("mace")
    ev_m = evaluate_candidates("mace", hull, sel)
    ev_c = load_evaluations("chgnet")
    thr = cfg["validation"]["agree_threshold"]
    out = {}
    for x in chosen:
        cid = x["id"]
        em = ev_m.get(cid, {})
        e_m = em.get("e_hull") if em.get("ok") else None
        e_c = ev_c[cid]["e_hull"]
        out[cid] = {"id": cid, "formula": x["formula"], "chgnet_e_hull": e_c, "mace_e_hull": e_m,
                    "agreed": bool(e_m is not None and e_c <= thr and e_m <= thr),
                    "mace_decomposition": em.get("decomposition"), "chgnet_decomposition": ev_c[cid].get("decomposition")}
    dump_json(out, RESULTS / "second_model.json")
    rep = validate("mace", hull)
    log.info("MACE agreed on %d/%d; MACE ref MAE %.3f over %d compounds", sum(v["agreed"] for v in out.values()),
             len(out), rep["mae_compounds"] or -1, rep["n_compounds"])
    return out


if __name__ == "__main__":
    {"prepare": prepare, "analyze": analyze}[sys.argv[1]]()
