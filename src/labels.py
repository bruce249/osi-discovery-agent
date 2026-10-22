"""Ground-truth labels. NEVER passed to strategies; only the campaign scorer reads them.

  hit    : MLIP e_hull <= threshold (0.03 eV/atom headline; 0.0 and 0.05 also reported)
  known  : reduced formula exists in MP with theoretical == False (experimental ICSD record), OR the exact formula
           appears in a paper title/abstract -> rediscovery target. (MP flags every Li3MCl6/Li2MCl6 entry as
           theoretical, because the experimental structures are disordered, so the MP flag alone finds almost none.)
  novel  : hit, reduced formula absent from MP, Alexandria, OQMD and JARVIS (checked providers), and no literature hit
"""
from __future__ import annotations

import pandas as pd

from .utils import CACHE, RESULTS, dump_json, get_logger, load_config, load_json

log = get_logger("labels")
LABEL_FILE = CACHE / "labels.json"


def thr_key(t: float) -> str:
    return f"{t:.2f}"


def build_labels() -> dict:
    from .candidates import load_candidates
    from .hull import load_evaluations
    from .mp_fetch import load_mp_index
    from .novelty import load_novelty

    cfg = load_config()
    thresholds = sorted({cfg["labels"]["hit_threshold"], *cfg["labels"]["extra_thresholds"]})
    ev = load_evaluations("chgnet")
    mp_index = load_mp_index()
    nov = load_novelty()
    labels, rows = {}, []
    for c in load_candidates():
        r = ev.get(c["id"], {})
        f = c["reduced_formula"]
        mp_rows = mp_index.get(f, [])
        n = nov.get(f)
        known_mp = any(not x["theoretical"] for x in mp_rows)
        known_lit = bool(n and n["lit"].get("exact_reported"))
        known = known_mp or known_lit
        db_absent = bool(n and n["db"]["absent"])
        lit = n["lit"]["lit_status"] if n else None
        novel_comp = db_absent and lit in ("none found",)
        lab = {"id": c["id"], "formula": f, "display_formula": c["display_formula"], "family": c["family"],
               "prototype": c["prototype"], "ok": bool(r.get("ok")), "e_hull": r.get("e_hull"),
               "known": known, "known_source": "MP experimental" if known_mp else ("literature (exact formula)" if known_lit else ""),
               "in_mp": bool(mp_rows), "db_status": n["db"]["summary"] if n else None,
               "lit_status": lit, "novel_composition": novel_comp}
        for t in thresholds:
            hit = bool(r.get("ok")) and r["e_hull"] <= t + 1e-9
            lab[f"hit_{thr_key(t)}"] = hit
            if hit and n is None:
                log.warning("hit %s (%s) has no novelty record: run src.novelty first", c["id"], f)
            lab[f"novel_{thr_key(t)}"] = hit and novel_comp
            lab[f"known_hit_{thr_key(t)}"] = hit and known
        labels[c["id"]] = lab
        rows.append(lab)
    dump_json(labels, LABEL_FILE)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "labels.csv", index=False)
    t0 = thr_key(cfg["labels"]["hit_threshold"])
    ok = df[df.ok]
    log.info("labels: %d ok candidates, hits %d (%d formulas), known formulas %d, known hits %d, novel hits %d formulas",
             len(ok), ok[f"hit_{t0}"].sum(), ok[ok[f"hit_{t0}"]].formula.nunique(), ok[ok.known].formula.nunique(),
             ok[ok[f"known_hit_{t0}"]].formula.nunique(), ok[ok[f"novel_{t0}"]].formula.nunique())
    return labels


def load_labels() -> dict:
    lab = load_json(LABEL_FILE)
    if lab is None:
        raise RuntimeError("labels.json missing: run `python -m src.labels` first")
    return lab


if __name__ == "__main__":
    build_labels()
