"""Novelty checks for hit compositions.

Database: Materials Project (local index from mp_fetch) + OPTIMADE (MP, Alexandria, OQMD, JARVIS), exact reduced
formula. A provider that errors is recorded as "unchecked", never as "absent".
Literature: OpenAlex, arXiv and Semantic Scholar searches by formula and by plain description. A paper counts as
"possibly reported" if its title/abstract contains a formula-like token whose element set equals the candidate's
(so solid-solution notation such as Li3Y1-xErxCl6 or Li3YCl6-xBrx counts).
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.parse
import xml.etree.ElementTree as ET

import requests
from pymatgen.core import Composition, Element

from .utils import CACHE, chemsys_key, dump_json, get_logger, load_config, load_json, text_hash

log = get_logger("novelty")
OPT_DIR = CACHE / "optimade"
LIT_DIR = CACHE / "literature"
UA = {"User-Agent": "osi-discovery-agent/1.0 (hackathon research; mailto:osi-hackathon@example.org)"}


def optimade_formula(reduced_formula: str) -> str:
    comp = Composition(reduced_formula).reduced_composition
    parts = []
    for el in sorted(e.symbol for e in comp.elements):
        n = comp[el]
        parts.append(el + ("" if abs(n - 1) < 1e-8 else str(int(round(n)))))
    return "".join(parts)


_FAILS: dict[str, int] = {}  # consecutive failures per provider (circuit breaker: skip a server that is down)


def optimade_lookup(provider: str, base: str, formula: str, timeout: float) -> dict:
    f = optimade_formula(formula)
    path = OPT_DIR / provider.replace(" ", "_") / f"{f}.json"
    cached = load_json(path)
    if cached is not None and cached.get("status") != "unchecked":
        return cached
    if _FAILS.get(provider, 0) >= 3:
        return {"provider": provider, "formula": f, "status": "unchecked", "ids": [], "error": "server down this run"}
    url = f"{base.rstrip('/')}/v1/structures"
    params = {"filter": f'chemical_formula_reduced="{f}"', "response_fields": "id,chemical_formula_reduced",
              "page_limit": 10}
    out = {"provider": provider, "formula": f, "status": "unchecked", "ids": []}
    for attempt in range(2):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=UA)
            r.raise_for_status()
            data = r.json().get("data", [])
            ids = [d.get("id") for d in data]
            n_more = r.json().get("meta", {}).get("data_returned", len(ids))
            out = {"provider": provider, "formula": f, "status": "present" if ids else "absent",
                   "ids": ids[:10], "n_returned": n_more}
            break
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(2)
    _FAILS[provider] = 0 if out["status"] != "unchecked" else _FAILS.get(provider, 0) + 1
    dump_json(out, path)
    return out


_JARVIS_LOCK = threading.Lock()
_JARVIS: set | None = None
_MP_LOCK = threading.Lock()


def jarvis_formulas() -> set | None:
    """Reduced formulas of the whole JARVIS-DFT 3D dataset (figshare download via jarvis-tools), cached."""
    global _JARVIS
    with _JARVIS_LOCK:
        if _JARVIS is not None:
            return _JARVIS
        path = CACHE / "jarvis_dft3d_formulas.json"
        cached = load_json(path)
        if cached:
            _JARVIS = set(cached)
            return _JARVIS
        try:
            from jarvis.db.figshare import data

            forms = set()
            for x in data("dft_3d"):
                try:
                    forms.add(Composition(x["formula"]).reduced_formula)
                except Exception:  # noqa: BLE001
                    pass
            dump_json(sorted(forms), path)
            _JARVIS = forms
            log.info("JARVIS dft_3d: %d distinct formulas", len(forms))
        except Exception as e:  # noqa: BLE001
            log.warning("JARVIS figshare download failed: %s", e)
            _JARVIS = None
        return _JARVIS


def gnome_status(reduced_formula: str) -> str:
    """Materials Project API with include_gnome=True (GNoME hypothetical structures). Cached per formula."""
    path = CACHE / "gnome" / f"{reduced_formula}.json"
    cached = load_json(path)
    if cached is not None:
        return cached["status"]
    from mp_api.client import MPRester

    from .utils import require_env

    try:
        with _MP_LOCK, MPRester(require_env("MP_API_KEY")) as mpr:
            docs = mpr.materials.summary.search(formula=reduced_formula, include_gnome=True, fields=["material_id"])
        status = "present" if docs else "absent"
        ids = [str(d.material_id) for d in docs][:10]
    except Exception as e:  # noqa: BLE001
        status, ids = "unchecked", [f"{type(e).__name__}: {str(e)[:150]}"]
    dump_json({"status": status, "ids": ids}, path)
    return status


def database_status(reduced_formula: str, mp_index: dict) -> dict:
    cfg = load_config()["novelty"]
    res = {"MP (API)": "present" if reduced_formula in mp_index else "absent",
           "MP+GNoME (API)": gnome_status(reduced_formula)}
    detail = {}
    for name, base in cfg["optimade"].items():
        r = optimade_lookup(name, base, reduced_formula, cfg["timeout_s"])
        res[name] = r["status"]
        detail[name] = r.get("ids", [])
    # JARVIS OPTIMADE is flaky (redirects / 502): the full figshare dft_3d dataset is the authoritative check
    jf = jarvis_formulas()
    if jf is not None:
        res["JARVIS"] = "present" if (reduced_formula in jf or res.get("JARVIS") == "present") else "absent"
        detail["JARVIS"] = detail.get("JARVIS", []) + ["figshare dft_3d"]
    present = [k for k, v in res.items() if v == "present"]
    unchecked = [k for k, v in res.items() if v == "unchecked"]
    if present:
        summary = "in " + ", ".join(present)
    else:
        summary = "absent" + (f" (unchecked: {', '.join(unchecked)})" if unchecked else "")
    return {"by_provider": res, "ids": detail, "summary": summary, "absent": not present}


# ----------------------------------------------------------------------------- literature

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉−–₋ₓ₊", "0123456789---x+")
_TOKEN = re.compile(r"[A-Z][A-Za-z0-9.()\[\]\-+,/δ]*")
_VALID = re.compile(r"(?:[A-Z][a-z]?|[0-9.]+|[xyzδ]|[()\[\]\-+,/])+")


def formula_tokens(text: str) -> list[set[str]]:
    text = text.translate(_SUB)
    out = []
    for raw in _TOKEN.findall(text):
        # split lists such as "LiCl,YCl3" or "LiCl/YCl3" (but not "Li3(Y,Er)Cl6"), drop "-based"-style suffixes
        for tok in re.split(r"[,/](?![^()\[\]]*[)\]])", raw):
            tok = re.sub(r"-[a-z]{2,}.*$", "", tok).rstrip(".,-")
            if not tok or not _VALID.fullmatch(tok):
                continue
            els = {e for e in re.findall(r"[A-Z][a-z]?", tok)}
            if len(els) >= 3 and all(Element.is_valid_symbol(e) for e in els):
                out.append(els)
    return out


def _openalex(q: str) -> list[dict]:
    r = requests.get("https://api.openalex.org/works", params={"search": q, "per_page": 25,
                     "mailto": "osi-hackathon@example.org"}, timeout=30, headers=UA)
    r.raise_for_status()
    out = []
    for w in r.json().get("results", []):
        abst = ""
        inv = w.get("abstract_inverted_index") or {}
        if inv:
            pos = sorted((p, word) for word, ps in inv.items() for p in ps)
            abst = " ".join(word for _, word in pos)
        out.append({"title": w.get("title") or "", "abstract": abst, "url": w.get("doi") or w.get("id"),
                    "year": w.get("publication_year")})
    return out


_LOCKS = {"arxiv": threading.Lock(), "semantic_scholar": threading.Lock(), "openalex": threading.Semaphore(4)}


def _arxiv(q: str) -> list[dict]:
    query = f'all:"{q}"' if " " not in q else "all:" + "+AND+all:".join(urllib.parse.quote(t) for t in q.split())
    r = requests.get(f"http://export.arxiv.org/api/query?search_query={query}&max_results=20", timeout=30, headers=UA)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text)
    out = []
    for e in root.findall("a:entry", ns):
        out.append({"title": (e.findtext("a:title", "", ns) or "").strip(),
                    "abstract": (e.findtext("a:summary", "", ns) or "").strip(),
                    "url": e.findtext("a:id", "", ns), "year": (e.findtext("a:published", "", ns) or "")[:4]})
    time.sleep(3)  # arXiv API etiquette
    return out


def _s2(q: str) -> list[dict]:
    r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
                     params={"query": q, "fields": "title,abstract,url,year", "limit": 20}, timeout=30, headers=UA)
    r.raise_for_status()  # 429 (shared unauthenticated quota) -> recorded as a failed query, other sources still count
    time.sleep(1.1)
    return [{"title": p.get("title") or "", "abstract": p.get("abstract") or "", "url": p.get("url"),
             "year": p.get("year")} for p in r.json().get("data", [])]


SOURCES = {"openalex": _openalex, "arxiv": _arxiv, "semantic_scholar": _s2}


_LIT_FAILS: dict[str, int] = {}


def _search(source: str, q: str) -> dict:
    path = LIT_DIR / source / f"{text_hash(q)}.json"
    cached = load_json(path)
    if cached is not None and cached.get("ok"):
        return cached
    if _LIT_FAILS.get(source, 0) >= 5:  # circuit breaker: source rate-limited or down for this run
        return {"ok": False, "query": q, "error": "source unavailable this run", "papers": []}
    try:
        with _LOCKS[source]:
            res = {"ok": True, "query": q, "papers": SOURCES[source](q)}
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "query": q, "error": f"{type(e).__name__}: {str(e)[:200]}", "papers": []}
    _LIT_FAILS[source] = 0 if res["ok"] else _LIT_FAILS.get(source, 0) + 1
    dump_json(res, path)
    return res


def literature_queries(cand: dict) -> list[str]:
    metals = [e for e in cand["elements"] if e not in ("Li", "Cl", "Br")]
    hal = [h for h in ("Cl", "Br") if h in cand["elements"]]
    names = {"Cl": "chloride", "Br": "bromide"}
    hal_word = " ".join(names[h] for h in hal) if len(hal) == 1 else "chloride bromide mixed halide"
    qs = [cand["display_formula"], cand["reduced_formula"]]
    plain = f"{' '.join(metals)} {'mixed ' if len(metals) > 1 else ''}lithium {hal_word} solid electrolyte"
    qs.append(plain)
    return list(dict.fromkeys(qs))


def _exact_in(formula: str, compact: str) -> bool:
    """Formula appears as a whole token (Li3YCl6 must not match the start of Li3YCl6-xBrx-like longer numbers)."""
    return re.search(re.escape(formula) + r"(?![0-9.])", compact) is not None


def literature_status(cand: dict) -> dict:
    target = set(cand["elements"])
    exact = {cand["display_formula"], cand["reduced_formula"]}
    hits, checked, failed = [], [], []
    for q in literature_queries(cand):
        for src in SOURCES:
            if src == "arxiv" and q not in exact:  # arXiv etiquette (3 s/query): formula queries only
                continue
            res = _search(src, q)
            (checked if res["ok"] else failed).append(f"{src}:{q}")
            for p in res["papers"]:
                # titles/abstracts often carry markup such as Li<sub>3</sub>YCl<sub>6</sub> or $_3$
                text = re.sub(r"<[^>]+>|\$|_\{?|\}", "", f"{p['title']} {p['abstract']}")
                compact = text.translate(_SUB).replace(" ", "")
                is_exact = any(_exact_in(f, compact) for f in exact)
                match = is_exact or any(t == target for t in formula_tokens(text))
                if match and p["url"] not in {h["url"] for h in hits}:
                    hits.append({"title": p["title"][:200], "url": p["url"], "year": p.get("year"), "source": src,
                                 "exact_formula": is_exact})
    if hits:
        status = "possibly reported"
    elif checked:
        status = "none found"
    else:
        status = "unchecked"
    hits.sort(key=lambda h: not h["exact_formula"])
    return {"lit_status": status, "exact_reported": any(h["exact_formula"] for h in hits), "links": hits[:8],
            "n_queries_ok": len(checked), "n_queries_failed": len(failed)}


def run(threshold: float | None = None) -> dict:
    """Check every unique hit composition (at the loosest threshold) against databases and literature."""
    from .candidates import load_candidates
    from .hull import load_evaluations
    from .mp_fetch import load_mp_index

    cfg = load_config()
    thr = threshold if threshold is not None else max([cfg["labels"]["hit_threshold"]] + cfg["labels"]["extra_thresholds"])
    ev = load_evaluations("chgnet")
    mp_index = load_mp_index()
    cands = {c["id"]: c for c in load_candidates()}
    by_formula = {}
    for cid, r in ev.items():
        if r.get("ok") and r["e_hull"] <= thr:
            by_formula.setdefault(cands[cid]["reduced_formula"], cands[cid])
    log.info("%d unique hit formulas at e_hull <= %.3f", len(by_formula), thr)
    out = load_json(CACHE / "novelty.json", {})
    todo = [(f, c) for f, c in sorted(by_formula.items())
            if not (f in out and "unchecked" not in out[f]["db"]["summary"]
                    and out[f].get("lit", {}).get("lit_status") != "unchecked")]

    def check(item):
        f, cand = item
        db = database_status(f, mp_index)
        lit = literature_status(cand)  # all hits: also decides "known" (exact formula reported)
        return f, {"formula": f, "display_formula": cand["display_formula"], "chemsys": chemsys_key(cand["elements"]),
                   "db": db, "lit": lit}

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (f, rec) in enumerate(pool.map(check, todo)):
            out[f] = rec
            log.info("[%d/%d] %s: db=%s lit=%s exact=%s", i + 1, len(todo), rec["display_formula"],
                     rec["db"]["summary"], rec["lit"]["lit_status"], rec["lit"]["exact_reported"])
            if i % 10 == 0:
                dump_json(out, CACHE / "novelty.json")
    dump_json(out, CACHE / "novelty.json")
    return out


def load_novelty() -> dict:
    return load_json(CACHE / "novelty.json", {})


if __name__ == "__main__":
    import sys

    run(float(sys.argv[1]) if len(sys.argv) > 1 else None)
