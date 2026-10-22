"""ML interatomic potential relaxations (CHGNet primary, MACE-MP-0 second), cached by input-structure hash.

The heavy work runs on Modal workers (see modal_app.py). This module holds:
  * relax_structure(): the worker-side relaxation (ASE FIRE + FrechetCellFilter, step and wall-clock limits)
  * cache helpers used by every downstream module
  * collect_tasks(): every structure that must be relaxed (candidate orderings + reference phases)
"""
from __future__ import annotations

import math
import os
import time
from pathlib import Path

from .utils import CACHE, get_logger, load_config, load_json, dump_json, structure_hash

log = get_logger("mlip")

_CALCS: dict = {}


def mlip_dir(model: str) -> Path:
    d = CACHE / "mlip" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def result_path(model: str, h: str) -> Path:
    return mlip_dir(model) / f"{h}.json"


def load_result(model: str, h: str) -> dict | None:
    return load_json(result_path(model, h))


def all_results(model: str) -> dict[str, dict]:
    out = {}
    for p in mlip_dir(model).glob("*.json"):
        r = load_json(p)
        if r:
            out[p.stem] = r
    return out


# ----------------------------------------------------------------------------- worker side

def _get_calc(model: str, mace_model: str = "medium"):
    if model in _CALCS:
        return _CALCS[model]
    import torch

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    if model == "chgnet":
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        calc = CHGNetCalculator(model=CHGNet.load(), use_device="cpu")
    elif model == "mace":
        from mace.calculators import mace_mp

        calc = mace_mp(model=mace_model, default_dtype="float64", device="cpu")
    else:
        raise ValueError(model)
    _CALCS[model] = calc
    return calc


class _WallClock(Exception):
    pass


def relax_structure(struct_dict: dict, model: str, fmax: float, steps: int, timeout_s: float,
                    mace_model: str = "medium") -> dict:
    """Relax positions + cell. Never raises: failures come back as {"ok": False, "error": ...}."""
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    t0 = time.time()
    try:
        from ase.filters import FrechetCellFilter
    except ImportError:  # older ASE
        from ase.constraints import ExpCellFilter as FrechetCellFilter
    from ase.optimize import FIRE

    try:
        s = Structure.from_dict(struct_dict)
        s.remove_oxidation_states()
        h = structure_hash(s)
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = _get_calc(model, mace_model)
        v0 = atoms.get_volume()
        opt = FIRE(FrechetCellFilter(atoms), logfile=None)

        def _check():
            if time.time() - t0 > timeout_s:
                raise _WallClock(f"wall-clock limit {timeout_s}s")

        opt.attach(_check, interval=1)
        converged = bool(opt.run(fmax=fmax, steps=steps))
        e_pa = float(atoms.get_potential_energy()) / len(atoms)
        if not math.isfinite(e_pa):
            raise ValueError("non-finite energy")
        final = AseAtomsAdaptor.get_structure(atoms)
        return {
            "ok": True, "hash": h, "model": model, "e_per_atom": e_pa,
            "natoms": len(atoms), "steps": int(opt.nsteps), "converged": converged,
            "volume_ratio": float(atoms.get_volume() / v0),
            "final_structure": final.as_dict(), "seconds": time.time() - t0,
        }
    except Exception as e:  # noqa: BLE001  - any failure is recorded and skipped
        return {"ok": False, "model": model, "error": f"{type(e).__name__}: {e}",
                "seconds": time.time() - t0,
                "hash": structure_hash(Structure.from_dict(struct_dict))}


# ----------------------------------------------------------------------------- task collection

def collect_tasks(model: str = "chgnet") -> list[dict]:
    """All structures needing a primary-model relaxation: references first (validation needs them), then candidates."""
    from .candidates import load_candidates
    from .mp_fetch import load_references

    cfg = load_config()
    tasks, seen = [], set()
    for ref in load_references().values():
        if ref["hash"] not in seen:
            seen.add(ref["hash"])
            tasks.append({"hash": ref["hash"], "kind": "reference", "label": ref["material_id"],
                          "structure": ref["structure"], "natoms": ref["nsites"]})
    for c in load_candidates():
        for o in c["orderings"]:
            if o["hash"] not in seen:
                seen.add(o["hash"])
                tasks.append({"hash": o["hash"], "kind": "candidate", "label": c["id"],
                              "structure": o["structure"], "natoms": o["natoms"]})
    for t in tasks:
        t.update(model=model, fmax=cfg["mlip"]["fmax"], steps=cfg["mlip"]["steps"],
                 timeout_s=cfg["mlip"]["timeout_s"], mace_model=cfg["mlip"]["mace_model"])
    return tasks


def pending(tasks: list[dict]) -> list[dict]:
    return [t for t in tasks if not result_path(t["model"], t["hash"]).exists()]


def save_result(task: dict, res: dict) -> None:
    res = dict(res)
    res["hash"] = task["hash"]  # key by the task hash so lookups never depend on re-hashing
    res["kind"] = task.get("kind")
    res["label"] = task.get("label")
    dump_json(res, result_path(task["model"], task["hash"]))


def status(model: str = "chgnet") -> dict:
    tasks = collect_tasks(model)
    res = all_results(model)
    done = [t for t in tasks if t["hash"] in res]
    ok = [t for t in done if res[t["hash"]].get("ok")]
    secs = [res[t["hash"]]["seconds"] for t in ok]
    return {"tasks": len(tasks), "done": len(done), "ok": len(ok),
            "mean_seconds": sum(secs) / len(secs) if secs else None,
            "unconverged": sum(1 for t in ok if not res[t["hash"]].get("converged"))}


if __name__ == "__main__":
    import json

    print(json.dumps(status(), indent=1))
