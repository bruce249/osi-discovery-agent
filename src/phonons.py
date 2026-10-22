"""Third check for the top new candidates: are they dynamically stable (no imaginary phonon modes) with CHGNet?

Idea from the Lattice studio (phonopy finite displacements on an ML potential). For each candidate in
results/top_candidates.csv: take its lowest-energy CHGNet structure, re-relax tightly (fmax 0.01 eV/Å),
build a supercell with every axis >= 9 Å, finite displacements of 0.01 Å, CHGNet forces, phonopy force constants,
frequencies on a Gamma-centred 8x8x8 mesh. Verdict tiers follow the Lattice studio's phonon diagnostics:
stable (lowest frequency >= -0.1 THz), marginal (-1 to -0.1 THz: soft modes that need convergence checks),
unstable (< -1 THz, where Lattice stops further screening).

    python -m src.phonons prepare   # writes data/cache/phonon_tasks.json
    python -m src.phonons collect   # writes results/phonons.json after modal_app.py::phonons
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from .utils import CACHE, RESULTS, dump_json, get_logger, load_json

log = get_logger("phonons")
TASKS = CACHE / "phonon_tasks.json"
OUT = CACHE / "phonons"
STABLE_THZ = -0.1
SEVERE_THZ = -1.0


def prepare() -> list[dict]:
    import pandas as pd

    from .hull import load_evaluations
    from .mlip import load_result

    top = pd.read_csv(RESULTS / "top_candidates.csv")
    evals = load_evaluations("chgnet")
    tasks = []
    for _, r in top.iterrows():
        ev = evals[r["id"]]
        res = load_result("chgnet", ev["best_ordering"])
        tasks.append({"id": r["id"], "formula": r["formula"], "structure": res["final_structure"]})
    dump_json(tasks, TASKS)
    log.info("%d phonon tasks", len(tasks))
    return tasks


def run_phonons(task: dict, min_len: float = 9.0) -> dict:
    """Worker: tight relax + finite-displacement phonons with CHGNet. Never raises."""
    t0 = time.time()
    try:
        from ase import Atoms
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE
        from phonopy import Phonopy
        from phonopy.structure.atoms import PhonopyAtoms
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor

        from .mlip import _get_calc

        calc = _get_calc("chgnet")
        s = Structure.from_dict(task["structure"])
        s.remove_oxidation_states()
        s = s.get_reduced_structure()  # Niggli-reduced cell (spglib failed on one skewed cell otherwise)
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = calc
        FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.01, steps=400)
        fmax = float(np.abs(atoms.get_forces()).max())
        unit = PhonopyAtoms(symbols=atoms.get_chemical_symbols(), cell=atoms.cell[:],
                            scaled_positions=atoms.get_scaled_positions())
        sc = [max(1, math.ceil(min_len / x)) for x in atoms.cell.lengths()]
        ph = Phonopy(unit, supercell_matrix=np.diag(sc), primitive_matrix=None)
        ph.generate_displacements(distance=0.01, is_plusminus=True)
        forces = []
        for d in ph.supercells_with_displacements:
            a = Atoms(d.symbols, cell=d.cell, scaled_positions=d.scaled_positions, pbc=True)
            a.calc = calc
            f = a.get_forces()
            forces.append(f - f.mean(0))
        ph.forces = forces
        ph.produce_force_constants()
        ph.symmetrize_force_constants()
        ph.run_mesh([8, 8, 8], is_gamma_center=True, with_eigenvectors=True)
        mesh = ph.get_mesh_dict()
        freqs = np.asarray(mesh["frequencies"])
        fmin = float(freqs.min())
        # share of the softest mode's motion carried by Li atoms (Li is 30-33% of the atoms)
        iq, ib = np.unravel_index(int(freqs.argmin()), freqs.shape)
        w = (np.abs(np.asarray(mesh["eigenvectors"])[iq][:, ib]) ** 2).reshape(-1, 3).sum(1)
        is_li = np.array([e == "Li" for e in atoms.get_chemical_symbols()])
        li_share = float(w[is_li].sum() / w.sum())
        n_imag = int((freqs < STABLE_THZ).sum())
        return {"id": task["id"], "formula": task["formula"], "ok": True, "min_freq_thz": fmin,
                "n_modes": int(freqs.size), "n_imag_below_tol": n_imag, "frac_imag": n_imag / freqs.size,
                "stable": fmin >= STABLE_THZ,
                "verdict": "stable" if fmin >= STABLE_THZ else ("marginal" if fmin >= SEVERE_THZ else "unstable"),
                "supercell": sc, "natoms_unit": len(atoms),
                "n_displacements": len(forces),
                "soft_mode_li_share": li_share, "li_atom_share": float(is_li.mean()), "residual_fmax": fmax, "seconds": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"id": task["id"], "formula": task.get("formula"), "ok": False,
                "error": f"{type(e).__name__}: {e}"[:400], "seconds": time.time() - t0}


def collect() -> dict:
    rows = [load_json(p) for p in sorted(OUT.glob("*.json"))]
    ok = [r for r in rows if r and r.get("ok")]
    out = {"tolerance_thz": STABLE_THZ, "severe_thz": SEVERE_THZ,
           "n_marginal": sum(r["verdict"] == "marginal" for r in ok), "n": len(rows), "n_ok": len(ok),
           "n_stable": sum(r["stable"] for r in ok), "rows": rows}
    dump_json(out, RESULTS / "phonons.json")
    log.info("phonons: %d/%d dynamically stable (%d computed)", out["n_stable"], len(ok), len(rows))
    for r in rows:
        log.info("  %s %s", r.get("formula"), f"min {r['min_freq_thz']:.2f} THz" if r.get("ok") else r.get("error"))
    return out


if __name__ == "__main__":
    {"prepare": prepare, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "prepare"]()
