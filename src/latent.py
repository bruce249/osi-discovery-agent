"""Inside the simulator: what MACE-MP-0's internal features already know about stability, before any relaxation.

Method follows the osi-draft probing study (MACE-MP-0 site descriptors vs geometry / random-weight controls,
splits grouped by chemistry): one forward pass of MACE-MP-0 medium on each candidate's UNRELAXED structure,
invariant descriptors averaged over Li, metal and halide sites (3 x 256 dims). Controls:
  mace_random : the same network with every parameter tensor randomly permuted (seed 0): architecture, no training
  magpie      : the composition features the GP-BO arm already uses
  shuffled    : trained MACE features with training labels permuted (pipeline leak check)
Probe: StandardScaler + RidgeCV, 5-fold splits grouped by chemistry (metal set + halide set), 5 repeats.
Targets: CHGNet e_hull (clipped at 1 eV/atom, as in GP-BO) and the hit label e_hull <= threshold (ROC-AUC).

    python -m src.latent            # features (cached) + probe + figure
"""
from __future__ import annotations

import numpy as np

from .utils import CACHE, RESULTS, dump_json, get_logger, load_config, load_json

log = get_logger("latent")
KINDS = ("mace_trained", "mace_random")
HALIDES = {"F", "Cl", "Br", "I"}


def _file(kind: str):
    return CACHE / f"latent_{kind}.json"


def _calc(random_weights: bool):
    import torch
    from mace.calculators import mace_mp

    calc = mace_mp(model="medium", default_dtype="float64", device="cpu")
    if random_weights:
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for m in calc.models:
                for p in m.parameters():
                    flat = p.data.view(-1)
                    flat.copy_(flat[torch.randperm(flat.numel(), generator=g)])
    return calc


def _pooled(calc, struct_dict: dict) -> list[float]:
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor

    s = Structure.from_dict(struct_dict)
    s.remove_oxidation_states()
    atoms = AseAtomsAdaptor.get_atoms(s)
    d = np.asarray(calc.get_descriptors(atoms, invariants_only=True))
    sym = atoms.get_chemical_symbols()
    roles = [[i for i, e in enumerate(sym) if e == "Li"],
             [i for i, e in enumerate(sym) if e != "Li" and e not in HALIDES],
             [i for i, e in enumerate(sym) if e in HALIDES]]
    return np.concatenate([d[r].mean(0) if r else np.zeros(d.shape[1]) for r in roles]).tolist()


def build(force: bool = False) -> None:
    """Descriptors of every candidate's first (unrelaxed) ordering, for the trained and the random-weight network."""
    from .candidates import load_candidates

    cands = load_candidates()
    for kind in KINDS:
        if load_json(_file(kind)) and not force:
            continue
        calc = _calc(kind == "mace_random")
        out = {}
        for i, c in enumerate(cands):
            out[c["id"]] = _pooled(calc, c["orderings"][0]["structure"])
            if i % 50 == 0:
                log.info("%s %d/%d", kind, i, len(cands))
        dump_json(out, _file(kind))
        log.info("%s: %d candidates x %d dims", kind, len(out), len(next(iter(out.values()))))


def latent_matrix(ids: list[str], kind: str = "mace_trained", n_pca: int = 20) -> np.ndarray:
    """Standardised descriptors reduced by PCA (unsupervised, no labels) for the GP arms."""
    from sklearn.decomposition import PCA

    feats = load_json(_file(kind))
    if not feats:
        raise RuntimeError(f"no {kind} features: run `python -m src.latent` first")
    X = np.array([feats[i] for i in ids], dtype=float)
    sd = X.std(0)
    X = (X[:, sd > 1e-9] - X[:, sd > 1e-9].mean(0)) / sd[sd > 1e-9]
    return PCA(n_components=min(n_pca, X.shape[1], len(ids)), random_state=0).fit_transform(X)


def _group(c: dict) -> str:
    from pymatgen.core import Composition

    els = {e.symbol for e in Composition(c["reduced_formula"]).elements} - {"Li"}
    return "-".join(sorted(els))


def _splits(groups: np.ndarray, seed: int, k: int = 5):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    for fold in np.array_split(uniq, k):
        test = np.isin(groups, fold)
        yield np.where(~test)[0], np.where(test)[0]


def probe() -> dict:
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .evaluator import MLIPEvaluator
    from .features import feature_matrix

    cfg = load_config()
    thr = cfg["labels"]["hit_threshold"]
    ev = MLIPEvaluator("chgnet")
    ids = sorted(ev.candidate_ids())
    y = np.array([min(ev.evals[i]["e_hull"], 1.0) for i in ids])
    hit = (y <= thr + 1e-9).astype(int)
    groups = np.array([_group(ev.candidate(i)) for i in ids])
    reps = {"mace_trained": np.array([load_json(_file("mace_trained"))[i] for i in ids]),
            "mace_random": np.array([load_json(_file("mace_random"))[i] for i in ids]),
            "magpie": feature_matrix(ids)}
    rows = {}
    for name, X in [*reps.items(), ("shuffled", reps["mace_trained"])]:
        r2s, rhos, aucs, fold_r2 = [], [], [], []
        for rep in range(5):
            pred = np.zeros_like(y)
            rng = np.random.default_rng(100 + rep)
            for tr, te in _splits(groups, rep):
                ytr = rng.permutation(y[tr]) if name == "shuffled" else y[tr]
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 6, 30)))
                m.fit(X[tr], ytr)
                pred[te] = m.predict(X[te])
                if len(te) > 2:
                    fold_r2.append(r2_score(y[te], pred[te]))
            r2s.append(r2_score(y, pred))
            rhos.append(spearmanr(y, pred).statistic)
            aucs.append(roc_auc_score(hit, -pred))
        rows[name] = {"r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)), "r2_median": float(np.median(r2s)),
                      "r2_folds_median": float(np.median(fold_r2)),
                      "spearman_mean": float(np.mean(rhos)), "spearman_std": float(np.std(rhos)),
                      "auc_hit_mean": float(np.mean(aucs)), "auc_hit_std": float(np.std(aucs))}
        log.info("%-13s R2 %.2f ± %.2f  rho %.2f  AUC(hit) %.2f", name, rows[name]["r2_mean"], rows[name]["r2_std"],
                 rows[name]["spearman_mean"], rows[name]["auc_hit_mean"])
    out = {"n_candidates": len(ids), "n_groups": int(len(np.unique(groups))), "hit_threshold": thr,
           "base_rate_hit": float(hit.mean()), "probe": rows,
           "protocol": "ridge on unrelaxed-structure descriptors, 5-fold chemistry-grouped splits x 5 repeats"}
    dump_json(out, RESULTS / "latent_probe.json")
    _figure(ids, y, reps, out)
    return out


def _figure(ids, y, reps, out) -> None:
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for ax, name, title in zip(axes, ("mace_trained", "mace_random"),
                               ("MACE-MP-0, trained weights", "same network, scrambled weights")):
        Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(reps[name]))
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.clip(y, -0.1, 0.2), cmap="coolwarm", s=22, edgecolor="k", lw=0.3)
        p = out["probe"][name]
        ax.set_title(f"{title}\nheld-out R² {p['r2_mean']:.2f}, AUC(stable) {p['auc_hit_mean']:.2f}", fontsize=10)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label("CHGNet energy above hull after relaxation (eV/atom)")
    fig.suptitle("What the simulator 'sees' before it simulates: unrelaxed candidates in MACE's internal feature space",
                 fontsize=11)
    fig.savefig(RESULTS / "fig3_latent.png", dpi=160)
    plt.close(fig)


def main():
    build()
    probe()


if __name__ == "__main__":
    main()
