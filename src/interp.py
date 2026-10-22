"""Interpretability of the stability signal inside MACE-MP-0 (builds on src/latent.py's cached descriptors).

1. Where: probe each layer (L1 = first message-passing layer, L2 = second) x each site role (Li, metal, halide).
   Descriptor layout from latent.py: [Li | metal | halide] x [L1 (128) | L2 (128)] invariants.
2. Causal test: iteratively remove the linear "stability directions" (INLP: refit ridge, project out its direction,
   10 times) from the trained features, then rerun the latent GP campaign on the ablated features
   (gp_latent_inlp). Control: remove 10 random directions (gp_latent_ablrand). If the search depends on this
   information, INLP should hurt it and random removal should not.
3. Meaning: correlate the probe's stability score with textbook descriptors (metal ionic radius and
   electronegativity, Br fraction, volume per atom, Li2MX6 vs Li3MX6) and measure how much of the score they explain.

    python -m src.interp        # writes ablated feature files, results/interp.json, results/fig4_interp.png
"""
from __future__ import annotations

import numpy as np

from .latent import _file, _group, _splits
from .utils import RESULTS, dump_json, get_logger, load_config, load_json

log = get_logger("interp")
N_ABLATE = 10
ROLES = ("Li", "metal", "halide")


def _std(X):
    sd = X.std(0)
    sd[sd < 1e-9] = 1.0
    return (X - X.mean(0)) / sd


def _probe(X, y, hit, groups, reps=5):
    from scipy.stats import spearmanr
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rho, auc, r2 = [], [], []
    for rep in range(reps):
        pred = np.zeros_like(y)
        for tr, te in _splits(groups, rep):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 6, 30)))
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        rho.append(spearmanr(y, pred).statistic)
        auc.append(roc_auc_score(hit, -pred))
        r2.append(r2_score(y, pred))
    return {"spearman": float(np.mean(rho)), "auc": float(np.mean(auc)), "r2_median": float(np.median(r2))}


def _cols(role: str, layer: str) -> np.ndarray:
    r = ROLES.index(role)
    base = r * 256
    if layer == "L1":
        return np.arange(base, base + 128)
    if layer == "L2":
        return np.arange(base + 128, base + 256)
    return np.arange(base, base + 256)


def _inlp(Xs, y, k):
    from sklearn.linear_model import RidgeCV

    X = Xs.copy()
    for _ in range(k):
        w = RidgeCV(alphas=np.logspace(-3, 6, 30)).fit(X, y).coef_
        w = w / np.linalg.norm(w)
        X = X - np.outer(X @ w, w)
    return X


def _rand_ablate(Xs, k, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(Xs.shape[1], k)))
    return Xs - (Xs @ Q) @ Q.T


def _descriptors(ev, ids):
    from pymatgen.core import Composition, Element, Species, Structure

    rows = []
    for i in ids:
        c = ev.candidate(i)
        comp = Composition(c["reduced_formula"])
        ox = c.get("oxidation_states", {})
        metals = [e for e in comp.elements if e.symbol not in ("Li", "Cl", "Br", "F", "I")]
        wts = np.array([comp[e] for e in metals], dtype=float)
        radius = np.array([Species(e.symbol, ox.get(e.symbol, 3)).ionic_radius or np.nan for e in metals], dtype=float)
        chi = np.array([Element(e.symbol).X for e in metals], dtype=float)
        hal = comp.get("Cl", 0) + comp.get("Br", 0)
        s = Structure.from_dict(c["orderings"][0]["structure"])
        rows.append({"metal ionic radius": float(np.nansum(radius * wts) / wts.sum()),
                     "metal electronegativity": float((chi * wts).sum() / wts.sum()),
                     "Br fraction of halide": float(comp.get("Br", 0) / hal) if hal else 0.0,
                     "volume per atom (unrelaxed)": float(s.volume / len(s)),
                     "Li2MX6 type (vs Li3MX6)": float(comp["Li"] / comp.num_atoms < 0.27)})
    return rows


def analyse() -> dict:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression, RidgeCV

    from .evaluator import MLIPEvaluator

    cfg = load_config()
    thr = cfg["labels"]["hit_threshold"]
    ev = MLIPEvaluator("chgnet")
    ids = sorted(ev.candidate_ids())
    y = np.array([min(ev.evals[i]["e_hull"], 1.0) for i in ids])
    hit = (y <= thr + 1e-9).astype(int)
    groups = np.array([_group(ev.candidate(i)) for i in ids])
    feats = {k: np.array([load_json(_file(k))[i] for i in ids]) for k in ("mace_trained", "mace_random")}
    out = {"n": len(ids)}

    # 1. layer x role
    grid = {}
    for kind, X in feats.items():
        for role in (*ROLES, "all"):
            for layer in ("L1", "L2", "both"):
                cols = np.concatenate([_cols(r, layer) for r in ROLES]) if role == "all" else _cols(role, layer)
                grid[f"{kind}|{role}|{layer}"] = _probe(X[:, cols], y, hit, groups, reps=3)
    out["layer_role"] = grid
    for k, v in grid.items():
        if k.startswith("mace_trained"):
            log.info("%-28s rho %.2f AUC %.2f", k, v["spearman"], v["auc"])

    # 2. ablations (features saved for the campaign arms)
    Xs = _std(feats["mace_trained"])
    X_inlp, X_rand = _inlp(Xs, y, N_ABLATE), _rand_ablate(Xs, N_ABLATE)
    dump_json({i: x.tolist() for i, x in zip(ids, X_inlp)}, _file("mace_trained_inlp"))
    dump_json({i: x.tolist() for i, x in zip(ids, X_rand)}, _file("mace_trained_ablrand"))
    out["ablation_probe"] = {"none": _probe(Xs, y, hit, groups), "inlp": _probe(X_inlp, y, hit, groups),
                             "random_dirs": _probe(X_rand, y, hit, groups), "k": N_ABLATE}
    log.info("ablation probe: %s", {k: round(v["auc"], 3) for k, v in out["ablation_probe"].items() if k != "k"})

    # 3. chemical meaning of the stability direction
    w = RidgeCV(alphas=np.logspace(-3, 6, 30)).fit(Xs, y).coef_
    score = Xs @ w  # higher = predicted less stable
    desc = _descriptors(ev, ids)
    names = list(desc[0])
    D = np.array([[d[n] for n in names] for d in desc])
    D = np.where(np.isfinite(D), D, np.nanmean(D, 0))
    out["meaning"] = {n: {"rho_with_probe_score": float(spearmanr(D[:, j], score).statistic),
                          "rho_with_e_hull": float(spearmanr(D[:, j], y).statistic)} for j, n in enumerate(names)}
    out["meaning_r2_descriptors_explain_score"] = float(LinearRegression().fit(D, score).score(D, score))
    out["meaning_r2_descriptors_explain_e_hull"] = float(LinearRegression().fit(D, y).score(D, y))
    log.info("descriptors explain %.2f of probe score, %.2f of e_hull", out["meaning_r2_descriptors_explain_score"],
             out["meaning_r2_descriptors_explain_e_hull"])
    for n, v in out["meaning"].items():
        log.info("  %-28s rho(score) %+.2f  rho(e_hull) %+.2f", n, v["rho_with_probe_score"], v["rho_with_e_hull"])
    return out


def figure(out: dict) -> None:
    import matplotlib.pyplot as plt

    summ = load_json(RESULTS / "summary.json", {})
    tk = f"{summ.get('headline_threshold', 0.0):.2f}"
    st = summ.get("strategies", {})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)

    ax = axes[0]
    M = np.array([[out["layer_role"][f"mace_trained|{r}|{l}"]["spearman"] for l in ("L1", "L2", "both")]
                  for r in (*ROLES, "all")])
    R = np.array([[out["layer_role"][f"mace_random|{r}|{l}"]["spearman"] for l in ("L1", "L2", "both")]
                  for r in (*ROLES, "all")])
    im = ax.imshow(M, cmap="viridis", vmin=0.4, vmax=0.9)
    for a in range(M.shape[0]):
        for b in range(M.shape[1]):
            ax.text(b, a, f"{M[a, b]:.2f}\n({R[a, b]:.2f})", ha="center", va="center", fontsize=8,
                    color="w" if M[a, b] < 0.7 else "k")
    ax.set_xticks(range(3), ["layer 1", "layer 2", "both"])
    ax.set_yticks(range(4), ["Li sites", "metal sites", "halide sites", "all sites"])
    ax.set_title("Where is stability encoded?\nprobe rank corr., trained (scrambled)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    arms = [("gp_latent", "trained features"), ("gp_latent_ablrand", f"minus {N_ABLATE} random dirs"),
            ("gp_latent_inlp", f"minus {N_ABLATE} stability dirs"), ("gp_latent_rand", "scrambled weights"),
            ("gp_bo", "GP-BO (composition)"), ("random", "random search")]
    arms = [(k, l) for k, l in arms if k in st]
    vals = [st[k][tk]["hit"]["at_budget_mean"] for k, _ in arms]
    errs = [st[k][tk]["hit"]["at_budget_std"] for k, _ in arms]
    cols = ["#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#7f7f7f"][:len(arms)]
    ax.barh(range(len(arms)), vals, xerr=errs, color=cols, capsize=3)
    ax.set_yticks(range(len(arms)), [l for _, l in arms])
    ax.invert_yaxis()
    ax.set_xlabel(f"stable compositions found in {summ.get('budget', 90)} simulations")
    ab = out["ablation_probe"]
    ax.set_title(f"Causal test: delete the stability directions, re-run the search\nprobe AUC {ab['none']['auc']:.2f}"
                 f" → {ab['inlp']['auc']:.2f} (random dirs: {ab['random_dirs']['auc']:.2f})", fontsize=10)

    ax = axes[2]
    mn = out["meaning"]
    names = list(mn)
    ax.barh(np.arange(len(names)) - 0.2, [mn[n]["rho_with_probe_score"] for n in names], 0.38,
            color="#1f77b4", label="vs probe's stability score")
    ax.barh(np.arange(len(names)) + 0.2, [mn[n]["rho_with_e_hull"] for n in names], 0.38,
            color="#ff7f0e", label="vs simulated e_hull")
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("rank correlation (positive = less stable)")
    ax.set_title(f"What does the direction mean?\ntextbook descriptors explain "
                 f"{out['meaning_r2_descriptors_explain_score']:.0%} of it", fontsize=10)
    ax.legend(fontsize=8)
    fig.savefig(RESULTS / "fig4_interp.png", dpi=150)
    plt.close(fig)


def main():
    out = analyse()
    dump_json(out, RESULTS / "interp.json")
    figure(out)


if __name__ == "__main__":
    main()
