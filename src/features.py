"""Composition features (matminer Magpie) + prototype one-hot + family flags for the GP-BO arm. Cached."""
from __future__ import annotations

import numpy as np

from .utils import CACHE, dump_json, get_logger, load_json

log = get_logger("features")
FEAT_FILE = CACHE / "features.json"


def build_features(force: bool = False) -> dict[str, list[float]]:
    cached = load_json(FEAT_FILE)
    if cached and not force:
        return cached
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    from .candidates import load_candidates

    cands = load_candidates()
    ep = ElementProperty.from_preset("magpie")
    protos = sorted({c["prototype"] for c in cands})
    fams = sorted({c["family"] for c in cands})
    feats = {}
    for c in cands:
        comp = Composition(c["reduced_formula"])
        x = list(ep.featurize(comp))
        x += [1.0 if c["prototype"] == p else 0.0 for p in protos]
        x += [1.0 if c["family"] == f else 0.0 for f in fams]
        feats[c["id"]] = [float(v) if v is not None and np.isfinite(v) else 0.0 for v in x]
    dump_json(feats, FEAT_FILE)
    log.info("features: %d candidates x %d dims", len(feats), len(next(iter(feats.values()))))
    return feats


def feature_matrix(ids: list[str]) -> np.ndarray:
    feats = build_features()
    X = np.array([feats[i] for i in ids], dtype=float)
    return X


if __name__ == "__main__":
    build_features(force=True)
