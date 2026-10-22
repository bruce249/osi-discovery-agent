"""GP Bayesian optimisation: Matern GP on Magpie + prototype + family features. First round is random.

Acquisition (config campaign.gp_acquisition):
  p_hit (default) : probability that e_hull <= hit threshold under the GP posterior. For an objective that counts
                    hits, picking the k most likely hits is the greedy-optimal batch.
  ei              : expected improvement below the best e_hull seen (the brief's original choice; it chases the
                    single lowest value and over-explores here, kept for the record).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

from ..features import feature_matrix
from .base import Strategy


class GPBOStrategy(Strategy):
    name = "gp_bo"

    def __init__(self, seed, evaluator, batch_size=10, n_init=10, xi=0.01):
        super().__init__(seed, evaluator, batch_size)
        from ..utils import load_config

        cfg = load_config()
        self.n_init = n_init
        self.xi = xi
        self.acq = cfg["campaign"].get("gp_acquisition", "p_hit")
        self.thr = cfg["labels"]["hit_threshold"]
        ids = sorted(evaluator.candidate_ids())
        X = feature_matrix(ids)
        mu, sd = X.mean(0), X.std(0)
        keep = sd > 1e-9
        self.X = {i: x for i, x in zip(ids, (X[:, keep] - mu[keep]) / sd[keep])}

    def propose(self, pool_ids, history, k):
        pool = sorted(pool_ids)
        score = self.scores(pool, history)
        if score is None:
            return self.rng.sample(pool, min(k, len(pool)))
        order = np.lexsort((np.array([self.rng.random() for _ in pool]), -score))
        return [pool[i] for i in order[:k]]

    def scores(self, pool: list[str], history) -> np.ndarray | None:
        """Acquisition value for every id in `pool` (None until n_init observations exist)."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

        obs = [(c, e) for c, e in history if e is not None]
        if len(obs) < self.n_init:
            return None
        Xtr = np.array([self.X[c] for c, _ in obs])
        y = np.array([min(e, 1.0) for _, e in obs])  # clip extreme outliers so they do not dominate the GP
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.sqrt(Xtr.shape[1]), nu=2.5) + WhiteKernel(1e-3)
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=self.seed, n_restarts_optimizer=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(Xtr, y)
        Xp = np.array([self.X[c] for c in pool])
        m, s = gp.predict(Xp, return_std=True)
        s = np.maximum(s, 1e-9)
        if self.acq == "ei":
            best = y.min()
            z = (best - m - self.xi) / s
            score = (best - m - self.xi) * norm.cdf(z) + s * norm.pdf(z)
        else:
            score = norm.cdf((self.thr - m) / s)
        return score


class GPLatentStrategy(GPBOStrategy):
    """Same GP and acquisition, but the features are MACE-MP-0's internal representation of the UNRELAXED
    structure (src/latent.py, PCA to 20 dims): the surrogate reads what the simulator 'sees' before simulating."""
    name = "gp_latent"
    kind = "mace_trained"

    def __init__(self, seed, evaluator, batch_size=10, n_init=10, xi=0.01):
        super().__init__(seed, evaluator, batch_size, n_init, xi)
        from ..latent import latent_matrix

        ids = sorted(evaluator.candidate_ids())
        self.X = dict(zip(ids, latent_matrix(ids, self.kind)))


class GPLatentRandomStrategy(GPLatentStrategy):
    """Control: identical, but features from the same network with scrambled weights."""
    name = "gp_latent_rand"
    kind = "mace_random"
