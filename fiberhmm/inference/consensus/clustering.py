# Numerical kernel promoted from the validated September 2026 consensus experiments.
from __future__ import annotations

import hashlib
import warnings
import numpy as np
from sklearn.mixture import GaussianMixture


def group_hash(group, salt):
    return int(hashlib.sha256((salt + "|" + str(group)).encode()).hexdigest()[:16], 16)


def normalized_centers(centers):
    centers = np.rint(np.asarray(centers)).astype(int)
    if centers.ndim != 2 or centers.shape[1] != 2 or np.any(centers[:, 1] <= centers[:, 0]):
        raise ValueError("Invalid rounded geometry center")
    centers = np.unique(centers, axis=0)
    return centers[np.lexsort((centers[:, 1], centers[:, 0]))]


def gmm_centers(edges, k, seed, *, regularization=1., restarts=4, max_iter=400):
    origin = edges.min(axis=0)
    if k == 1:
        return normalized_centers([np.mean(edges, axis=0)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm = GaussianMixture(k, covariance_type="full", reg_covar=regularization, n_init=restarts,
                              max_iter=max_iter, random_state=seed).fit(edges - origin)
    if not gmm.converged_:
        raise RuntimeError("GMM nomination did not converge")
    return normalized_centers(gmm.means_ + origin)
