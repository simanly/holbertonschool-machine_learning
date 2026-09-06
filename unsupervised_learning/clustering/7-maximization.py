#!/usr/bin/env python3
"""Calculates maximization step in EM algorithm for a GMM."""
import numpy as np


def maximization(X, g):
    """Calculates maximization step in EM algorithm for a GMM."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k, n_g = g.shape

    if n != n_g:
        return None, None, None

    if not np.isclose(np.sum(g, axis=0), 1).all():
        return None, None, None

    N_k = np.sum(g, axis=1)
    pi = N_k / n
    m = (g @ X) / N_k[:, np.newaxis]

    S = np.zeros((k, d, d))
    for i in range(k):
        x_m = X - m[i]
        S[i] = (g[i] * x_m.T) @ x_m / N_k[i]

    return pi, m, S
