#!/usr/bin/env python3
"""
Calculates the Bayesian Information Criterion for a Gaussian Mixture Model
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Finds the best number of clusters for a GMM using BIC."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None

    n, d = X.shape

    if type(kmin) is not int or kmin <= 0 or kmin > n:
        return None, None, None, None

    if kmax is None:
        kmax = n

    if type(kmax) is not int or kmax <= 0 or kmax > n:
        return None, None, None, None

    if kmin > kmax:
        return None, None, None, None

    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None

    if type(tol) not in (int, float) or type(tol) is bool or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    num_k = kmax - kmin + 1
    likes = np.zeros(num_k)
    bics = np.zeros(num_k)
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, like = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or like is None:
            return None, None, None, None

        results.append((pi, m, S))
        idx = k - kmin
        likes[idx] = like

        p = (k - 1) + (k * d) + (k * d * (d + 1) // 2)
        bics[idx] = p * np.log(n) - (2 * like)

    best_idx = np.argmin(bics)
    best_k = kmin + best_idx
    best_res = results[best_idx]

    return best_k, best_res, likes, bics
