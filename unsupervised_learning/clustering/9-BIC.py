#!/usr/bin/env python3
"""
Calculates the Bayesian Information Criterion for a Gaussian Mixture Model
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM using
    the Bayesian Information Criterion
    """
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

    if type(tol) not in [int, float] or tol < 0:
        return None, None, None, None

    if type(verbose) is not bool:
        return None, None, None, None

    num_k = kmax - kmin + 1
    a = np.zeros(num_k)
    b = np.zeros(num_k)
    results = []
    for k in range(kmin, kmax + 1):
        pi, m, S, g, log_l = expectation_maximization(
            X, k, iterations, tol, verbose
        )
        if pi is None or m is None or S is None or log_l is None:
            return None, None, None, None

        results.append((pi, m, S))
        idx = k - kmin
        a[idx] = log_l

        p = (k - 1) + (k * d) + (k * d * (d + 1) / 2)
        b[idx] = p * np.log(n) - (2 * log_l)

    best_index = np.argmin(b)
    best_k = kmin + best_index
    best_result = results[best_index]

    return best_k, best_result, a, b
