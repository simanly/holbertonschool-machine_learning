#!/usr/bin/env python3
"""
Expectation Maximization for a Gaussian Mixture Model (GMM)
"""
import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a GMM.

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing data set
    - k: positive integer containing number of clusters
    - iterations: positive integer containing max iterations
    - tol: non-negative float containing tolerance of log likelihood
    - verbose: boolean determining if info should be printed

    Returns:
    - pi, m, S, g, l or None, None, None, None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or type(k) is bool or k <= 0:
        return None, None, None, None, None
    if (not isinstance(iterations, int)
        or type(iterations) is bool
            or iterations <= 0):
        return None, None, None, None, None
    if not isinstance(tol, (int, float)) or type(tol) is bool or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    l_old = 0
    for i in range(iterations + 1):
        g, b = expectation(X, pi, m, S)
        if g is None or b is None:
            return None, None, None, None, None

        is_last = (i > 0 and abs(b - l_old) <= tol) or (i == iterations)

        if verbose and (i % 10 == 0 or is_last):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, b))

        if is_last:
            break

        l_old = b
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

    return pi, m, S, g, b
