#!/usr/bin/env python3
"""
Contains the intersection function for Bayesian probability
"""
import numpy as np


def intersection(x, n, P, Pr):
    """
    Calculates the intersection of obtaining data x and n
    with various hypothetical probabilities
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Helper function to calculate factorial
    def factorial(num):
        res = 1
        for i in range(1, num + 1):
            res *= i
        return res

    # Calculate binomial coefficient: n! / (x! * (n - x)!)
    n_fact = factorial(n)
    x_fact = factorial(x)
    n_x_fact = factorial(n - x)
    binomial_coefficient = n_fact / (x_fact * n_x_fact)

    # Calculate likelihood
    likelihood = binomial_coefficient * (P ** x) * ((1 - P) ** (n - x))

    # Intersection = Likelihood * Prior
    return likelihood * Pr
