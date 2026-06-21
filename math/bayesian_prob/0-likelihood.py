#!/usr/bin/env python3
"""
Contains the likelihood function for a binomial distribution
"""
import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data x and n given
    various hypothetical probabilities in P
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

    if np.any(P < 0) or np.any(P > 1):
        raise ValueError("All values in P must be in the range [0, 1]")

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

    fact_part = binomial_coefficient
    prob_part = (P ** x) * ((1 - P) ** (n - x))

    return fact_part * prob_part
