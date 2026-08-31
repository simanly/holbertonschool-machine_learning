#!/usr/bin/env python3
"""BayesianOptimization class initialization."""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """Initializes the BayesianOptimization class."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        min_val, max_val = bounds
        self.X_s = np.linspace(min_val, max_val, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
