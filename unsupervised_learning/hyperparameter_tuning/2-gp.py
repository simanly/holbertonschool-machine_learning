#!/usr/bin/env python3
"""GaussianProcess class update method."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """Initializes the GaussianProcess class."""
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """Calculates the covariance kernel matrix using RBF."""
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - \
            2 * np.dot(X1, X2.T)
        return (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)

    def predict(self, X_s):
        """Predicts the mean and variance of points in a Gaussian process."""
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu = np.matmul(K_s.T, np.matmul(K_inv, self.Y)).reshape(-1)
        sigma = np.diag(K_ss - np.matmul(K_s.T, np.matmul(K_inv, K_s)))

        return mu, sigma

    def update(self, X_new, Y_new):
        """Updates a Gaussian Process with a new sample point.

        X_new: numpy.ndarray (1,) representing the new sample point
        Y_new: numpy.ndarray (1,) representing the new sample function value
        """
        self.X = np.vstack((self.X, X_new))
        self.Y = np.vstack((self.Y, Y_new))
        self.K = self.kernel(self.X, self.X)
