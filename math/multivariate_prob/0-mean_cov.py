#!/usr/bin/env python3
"""
Module to calculate the mean and covariance of a 2D dataset.
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set X.
    """
    # Validation checks
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Calculate mean with shape (1, d)
    mean = np.mean(X, axis=0, keepdims=True)

    # Calculate sample covariance with shape (d, d)
    X_centered = X - mean
    cov = np.dot(X_centered.T, X_centered) / (n - 1)

    return mean, cov
