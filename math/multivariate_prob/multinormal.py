#!/usr/bin/env python3
"""
Module defining the MultiNormal class
representing a Multivariate Normal distribution.
"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution.
    """

    def __init__(self, data):
        """
        Initializes the MultiNormal distribution with a data set.
        """
        # Type validation
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        # Extract shapes where rows are dimensions (d)
        # and columns are samples(n)
        d, n = data.shape

        # Value validation
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate mean along axis 1 (across data points),
        # keeping shape (d, 1)
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Center the data: (d, n) - (d, 1) -> (d, n)
        data_centered = data - self.mean

        # Calculate sample covariance matrix: (d, n) dot (n, d) -> (d, d)
        # Using Bessel's correction (n - 1)
        self.cov = np.dot(data_centered, data_centered.T) / (n - 1)
