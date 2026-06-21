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

    def pdf(self, x):
        """
        Calculates the PDF at a data point x.

        Parameters:
        x (numpy.ndarray): shape (d, 1) containing the data point

        Returns:
        float: The value of the PDF at x
        """
        # Determine the dimensions (d) from the covariance matrix
        d = self.cov.shape[0]

        # Type validation
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        # Shape validation: must be exactly (d, 1)
        if len(x.shape) != 2 or x.shape[0] != d or x.shape[1] != 1:
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Compute components of the multivariate normal PDF formula
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)

        norm_factor = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv), x_centered)

        # Exponent is a 1x1 matrix; extract the scalar value using .item()
        pdf_val = norm_factor * np.exp(exponent.item())

        return pdf_val
