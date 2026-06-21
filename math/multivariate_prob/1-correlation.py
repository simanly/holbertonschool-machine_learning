#!/usr/bin/env python3
"""
Module to calculate the correlation matrix from a covariance matrix.
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix C.
    """
    # Type validation
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    # Shape validation (must be a 2D square matrix)
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    # Extract the diagonal elements (variances)
    variances = np.diag(C)

    # Calculate standard deviations from the variances
    std_devs = np.sqrt(variances)

    # Use outer product to create a matrix of (std_dev_i * std_dev_j)
    # shape: (d, 1) * (1, d) -> (d, d)
    std_matrix = np.outer(std_devs, std_devs)

    # Element-wise division to get the correlation matrix
    corr_matrix = C / std_matrix

    return corr_matrix
