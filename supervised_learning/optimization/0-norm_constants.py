#!/usr/bin/env python3
"""
Module for calculating feature normalization constants.
"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Parameters:
    X (numpy.ndarray): Matrix of shape (m, nx) to normalize, where
                       m is the number of data points and
                       nx is the number of features.

    Returns:
    tuple: (mean, std)
           - mean: numpy.ndarray containing the mean of each feature
           - std: numpy.ndarray containing the standard deviation of each feature
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
