#!/usr/bin/env python3

import numpy as np

def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix X using mean and standard deviation.

    Parameters:
    - X: numpy.ndarray of shape (d, nx) to normalize
    - m: numpy.ndarray of shape (nx,) containing the mean of all features
    - s: numpy.ndarray of shape (nx,) containing the standard deviation of all features

    Returns:
    - Normalized numpy.ndarray of shape (d, nx)
    """
    return (X - m) / s
