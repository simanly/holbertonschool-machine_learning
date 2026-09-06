#!/usr/bin/env python3
"""Calculates the total intra-cluster variance for a data set."""
import numpy as np


def variance(X, C):
    """Calculates the total intra-cluster variance for a data set."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    try:
        distances = np.sum((X[:, np.newaxis] - C) ** 2, axis=-1)
        min_distances = np.min(distances, axis=-1)
        return np.sum(min_distances)
    except Exception:
        return None
