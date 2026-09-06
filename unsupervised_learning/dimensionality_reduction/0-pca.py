#!/usr/bin/env python3
"""Performs Principal Component Analysis (PCA) on a dataset."""
import numpy as np


def pca(X, var=0.95):
    """
    Calculates the PCA weights matrix maintaining var fraction of variance
    """
    u, s, vh = np.linalg.svd(X)
    cum_var = np.cumsum(s) / np.sum(s)
    nd = np.argmax(cum_var >= var) + 1
    W = vh[:nd].T
    return W
