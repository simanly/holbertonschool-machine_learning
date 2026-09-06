#!/usr/bin/env python3
"""Performs K-means clustering on a dataset."""
import numpy as np


def initialize(X, k):
    """Initializes cluster centroids for K-means"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None
    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    return np.random.uniform(low, high, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering and returns
    the cluster centroids and assignments
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = initialize(X, k)
    if C is None:
        return None, None

    for _ in range(iterations):
        C_prev = np.copy(C)

        distances = np.zeros((X.shape[0], k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(X - C[j], axis=1)
        clss = np.argmin(distances, axis=1)

        for j in range(k):
            points = X[clss == j]
            if len(points) == 0:
                C[j] = np.random.uniform(low, high)
            else:
                C[j] = np.mean(points, axis=0)

        distances = np.zeros((X.shape[0], k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(X - C[j], axis=1)
        clss = np.argmin(distances, axis=1)

        if (C == C_prev).all():
            break

    return C, clss
