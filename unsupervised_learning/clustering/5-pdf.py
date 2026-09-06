#!/usr/bin/env python3
"""Calculates the PDF of a Gaussian distribution."""
import numpy as np


def pdf(X, m, S):
    """Calculates the PDF of a Gaussian distribution."""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None

    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)

        norm_const = 1.0 / (np.sqrt(((2 * np.pi) ** d) * det))
        x_m = X - m
        quad = np.sum((x_m @ inv) * x_m, axis=1)

        P = norm_const * np.exp(-0.5 * quad)
        return np.maximum(P, 1e-300)
    except Exception:
        return None
