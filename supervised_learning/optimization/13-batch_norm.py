#!/usr/bin/env python3
"""
Module containing function to perform batch normalization.
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of
    a neural network using batch normalization.

    Parameters:
    - Z: numpy.ndarray of shape (m, n) containing the unactivated outputs
    - gamma: numpy.ndarray of shape (1, n) containing scale factors
    - beta: numpy.ndarray of shape (1, n) containing offset parameters
    - epsilon: small constant to avoid division by zero

    Returns:
    - Z_norm: normalized and scaled Z matrix
    """
    # 1. Compute mean across mini-batch (axis=0)
    mean = np.mean(Z, axis=0, keepdims=True)

    # 2. Compute variance across mini-batch (axis=0)
    variance = np.var(Z, axis=0, keepdims=True)

    # 3. Normalize Z to zero mean and unit variance
    Z_hat = (Z - mean) / np.sqrt(variance + epsilon)

    # 4. Scale and shift (learnable parameters)
    Z_norm = gamma * Z_hat + beta

    return Z_norm
