#!/usr/bin/env python3
"""L2 Regularization Gradient Descent."""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights and biases with L2 regularization."""
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f'A{i - 1}']

        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * weights[f'W{i}'])
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dZ = np.matmul(weights[f'W{i}'].T, dZ) * (1 - np.square(A_prev))

        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db
