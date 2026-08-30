#!/usr/bin/env python3
"""Forward propagation with dropout."""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Conducts forward propagation using Dropout."""
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights[f'W{i}']
        b = weights[f'b{i}']
        A_prev = cache[f'A{i - 1}']
        Z = np.matmul(W, A_prev) + b

        if i == L:
            exp_Z = np.exp(Z)
            cache[f'A{i}'] = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            D = D.astype(int)
            A = (A * D) / keep_prob
            cache[f'D{i}'] = D
            cache[f'A{i}'] = A

    return cache
