#!/usr/bin/env python3
"""Gradient descent with dropout."""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights and biases using gradient descent with Dropout."""
    m = Y.shape[1]
    dZ = cache[f'A{L}'] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f'A{i - 1}']

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dZ = np.matmul(weights[f'W{i}'].T, dZ) * (1 - np.square(A_prev))
            dZ *= cache[f'D{i - 1}']
            dZ /= keep_prob

        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db
