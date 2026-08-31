#!/usr/bin/env python3
"""
Module 4-deep_rnn
Defines the function deep_rnn that performs
forward propagation for a deep RNN.
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Parameters:
    - rnn_cells: list of RNNCell instances of length l
    - X: numpy.ndarray of shape (t, m, i) containing data to be used
    - h_0: numpy.ndarray of shape (l, m, h) containing initial hidden states

    Returns:
    - H: numpy.ndarray containing all hidden states, shape (t + 1, l, m, h)
    - Y: numpy.ndarray containing all outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    h_prev = np.copy(h_0)

    for step in range(t):
        x_t = X[step]
        for layer in range(l):
            cell = rnn_cells[layer]
            h_next, y_t = cell.forward(h_prev[layer], x_t)
            h_prev[layer] = h_next
            x_t = h_next

        H[step + 1] = h_prev
        Y[step] = y_t

    return H, Y
