#!/usr/bin/env python3
"""
Module 1-rnn
Defines the function rnn that performs forward
propagation for a simple RNN.
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Parameters:
    - rnn_cell: instance of RNNCell used for forward propagation
    - X: numpy.ndarray of shape (t, m, i) containing data to be used
    - h_0: numpy.ndarray of shape (m, h) containing initial hidden state

    Returns:
    - H: numpy.ndarray containing all hidden states, shape (t + 1, m, h)
    - Y: numpy.ndarray containing all outputs, shape (t, m, o)
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    Y = np.zeros((t, m, o))

    h_prev = h_0
    for step in range(t):
        h_prev, y_t = rnn_cell.forward(h_prev, X[step])
        H[step + 1] = h_prev
        Y[step] = y_t

    return H, Y
