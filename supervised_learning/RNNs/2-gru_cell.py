#!/usr/bin/env python3
"""
Module 2-gru_cell
Defines the class GRUCell that represents a gated recurrent unit.
"""
import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit.
    """

    def __init__(self, i, h, o):
        """
        Class constructor for GRUCell.

        Parameters:
        - i: dimensionality of the input data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h) with previous hidden state
        - x_t: numpy.ndarray of shape (m, i) with input data for the cell

        Returns:
        - h_next: next hidden state, shape (m, h)
        - y: output of the cell, shape (m, o)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        z = 1 / (1 + np.exp(-(np.matmul(concat, self.Wz) + self.bz)))
        r = 1 / (1 + np.exp(-(np.matmul(concat, self.Wr) + self.br)))

        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_cand = np.tanh(np.matmul(concat_r, self.Wh) + self.bh)

        h_next = (1 - z) * h_prev + z * h_cand

        y_logit = np.matmul(h_next, self.Wy) + self.by
        e_x = np.exp(y_logit - np.max(y_logit, axis=1, keepdims=True))
        y = e_x / np.sum(e_x, axis=1, keepdims=True)

        return h_next, y
