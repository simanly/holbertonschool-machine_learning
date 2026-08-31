#!/usr/bin/env python3
"""Module 0-rnn_cell"""
import numpy as np


class RNNCell:
    """Class RNNCell"""

    def __init__(self, i, h, o):
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y_logit = np.matmul(h_next, self.Wy) + self.by
        e_x = np.exp(y_logit - np.max(y_logit, axis=1, keepdims=True))
        y = e_x / np.sum(e_x, axis=1, keepdims=True)
        return h_next, y
