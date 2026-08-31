#!/usr/bin/env python3
"""
Module 3-lstm_cell
Defines the class LSTMCell that represents an LSTM unit.
"""
import numpy as np


class LSTMCell:
    """
    Represents an LSTM unit.
    """

    def __init__(self, i, h, o):
        """
        Class constructor for LSTMCell.

        Parameters:
        - i: dimensionality of the input data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h) with previous hidden state
        - c_prev: numpy.ndarray of shape (m, h) with previous cell state
        - x_t: numpy.ndarray of shape (m, i) with input data for the cell

        Returns:
        - h_next: next hidden state, shape (m, h)
        - c_next: next cell state, shape (m, h)
        - y: output of the cell, shape (m, o)
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        f = 1 / (1 + np.exp(-(np.matmul(concat, self.Wf) + self.bf)))
        u = 1 / (1 + np.exp(-(np.matmul(concat, self.Wu) + self.bu)))
        c_cand = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        c_next = f * c_prev + u * c_cand

        o = 1 / (1 + np.exp(-(np.matmul(concat, self.Wo) + self.bo)))
        h_next = o * np.tanh(c_next)

        y_logit = np.matmul(h_next, self.Wy) + self.by
        e_x = np.exp(y_logit - np.max(y_logit, axis=1, keepdims=True))
        y = e_x / np.sum(e_x, axis=1, keepdims=True)

        return h_next, c_next, y
