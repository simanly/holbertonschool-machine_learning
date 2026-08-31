#!/usr/bin/env python3
"""
Module 0-rnn_cell
Defines the class RNNCell that represents a cell of a simple RNN.
"""
import numpy as np


class RNNCell:
    """
    Represents a cell of a simple RNN.
    """

    def __init__(self, i, h, o):
        """
        Class constructor for RNNCell.

        Parameters:
        - i: dimensionality of the input data
        - h: dimensionality of the hidden state
        - o: dimensionality of the outputs
        """
        # Concatenated weights for hidden state
        # and input data: shape (h + i, h)
        self.Wh = np.random.randn(h + i, h)
        # Weights for output: shape (h, o)
        self.Wy = np.random.randn(h, o)

        # Biases initialized to zeros
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
        # Concatenate previous hidden state and input along columns (axis 1)
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Next hidden state using tanh activation function
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Raw output logits
        y_logit = np.matmul(h_next, self.Wy) + self.by

        # Softmax activation for output
        e_x = np.exp(y_logit - np.max(y_logit, axis=1, keepdims=True))
        y = e_x / np.sum(e_x, axis=1, keepdims=True)

        return h_next, y
