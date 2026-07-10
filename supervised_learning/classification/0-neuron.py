#!/usr/bin/env python3
"""
Module that defines a single neuron performing binary classification.
"""

import numpy as np


class Neuron:
    """
    Class that defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Class constructor for the Neuron.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # The weights vector initialized using a random normal distribution
        self.W = np.random.randn(1, nx)
        # The bias for the neuron
        self.b = 0
        # The activated output of the neuron (prediction)
        self.A = 0
