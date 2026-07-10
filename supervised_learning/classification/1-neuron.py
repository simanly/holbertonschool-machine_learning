#!/usr/bin/env python3
"""
Module defining a single neuron used in binary classification.
"""

import numpy as np


class Neuron:
    """
    Class representing a single neuron in a neural network.
    """

    def __init__(self, nx):
        """
        Initializes a Neuron instance.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Private weights vector initialized with random normal distribution
        self.__W = np.random.randn(1, nx)
        # Private bias of the neuron
        self.__b = 0
        # Private activated output of the neuron
        self.__A = 0

    @property
    def W(self):
        """
        Getter method retrieving the weights vector.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method retrieving the bias.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method retrieving the activated output.
        """
        return self.__A
