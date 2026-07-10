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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            The private attribute __A.
        """
        # Calculate Z using numpy.matmul and add the bias
        Z = np.matmul(self.__W, X) + self.__b

        # Apply the sigmoid activation function: 1 / (1 + e^-Z)
        self.__A = 1 / (1 + np.exp(-Z))

        return self.__A
