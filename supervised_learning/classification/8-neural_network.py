#!/usr/bin/env python3
"""
Module defining a Deep Neural Network class.
"""

import numpy as np


class DeepNeuralNetwork:
    """
    Class representing a deep neural network.
    """

    def __init__(self, nx, layers):
        """
        Initializes the Deep Neural Network.

        Args:
            nx (int): Number of input features.
            layers (list): List containing nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list.
            ValueError: If nx <= 0 or layers is empty/invalid.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise ValueError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Recursively initialize weights and biases
        self.__recursive_init(nx, layers, 0, nx)

    def __recursive_init(self, nx, layers, idx, prev_size):
        """
        Recursively setup weights and biases per layer.
        """
        if idx == self.__L:
            return

        layer_size = layers[idx]
        if type(layer_size) is not int or layer_size < 1:
            raise TypeError("layers must be a list of positive integers")

        w_key = "W{}".format(idx + 1)
        b_key = "b{}".format(idx + 1)

        # He initialization scaled by the previous layer size
        scale = np.sqrt(2 / prev_size)
        self.__weights[w_key] = np.random.randn(layer_size, prev_size) * scale
        self.__weights[b_key] = np.zeros((layer_size, 1))

        self.__recursive_init(nx, layers, idx + 1, layer_size)

    @property
    def L(self):
        """
        Getter method retrieving the number of layers.
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter method retrieving the cache dictionary.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter method retrieving the weights dictionary.
        """
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the activation of the network using recursion.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: Activation and cache.
        """
        self.__cache['A0'] = X
        return self.__recursive_prop(X, 1)

    def __recursive_prop(self, A, idx):
        """
        Recursively apply activation across network layers.
        """
        if idx > self.__L:
            return A, self.__cache

        w_key = "W{}".format(idx)
        b_key = "b{}".format(idx)
        a_key = "A{}".format(idx)

        z_val = np.matmul(self.__weights[w_key], A) + self.__weights[b_key]
        activated = 1 / (1 + np.exp(-z_val))
        self.__cache[a_key] = activated

        return self.__recursive_prop(activated, idx + 1)

    def cost(self, Y, A):
        """
        Compute the logistic regression cost of the neural network.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]

        # Vectorized cost computation via matrix operations
        loss = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost_val = -(1 / m) * np.sum(loss)

        return float(cost_val)
