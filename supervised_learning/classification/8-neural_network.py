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
            layers (list): List containing the number of nodes in each layer.

        Raises:
            TypeError: If nx is not an integer or layers is not a list.
            ValueError: If nx <= 0 or layers is empty or contains non-positives.
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

        # Initialization logic
        prev = nx
        idx = 0
        # Iterate through layers to set weights and biases
        while idx < self.__L:
            layer_size = layers[idx]
            if type(layer_size) is not int or layer_size < 1:
                raise TypeError("layers must be a list of positive integers")

            w_key = "W{}".format(idx + 1)
            b_key = "b{}".format(idx + 1)

            # He initialization scaling
            self.__weights[w_key] = np.random.randn(layer_size, prev) * \
                np.sqrt(2 / prev)
            self.__weights[b_key] = np.zeros((layer_size, 1))

            prev = layer_size
            idx += 1

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
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: The activated output of the last layer and the cache.
        """
        self.__cache['A0'] = X
        A = X
        # Calculate output at each layer
        for i in range(1, self.__L + 1):
            W = self.__weights['W{}'.format(i)]
            b = self.__weights['b{}'.format(i)]

            # Z = W * A + b
            Z = np.matmul(W, A) + b

            # Sigmoid activation: 1 / (1 + e^-Z)
            A = 1 / (1 + np.exp(-Z))

            # Store activation result in cache
            self.__cache['A{}'.format(i)] = A

        return A, self.__cache
