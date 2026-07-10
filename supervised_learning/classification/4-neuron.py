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
        if type(nx) is not int:
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
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
            float: The cost.
        """
        m = Y.shape[1]

        # Calculate the logistic regression cost function
        term1 = Y * np.log(A)
        term2 = (1 - Y) * np.log(1.0000001 - A)
        cost = -(1 / m) * np.sum(term1 + term2)

        return float(cost)

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            tuple: The neuron's prediction and the cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)

        # Convert probabilities to
        # binary predictions (1 if >= 0.5, 0 otherwise)
        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost
