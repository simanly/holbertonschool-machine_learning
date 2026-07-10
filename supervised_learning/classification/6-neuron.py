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

        self.__W = np.random.randn(1, nx)
        self.__b = 0
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

        prediction = np.where(A >= 0.5, 1, 0)

        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).
            alpha (float): The learning rate.
        """
        m = Y.shape[1]
        dz = A - Y

        dW = (1 / m) * np.matmul(dz, X.T)
        db = (1 / m) * np.sum(dz)

        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron to optimize weights and bias.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            iterations (int): Number of training iterations.
            alpha (float): The learning rate.

        Returns:
            tuple: The evaluation of the training data after training.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Run the training loop
        for _ in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
