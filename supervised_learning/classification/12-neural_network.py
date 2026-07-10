#!/usr/bin/env python3
"""
Module defining a neural network with one hidden layer
used in binary classification.
"""
import numpy as np


class NeuralNetwork:
    """
    Class representing a neural network with one hidden layer
    performing binary classification.
    """

    def __init__(self, nx, nodes):
        """
        Initializes a NeuralNetwork instance.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If nodes is not an integer.
            ValueError: If nodes is less than 1.
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        # Hidden layer parameters
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        # Output neuron parameters
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        Getter method retrieving the hidden layer weights.
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter method retrieving the hidden layer biases.
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter method retrieving the hidden layer activated output.
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter method retrieving the output neuron weights.
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter method retrieving the output neuron bias.
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter method retrieving the output neuron activated output.
        """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).

        Returns:
            tuple: The private attributes __A1 and __A2.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

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
	    Evaluates the neural network's predictions.

	    Args:
	        X (numpy.ndarray): Input data with shape (nx, m).
	        Y (numpy.ndarray): Correct labels with shape (1, m).

	    Returns:
	        tuple: (prediction, cost)
	            prediction is a numpy.ndarray of shape (1, m)
	            containing 0s and 1s.
	            cost is the cost of the network.
	    """
	    self.forward_prop(X)
	    prediction = np.where(self.__A2 >= 0.5, 1, 0)
	    cost = self.cost(Y, self.__A2)
	    return prediction, cost
