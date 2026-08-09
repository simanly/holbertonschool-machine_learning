#!/usr/bin/env python3
"""
Module containing the RMSProp optimization algorithm function.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta2: RMSProp weight (discount factor / decay rate)
    - epsilon: small number to avoid division by zero
    - var: numpy.ndarray containing the variable to be updated
    - grad: numpy.ndarray containing the gradient of var
    - s: previous second moment of var

    Returns:
    - var: updated variable
    - s: new second moment
    """
    # Calculate the new moving average of squared gradients (second moment)
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # Update the variable
    var = var - alpha * (grad / (np.sqrt(s) + epsilon))

    return var, s
