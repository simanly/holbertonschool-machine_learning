#!/usr/bin/env python3
"""
Module containing the Adam optimization algorithm function.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta1: weight used for the first moment
    - beta2: weight used for the second moment
    - epsilon: small number to avoid division by zero
    - var: numpy.ndarray containing the variable to be updated
    - grad: numpy.ndarray containing the gradient of var
    - v: previous first moment of var
    - s: previous second moment of var
    - t: time step used for bias correction

    Returns:
    - var: updated variable
    - v: new first moment
    - s: new second moment
    """
    # 1. Update biased first moment estimate
    v = beta1 * v + (1 - beta1) * grad

    # 2. Update biased second raw moment estimate
    s = beta2 * s + (1 - beta2) * (grad ** 2)

    # 3. Compute bias-corrected first moment estimate
    v_corrected = v / (1 - beta1 ** t)

    # 4. Compute bias-corrected second raw moment estimate
    s_corrected = s / (1 - beta2 ** t)

    # 5. Update parameters in place
    var -= alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v, s
