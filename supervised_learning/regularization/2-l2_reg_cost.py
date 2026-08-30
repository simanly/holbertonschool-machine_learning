#!/usr/bin/env python3
"""
Calculates the cost of a neural network
with L2 regularization in Keras.
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates total cost for
    each layer accounting for L2 regularization.
    """
    return cost + model.losses
