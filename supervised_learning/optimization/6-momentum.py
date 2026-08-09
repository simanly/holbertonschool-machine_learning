#!/usr/bin/env python3
"""Sets up gradient descent with momentum optimization in TensorFlow."""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Sets up the gradient descent with momentum optimization algorithm.

    Parameters:
    - alpha: learning rate
    - beta1: momentum weight

    Returns:
    - optimizer: tf.keras.optimizers.SGD initialized with momentum
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
