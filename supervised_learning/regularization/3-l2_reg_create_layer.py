#!/usr/bin/env python3
"""Create L2 regularized layer."""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Creates a Dense layer with L2 regularization."""
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg'
    )
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )
    return layer(prev)
