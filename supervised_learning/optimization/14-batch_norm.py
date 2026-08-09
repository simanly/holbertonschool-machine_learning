#!/usr/bin/env python3
"""
Module containing function to create a Batch Normalization layer in TensorFlow.
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    - prev: activated output of the previous layer
    - n: number of nodes in the layer to be created
    - activation: activation function to be used on the output of the layer

    Returns:
    - tensor of the activated output for the layer
    """
    # 1. Base Dense layer without activation
    base_layer = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'
        )
    )
    Z = base_layer(prev)

    # 2. Batch Normalization layer
    batch_norm = tf.keras.layers.BatchNormalization(
        gamma_initializer='ones',
        beta_initializer='zeros',
        epsilon=1e-7
    )
    Z_norm = batch_norm(Z)

    # 3. Apply activation function if provided
    if activation is not None:
        return activation(Z_norm)

    return Z_norm
