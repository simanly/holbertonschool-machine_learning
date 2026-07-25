#!/usr/bin/env python3
"""
Module to build a neural network model using Keras.
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library using Functional API.

    Args:
        nx (int): Number of input features to the network.
        layers (list): List containing the number of nodes in each layer.
        activations (list): List containing the activation functions for
                            each layer.
        lambtha (float): L2 regularization parameter.
        keep_prob (float): Probability that a node will be kept for dropout.

    Returns:
        K.Model: The compiled Keras model instance.
    """
    inputs = K.Input(shape=(nx,))
    regularizer = K.regularizers.l2(lambtha)

    x = inputs
    for i in range(len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer
        )(x)

        # Apply dropout to hidden layers only (not output layer)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)
    return model
