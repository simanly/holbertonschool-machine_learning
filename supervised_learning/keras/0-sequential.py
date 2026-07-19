#!/usr/bin/env python3
"""
Module that contains the build_model function
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

    Args:
        nx: The number of input features to the network.
        layers: A list containing the number of nodes in each layer.
        activations: A list containing the activation functions used for
                     each layer of the network.
        lambtha: The L2 regularization parameter.
        keep_prob: The probability that a node will be kept for dropout.

    Returns:
        The Keras model.
    """
    model = K.Sequential()

    for i in range(len(layers)):
        # Define the L2 regularizer
        regularizer = K.regularizers.L2(lambtha)

        # Determine if it's the first layer to specify input_shape
        if i == 0:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(units=layers[i],
                                     activation=activations[i],
                                     kernel_regularizer=regularizer))

        # Add Dropout to every layer except the last one
        if i < len(layers) - 1:
            # Keras Dropout takes the drop rate, which is 1 - keep_prob
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
