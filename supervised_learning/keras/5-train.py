#!/usr/bin/env python3
"""
Module to train a Keras model using mini-batch gradient descent
with optional validation data.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent and analyzes
    validation data if provided.

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: tuple of (X_valid, Y_valid) data to validate with
        verbose: boolean that determines if output should be printed
        shuffle: boolean determining whether to shuffle batches every epoch

    Returns:
        The History object generated after training the model
    """
    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
