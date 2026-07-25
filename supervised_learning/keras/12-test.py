#!/usr/bin/env python3
"""
Module to test a Keras neural network model.
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model on testing data.

    Args:
        network: the network model to test
        data: input data to test the model with
        labels: correct one-hot labels of data
        verbose: boolean determining if output should be printed

    Returns:
        The loss and accuracy of the model with the testing data
    """
    return network.evaluate(x=data, y=labels, verbose=verbose)
