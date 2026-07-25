#!/usr/bin/env python3
"""
Module containing functions to save and load a Keras model's weights.
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model's weights.

    Args:
        network: the model whose weights should be saved
        filename: path of the file where the weights should be saved
        save_format: format in which the weights should be saved

    Returns:
        None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads a model's weights.

    Args:
        network: the model to which the weights should be loaded
        filename: path of the file from which the weights should be loaded

    Returns:
        None
    """
    network.load_weights(filename)
