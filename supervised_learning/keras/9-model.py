#!/usr/bin/env python3
"""
Module containing functions to save and load an entire Keras model.
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model to a file.

    Args:
        network: the Keras model to save
        filename: path of the file where the model should be saved

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire Keras model from a file.

    Args:
        filename: path of the file where the model should be loaded from

    Returns:
        The loaded Keras model
    """
    return K.models.load_model(filename)
