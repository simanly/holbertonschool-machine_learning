#!/usr/bin/env python3
"""
Module to convert a label vector into a one-hot matrix.
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix.

    Args:
        labels: label vector to be converted
        classes: maximum number of classes (optional)

    Returns:
        The one-hot matrix with classes as the last dimension
    """
    return K.utils.to_categorical(labels, num_classes=classes)o
