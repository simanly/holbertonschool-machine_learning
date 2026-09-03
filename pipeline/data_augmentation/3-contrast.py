#!/usr/bin/env python3
"""Random contrast adjustment module."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Parameters:
    - image: A 3D tf.Tensor representing the input image
    - lower: Float, lower bound of the contrast factor range
    - upper: Float, upper bound of the contrast factor range

    Returns:
    - The contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
