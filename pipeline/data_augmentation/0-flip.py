#!/usr/bin/env python3
"""Horizontal image flipping module."""
import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Parameters:
    - image: 3D tf.Tensor

    Returns:
    - flipped image
    """
    return tf.image.flip_left_right(image)
