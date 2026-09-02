#!/usr/bin/env python3
"""Image rotation module."""
import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image by 90 degrees counter-clockwise.

    Parameters:
    - image: 3D tf.Tensor

    Returns:
    - rotated image
    """
    return tf.image.rot90(image, k=1)
