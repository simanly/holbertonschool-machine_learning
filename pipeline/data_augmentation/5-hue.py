#!/usr/bin/env python3
"""Image hue adjustment module."""
import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Parameters:
    - image: 3D tf.Tensor containing the image to change
    - delta: float, amount the hue should change

    Returns:
    - The altered image
    """
    return tf.image.adjust_hue(image, delta)
