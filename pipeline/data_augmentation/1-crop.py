#!/usr/bin/env python3
"""Random image cropping module."""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Parameters:
    - image: 3D tf.Tensor
    - size: tuple containing the size of the crop

    Returns:
    - cropped image
    """
    return tf.image.random_crop(image, size=size, seed=None)
