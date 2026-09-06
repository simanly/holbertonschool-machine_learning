#!/usr/bin/env python3
"""Neural Style Transfer Module"""

import numpy as np
import tensorflow as tf


class NST:
    """Class NST that performs Neural Style Transfer"""

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes the NST class instance"""
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) \
                or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or isinstance(beta, bool) \
                or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image, max_dim=512):
        """Rescales an image so that its maximum dimension is max_dim"""
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if type(max_dim) is not int or max_dim <= 0:
            raise TypeError("max_dim must be a positive integer")

        h, w, _ = image.shape
        if h > w:
            h_new = max_dim
            w_new = int((w * max_dim) / h)
        else:
            w_new = max_dim
            h_new = int((h * max_dim) / w)

        image_scaled = image / 255.0
        image_resized = tf.image.resize(
            image_scaled,
            size=[h_new, w_new],
            method='bicubic'
        )
        image_resized = tf.clip_by_value(image_resized, 0.0, 1.0)

        return tf.expand_dims(image_resized, axis=0)
