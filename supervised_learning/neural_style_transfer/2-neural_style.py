#!/usr/bin/env python3
"""
Contains the NST class for Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for Neural Style Transfer
    """

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for Neural Style Transfer
        """
        if (not isinstance(style_image, np.ndarray)
                or style_image.ndim != 3
                or style_image.shape[2] != 3):
            msg = "style_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        if (not isinstance(content_image, np.ndarray)
                or content_image.ndim != 3
                or content_image.shape[2] != 3):
            msg = "content_image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels
        """
        if (not isinstance(image, np.ndarray)
                or image.ndim != 3
                or image.shape[2] != 3):
            msg = "image must be a numpy.ndarray with shape (h, w, 3)"
            raise TypeError(msg)

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim

        h_new = int(round(h * scale))
        w_new = int(round(w * scale))

        resized_image = tf.image.resize(
            image,
            size=[h_new, w_new],
            method='bicubic'
        )

        rescaled_image = resized_image / 255.0
        rescaled_image = tf.clip_by_value(rescaled_image, 0.0, 1.0)

        return tf.expand_dims(rescaled_image, axis=0)

    def load_model(self):
        """
        Creates the model used to calculate cost
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        x = vgg.input
        layer_outputs = {}

        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name
                )(x)
            else:
                x = layer(x)

            layer_outputs[layer.name] = x

        outputs = [layer_outputs[layer] for layer in self.style_layers]
        outputs.append(layer_outputs[self.content_layer])

        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the Gram matrix of a given layer output.

        Parameters:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)

        Returns:
            tf.Tensor of shape (1, c, c) containing the Gram matrix
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Compute inner product over height and width dimensions
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Normalize by feature map area (h * w)
        num_locations = tf.cast(tf.shape(input_layer)[1] * tf.shape(input_layer)[2], tf.float32)

        return gram / num_locations
