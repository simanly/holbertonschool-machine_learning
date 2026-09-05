#!/usr/bin/env python3
"""
Neural Style Transfer Module
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    Class NST that performs Neural Style Transfer
    """
    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block4_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor for NST class

        Parameters:
            style_image: numpy.ndarray - image used as style reference
            content_image: numpy.ndarray - image used as content reference
            alpha: float/int - weight for content cost
            beta: float/int - weight for style cost
        """
        if (not isinstance(style_image, np.ndarray)
            or style_image.ndim != 3
                or style_image.shape[2] != 3):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        if (not isinstance(content_image, np.ndarray)
            or content_image.ndim != 3
                or content_image.shape[2] != 3):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

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
        Rescales an image such that its pixel values are in [0, 1]
        and its largest side is 512 pixels.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize(image, (h_new, w_new), method='bicubic')
        scaled = resized / 255.0
        clipped = tf.clip_by_value(scaled, 0.0, 1.0)
        return tf.expand_dims(clipped, axis=0)

    def load_model(self):
        """
        Creates the model used to calculate cost.
        Uses VGG19 as base and replaces MaxPooling2D with AveragePooling2D.
        Sets outputs to style layers followed by content layer.
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

        # Replace MaxPooling2D layers with AveragePooling2D layers
        x = vgg.input
        outputs_dict = {}

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

            if layer.name in self.style_layers or layer.name == self.content_layer:
                outputs_dict[layer.name] = x

        # Ensure correct output order: style_layers followed by content_layer
        outputs = [outputs_dict[layer_name] for layer_name in self.style_layers]
        outputs.append(outputs_dict[self.content_layer])

        model = tf.keras.Model(inputs=vgg.input, outputs=outputs)
        model.trainable = False

        self.model = model
        return model
    