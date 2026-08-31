#!/usr/bin/env python3
"""
Module 3-generate_faces
Defines the convolutional_GenDiscr function to build generator and discriminator
models using Keras.
"""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds and returns the generator and discriminator Keras models
    matching the target summary architectures.
    """

    def generator():
        inputs = keras.Input(shape=(16,))
        x = keras.layers.Dense(2048, activation="tanh")(inputs)
        x = keras.layers.Reshape((2, 2, 512))(x)

        # Block 1
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        # Block 2
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(16, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("tanh")(x)

        # Block 3
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(1, (3, 3), padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Activation("tanh")(x)

        return keras.Model(inputs, outputs, name="generator")

    def get_discriminator():
        inputs = keras.Input(shape=(16, 16, 1))

        # Block 1
        x = keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        # Block 2
        x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        # Block 3
        x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        # Block 4
        x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Activation("tanh")(x)

        # Output Head
        x = keras.layers.Flatten()(x)
        outputs = keras.layers.Dense(1, activation="tanh")(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return generator(), get_discriminator()
