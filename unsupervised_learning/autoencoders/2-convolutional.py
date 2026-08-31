#!/usr/bin/env python3
"""Creates a convolutional autoencoder model using keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder.

    input_dims: tuple of integers containing dimensions of model input
    filters: list containing number of filters for each conv layer in encoder
    latent_dims: tuple of integers containing dimensions of latent space

    Returns: encoder, decoder, auto
    """
    inputs = keras.Input(shape=input_dims)
    x = inputs

    for f in filters:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(x)

    latent = x
    encoder = keras.Model(inputs=inputs, outputs=latent)

    latent_inputs = keras.Input(shape=latent_dims)
    x = latent_inputs

    reversed_filters = list(reversed(filters))

    for f in reversed_filters[:-1]:
        x = keras.layers.Conv2D(
            filters=f,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    outputs = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(x)

    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
