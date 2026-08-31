#!/usr/bin/env python3
"""Creates a vanilla autoencoder model using keras."""
import keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a vanilla autoencoder.

    input_dims: integer containing the dimensions of the model input
    hidden_layers: list with number of nodes for each hidden layer in encoder
    latent_dims: integer containing dimensions of latent space representation

    Returns: encoder, decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=inputs, outputs=latent)

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    auto_outputs = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
