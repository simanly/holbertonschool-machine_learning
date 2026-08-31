#!/usr/bin/env python3
"""
Module 3-variational
Contains the autoencoder function to create a Variational Autoencoder (VAE).
"""
import tensorflow as tf


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates a Variational Autoencoder (VAE) using Keras.

    Args:
        input_dims (int): Dimension of the model input.
        hidden_layers (list): Number of nodes for each hidden layer
                              in the encoder.
        latent_dims (int): Dimension of the latent space representation.

    Returns:
        encoder (tf.keras.Model): The encoder model, outputting (z, mean, log_var).
        decoder (tf.keras.Model): The decoder model.
        auto (tf.keras.Model): The full variational autoencoder model.
    """
    # ------------------- ENCODER -------------------
    inputs = tf.keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = tf.keras.layers.Dense(nodes, activation='relu')(x)

    z_mean = tf.keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = tf.keras.layers.Dense(latent_dims, activation=None)(x)

    # Reparameterization trick via Lambda layer
    def sampling(args):
        mean, log_var = args
        epsilon = tf.keras.backend.random_normal(
            shape=tf.shape(mean), mean=0.0, stddev=1.0
        )
        return mean + tf.exp(log_var / 2) * epsilon

    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dims,))(
        [z_mean, z_log_var]
    )

    # Encoder outputs z (latent representation), mean, and log_variance
    encoder = tf.keras.Model(
        inputs, [z, z_mean, z_log_var], name="encoder"
    )

    # ------------------- DECODER -------------------
    latent_inputs = tf.keras.Input(shape=(latent_dims,))
    x_dec = latent_inputs

    for nodes in reversed(hidden_layers):
        x_dec = tf.keras.layers.Dense(nodes, activation='relu')(x_dec)

    outputs = tf.keras.layers.Dense(input_dims, activation='sigmoid')(x_dec)

    decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")

    # ------------------- AUTOENCODER -------------------
    # Connect encoder sampling output directly to decoder
    encoder_output = encoder(inputs)[0]
    auto_outputs = decoder(encoder_output)

    auto = tf.keras.Model(inputs, auto_outputs, name="autoencoder")

    # Calculation of reconstruction and KL divergence loss
    reconstruction_loss = tf.keras.losses.binary_crossentropy(
        inputs, auto_outputs
    )
    reconstruction_loss *= input_dims

    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)

    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
