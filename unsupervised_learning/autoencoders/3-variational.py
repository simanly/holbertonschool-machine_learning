#!/usr/bin/env python3
"""Creates a variational autoencoder model using keras."""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates a variational autoencoder.

    input_dims: integer containing dimensions of model input
    hidden_layers: list with number of nodes for hidden layers in encoder
    latent_dims: integer containing dimensions of latent space
    Returns: encoder, decoder, auto
    """
    inputs = keras.Input(shape=(input_dims,))
    x = inputs

    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_sig = keras.layers.Dense(latent_dims, activation=None)(x)

    def sampling(args):
        """Sampling trick for VAE latent space."""
        z_mean, z_log_sig = args
        epsilon = keras.backend.random_normal(
            shape=(keras.backend.shape(z_mean)[0], latent_dims)
        )
        return z_mean + keras.backend.exp(z_log_sig / 2) * epsilon

    z = keras.layers.Lambda(sampling)([mean, log_sig])

    encoder = keras.Model(inputs=inputs, outputs=[z, mean, log_sig])

    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs

    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)

    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    auto_outputs = decoder(encoder(inputs)[0])
    auto = keras.Model(inputs=inputs, outputs=auto_outputs)

    kl_loss = 1 + log_sig - keras.backend.square(mean) - \
        keras.backend.exp(log_sig)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss = keras.backend.mean(kl_loss) * -0.5

    auto.add_loss(kl_loss)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
