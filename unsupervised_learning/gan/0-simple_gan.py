#!/usr/bin/env python3
"""
Module containing the Simple_GAN class implementation using Keras.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    Simple GAN model class inheriting from keras.Model.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
    ):
        """
        Initializes the Simple_GAN model, setting up loss functions and optimizers.
        """
        super().__init__()

        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta1 = 0.5
        self.beta2 = 0.9

        # Define generator loss and optimizer
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer, loss=self.generator.loss
        )

        # Define discriminator loss and optimizer
        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
            + tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta1,
            beta_2=self.beta2,
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss,
        )

    def get_fake_sample(self, size=None, training=False):
        """
        Generates fake samples from latent vectors using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(
            self.latent_generator(size), training=training
        )

    def get_real_sample(self, size=None):
        """
        Retrieves a random sample batch from real_examples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, useless_argument):
        """
        Executes one training step:
        Applies gradient descent disc_iter times for the discriminator,
        then once for the generator.
        """
        # 1. Train Discriminator disc_iter times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample(training=True)

                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)

                discr_loss = self.discriminator.loss(real_output, fake_output)

            grads = tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_variables)
            )

        # 2. Train Generator once
        with tf.GradientTape() as tape:
            fake_sample = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_sample, training=True)

            gen_loss = self.generator.loss(fake_output)

        grads = tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(grads, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
