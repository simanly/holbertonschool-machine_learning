#!/usr/bin/env python3
"""
Module to train a Keras model using mini-batch gradient descent
with optional validation data, early stopping, learning rate decay,
and saving the best model checkpoint.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with options for
    validation data, early stopping, learning rate decay, and saving
    the best model based on validation loss.

    Args:
        network: the model to train
        data: numpy.ndarray of shape (m, nx) containing input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing labels
        batch_size: size of batch used for mini-batch gradient descent
        epochs: number of passes through data
        validation_data: tuple of (X_valid, Y_valid) data to validate with
        early_stopping: boolean indicating whether to use early stopping
        patience: patience used for early stopping
        learning_rate_decay: boolean indicating whether to use LR decay
        alpha: initial learning rate
        decay_rate: decay rate
        save_best: boolean indicating whether to save the best model
        filepath: file path where the model should be saved
        verbose: boolean that determines if output should be printed
        shuffle: boolean determining whether to shuffle batches every epoch

    Returns:
        The History object generated after training the model
    """
    callbacks = []

    if validation_data is not None:
        if early_stopping:
            callbacks.append(K.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience
            ))
        if learning_rate_decay:
            def schedule(epoch):
                """Calculates inverse time decay for the given epoch."""
                return alpha / (1 + decay_rate * epoch)

            callbacks.append(K.callbacks.LearningRateScheduler(
                schedule,
                verbose=1
            ))
        if save_best and filepath:
            callbacks.append(K.callbacks.ModelCheckpoint(
                filepath=filepath,
                monitor='val_loss',
                save_best_only=True
            ))

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
