#!/usr/bin/env python3
"""
Module to calculate a confusion matrix.
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from one-hot encoded labels and logits.

    Args:
        labels (numpy.ndarray): One-hot array of shape (m, classes).
        logits (numpy.ndarray): One-hot array of shape (m, classes).

    Returns:
        numpy.ndarray: Confusion matrix of shape (classes, classes).
    """
    # Convert one-hot vectors to 1D integer arrays of class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    # Determine total number of classes
    num_classes = labels.shape[1]

    # Calculate confusion matrix using matrix multiplication
    # row index = correct label, column index = predicted label
    confusion = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            confusion[i, j] = np.sum((true_classes == i) & (pred_classes == j))

    return confusion
