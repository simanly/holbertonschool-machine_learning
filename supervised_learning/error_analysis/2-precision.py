#!/usr/bin/env python3
"""
Module to calculate precision for each class.
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where rows are correct labels and columns are predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing precision.
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Total predicted positives for each class is the sum of each column
    predicted_positives = np.sum(confusion, axis=0)

    # Precision = True Positives / Total Predicted Positives
    return true_positives / predicted_positives
