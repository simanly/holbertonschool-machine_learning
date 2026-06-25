#!/usr/bin/env python3
"""
Module to calculate sensitivity for each class.
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Confusion matrix of shape (classes, classes)
            where rows are correct labels and columns are predicted labels.

    Returns:
        numpy.ndarray: Array of shape (classes,) containing sensitivity.
    """
    # True positives are the diagonal elements
    true_positives = np.diag(confusion)

    # Actual positives for each class is the sum of each row
    actual_positives = np.sum(confusion, axis=1)

    # Sensitivity = True Positives / Actual Positives
    return true_positives / actual_positives
