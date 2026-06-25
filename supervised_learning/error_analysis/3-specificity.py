#!/usr/bin/env python3
"""
Module to calculate specificity for each class.
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.
    """
    total_samples = np.sum(confusion)
    true_positives = np.diag(confusion)

    # False positives: sum of each column minus the true positives
    false_positives = np.sum(confusion, axis=0) - true_positives

    # False negatives: sum of each row minus the true positives
    false_negatives = np.sum(confusion, axis=1) - true_positives

    # True negatives for each class broken down to pass line length limit
    true_negatives = total_samples - (
        true_positives + false_positives + false_negatives
    )

    # Actual negatives = TN + FP
    actual_negatives = true_negatives + false_positives

    return true_negatives / actual_negatives
