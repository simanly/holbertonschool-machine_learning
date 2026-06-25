#!/usr/bin/env python3
"""
Module to calculate the F1 score for each class.
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class.
    """
    # Calculate precision and sensitivity
    prec = precision(confusion)
    sens = sensitivity(confusion)

    # Return harmonic mean
    return 2 * (prec * sens) / (prec + sens)
