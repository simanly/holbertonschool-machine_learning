#!/usr/bin/env python3
"""
Module containing function to calculate learning rate inverse time decay.
"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in a stepwise fashion.

    Parameters:
    - alpha: original learning rate
    - decay_rate: weight used to determine the decay rate
    - global_step: number of passes of gradient descent that have elapsed
    - decay_step: number of passes that should occur before decay

    Returns:
    - updated value for alpha
    """
    # Stepwise decay factor using floor integer division
    step_factor = global_step // decay_step

    # Inverse time decay formula
    alpha_updated = alpha / (1 + decay_rate * step_factor)

    return alpha_updated
