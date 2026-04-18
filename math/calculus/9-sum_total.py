#!/usr/bin/env python3
'''
Module to calculate the sum of squares of integers up to n.
This module provides a function that uses a mathematical formula.
'''


def summation_i_squared(n):
    '''
    Calculates the sum of squares from 1 to n.
    Args:
        n (int): The stop value for the sum.
    Returns:
        int: The sum of squares, or 0 if n is not valid.
    '''
    if not isinstance(n, int) or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
