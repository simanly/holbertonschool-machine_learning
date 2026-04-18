#!/usr/bin/env python3
'''
Module to calculate the sum of squares of integers up to n.
This module provides a function that uses a mathematical formula
to achieve O(1) time complexity.
'''
def summation_i_squared(n):
    '''
    Calculates the sum of squares from 1 to n.

        Args:
            n (int): The stop value for the sum.

        Returns:
            int: The sum of squares if n is a valid positive integer, 0 otherwise.
    '''
    if not isinstance(n, int):
        return 0
    return (n * (n + 1) * (2 * n + 1)) // 6
