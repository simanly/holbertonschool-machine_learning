#!/usr/bin/env python3
'''
Module to calculate the derivative of a polynomial.
'''


def poly_derivative(poly):
    '''
    Calculates the derivative of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.

    Returns:
        list: Coefficients representing the derivative, or None if invalid.
    '''
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    derivative = [poly[i] * i for i in range(1, len(poly))]
    return derivative
