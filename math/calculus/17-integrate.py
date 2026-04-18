#!/usr/bin/env python3
'''
Module to calculate the integral of a polynomial.
'''

def poly_integral(poly, C=0):
    '''
    Calculates the integral of a polynomial.

    Args:
        poly (list): List of coefficients representing a polynomial.
        C (int): Integration constant.

    Returns:
        list: Coefficients of the integral, or None if invalid.
    '''
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None
    integral = [C]
    for i in range(len(poly)):
        val = poly[i] / (i+1)
        integral.append(int(val) if val == int(val) else val)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
