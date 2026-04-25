#!/usr/bin/env python3
'''
Defines a function that performs element-wise
addition, subtraction, multiplication, and division
'''


def np_elementwise(mat1, mat2):
    '''
    Returns a tuple containing the element-wise
    sum, difference, product, and quotient, respectively
    '''
    sum = mat1 + mat2
    diff = mat1 - mat2
    prod = mat1 * mat2
    div = mat1 / mat2
    return (sum, diff, prod, div)
