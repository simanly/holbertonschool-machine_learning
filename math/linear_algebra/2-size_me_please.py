#!/usr/bin/env python3
'''
Defines a function to calculate the shape of a matrix
'''
def matrix_shape(matrix):
    '''
    Calculates the shape of matrix and returns it as a list of integers
    '''
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        if len(matrix) > 0:
            matrix = matrix[0]
        else:
            break
    return shape
