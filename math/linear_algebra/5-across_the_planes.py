#!/usr/bin/env python3
'''
Defines a function that adds two matrices element-wise
'''


def add_matrices2D(mat1, mat2):
    '''
    Returns the sum of matrix
    '''
    matrix = []
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    else:
        for i in range(len(mat1)):
            row = []
            for j in range(len(mat1[0])):
                row.append(mat1[i][j] + mat2[i][j])
            matrix.append(row)
        return matrix
