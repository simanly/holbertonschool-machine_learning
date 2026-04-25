#!/usr/bin/env python3
'''
Defines a function that performs matrix multiplication
'''


def mat_mul(mat1, mat2):
    '''
    Returns a multiplied new matrix
    '''
    if len(mat1[0]) != len(mat2):
        return None
    mat12 = []
    for i in range(len(mat1)):
        row12 = []
        for j in range(len(mat2[0])):
            sum12 = 0
            for k in range(len(mat1[0])):
                sum12 += mat1[i][k] * mat2[k][j]
            row12.append(sum12)
        mat12.append(row12)
    return mat12
