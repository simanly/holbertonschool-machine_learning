#!/usr/bin/env python3
'''
Defines a function that adds 2 arrays element-wise
'''


def add_arrays(arr1, arr2):
    '''
    Returns the sum of elements
    '''
    matrix = []
    if len(arr1) != len(arr2):
        return None
    else:
        for i in range(len(arr1)):
            matrix.append(arr1[i] + arr2[i])
    return matrix
