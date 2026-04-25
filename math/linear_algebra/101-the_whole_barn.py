#!/usr/bin/env python3
'''
Defines a function that adds two matrices
'''
import numpy as np


def add_matrices(mat1, mat2):
    '''
    Return the sum like new matrix
    '''
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    if m1.shape != m2.shape:
       return None
    return (m1 + m2)
