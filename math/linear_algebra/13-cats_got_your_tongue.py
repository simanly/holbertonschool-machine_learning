#!/usr/bin/env python3
'''
Defines a function that concatenates
Two matrices along a specific axis
'''
import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''
    Returns a concatenated new numpy array
    '''
    return np.concatenate((mat1, mat2), axis=axis)
