#!/usr/bin/env python3
'''
Defines a function that slices a matrix along specific axes
'''
import numpy as np


def np_slice(matrix, axes={}):
    '''
    Returns slices a matrix along specific axes provided in a dictionary
    '''
    slices = [slice(None)] * len(matrix.shape)
    for axis, slice_tuple in axes.items():
        slices[axis] = slice(*slice_tuple)
    return matrix[tuple(slices)]
