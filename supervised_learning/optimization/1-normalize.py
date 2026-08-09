#!/usr/bin/env python3
'''
Normalizes a matrix.
'''
import numpy as np


def normalize(X, m, s):
    '''
    Normalizes a matrix X using mean m and standard deviation s.
    '''
    return (X - m) / s
