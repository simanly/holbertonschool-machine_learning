#!/usr/bin/env python3
'''
Defines a function that concatenates two arrays
'''


def cat_arrays(arr1, arr2):
    '''
    Returns a new concatenated list
    '''
    if len(arr1) >= len(arr2):
        arr12 = arr1.copy()
        for i in arr2:
            arr12.append(i)
    else:
        arr12 = arr2.copy()
        for i in arr1:
            arr12.append(i)
    return arr12
