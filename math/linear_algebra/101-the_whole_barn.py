#!/usr/bin/env python3
'''
Defines a function that adds two matrices
'''


def add_matrices(mat1, mat2):
    '''
    Return the sum like new matrix
    '''
    is_list1 = isinstance(mat1, list)
    is_list2 = isinstance(mat2, list)
    if is_list1 != is_list2:
       return None
    if is_list1 and is_list2:
        if len(mat1) != len(mat2):
            return None
        mat12 = []
        for i in range(len(mat1)):
            result = add_matrices(mat1[i], mat2[i])
            if result is None:
                return None
            mat12.append(result)
        return mat12
    return (mat1 + mat2)
