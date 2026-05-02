#!/usr/bin/env python3
"""
Module to calculate the inverse of a matrix
"""


def determinant(matrix):
    """ Helper function to calculate determinant """
    n = len(matrix)
    if n == 0:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(n):
        sub = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det


def adjugate(matrix):
    """ Helper function to calculate adjugate matrix """
    n = len(matrix)
    if n == 1:
        return [[1]]

    cofactors = []
    for i in range(n):
        row_cof = []
        for j in range(n):
            sub = [row[:j] + row[j + 1:] for k, row in
                   enumerate(matrix) if k != i]
            sign = (-1) ** (i + j)
            row_cof.append(sign * determinant(sub))
        cofactors.append(row_cof)

    return [[cofactors[i][j] for i in range(n)] for j in range(n)]


def inverse(matrix):
    """
    Calculates the inverse of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)

    if det == 0:
        return None

    if n == 1:
        return [[1 / matrix[0][0]]]

    adj = adjugate(matrix)

    return [[cell / det for cell in row] for row in adj]
