#!/usr/bin/env python3
"""
Module to calculate the adjugate matrix of a matrix
"""


def determinant(matrix):
    """ Helper function to calculate determinant """
    if len(matrix) == 0:
        return 1
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for j in range(len(matrix)):
        sub = [row[:j] + row[j + 1:] for row in matrix[1:]]
        det += ((-1) ** j) * matrix[0][j] * determinant(sub)
    return det


def adjugate(matrix):
    """
    Calculates the adjugate matrix of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cofactors = []
    for i in range(n):
        row_cofactors = []
        for j in range(n):
            sub = [row[:j] + row[j + 1:] for k, row in
                   enumerate(matrix) if k != i]
            sign = (-1) ** (i + j)
            row_cofactors.append(sign * determinant(sub))
        cofactors.append(row_cofactors)

    adjugate_matrix = []
    for j in range(n):
        new_row = []
        for i in range(n):
            new_row.append(cofactors[i][j])
        adjugate_matrix.append(new_row)

    return adjugate_matrix
