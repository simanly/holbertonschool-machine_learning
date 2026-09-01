#!/usr/bin/env python3
"""Pooling forward propagation module."""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer.

    Parameters:
    - A_prev: numpy.ndarray, shape (m, h_prev, w_prev, c_prev)
    - kernel_shape: tuple, (kh, kw)
    - stride: tuple, (sh, sw)
    - mode: str, 'max' or 'avg'

    Returns:
    - output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    out_h = int((h_prev - kh) / sh) + 1
    out_w = int((w_prev - kw) / sw) + 1

    A = np.zeros((m, out_h, out_w, c_prev))

    for h in range(out_h):
        for w in range(out_w):
            h_start = h * sh
            h_end = h_start + kh
            w_start = w * sw
            w_end = w_start + kw

            slice_A = A_prev[:, h_start:h_end, w_start:w_end, :]

            if mode == 'max':
                A[:, h, w, :] = np.max(slice_A, axis=(1, 2))
            elif mode == 'avg':
                A[:, h, w, :] = np.mean(slice_A, axis=(1, 2))

    return A
