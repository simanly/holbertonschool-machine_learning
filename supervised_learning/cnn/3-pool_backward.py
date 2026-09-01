#!/usr/bin/env python3
"""Pooling back propagation module."""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs back propagation over a pooling layer.

    Parameters:
    - dA: numpy.ndarray, shape (m, h_new, w_new, c)
    - A_prev: numpy.ndarray, shape (m, h_prev, w_prev, c)
    - kernel_shape: tuple, (kh, kw)
    - stride: tuple, (sh, sw)
    - mode: str, 'max' or 'avg'

    Returns:
    - dA_prev: partial derivatives with respect to previous layer
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    da = dA[i, h, w, k]

                    if mode == 'max':
                        slice_a = a_prev[h_start:h_end, w_start:w_end, k]
                        mask = (slice_a == np.max(slice_a))
                        dA_prev[i, h_start:h_end, w_start:w_end, k] += (
                            mask * da
                        )
                    elif mode == 'avg':
                        avg_da = da / (kh * kw)
                        dA_prev[i, h_start:h_end, w_start:w_end, k] += (
                            np.ones((kh, kw)) * avg_da
                        )

    return dA_prev
