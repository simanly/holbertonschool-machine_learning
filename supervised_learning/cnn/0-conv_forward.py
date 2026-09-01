#!/usr/bin/env python3
"""Convolutional forward propagation module."""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer.

    Parameters:
    - A_prev: numpy.ndarray, shape (m, h_prev, w_prev, c_prev)
    - W: numpy.ndarray, shape (kh, kw, c_prev, c_new)
    - b: numpy.ndarray, shape (1, 1, 1, c_new)
    - activation: function, activation function
    - padding: str, "same" or "valid"
    - stride: tuple, (sh, sw)

    Returns:
    - output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph, pw = 0, 0

    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    out_h = int((h_prev + 2 * ph - kh) / sh) + 1
    out_w = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, out_h, out_w, c_new))

    for h in range(out_h):
        for w in range(out_w):
            h_start = h * sh
            h_end = h_start + kh
            w_start = w * sw
            w_end = w_start + kw

            slice_A = A_prev_padded[:, h_start:h_end, w_start:w_end, :]

            for k in range(c_new):
                Z[:, h, w, k] = np.sum(
                    slice_A * W[:, :, :, k],
                    axis=(1, 2, 3)
                )

    Z = Z + b
    A = activation(Z)

    return A
