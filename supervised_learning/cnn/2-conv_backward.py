#!/usr/bin/env python3
"""Convolutional back propagation module."""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer.

    Parameters:
    - dZ: numpy.ndarray, shape (m, h_new, w_new, c_new)
    - A_prev: numpy.ndarray, shape (m, h_prev, w_prev, c_prev)
    - W: numpy.ndarray, shape (kh, kw, c_prev, c_new)
    - b: numpy.ndarray, shape (1, 1, 1, c_new)
    - padding: str, "same" or "valid"
    - stride: tuple, (sh, sw)

    Returns:
    - dA_prev: partial derivatives with respect to the previous layer
    - dW: partial derivatives with respect to the kernels
    - db: partial derivatives with respect to the biases
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

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
    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(m):
        a_prev_pad = A_prev_padded[i]
        da_prev_pad = dA_prev_padded[i]
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    slice_a = a_prev_pad[h_start:h_end, w_start:w_end, :]
                    dz = dZ[i, h, w, k]

                    da_prev_pad[h_start:h_end, w_start:w_end, :] += (
                        W[:, :, :, k] * dz
                    )
                    dW[:, :, :, k] += slice_a * dz

    if padding == "same":
        if ph != 0 and pw != 0:
            dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
        elif ph != 0:
            dA_prev = dA_prev_padded[:, ph:-ph, :, :]
        elif pw != 0:
            dA_prev = dA_prev_padded[:, :, pw:-pw, :]
        else:
            dA_prev = dA_prev_padded
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
