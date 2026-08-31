#!/usr/bin/env python3
"""Convolution on grayscale images with custom padding."""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """Performs a convolution on grayscale images with custom padding.

    images: numpy.ndarray (m, h, w)
    kernel: numpy.ndarray (kh, kw)
    padding: tuple of (ph, pw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    images_padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant',
        constant_values=0
    )

    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            slice_images = images_padded[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(slice_images * kernel, axis=(1, 2))

    return convolved
