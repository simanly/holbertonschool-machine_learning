#!/usr/bin/env python3
"""Valid convolution on grayscale images."""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """Performs a valid convolution on grayscale images.

    images: numpy.ndarray (m, h, w)
    kernel: numpy.ndarray (kh, kw)

    Returns: numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    output_h = h - kh + 1
    output_w = w - kw + 1

    convolved = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            slice_images = images[:, i:i + kh, j:j + kw]
            convolved[:, i, j] = np.sum(slice_images * kernel, axis=(1, 2))

    return convolved
