"""
Module related to ForGrad method enchancement
"""

from functools import lru_cache

import tensorflow as tf
import numpy as np


@lru_cache(maxsize=32)
def _low_pass_real_signal_mask(size : int, bandwith : int) -> tf.Tensor:
    """
    Create a low pass filter mask to be applied on a real signal.
    Since the Discrete Fourier Transform of a real signal is Hermitian-symmetric, only returns the
    fft_length / 2 + 1 unique components of the transform.

    Parameters
    ----------
    size
        Size of the mask.
    bandwith
        Bandwith of the low pass filter. The higher the bandwith, the more frequencies are kept.

    Returns
    -------
    mask
        Low pass filter mask of shape (height, width).
    """
    center = (int(size/2), int(size/2))

    y_grid, x_grid = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((x_grid - center[0])**2 + (y_grid-center[1])**2)

    mask = dist_from_center <= bandwith
    # un-center the mask
    mask = tf.signal.fftshift(mask)
    # keep only the unique components
    mask = tf.cast(mask, tf.float32)[:, :size//2 + 1]

    return mask

def forgrad(explanations : tf.Tensor, sigma: int = 15) -> tf.Tensor:
    """
    ForGRAD is a method that enhances any attributions explanations (particularly useful on
    gradients based attribution method) by eliminating high frequencies in the explanations.

    Ref. Gradient strikes back: How filtering out high frequencies improves explanations (2023).
         https://arxiv.org/pdf/2307.09591.pdf

    Parameters
    ----------
    explanations
        List of explanations to filter. Explanation should be at least 3D (batch, height, width)
        and should have the same height and width.
    sigma
        Bandwith of the low pass filter. The higher the sigma, the more frequencies are kept.
        Sigma should be positive and less than image size.
        Default to paper recommendation, 15 for image size 224.

    Returns
    -------
    filtered_explanations
        Explanations low-pass filtered.
    """
    image_size = explanations.shape[1]

    assert image_size == explanations.shape[2], "Explanations should be square."
    assert len(explanations.shape) > 2, "Explanations should be at least 3D (batch, height, width)."
    assert 0 < sigma <= image_size, "Sigma should be positive and less than image size."

    if len(explanations.shape) == 4:
        explanations = tf.reduce_mean(explanations, -1)
    explanations = tf.abs(explanations)

    spectrums = tf.signal.rfft2d(explanations)

    filter_mask = _low_pass_real_signal_mask(explanations.shape[1], sigma)
    filter_mask = tf.cast(filter_mask, tf.complex64)
    filtered_spectrums = spectrums * filter_mask

    filtered_explanations = tf.signal.irfft2d(filtered_spectrums)
    filtered_explanations = tf.abs(filtered_explanations)

    # resize if odd dimension (cut-off pixel)
    if image_size % 2 == 1:
        filtered_explanations = tf.image.resize(filtered_explanations[:, :, :, None],
                                                (image_size, image_size), method="bicubic")

    if len(explanations.shape) == 4 and len(filtered_explanations.shape) == 3:
        filtered_explanations = tf.expand_dims(filtered_explanations, -1)

    return filtered_explanations
