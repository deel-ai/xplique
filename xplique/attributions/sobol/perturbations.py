import cv2
import numpy as np
import tensorflow as tf


@tf.function
def _baseline_ponderation(x, masks, x0):
    return tf.expand_dims(x, 0) * masks + (1.0 - masks) * tf.expand_dims(x0, 0)


@tf.function
def _amplitude_operator(x, masks, sigma):
    return x[None, :, :, :] * (masks - 0.5) * sigma


def inpainting(input):
    """
    Tensorflow inpainting perturbation function.

    X_perturbed = X * M

    Parameters
    ----------
    input: tf.Tensor
        Image to perform perturbation on.

    Returns
    -------
    f: callable
        Inpainting perturbation function.
    """
    x0 = np.zeros(input.shape)
    x0 = tf.cast(x0, tf.float32)

    def f(masks):
        return _baseline_ponderation(input, masks, x0)
    return f


def blurring(input, sigma=10):
    """
    Tensorflow blur perturbation function.

    X_perturbed = blur(X, M)

    Parameters
    ----------
    input: tf.Tensor
        Image to perform perturbation on.
    sigma: int
        Blurring operator intensity.

    Returns
    -------
    f: callable
        Blur perturbation function.
    """
    x0 = cv2.blur(input.copy(), (sigma, sigma))
    x0 = tf.cast(x0, tf.float32)

    def f(masks):
        return _baseline_ponderation(input, masks, x0)
    return f


def amplitude(input, sigma=1.0):
    """
    Tensorflow amplitude perturbation function.

    X_perturbed = X + (M - 0.5) * sigma

    Parameters
    ----------
    input: tf.Tensor
        Image to perform perturbation on.
    sigma: int
        Amplitude operator intensity.

    Returns
    -------
    f: callable
        Blur perturbation function.
    """

    def f(masks):
        return _amplitude_operator(input, masks, sigma)
    return f