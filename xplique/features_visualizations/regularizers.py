"""
Image regularizers
"""

import tensorflow as tf

from ..types import Callable


def l1_reg(factor: float = 1.0) -> Callable:
    """
    Mean L1 regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    reg
        Mean L1 of the images.
    """
    def reg(images: tf.Tensor) -> tf.Tensor:
        return factor * tf.reduce_mean(tf.abs(images), (1, 2, 3))
    return reg


def l2_reg(factor: float = 1.0) -> Callable:
    """
    Mean L2 regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    reg
        Mean L2 of the images.
    """
    def reg(images: tf.Tensor) -> tf.Tensor:
        return factor * tf.sqrt(tf.reduce_mean(images**2, (1, 2, 3)))
    return reg


def l_inf_reg(factor: float = 1.0) -> Callable:
    """
    Mean L-inf regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    l_inf
        Max of the images.
    """
    def l_inf(images: tf.Tensor) -> tf.Tensor:
        return factor * tf.reduce_max(tf.abs(images), (1, 2, 3))
    return l_inf


def total_variation_reg(factor: float = 1.0) -> Callable:
    """
    Total variation regularization.

    Parameters
    ----------
    factor
        Weight that controls the importance of the regularization term.

    Returns
    -------
    tv_reg
        Total variation of the images.
    """
    def tv_reg(images: tf.Tensor) -> tf.Tensor:
        return factor * tf.image.total_variation(images)
    return tv_reg
