"""
Image regularizers
"""

import tensorflow as tf


def l1_reg(factor=1.0):
    """
    Mean L1 regularization.

    Parameters
    ----------
    factor : float, optional
        Weight that controls the importance of the regularization term.

    Returns
    -------
    reg : func
        Mean L1 of the images.
    """
    def reg(images):
        return factor * tf.reduce_mean(tf.abs(images), (1, 2, 3))
    return reg


def l2_reg(factor=1.0):
    """
    Mean L2 regularization.

    Parameters
    ----------
    factor : float, optional
        Weight that controls the importance of the regularization term.

    Returns
    -------
    reg : func
        Mean L2 of the images.
    """
    def reg(images):
        return factor * tf.sqrt(tf.reduce_mean(images**2, (1, 2, 3)))
    return reg


def l_inf_reg(factor=1.0):
    """
    Mean L-inf regularization.

    Parameters
    ----------
    factor : float, optional
        Weight that controls the importance of the regularization term.

    Returns
    -------
    l_inf : func
        Max of the images.
    """
    def l_inf(images):
        return factor * tf.reduce_max(tf.abs(images), (1, 2, 3))
    return l_inf


def total_variation_reg(factor=1.0):
    """
    Total variation regularization.

    Parameters
    ----------
    factor : float, optional
        Weight that controls the importance of the regularization term.

    Returns
    -------
    tv_reg : func
        Total variation of the images.
    """
    def tv_reg(images):
        return factor * tf.image.total_variation(images)
    return tv_reg
