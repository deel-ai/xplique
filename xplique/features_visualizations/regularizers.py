# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        return factor * tf.sqrt(tf.reduce_mean(images ** 2, (1, 2, 3)))

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
