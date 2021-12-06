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
Losses used for feature visualizations
"""
import tensorflow as tf


@tf.function
def cosine_similarity(tensor_a: tf.Tensor, tensor_b: tf.Tensor) -> tf.Tensor:
    """
    Return the cosine similarity for batchs of vectors passed.

    Parameters
    ----------
    tensor_a
        Batch of N tensors.
    tensor_b
        Batch of N tensors.

    Returns
    -------
    cosine_similarity
        The cosine similarity for each pairs of tensors : <x, y> / (|x|+|y|)
    """
    axis_to_norm = range(1, len(tf.shape(tensor_a)))

    tensor_a = tf.nn.l2_normalize(tensor_a, axis=axis_to_norm)
    tensor_b = tf.nn.l2_normalize(tensor_b, axis=axis_to_norm)

    return tf.reduce_sum(tensor_a * tensor_b, axis=axis_to_norm)


@tf.function
def dot_cossim(
    tensor_a: tf.Tensor, tensor_b: tf.Tensor, cossim_pow: float = 2.0
) -> tf.Tensor:
    """
    Return the product of the cosine similarity and the dot product for batchs of vectors passed.
    This original looking loss was proposed by the authors of lucid and seeks to both optimise
    the direction with cosine similarity, but at the same time exaggerate the feature (caricature)
    with the dot product.

    source: https://github.com/tensorflow/lucid/issues/116

    Parameters
    ----------
    tensor_a
        Batch of N tensors.
    tensor_b
        Batch of N tensors.
    cossim_pow
        Power of the cosine similarity, higher value encourage the objective to care more about
        the angle of the activations.

    Returns
    -------
    dot_cossim_value
        The product of the cosine similarity and the dot product for each pairs of tensors.
    """

    cosim = tf.maximum(cosine_similarity(tensor_a, tensor_b), 1e-1) ** cossim_pow
    dot = tf.reduce_sum(tensor_a * tensor_b)

    return dot * cosim
