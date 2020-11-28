"""
Losses used for feature visualizations
"""

import tensorflow as tf


def cosine_similarity(tensor_a, tensor_b):
    """
    Return the cosine similarity for batchs of vectors passed.

    Parameters
    ----------
    tensor_a : tf.Tensor (N, ...)
        Batch of N tensors.
    tensor_b : tf.tensor (N, ...)
        Batch of N tensors.

    Returns
    -------
    cosine_similarity : tf.Tensor (N)
        The cosine similarity for each pairs of tensors : <x, y> / (|x|+|y|)
    """
    axis_to_norm = range(1, len(tf.shape(tensor_a)))

    tensor_a = tf.nn.l2_normalize(tensor_a, axis=axis_to_norm)
    tensor_b = tf.nn.l2_normalize(tensor_b, axis=axis_to_norm)

    return tf.reduce_sum(tensor_a * tensor_b, axis=axis_to_norm)
