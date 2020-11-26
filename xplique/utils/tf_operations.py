"""
Custom tensorflow operations
"""

import tensorflow as tf


def repeat_labels(labels, nb_repetitions):
    """
    Duplicate each label nb_repetitions times.

    Parameters
    ----------
    labels : tf.tensor (N, L)
        One hot encoded labels to compute for each sample, with N the number of samples, and L
        the number of classes.
    nb_repetitions : int
        Number of times each labels should be duplicate.

    Returns
    -------
    repeated_labels : tf.tensor (N * nb_repetitions, L)
        Unchanged label repeated.
    """
    repeated_labels = tf.expand_dims(labels, axis=1)
    repeated_labels = tf.repeat(repeated_labels, repeats=nb_repetitions, axis=1)

    repeated_labels = tf.reshape(repeated_labels, (-1, *repeated_labels.shape[2:]))

    return repeated_labels
