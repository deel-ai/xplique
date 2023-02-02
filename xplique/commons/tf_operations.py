"""
Custom tensorflow operations
"""

import tensorflow as tf

from ..types import Optional, Union, Tuple


def repeat_labels(labels: tf.Tensor, nb_repetitions: int) -> tf.Tensor:
    """
    Duplicate each label nb_repetitions times.

    Parameters
    ----------
    labels
        One hot encoded labels (N, L) to compute for each sample, with N the number of samples,
        and L the number of classes.
    nb_repetitions
        Number of times each labels should be duplicate.

    Returns
    -------
    repeated_labels
        Unchanged label repeated (N*nb_repetitions, L).
    """
    repeated_labels = tf.expand_dims(labels, axis=1)
    repeated_labels = tf.repeat(repeated_labels, repeats=nb_repetitions, axis=1)

    repeated_labels = tf.reshape(repeated_labels, (-1, *repeated_labels.shape[2:]))

    return repeated_labels


def batch_tensor(tensors: Union[Tuple, tf.Tensor],
                 batch_size: Optional[int] = None):
    """
    Create a tensorflow dataset of tensors or series of tensors.

    Parameters
    ----------
    tensors
        Tuple of tensors or tensors to batch.
    batch_size
        Number of samples to iterate at once, if None process all at once.

    Returns
    -------
    dataset
        Tensorflow dataset batched.
    """
    dataset = tf.data.Dataset.from_tensor_slices(tensors)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    return dataset
