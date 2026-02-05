"""
Custom tensorflow operations
"""

import tensorflow as tf

from ..types import Optional, Tuple, Union


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


def batch_tensor(tensors: Union[Tuple, tf.Tensor], batch_size: Optional[int] = None):
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


def get_device(device: Optional[str] = None) -> str:
    """
    Gets the name of the device to use. If there are any available GPUs, it will use the first one
    in the system, otherwise, it will use the CPU.

    Parameters
    ----------
    device
        A string specifying the device on which to run the computations. If None, it will search
        for available GPUs, and if none are found, it will return the first CPU.

    Returns
    -------
    device
        A string with the name of the device on which to run the computations.
    """
    if device is not None:
        return device

    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices is None or len(physical_devices) == 0:
        return "cpu:0"
    return "GPU:0"
