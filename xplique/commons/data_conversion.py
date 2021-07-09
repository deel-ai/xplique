"""
Numpy from/to Tensorflow manipulation
"""

import tensorflow as tf
import numpy as np

from ..types import Union, Optional, Tuple


def tensor_sanitize(inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                    targets: Optional[Union[tf.Tensor, np.ndarray]]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Ensure the output as tf.Tensor, accept various inputs format including:
    tf.Tensor, List, numpy array, tf.data.Dataset (when label = None).

    Parameters
    ----------
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    inputs_tensors
        Inputs samples as tf.Tensor
    targets_tensors
        Targets as tf.Tensor
    """

    # deal with tf.data.Dataset
    if isinstance(inputs, tf.data.Dataset):
        # if the dataset as 4 dimensions, assume it is batched
        dataset_shape = inputs.element_spec[0].shape
        if len(dataset_shape) == 4:
            inputs = inputs.unbatch()
        # unpack the dataset, assume we have tuple of (input, target)
        targets = [target for inp, target in inputs]
        inputs  = [inp for inp, target in inputs]

    # deal with numpy array
    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)

    return inputs, targets


def numpy_sanitize(inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                   targets: Optional[Union[tf.Tensor, np.ndarray]]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Ensure the output as np.ndarray, accept various inputs format including:
    tf.Tensor, List, numpy array, tf.data.Dataset (when label = None).

    Parameters
    ----------
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    inputs_ndarray
        Inputs samples as np.ndarray
    targets_ndarray
        Targets as np.ndarray
    """
    inputs, targets = tensor_sanitize(inputs, targets)
    return inputs.numpy(), targets.numpy()
