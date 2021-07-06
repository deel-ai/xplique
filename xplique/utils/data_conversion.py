"""
Numpy from/to Tensorflow manipulation
"""

import tensorflow as tf
import numpy as np

from ..types import Union, Optional, Tuple


def tensor_sanitize(inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
                    targets: Optional[Union[tf.Tensor, np.array]]) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Ensure the output as tf.Tensor, accept various inputs format including:
    tf.Tensor, List, numpy array, tf.data.Dataset (when label = None).

    Parameters
    ----------
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target, one for each sample.

    Returns
    -------
    inputs_tensors
        Inputs samples as tf.Tensor
    targets_tensors
        Targets as tf.Tensor
    """

    # deal with tf.data.Dataset
    if isinstance(inputs, tf.data.Dataset):
        # unpack the dataset, assume we have tuple of (input, target)
        targets = [y for x,y in inputs.unbatch()]
        inputs  = [x for x,y in inputs.unbatch()]

    # deal with numpy array
    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)

    return inputs, targets
