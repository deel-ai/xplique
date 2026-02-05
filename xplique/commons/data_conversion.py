"""
Numpy from/to Tensorflow manipulation
"""

import numpy as np
import tensorflow as tf

from ..types import Callable, Optional, Tuple, Union


def tensor_sanitize(
    inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray], targets: Union[tf.Tensor, np.ndarray]
) -> Tuple[tf.Tensor, tf.Tensor]:
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
        # try to know if the dataset is batched, if it is the case we unbatch
        if hasattr(inputs, "_batch_size"):
            inputs = inputs.unbatch()
        # unpack the dataset, assume we have tuple of (input, target)
        targets = [target for _, target in inputs]
        inputs = [inp for inp, _ in inputs]

    inputs = tf.cast(inputs, tf.float32)
    targets = tf.cast(targets, tf.float32)

    return inputs, targets


def numpy_sanitize(
    inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
    targets: Optional[Union[tf.Tensor, np.ndarray]],
) -> Tuple[tf.Tensor, tf.Tensor]:
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


def sanitize_inputs_targets(explanation_method: Callable):
    """
    Wrap a method explanation function to ensure tf.Tensor as inputs and targets.
    But targets may be None.

    explanation_method
        Function to wrap, should return an tf.tensor.
    """

    def sanitize(
        self,
        inputs: Union[tf.Tensor, np.array],
        targets: Optional[Union[tf.Tensor, np.array]] = None,
        *args,
        **kwargs,
    ):
        # pylint: disable=keyword-arg-before-vararg
        # ensure we have tf.tensor
        inputs = tf.cast(inputs, tf.float32)
        if targets is not None:
            targets = tf.cast(targets, tf.float32)

        if args:
            args = [tf.cast(arg, tf.float32) for arg in args]

        if kwargs:
            kwargs = {key: tf.cast(value, tf.float32) for key, value in kwargs.items()}

        # then enter the explanation function
        return explanation_method(self, inputs, targets, *args, **kwargs)

    return sanitize
