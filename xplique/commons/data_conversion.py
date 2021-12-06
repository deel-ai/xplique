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
Numpy from/to Tensorflow manipulation
"""
import numpy as np
import tensorflow as tf

from ..types import Optional
from ..types import Tuple
from ..types import Union


def tensor_sanitize(
    inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
    targets: Optional[Union[tf.Tensor, np.ndarray]],
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
        # if the dataset as 4 dimensions, assume it is batched
        dataset_shape = inputs.element_spec[0].shape
        if len(dataset_shape) == 4:
            inputs = inputs.unbatch()
        # unpack the dataset, assume we have tuple of (input, target)
        targets = [target for inp, target in inputs]
        inputs = [inp for inp, target in inputs]

    # deal with numpy array
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
