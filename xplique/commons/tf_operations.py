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
Custom tensorflow operations
"""
import tensorflow as tf

from ..types import Callable
from ..types import Optional
from ..types import Tuple
from ..types import Union


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


@tf.function
def predictions_one_hot(
    model: Callable, inputs: tf.Tensor, targets: tf.Tensor
) -> tf.Tensor:
    """
    Compute predictions scores, only for the label class, for a batch of samples.

    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    scores
        Predictions scores computed, only for the label class.
    """
    scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
    return scores


@tf.function
def gradient(model: Callable, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """
    Compute gradients for a batch of samples.

    Parameters
    ----------
    model
        Model used for computing gradient.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    gradients
        Gradients computed, with the same shape as the inputs.
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape:  # type: ignore
        tape.watch(inputs)
        score = tf.reduce_sum(tf.multiply(model(inputs), targets), axis=1)
    return tape.gradient(score, inputs)


def inference_batching(
    operation: Callable,
    model: Callable,
    inputs: tf.Tensor,
    targets: tf.Tensor,
    batch_size: Optional[int],
) -> tf.Tensor:
    """
    Take care of batching an inference operation: (model, inputs, labels).

    Parameters
    ----------
    operation
        Any callable that take model, inputs and labels as parameters.
    model
        Callable that will be passed to the operation.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to explain at once, if None compute all at once.

    Returns
    -------
    results
        Results of the batched operations.
    """
    if batch_size is not None:
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        results = tf.concat(
            [operation(model, x, y) for x, y in dataset.batch(batch_size)], axis=0
        )
    else:
        results = operation(model, inputs, targets)

    return results


def batch_predictions_one_hot(
    model: Callable,
    inputs: tf.Tensor,
    targets: tf.Tensor,
    batch_size: Optional[int] = None,
) -> tf.Tensor:
    """
    Compute predictions scores, only for the label class, for the samples passed. Take
    care of splitting in multiple batches if batch_size is specified.

    Parameters
    ----------
    model
        Model used for computing predictions score.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to predict at once, if None compute all at once.

    Returns
    -------
    scores
        Predictions scores computed, only for the label class.
    """
    return inference_batching(predictions_one_hot, model, inputs, targets, batch_size)


def batch_gradient(
    model: Callable, inputs: tf.Tensor, targets: tf.Tensor, batch_size: Optional[int]
) -> tf.Tensor:
    """
    Compute the gradients of the sample passed, take care of splitting the samples in
    multiple batches if batch_size is specified.

    Parameters
    ----------
    model
        Model used for computing gradient.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
    batch_size
        Number of samples to explain at once, if None compute all at once.

    Returns
    -------
    gradients
        Gradients computed, with the same shape as the inputs.
    """
    return inference_batching(gradient, model, inputs, targets, batch_size)


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
