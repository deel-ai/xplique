"""
Custom tensorflow operator for Attributions
"""

import inspect
import tensorflow as tf
from ..types import Callable
from .exceptions import raise_invalid_operator


@tf.function
def predictions_operator(model: Callable,
                         inputs: tf.Tensor,
                         targets: tf.Tensor) -> tf.Tensor:
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
def binary_segmentation_operator(model: Callable,
                                 inputs: tf.Tensor,
                                 targets: tf.Tensor) -> tf.Tensor:
    """
    Compute the segmentation score for a batch of samples.

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
        Segmentation scores computed.
    """
    scores = tf.reduce_sum(model(inputs) * targets, axis=(1, 2))
    return scores


@tf.function
def segmentation_operator(model: Callable,
                          inputs: tf.Tensor,
                          targets: tf.Tensor) -> tf.Tensor:
    """
    Compute the segmentation score for a batch of samples.

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
        Segmentation scores computed.
    """
    scores = tf.reduce_sum(model(inputs) * targets, axis=(1, 2, 3))
    return scores


def get_gradient_of_operator(operator):
    """
    Get the gradient of an operator.

    Parameters
    ----------
    operator
        Operator to compute the gradient of.

    Returns
    -------
    gradient
        Gradient of the operator.
    """
    @tf.function
    def gradient(model, inputs, targets):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            scores = operator(model, inputs, targets)

        return tape.gradient(scores, inputs)

    return gradient


def operator_batching(operator: Callable) -> tf.Tensor:
    """
    Take care of batching an operator: (model, inputs, labels).

    Parameters
    ----------
    operator
        Any callable that take model, inputs and labels as parameters.

    Returns
    -------
    batched_operator
        Function that apply operator by batch.
    """

    def batched_operator(model, inputs, targets, batch_size=None):
        if batch_size is not None:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
            results = tf.concat([
                operator(model, x, y)
                for x, y in dataset.batch(batch_size)
            ], axis=0)
        else:
            results = operator(model, inputs, targets)

        return results

    return batched_operator


def check_operator(operator: Callable):
    """
    Check if the operator is valid g(f, x, y) -> tf.Tensor
    and raise an exception and return true if so.

    Parameters
    ----------
    operator
        Operator to check

    Returns
    -------
    is_valid
        True if the operator is valid, False otherwise.
    """
    is_callable = hasattr(operator, "__call__")
    args = inspect.getfullargspec(operator).args

    # we allow operator with optional arguments, but the first 3 must be present
    is_valid = is_callable and len(args) >= 3

    if not is_valid:
        raise_invalid_operator()


batch_predictions = operator_batching(predictions_operator)
gradients_predictions = get_gradient_of_operator(predictions_operator)
batch_gradients_predictions = operator_batching(gradients_predictions)
