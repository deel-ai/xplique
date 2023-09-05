"""
Custom tensorflow operator for Attributions
"""

import inspect
from enum import Enum

import tensorflow as tf

from ..types import Callable, Optional, Union, OperatorSignature
from .exceptions import raise_invalid_operator, no_gradients_available
from .callable_operations import predictions_one_hot_callable

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
def regression_operator(model: Callable,
                        inputs: tf.Tensor,
                        targets: tf.Tensor) -> tf.Tensor:
    """
    Compute the the mean absolute error between model prediction and the target.
    Target should the model prediction on non-perturbed input.
    This operator can be used to compute attributions for all outputs of a regression model.

    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        Model prediction on non-perturbed inputs.

    Returns
    -------
    scores
        MAE between model prediction and targets.
    """
    scores = tf.reduce_mean(tf.abs(model(inputs) - targets), axis=-1)
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

class Tasks(Enum):
    """
    Enumeration of different tasks for which we have defined operators
    """
    CLASSIFICATION = predictions_operator
    REGRESSION = regression_operator

    @staticmethod
    def from_string(operator_name: str) -> "Tasks":
        """
        Restore an operator from a string

        Parameters
        ----------
        operator_name
            String indicating the operator to restore: must be one
            of 'classification' or 'regression'

        Returns
        -------
        operator
            The Tasks object
        """
        assert operator_name in [
            "classification",
            "regression",
        ], "Only 'classification' and 'regression' are supported."

        if operator_name == "regression":
            return Tasks.REGRESSION
        return Tasks.CLASSIFICATION

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
    # handle tf functions
    # pylint: disable=protected-access
    if hasattr(operator, '_python_function'):
        return check_operator(operator._python_function)

    # the operator must be callable
    if not hasattr(operator, '__call__'):
        raise_invalid_operator()

    # the operator should take at least three arguments
    args = inspect.getfullargspec(operator).args
    if len(args) < 3:
        raise_invalid_operator()

    return True

def get_operator(
        operator: Optional[Union[Tasks, str, OperatorSignature]]):
    """
    This function allows to retrieve an operator from: a Tasks, a task name. If the operator
    is a custom one, we simply check if its signature is correct

    Parameters
    ----------
    operator
        An operator from the Tasks enum or the task name or a custom operator. If None, use a
        classification operator.

    Returns
    -------
    operator
        The operator requested
    """
    # case when no operator is provided
    if operator is None:
        return predictions_operator

    # case when the query is a string
    if isinstance(operator, str):
        return Tasks.from_string(operator)

    # case when the query belong to the Tasks enum
    if operator in [t.value for t in Tasks]:
        return operator

    # case when the operator is a custom one
    assert check_operator(operator)
    return operator

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


def operator_batching(operator: OperatorSignature) -> tf.Tensor:
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


batch_predictions = operator_batching(predictions_operator)
gradients_predictions = get_gradient_of_operator(predictions_operator)
batch_gradients_predictions = operator_batching(gradients_predictions)
batch_predictions_one_hot_callable = operator_batching(predictions_one_hot_callable)


def get_inference_function(
        model: Callable,
        operator: Optional[OperatorSignature] = None):
    """
    Define the inference function according to the model type

    Parameters
    ----------
    model
        Model used for computing explanations.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].

    Returns
    -------
    inference_function
        Same definition as the operator.
    batch_inference_function
        An inference function which treat inputs and targets by batch,
        it has an additionnal parameter `batch_size`.
    """
    if operator is not None:
        # user specified a string, an operator from the ones available or a
        # custom operator, we check if the operator is valid
        # and we wrap it to generate a batching version of this operator
        operator = get_operator(operator)
        inference_function = operator
        batch_inference_function = operator_batching(operator)

    elif isinstance(model, (tf.keras.Model, tf.Module, tf.keras.layers.Layer)):
        inference_function = predictions_operator
        batch_inference_function = batch_predictions

    else:
        # completely unknown model (e.g. sklearn), we can't backprop through it
        inference_function = predictions_one_hot_callable
        batch_inference_function = batch_predictions_one_hot_callable

    return inference_function, batch_inference_function


def get_gradient_functions(
        model: Callable,
        operator: Optional[OperatorSignature] = None):
    """
    Define the gradient function according to the model type

    Parameters
    ----------
    model
        Model used for computing explanations.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].

    Returns
    -------
    gradient
        Gradient function of the operator.
    batch_gradient
        An gradient function which treat inputs and targets by batch,
        it has an additionnal parameter `batch_size`.
    """
    if operator is not None:
        # user specified a string, an operator from the ones available or a
        # custom operator, we check if the operator is valid
        # and we wrap it to generate a batching version of this operator
        operator = get_operator(operator)
        gradient = get_gradient_of_operator(operator)
        batch_gradient = operator_batching(gradient)

    elif isinstance(model, tf.keras.Model):
        # no custom operator, for keras model we can backprop through the model
        gradient = gradients_predictions
        batch_gradient = batch_gradients_predictions

    else:
        # custom model or completely unknown model (e.g. sklearn), we can't backprop through it
        gradient = no_gradients_available
        batch_gradient = no_gradients_available

    return gradient, batch_gradient
    