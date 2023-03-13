"""
Custom callable operations
"""

import tensorflow as tf

from .operators import operator_batching
from ..types import Callable

def predictions_one_hot_callable(
    model: Callable,
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
    if isinstance(model, tf.lite.Interpreter):

        model.resize_tensor_input(0, [*inputs.shape], strict=False)
        model.allocate_tensors()
        model.set_tensor(model.get_input_details()[0]["index"], inputs)
        model.invoke()
        pred = model.get_tensor(model.get_output_details()[0]["index"])

    # can be a sklearn model or xgboost model
    elif hasattr(model, 'predict_proba'):
        pred = model.predict_proba(inputs.numpy())

    # can be another model thus it needs to implement a call function
    else:
        pred = model(inputs.numpy())

    # make sure that the prediction shape is coherent
    if inputs.shape[0] != 1:
        # a batch of prediction is required
        if len(pred.shape) == 1:
            # The prediction dimension disappeared
            pred = tf.expand_dims(pred, axis=1)

    pred = tf.cast(pred, dtype=tf.float32)
    scores = tf.reduce_sum(pred * targets, axis=-1)

    return scores

batch_predictions_one_hot_callable = operator_batching(predictions_one_hot_callable)
