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
Custom callable operations
"""
import tensorflow as tf

from ..types import Callable
from ..types import Optional
from .tf_operations import inference_batching


def predictions_one_hot_callable(
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
    if isinstance(model, tf.lite.Interpreter):

        model.resize_tensor_input(0, [*inputs.shape], strict=False)
        model.allocate_tensors()
        model.set_tensor(model.get_input_details()[0]["index"], inputs)
        model.invoke()
        pred = model.get_tensor(model.get_output_details()[0]["index"])

    # can be a sklearn model or xgboost model
    elif hasattr(model, "predict_proba"):
        pred = model.predict_proba(inputs.numpy())

    # can be another model thus it needs to implement a call function
    else:
        pred = model(inputs.numpy())

    pred = tf.cast(pred, dtype=tf.float32)
    scores = tf.reduce_sum(pred * targets, axis=-1)

    return scores


def batch_predictions_one_hot_callable(
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
    return inference_batching(
        predictions_one_hot_callable, model, inputs, targets, batch_size
    )
