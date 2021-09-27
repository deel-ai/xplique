"""
Module related to Integrated Gradients method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_gradient, batch_tensor
from ..types import Tuple, Union, Optional


class IntegratedGradients(WhiteBoxExplainer):
    """
    Used to compute the Integrated Gradients, by cumulating the gradients along a path from a
    baseline to the desired point.

    Ref. Sundararajan & al., Axiomatic Attribution for Deep Networks (2017).
    http://proceedings.mlr.press/v70/sundararajan17a/sundararajan17a.pdf

    Notes
    -----
    In order to approximate from a finite number of steps, the implementation here use the
    trapezoidal rule and not a Riemann sum (see the paper below for a comparison of the results).
    Ref. Computing Linear Restrictions of Neural Networks (2019).
    https://arxiv.org/abs/1908.06214

    Parameters
    ----------
    model
        Model used for computing explanations.
    output_layer
        Layer to target for the output (e.g logits or after softmax), if int, will be be interpreted
        as layer index, if string will look for the layer name. Default to the last layer, it is
        recommended to use the layer before Softmax.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    steps
        Number of points to interpolate between the baseline and the desired point.
    baseline_value
        Scalar used to create the the baseline point.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = -1,
                 batch_size: Optional[int] = 32,
                 steps: int = 50,
                 baseline_value: float = .0):
        super().__init__(model, output_layer, batch_size)
        self.steps = steps
        self.baseline_value = baseline_value

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute Integrated Gradients for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            Integrated gradients, same shape as the inputs.
        """
        integrated_gradients = None
        batch_size = self.batch_size or len(inputs)
        baseline = IntegratedGradients._get_baseline((*inputs.shape[1:],),
                                                     self.baseline_value)

        for x_batch, y_batch in batch_tensor((inputs, targets),
                                             max(batch_size // self.steps, 1)):
            # create the paths for every sample (interpolated points from baseline to sample)
            interpolated_inputs = IntegratedGradients._get_interpolated_points(
                x_batch, self.steps, baseline)
            repeated_targets = repeat_labels(y_batch, self.steps)

            # compute the gradient for each paths
            interpolated_gradients = batch_gradient(self.model, interpolated_inputs,
                                                    repeated_targets, batch_size)
            interpolated_gradients = tf.reshape(interpolated_gradients,
                                                (-1, self.steps, *interpolated_gradients.shape[1:]))

            # average the gradient using trapezoidal rule
            averaged_gradients = IntegratedGradients._average_gradients(interpolated_gradients)
            batch_integrated_gradients = (x_batch - baseline) * averaged_gradients

            integrated_gradients = batch_integrated_gradients if integrated_gradients is None else \
                tf.concat([integrated_gradients, batch_integrated_gradients], axis=0)

        return integrated_gradients

    @staticmethod
    def _get_baseline(shape: Tuple,
                      baseline_value: float) -> tf.Tensor:
        """
        Create the baseline point using a scalar value to fill the desired shape.

        Parameters
        ----------
        shape
            Desired shape, dimension of one sample.
        baseline_value
            Value defining the baseline state.

        Returns
        -------
        baseline_point
            A baseline point of the specified shape.
        """
        return tf.ones(shape) * baseline_value

    @staticmethod
    @tf.function
    def _get_interpolated_points(inputs: tf.Tensor,
                                 steps: int,
                                 baseline: tf.Tensor) -> tf.Tensor:
        """
        Create a path from baseline to sample for every samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        steps
            Number of points to interpolate between the baseline and the desired point.
        baseline
            Baseline point, start of the path.

        Returns
        -------
        interpolated_inputs
            Interpolated path for each inputs, the first dimension correspond to the number of
            samples multiplied by steps.
        """
        alpha = tf.reshape(tf.linspace(0.0, 1.0, steps), (1, -1, *(1,) * len(inputs.shape[1:])))

        interpolated_inputs = tf.expand_dims(inputs, axis=1)
        interpolated_inputs = tf.repeat(interpolated_inputs, repeats=steps, axis=1)
        interpolated_inputs = baseline + alpha * (interpolated_inputs - baseline)

        interpolated_inputs = tf.reshape(interpolated_inputs, (-1, *interpolated_inputs.shape[2:]))

        return interpolated_inputs

    @staticmethod
    @tf.function
    def _average_gradients(gradients: tf.Tensor) -> tf.Tensor:
        """
        Average the gradients obtained along the path using trapezoidal rule.

        Parameters
        ----------
        gradients
            Gradients obtained for each of the steps for each of the samples.

        Returns
        -------
        integrated_gradients
        """
        trapezoidal_gradients = gradients[:, :-1] + gradients[:, 1:]
        averaged_gradients = tf.reduce_mean(trapezoidal_gradients, axis=1) * 0.5

        return averaged_gradients
