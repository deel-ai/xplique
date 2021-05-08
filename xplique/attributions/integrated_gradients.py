"""
Module related to Integrated Gradients method
"""

import tensorflow as tf

from .base import BaseExplanation
from ..utils import sanitize_input_output, repeat_labels


class IntegratedGradients(BaseExplanation):
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
    model : tf.keras.Model
        Model used for computing explanations.
    output_layer_index : int, optional
        Index of the output layer, default to the last layer, it is recommended to use the layer
        before Softmax (often '-2').
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    steps : int, optional
        Number of points to interpolate between the baseline and the desired point.
    baseline_value : float, optional
        Scalar used to create the the baseline point.
    """

    def __init__(self, model, output_layer_index=-1, batch_size=32, steps=50, baseline_value=.0):
        super().__init__(model, output_layer_index, batch_size)
        self.steps = steps
        self.baseline_value = baseline_value

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute Integrated Gradients for a batch of samples.

        Parameters
        ----------
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        explanations : ndarray (N, W, H, C)
            Integrated gradients, same shape as the inputs.
        """
        return IntegratedGradients._compute(self.model,
                                            inputs,
                                            labels,
                                            self.batch_size,
                                            steps=self.steps,
                                            baseline_value=self.baseline_value)

    @staticmethod
    def _compute(model, inputs, labels, batch_size, steps, baseline_value):
        """
        Compute Integrated Gradients for a batch of samples.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing explanations.
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : tf.tensor (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.
        batch_size : int
            Number of samples to explain at once, if None compute all at once.
        steps : int
            Number of points to interpolate between the baseline and the desired point.
        baseline_value : float
            Value defining the baseline state.

        Returns
        -------
        explanations : tf.Tensor (N, W, H, C)
            Integrated gradients, same shape as the inputs.
        """
        integrated_gradients = None
        batch_size = batch_size or len(inputs)
        baseline = IntegratedGradients._get_baseline((*model.input.shape[1:],), baseline_value)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(
                batch_size):
            # create the paths for every sample (interpolated points from baseline to sample)
            interpolated_inputs = IntegratedGradients._get_interpolated_points(
                x_batch, steps, baseline)
            repeated_labels = repeat_labels(y_batch, steps)


            # compute the gradient for each paths
            interpolated_gradients = BaseExplanation._batch_gradient(model,
                                                                     interpolated_inputs,
                                                                     repeated_labels,
                                                                     batch_size)
            interpolated_gradients = tf.reshape(interpolated_gradients,
                                                (-1, steps, *interpolated_gradients.shape[1:]))

            # average the gradient using trapezoidal rule
            averaged_gradients = IntegratedGradients._average_gradients(interpolated_gradients)
            batch_integrated_gradients = (x_batch - baseline) * averaged_gradients

            integrated_gradients = batch_integrated_gradients if integrated_gradients is None else \
                tf.concat([integrated_gradients, batch_integrated_gradients], axis=0)

        return integrated_gradients

    @staticmethod
    def _get_baseline(shape, baseline_value):
        """
        Create the baseline point using a scalar value to fill the desired shape.

        Parameters
        ----------
        shape : tuple
            Desired shape, dimension of one sample.
        baseline_value : float
            Value defining the baseline state.

        Returns
        -------
        baseline_point : tf.tensor
            A baseline point of the specified shape.
        """
        return tf.ones(shape) * baseline_value

    @staticmethod
    @tf.function
    def _get_interpolated_points(inputs, steps, baseline):
        """
        Create a path from baseline to sample for every samples.

        Parameters
        ----------
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        steps : int
            Number of points to interpolate between the baseline and the desired point.
        baseline : tf.tensor (W, H, C)
            Baseline point, start of the path.

        Returns
        -------
        interpolated_inputs : tf.tensor (N * Steps, W, H, C)
            Interpolated path for each inputs.
        """
        alpha = tf.reshape(tf.linspace(0.0, 1.0, steps), (1, -1, 1, 1, 1))

        interpolated_inputs = tf.expand_dims(inputs, axis=1)
        interpolated_inputs = tf.repeat(interpolated_inputs, repeats=steps, axis=1)
        interpolated_inputs = baseline + alpha * (interpolated_inputs - baseline)

        interpolated_inputs = tf.reshape(interpolated_inputs, (-1, *interpolated_inputs.shape[2:]))

        return interpolated_inputs

    @staticmethod
    @tf.function
    def _average_gradients(gradients):
        """
        Average the gradients obtained along the path using trapezoidal rule.

        Parameters
        ----------
        gradients : tf.tensor (N, S, W, H, C)
            Gradients obtained for each of the S steps for each of the N samples.

        Returns
        -------
        integrated_gradients : tf.tensor (N, W, H, C)
        """
        trapezoidal_gradients = gradients[:, :-1] + gradients[:, 1:]
        averaged_gradients = tf.reduce_mean(trapezoidal_gradients, axis=1) * 0.5

        return averaged_gradients
