"""
Module related to SmoothGrad method
"""
from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..base import WhiteBoxExplainer, sanitize_input_output
from ...commons import repeat_labels, batch_tensor, Tasks
from ...types import Union, Optional, OperatorSignature


class GradientStatistic(WhiteBoxExplainer, ABC):
    """
    Abstract class generalizing SmoothGrad, VarGrad, and SquareGrad.
    It makes small perturbations around a sample,
    compute the gradient for each perturbed sample,
    then return a statistics of this gradient.

    The inheriting methods only differ on the statistic used, either mean, square mean, or variance.

    Ref. Smilkov & al., SmoothGrad: removing noise by adding noise (2017).
    https://arxiv.org/abs/1706.03825

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    online_statistic_class
        Class of an `OnlineStatistic`, used to compute means or variances online
        when such computations require too much memory.
    output_layer
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        Default to the last layer.
        It is recommended to use the layer before Softmax.
    batch_size
        Number of inputs to explain at once, if None compute all at once.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    reducer
        String, name of the reducer to use. Either "min", "mean", "max", "sum", or `None` to ignore.
        Used only for images to obtain explanation with shape (n, h, w, 1).
    nb_samples
        Number of noisy samples generated for the smoothing procedure.
    noise
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 reducer: Optional[str] = "mean",
                 nb_samples: int = 50,
                 noise: float = 0.2):
        super().__init__(model, output_layer, batch_size, operator, reducer)
        self.nb_samples = nb_samples
        self.noise = noise
        self._initialize_online_statistic()

    @sanitize_input_output
    @WhiteBoxExplainer._harmonize_channel_dimension
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute SmoothGrad for a batch of samples.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, output_size).
            More information in the documentation.

        Returns
        -------
        explanations
            Estimation of the gradients, same shape as the inputs.
        """
        batch_size = self.batch_size or (len(inputs) * self.nb_samples)
        perturbation_batch_size = min(batch_size, self.nb_samples)
        inputs_batch_size = max(1, batch_size // perturbation_batch_size)

        smoothed_gradients = []
        # loop over inputs (by batch if batch_size > nb_samples, one by one otherwise)
        for x_batch, y_batch in batch_tensor((inputs, targets), inputs_batch_size):
            total_perturbed_samples = 0

            # reset online statistic values
            self._initialize_online_statistic()

            # loop over perturbation (a single pass if batch_size > nb_samples, batched otherwise)
            while total_perturbed_samples < self.nb_samples:
                nb_perturbations = min(perturbation_batch_size,
                                       self.nb_samples - total_perturbed_samples)
                total_perturbed_samples += nb_perturbations

                # add noise to inputs
                perturbed_x_batch = GradientStatistic._perturb_samples(
                    x_batch, nb_perturbations, self.noise)
                repeated_targets = repeat_labels(y_batch, nb_perturbations)

                # compute the gradient of each noisy samples generated
                gradients = self.batch_gradient(
                    self.model, perturbed_x_batch, repeated_targets, batch_size)

                # group by inputs and compute the average gradient
                gradients = tf.reshape(
                    gradients, (x_batch.shape[0], nb_perturbations, *gradients.shape[1:]))

                # update online estimation
                self._update_online_statistic(gradients)

            # extract online estimation
            reduced_gradients = self._get_online_statistic_final_value()
            smoothed_gradients.append(reduced_gradients)

        smoothed_gradients = tf.concat(smoothed_gradients, axis=0)
        return smoothed_gradients

    @staticmethod
    @tf.function
    def _perturb_samples(inputs: tf.Tensor,
                         nb_perturbations: int,
                         noise: float) -> tf.Tensor:
        """
        Duplicate the samples and apply a noisy mask to each of them.

        Parameters
        ----------
        inputs
            Input samples to be explained. (n, ...)
        nb_perturbations
            Number of perturbations to apply for each input.
        noise
            Scalar, noise used as standard deviation of a normal law centered on zero.

        Returns
        -------
        perturbed_inputs
            Duplicated inputs perturbed with random noise. (n * nb_perturbations, ...)
        """
        perturbed_inputs = tf.repeat(inputs, repeats=nb_perturbations, axis=0)
        perturbed_inputs += tf.random.normal(perturbed_inputs.shape, 0.0, noise, dtype=tf.float32)
        return perturbed_inputs

    @abstractmethod
    def _initialize_online_statistic(self):
        """
        Initialize values for the online statistic.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_online_statistic(self, elements):
        """
        Update the running statistic by taking new elements into account.
        It reduces elements to a statistic and only saves the statistic.

        Parameters
        ----------
        elements
            Batch of batch of elements.
            Part of all the elements the statistic should be computed on.
            Shape: (inputs_batch_size, perturbation_batch_size, ...)
        """
        raise NotImplementedError

    @abstractmethod
    def _get_online_statistic_final_value(self):
        """
        Return the final value of the statistic.

        Returns
        -------
        statistic
            The statistic computed online.
        """
        raise NotImplementedError
