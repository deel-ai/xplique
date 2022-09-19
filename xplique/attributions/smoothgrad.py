"""
Module related to SmoothGrad method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, gradient
from ..types import Union, Optional


class SmoothGrad(WhiteBoxExplainer):
    """
    Used to compute the SmoothGrad, by averaging Saliency maps of noisy samples centered on the
    original sample.

    Ref. Smilkov & al., SmoothGrad: removing noise by adding noise (2017).
    https://arxiv.org/abs/1706.03825

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    output_layer
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        Default to the last layer.
        It is recommended to use the layer before Softmax.
    batch_size
        Number of inputs to explain at once, if None compute all at once.
    nb_samples
        Number of noisy samples generated for the smoothing procedure.
    noise
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 nb_samples: int = 50,
                 noise: float = 0.2):
        super().__init__(model, output_layer, batch_size)
        self.nb_samples = nb_samples
        self.noise = noise

    @sanitize_input_output
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
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, output_size).
            More information in the documentation.

        Returns
        -------
        explanations
            Smoothed gradients, same shape as the inputs.
        """
        smoothed_gradients = None
        batch_size = self.batch_size or self.nb_samples

        # since the number of masks is often very large, we process the entries one by one
        for single_input, single_target in zip(inputs, targets):
            accumulated_reduced_gradients = 0

            remaining_samples = self.nb_samples
            while remaining_samples > 0:
                tf.print("remaining_samples", remaining_samples)
                nb_noises = min(batch_size, remaining_samples)
                remaining_samples -= nb_noises
                # generate random noise
                batch_noises = tf.random.normal(
                    (nb_noises, *inputs.shape[1:]), 0.0, self.noise, dtype=tf.float32)

                # apply noise
                batch_noisy_inputs = single_input + batch_noises
                repeated_targets = repeat_labels(single_target[tf.newaxis, :], nb_noises)

                # compute the gradient of each noisy samples generated
                batch_gradients = gradient(self.model, batch_noisy_inputs, repeated_targets)

                # mean over a batch of gradients for one input
                reduced_gradients = self._reduce_gradients(batch_gradients)

                # accumulate weighted reduced gradient for weighted mean
                accumulated_reduced_gradients += nb_noises * reduced_gradients

            # weighted mean of the mean of batch of gradients for one input
            reduced_reduced_gradients = accumulated_reduced_gradients / self.nb_samples

            if smoothed_gradients is None:
                smoothed_gradients = reduced_reduced_gradients[tf.newaxis, :]
            else:
                smoothed_gradients = tf.concat(
                    [smoothed_gradients, reduced_reduced_gradients[tf.newaxis, :]], axis=0)

        return smoothed_gradients


    @staticmethod
    @tf.function
    def _reduce_gradients(gradients: tf.Tensor) -> tf.Tensor:
        """
        Average the gradients obtained on each noisy samples.

        Parameters
        ----------
        gradients
            Gradients to reduce the sampling dimension for each inputs.

        Returns
        -------
        reduced_gradients
            Single saliency map for each input.
        """
        return tf.reduce_mean(gradients, axis=0)
