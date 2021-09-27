"""
Module related to SmoothGrad method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_gradient, batch_tensor
from ..types import Tuple, Union, Optional


class SmoothGrad(WhiteBoxExplainer):
    """
    Used to compute the SmoothGrad, by averaging Saliency maps of noisy samples centered on the
    original sample.

    Ref. Smilkov & al., SmoothGrad: removing noise by adding noise (2017).
    https://arxiv.org/abs/1706.03825

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
    nb_samples
        Number of noisy samples generated for the smoothing procedure.
    noise
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = -1,
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
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            Smoothed gradients, same shape as the inputs.
        """
        smoothed_gradients = None
        batch_size = self.batch_size or len(inputs)

        noisy_mask = SmoothGrad._get_noisy_mask((self.nb_samples, *inputs.shape[1:]), self.noise)

        for x_batch, y_batch in batch_tensor((inputs, targets),
                                             max(batch_size // self.nb_samples, 1)):
            noisy_inputs = SmoothGrad._apply_noise(x_batch, noisy_mask)
            repeated_targets = repeat_labels(y_batch, self.nb_samples)
            # compute the gradient of each noisy samples generated
            gradients = batch_gradient(self.model, noisy_inputs, repeated_targets, batch_size)
            # group by inputs and compute the average gradient
            gradients = tf.reshape(gradients, (-1, self.nb_samples, *gradients.shape[1:]))
            reduced_gradients = self._reduce_gradients(gradients)

            smoothed_gradients = reduced_gradients if smoothed_gradients is None else tf.concat(
                [smoothed_gradients, reduced_gradients], axis=0)

        return smoothed_gradients

    @staticmethod
    def _get_noisy_mask(shape: Tuple[int, int, int, int],
                        noise: float) -> tf.Tensor:
        """
        Create a random noise mask of the specified shape.

        Parameters
        ----------
        shape
            Desired shape, dimension of one sample.
        noise
            Scalar, noise used as standard deviation of a normal law centered on zero.

        Returns
        -------
        noisy_mask
            Noise mask of the specified shape.
        """
        return tf.random.normal(shape, 0.0, noise, dtype=tf.float32)

    @staticmethod
    @tf.function
    def _apply_noise(inputs: tf.Tensor,
                     noisy_mask: tf.Tensor) -> tf.Tensor:
        """
        Duplicate the samples and apply a noisy mask to each of them.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        noisy_mask
            Mask of random noise to apply on a set of interpolations points. With S the number of
            samples, W & H the sample dimensions and C the number of channels.

        Returns
        -------
        noisy_inputs
            Duplicated inputs with noisy mask applied.
        """
        nb_samples = len(noisy_mask)

        noisy_inputs = tf.repeat(tf.expand_dims(inputs, axis=1), repeats=nb_samples, axis=1)
        noisy_inputs = noisy_inputs + noisy_mask
        noisy_inputs = tf.reshape(noisy_inputs, (-1, *noisy_inputs.shape[2:]))

        return noisy_inputs

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
        return tf.reduce_mean(gradients, axis=1)
