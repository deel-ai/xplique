"""
Module related to SmoothGrad method
"""

from functools import lru_cache

import numpy as np
import tensorflow as tf

from .base import BaseExplanation
from .utils import sanitize_input_output


class SmoothGrad(BaseExplanation):
    """
    Used to compute the SmoothGrad, by averaging Saliency maps of noisy samples centered on the
    original sample.

    Ref. SmoothGrad: removing noise by adding noise (2017).
    https://arxiv.org/abs/1706.03825

    Notes
    -----
    in order to speed up SmoothGrad computation as much as possible, a "noise mask" is generated
    once and then reused for each samples.

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    output_layer_index : int, optional
        Index of the output layer, default to the last layer, it is recommended to use the layer
        before Softmax (often '-2').
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    nb_samples : int, optional
        Number of noisy samples generated for the smoothing procedure.
    noise : float, optional
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    def __init__(self, model, output_layer_index=-1, batch_size=32, nb_samples=50, noise=0.5):
        super().__init__(model, output_layer_index, batch_size)
        self.nb_samples = nb_samples
        self.noise = noise

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute SmoothGrad for a batch of samples.

        Parameters
        ----------
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray(N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        explanations : ndarray (N, W, H)
            Explanations computed.
        """
        return SmoothGrad.compute(self.model,
                                  inputs,
                                  labels,
                                  self.batch_size,
                                  nb_samples=self.nb_samples,
                                  noise=self.noise)

    @staticmethod
    @lru_cache()
    def get_noisy_mask(shape, noise):
        """
        Create a random noise mask of the specified shape.

        Parameters
        ----------
        shape : tuple
            Desired shape, dimension of one sample.
        noise : float
            Scalar, noise used as standard deviation of a normal law centered on zero.

        Returns
        -------
        noisy_mask : ndarray
            Noise mask of the specified shape.
        """
        return np.random.normal(0, noise, shape).astype(np.float32)

    @staticmethod
    def compute(model, inputs, labels, batch_size, nb_samples, noise):
        """
        Compute SmoothGrad for a batch of samples.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing explanations.
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray(N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.
        batch_size : int
            Number of samples to explain at once, if None compute all at once.
        nb_samples : int
            Number of noisy samples generated for the smoothing procedure.
        noise : float, optional
            Scalar, noise used as standard deviation of a normal law centered on zero.

        Returns
        -------
        smoothed_gradients : tf.Tensor (N, W, H, C)
            Explanation computed, with the same shape as the inputs.
        """
        smoothed_gradients = None

        # re-evaluate batch_size to take into account the synthetic inputs that we create
        synthetic_batch_size = max(1, batch_size // nb_samples) if batch_size is not None else len(
            inputs)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(
                synthetic_batch_size):
            noisy_mask = SmoothGrad.get_noisy_mask((nb_samples, *x_batch.shape[1:]), noise)
            noisy_inputs, noisy_labels = SmoothGrad.apply_noise(x_batch, y_batch, nb_samples,
                                                                noisy_mask)
            # compute the gradient of each noisy samples generated
            gradients = BaseExplanation._gradient(model, noisy_inputs, noisy_labels)
            # group by inputs and compute the average gradient
            gradients = tf.reshape(gradients, (-1, nb_samples, *gradients.shape[1:]))
            batch_gradients = tf.reduce_mean(gradients, axis=1)

            smoothed_gradients = batch_gradients if smoothed_gradients is None else tf.concat(
                [smoothed_gradients, batch_gradients], axis=0)

        return smoothed_gradients

    @staticmethod
    @tf.function
    def apply_noise(inputs, labels, nb_samples, noisy_mask):
        """
        Duplicate the samples and apply a noisy mask to each of them.

        Parameters
        ----------
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray(N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.
        nb_samples : int
            Number of replications for each samples.
        noisy_mask : ndarray (S, W, H, C)
            Mask of random noise to apply on a set of interpolations points. With S the number of*
            samples, W & H the sample dimensions and C the number of channels.


        Returns
        -------
        noisy_inputs : ndarray (N * S, W, H, C)
            Duplicated inputs with noisy mask applied.
        labels : ndarray (N * S, L)
            Duplicated labels.
        """
        noisy_inputs = tf.repeat(tf.expand_dims(inputs, axis=1), repeats=nb_samples, axis=1)
        noisy_inputs = noisy_inputs + noisy_mask
        noisy_inputs = tf.reshape(noisy_inputs, (-1, *noisy_inputs.shape[2:]))

        labels = tf.repeat(tf.expand_dims(labels, axis=1), repeats=nb_samples, axis=1)
        labels = tf.reshape(labels, (-1, *labels.shape[2:]))

        return noisy_inputs, labels
