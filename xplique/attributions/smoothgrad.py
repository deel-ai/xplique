"""
Module related to SmoothGrad method
"""

import numpy as np
import tensorflow as tf

from .base import BaseExplanation
from ..utils import sanitize_input_output, repeat_labels


class SmoothGrad(BaseExplanation):
    """
    Used to compute the SmoothGrad, by averaging Saliency maps of noisy samples centered on the
    original sample.

    Ref. Smilkov & al., SmoothGrad: removing noise by adding noise (2017).
    https://arxiv.org/abs/1706.03825

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
            Smoothed gradients, same shape as the inputs.
        """
        return SmoothGrad.compute(self.model,
                                  inputs,
                                  labels,
                                  self.batch_size,
                                  nb_samples=self.nb_samples,
                                  noise=self.noise)

    @classmethod
    def compute(cls, model, inputs, labels, batch_size, nb_samples, noise):
        """
        Compute SmoothGrad for a batch of samples.

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
        nb_samples : int
            Number of noisy samples generated for the smoothing procedure.
        noise : float, optional
            Scalar, noise used as standard deviation of a normal law centered on zero.

        Returns
        -------
        smoothed_gradients : tf.Tensor (N, W, H, C)
            Smoothed gradients, same shape as the inputs.
        """
        smoothed_gradients = None
        batch_size = batch_size or len(inputs)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(
                batch_size):
            noisy_mask = SmoothGrad.get_noisy_mask((nb_samples, *x_batch.shape[1:]), noise)
            noisy_inputs = SmoothGrad.apply_noise(x_batch, noisy_mask)
            repeated_labels = repeat_labels(y_batch, nb_samples)
            # compute the gradient of each noisy samples generated
            gradients = BaseExplanation._batch_gradient(model, noisy_inputs, repeated_labels,
                                                        batch_size)
            # group by inputs and compute the average gradient
            gradients = tf.reshape(gradients, (-1, nb_samples, *gradients.shape[1:]))
            reduced_gradients = cls.reduce_gradients(gradients)

            smoothed_gradients = reduced_gradients if smoothed_gradients is None else tf.concat(
                [smoothed_gradients, reduced_gradients], axis=0)

        return smoothed_gradients

    @staticmethod
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
    @tf.function
    def apply_noise(inputs, noisy_mask):
        """
        Duplicate the samples and apply a noisy mask to each of them.

        Parameters
        ----------
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        noisy_mask : ndarray (S, W, H, C)
            Mask of random noise to apply on a set of interpolations points. With S the number of
            samples, W & H the sample dimensions and C the number of channels.

        Returns
        -------
        noisy_inputs : tf.tensor (N * S, W, H, C)
            Duplicated inputs with noisy mask applied.
        """
        nb_samples = len(noisy_mask)

        noisy_inputs = tf.repeat(tf.expand_dims(inputs, axis=1), repeats=nb_samples, axis=1)
        noisy_inputs = noisy_inputs + noisy_mask
        noisy_inputs = tf.reshape(noisy_inputs, (-1, *noisy_inputs.shape[2:]))

        return noisy_inputs

    @staticmethod
    @tf.function
    def reduce_gradients(gradients):
        """
        Average the gradients obtained on each noisy samples.

        Parameters
        ----------
        gradients : tf.tensor (N, S, W, H, C)
            Gradients to reduce for each of the S samples of each of the N samples. SmoothGrad use
            an average of all the gradients.

        Returns
        -------
        reduced_gradients : tf.tensor (N, W, H, C)
            Single saliency map for each input.
        """
        return tf.reduce_mean(gradients, axis=1)
