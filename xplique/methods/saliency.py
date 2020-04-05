"""
Module related to Saliency maps method
"""

import tensorflow as tf

from .base import BaseExplanation
from .utils import sanitize_input_output


class Saliency(BaseExplanation):
    """
    Used to compute the absolute gradient of the output relative to the input.

    Ref. Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency
    Maps (2013).
    https://arxiv.org/abs/1312.6034

    Notes
    -----
    As specified in the original paper, the Saliency map method should return the magnitude of the
    gradient (absolute value), and the maximum magnitude over the channels in case of RGB images.
    However it is not uncommon to find definitions that don't apply the L1 norm, in this case one
    can simply calculate the gradient relative to the input using the BaseExplanation method.

    """

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute saliency maps for a batch of samples.

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
        return Saliency.compute(self.model, inputs, labels, self.batch_size)

    @staticmethod
    def compute(model, inputs, labels, batch_size):
        """
        Compute saliency maps for a batch of samples.

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

        Returns
        -------
        saliency_map : tf.Tensor (N, W, H, C)
            Explanation computed, with the same shape as the inputs.
        """
        gradients = BaseExplanation._batch_gradient(model, inputs, labels, batch_size)
        gradients = tf.abs(gradients)

        # if the image is a RGB, take the maximum magnitude across the channels
        if len(gradients.shape) == 4:
            gradients = tf.reduce_max(gradients, axis=-1)

        return gradients
