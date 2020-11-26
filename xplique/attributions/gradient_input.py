"""
Module related to Gradient x Input method
"""

import tensorflow as tf

from .base import BaseExplanation
from ..utils import sanitize_input_output


class GradientInput(BaseExplanation):
    """
    Used to compute elementwise product between the saliency maps of Simonyan et al. and the
    input (Gradient x Input).

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    output_layer_index : int, optional
        Index of the output layer, default to the last layer, it is recommended to use the layer
        before Softmax (often '-2').
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    """

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute gradients x inputs for a batch of samples.

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
        explanations : ndarray (N, W, H)
            Gradients x Inputs, with the same shape as the inputs.
        """
        return GradientInput.compute(self.model, inputs, labels, self.batch_size)

    @staticmethod
    def compute(model, inputs, labels, batch_size):
        """
        Compute gradients x inputs for a batch of samples.

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

        Returns
        -------
        gradients_inputs : tf.Tensor (N, W, H, C)
            Gradients x Inputs, with the same shape as the inputs.
        """
        gradients = BaseExplanation._batch_gradient(model, inputs, labels, batch_size)
        gradients_inputs = tf.multiply(gradients, inputs)

        return gradients_inputs
