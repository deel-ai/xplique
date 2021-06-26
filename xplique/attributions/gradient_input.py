"""
Module related to Gradient x Input method
"""

import tensorflow as tf

from .base import WhiteBoxExplainer
from ..utils import sanitize_input_output, batch_gradient


class GradientInput(WhiteBoxExplainer):
    """
    Used to compute elementwise product between the saliency maps of Simonyan et al. and the
    input (Gradient x Input).

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
    """

    @sanitize_input_output
    def explain(self,
                inputs: tf.Tensor,
                labels: tf.Tensor) -> tf.Tensor:
        """
        Compute gradients x inputs for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        labels
            One-hot encoded labels, one for each sample.

        Returns
        -------
        explanations
            Gradients x Inputs, with the same shape as the inputs.
        """
        gradients = batch_gradient(self.model, inputs, labels, self.batch_size)
        gradients_inputs = tf.multiply(gradients, inputs)

        return gradients_inputs
