"""
Module related to Gradient x Input method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import batch_gradient
from ..types import Optional, Union


class GradientInput(WhiteBoxExplainer):
    """
    Used to compute elementwise product between the saliency maps of Simonyan et al. and the
    input (Gradient x Input).

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
    """

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute gradients x inputs for a batch of samples.

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
            Gradients x Inputs, with the same shape as the inputs.
        """
        gradients = batch_gradient(self.model, inputs, targets, self.batch_size)
        gradients_inputs = tf.multiply(gradients, inputs)

        return gradients_inputs
