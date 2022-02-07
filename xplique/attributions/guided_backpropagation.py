"""
Module related to Guided Backpropagation method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import guided_relu_policy, override_relu_gradient, batch_gradient
from ..types import Union, Optional


class GuidedBackprop(WhiteBoxExplainer):
    """
    Used to compute the Guided Backpropagation, which modifies the classic Saliency procedure on
    ReLU's non linearities, allowing only the positive gradients from positive activations to pass
    through.

    Ref. Tobias & al., Striving for Simplicity: The All Convolutional Net (2014).
    https://arxiv.org/abs/1412.6806

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

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = -1,
                 batch_size: Optional[int] = 32):
        super().__init__(model, output_layer, batch_size)
        self.model = override_relu_gradient(self.model, guided_relu_policy)

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute Guided Backpropagation for a batch of samples.

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
            Guided Backpropagation maps.
        """
        gradients = batch_gradient(self.model, inputs, targets, self.batch_size)
        return gradients
