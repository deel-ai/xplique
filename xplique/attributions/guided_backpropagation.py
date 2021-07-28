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
        Model used for computing explanations.
    output_layer
        Layer to target for the output (e.g logits or after softmax), if int, will be be interpreted
        as layer index, if string will look for the layer name. Default to the last layer, it is
        recommended to use the layer before Softmax.
    batch_size
        Number of samples to explain at once, if None compute all at once.
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
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            Guided Backpropagation maps.
        """
        gradients = batch_gradient(self.model, inputs, targets, self.batch_size)
        return gradients
