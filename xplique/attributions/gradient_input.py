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
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute gradients x inputs for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            Gradients x Inputs, with the same shape as the inputs.
        """
        gradients = batch_gradient(self.model, inputs, targets, self.batch_size)
        gradients_inputs = tf.multiply(gradients, inputs)

        return gradients_inputs
