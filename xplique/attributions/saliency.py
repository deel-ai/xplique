"""
Module related to Saliency maps method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import batch_gradient
from ..types import Optional, Union


class Saliency(WhiteBoxExplainer):
    """
    Used to compute the absolute gradient of the output relative to the input.

    Ref. Simonyan & al., Deep Inside Convolutional Networks: Visualising Image Classification
    Models and Saliency Maps (2013).
    https://arxiv.org/abs/1312.6034

    Notes
    -----
    As specified in the original paper, the Saliency map method should return the magnitude of the
    gradient (absolute value), and the maximum magnitude over the channels in case of RGB images.
    However it is not uncommon to find definitions that don't apply the L1 norm, in this case one
    can simply calculate the gradient relative to the input using the BaseExplanation method.

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
        Compute saliency maps for a batch of samples.

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
            Saliency maps.
        """
        gradients = batch_gradient(self.model, inputs, targets, self.batch_size)
        gradients = tf.abs(gradients)

        # if the image is a RGB, take the maximum magnitude across the channels (see Ref.)
        if len(gradients.shape) == 4:
            gradients = tf.reduce_max(gradients, axis=-1)

        return gradients
