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
        Compute saliency maps for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

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
