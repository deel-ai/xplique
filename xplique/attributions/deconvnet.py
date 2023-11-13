"""
Module related to DeconvNet method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import deconv_relu_policy, override_relu_gradient, Tasks
from ..types import Union, Optional, OperatorSignature

class DeconvNet(WhiteBoxExplainer):  # pylint: disable=duplicate-code
    """
    Used to compute the DeconvNet method, which modifies the classic Saliency procedure on
    ReLU's non linearities, allowing only the positive gradients (even from negative inputs) to
    pass through.

    Ref. Zeiler & al., Visualizing and Understanding Convolutional Networks (2013).
    https://arxiv.org/abs/1311.2901

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
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    reducer
        String, name of the reducer to use. Either "min", "mean", "max", "sum", or `None` to ignore.
        Used only for images to obtain explanation with shape (n, h, w, 1).
    """

    def __init__(self,
                model: tf.keras.Model,
                output_layer: Optional[Union[str, int]] = None,
                batch_size: Optional[int] = 32,
                operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                reducer: Optional[str] = "mean",):
        super().__init__(model, output_layer, batch_size, operator, reducer)
        self.model = override_relu_gradient(self.model, deconv_relu_policy)

    @sanitize_input_output
    @WhiteBoxExplainer._harmonize_channel_dimension
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute DeconvNet for a batch of samples.
        Accept Tensor, numpy array or tf.data.Dataset (in that case targets is None)

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, output_size).
            More information in the documentation.

        Returns
        -------
        explanations
            Deconv maps.
        """
        gradients = self.batch_gradient(self.model, inputs, targets, self.batch_size)
        return gradients
