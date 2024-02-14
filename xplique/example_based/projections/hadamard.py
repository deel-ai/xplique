"""
Attribution, a projection from example based module
"""
import warnings

import tensorflow as tf
import numpy as np
from xplique.types import Optional

from ...commons import get_gradient_functions
from ...types import Callable, Union, Optional, OperatorSignature

from .base import Projection
from .commons import model_splitting


class HadamardProjection(Projection):
    """
    Projection build on an the latent space and the gradient.
    This class is used as the projection of the `Cole` similar examples method.

    Depending on the `latent_layer`, the model will be splitted between
    the feature extractor and the predictor.
    The feature extractor will become the `space_projection()` method, then
    the predictor will be used to build the attribution method explain, and
    its `explain()` method will become the `get_weights()` method.

    If no `latent_layer` is provided, the model is not splitted,
    the `space_projection()` is the identity function, and
    the attributions (`get_weights()`) are compute on the whole model.

    Parameters
    ----------
    model
        The model from which we want to obtain explanations.
    latent_layer
        Layer used to split the model, the first part will be used for projection and
        the second to compute the attributions. By default, the model is not split.
        For such split, the `model` should be a `tf.keras.Model`.

        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        The method as described in the paper apply the separation on the last convolutional layer.
        To do so, the `"last_conv"` parameter will extract it.
        Otherwise, `-1` could be used for the last layer before softmax.
    operator
        Operator to use to compute the explanation, if None use standard predictions.
    """

    def __init__(
        self,
        model: Callable,
        latent_layer: Optional[Union[str, int]] = None,
        operator: Optional[OperatorSignature] = None,
    ):
        if latent_layer is None:
            # no split
            self.latent_layer = None
            space_projection = None
            self.predictor = model
        else:
            # split the model if a latent_layer is provided
            space_projection, self.predictor = model_splitting(model, latent_layer)
        
        # the weights are given be the gradient of the operator
        gradients, _ = get_gradient_functions(self.predictor, operator)
        get_weights = lambda inputs, targets: gradients(self.predictor, inputs, targets)  # TODO check usage of gpu

        # set methods
        super().__init__(get_weights, space_projection)

    def get_input_weights(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """
        For visualization purpose (and only), we may be interested to project weights
        from the projected space to the input space.
        This is applied only if their is a difference in dimension.
        We assume here that we are treating images and an upsampling is applied.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Additional parameter for `self.get_weights` function.

        Returns
        -------
        input_weights
            Tensor with the same dimension as `inputs` modulo the channels.
            They are an upsampled version of the actual weights used in the projection.
        """
        projected_inputs = self.space_projection(inputs)
        weights = self.get_weights(projected_inputs, targets)

        # take mean over channels for images
        channel_mean_fn = lambda: tf.reduce_mean(weights, axis=-1, keepdims=True)
        weights = tf.cond(
            pred=tf.shape(weights).shape[0] < 4,
            true_fn=lambda: weights,
            false_fn=channel_mean_fn,
        )

        # resizing
        resize_fn = lambda: tf.image.resize(
            weights, inputs.shape[1:-1], method="bicubic"
        )
        input_weights = tf.cond(
            pred=projected_inputs.shape == inputs.shape,
            true_fn=lambda: weights,
            false_fn=resize_fn,
        )
        return input_weights
