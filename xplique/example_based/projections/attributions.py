"""
Base model for example-based 
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ...attributions.base import BlackBoxExplainer
from ...attributions import Saliency
from ...commons import find_layer
from ...types import Callable, Dict, Tuple, Union, Optional

from .base import Projection
        

class AttributionProjection(Projection):
    """
    Projection build on an attribution function to provide local projections.
    This class is used as the projection of the `Cole` similar examples method.

    Depending on the `latent_layer`, the model will be splited between
    the feature extractor and the predictor.
    The feature extractor will become the `space_projection()` method, then
    the predictor will be used to build the attribution method explain, and
    its `explain()` method will become the `get_weights()` method.

    If no `latent_layer` is provided, the model is not splited,
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
        
        The method as described in the paper apply the separation on the last convolutionnal layer.
        To do so, the `"last_conv"` parameter will extract it.
        Otherwise, `-1` could be used for the last layer before softmax.
    attribution_method
        Class of the attribution method to use for projection.
        It should inherit from `xplique.attributions.base.BlackBoxExplainer`.
        Ignored if a projection is given.
    attribution_kwargs
        Parameters to be passed at the construction of the `attribution_method`.
    """
    def __init__(self, 
                 model: Callable,
                 method: BlackBoxExplainer = Saliency,
                 latent_layer:  Optional[Union[str, int]] = None,
                 **attribution_kwargs):
        self.model = model

        if latent_layer is None:
            # no split
            self.latent_layer = None
            space_projection = lambda inputs: inputs
            get_weights = method(model, **attribution_kwargs)
        else:
            # split the model if a latent_layer is provided
            if latent_layer == "last_conv":
                self.latent_layer = next(layer for layer in 
                                       model.layers[::-1] if hasattr(layer, 'filters'))
            else:
                self.latent_layer = find_layer(model, latent_layer)

            space_projection = tf.keras.Model(model.input, self.latent_layer.output,
                                                   name="features_extractor")
            self.predictor = tf.keras.Model(self.latent_layer.output, model.output,
                                            name="predictor")
            get_weights = method(self.predictor, **attribution_kwargs)
        
        # set methods
        super().__init__(get_weights, space_projection)

        # attribution methods output do not have channel
        # we wrap get_weights to expend dimensions if needed
        self.__wrap_get_weights_to_extend_channels(self.get_weights)

    def __wrap_get_weights_to_extend_channels(self, get_weights: Callable):
        """
        Extend channel if miss match between inputs and weights
        """
        def wrapped_get_weights(inputs, targets):
            weights = get_weights(inputs, targets)
            weights = tf.cond(pred=weights.shape==inputs.shape,
                          true_fn=lambda: weights,
                          false_fn=lambda: tf.expand_dims(weights, axis=-1))
            return weights
        self.get_weights = wrapped_get_weights
        
    def get_input_weights(self,
                          inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                          targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
        """
        For visualization purpose (and only), we may be interested to project weights
        from the projected space to the input space.
        This is applied only if their is a difference in dimension.
        We assume here that we are treating images and an upsampling is applied.
        
        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
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
        weights = tf.cond(pred=tf.shape(weights).shape[0] < 4,
                          true_fn=lambda: weights,
                          false_fn=channel_mean_fn)

        # resizing
        resize_fn = lambda: tf.image.resize(weights, inputs.shape[1:-1], method="bicubic")
        input_weights = tf.cond(pred=projected_inputs.shape==inputs.shape,
                                true_fn=lambda: weights,
                                false_fn=resize_fn,)
        return input_weights