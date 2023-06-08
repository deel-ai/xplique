"""
Base model for example-based 
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ...attributions.base import BlackBoxExplainer, sanitize_input_output
from ...attributions import Saliency
from ...commons import find_layer
from ...types import Callable, Dict, Tuple, Union, Optional

from .base import Projection
        

class AttributionProjection(Projection):
    """
    ...
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
            self.space_projection = lambda inputs: inputs
            self.get_weights = method(model, **attribution_kwargs)
        else:
            # split the model if a latent_layer is provided
            if latent_layer == "last_conv":
                self.latent_layer = next(layer for layer in 
                                       model.layers[::-1] if hasattr(layer, 'filters'))
            else:
                self.latent_layer = find_layer(model, latent_layer)

            self.space_projection = tf.keras.Model(model.input, self.latent_layer.output,
                                                   name="features_extractor")
            self.predictor = tf.keras.Model(self.latent_layer.output, model.output,
                                       name="predictor")
            self.get_weights = method(self.predictor, **attribution_kwargs)
        
        # attribution methods output do not have channel
        # we wrap get_weights to expend dimensions if needed
        self.wrap_get_weights_to_extend_channels(self.get_weights)

    def wrap_get_weights_to_extend_channels(self, get_weights: Callable):
        """
        extend channel if miss match between inputs and weights
        """
        def wrapped_get_weights(inputs, targets):
            weights = get_weights(inputs, targets)
            weights = tf.cond(pred=weights.shape==inputs.shape,
                          true_fn=lambda: weights,
                          false_fn=lambda: tf.expand_dims(weights, axis=-1))
            return weights
        self.get_weights = wrapped_get_weights