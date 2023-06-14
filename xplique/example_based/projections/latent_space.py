"""
Base model for example-based 
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ...commons import find_layer
from ...types import Callable, Dict, Tuple, Union, Optional

from .base import Projection
        

class LatentSpaceProjection(Projection):
    """
    Projection that project inputs in the model latent space.
    It does not have weighting.
    
    Parameters
    ----------
    model
        The model from which we want to obtain explanations.
    latent_layer
        Layer used to split the `model`.
        
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.
        
        To separate after the last convolution, `"last_conv"` can be used.
        Otherwise, `-1` could be used for the last layer before softmax.
    """
    def __init__(self, 
                 model: Callable,
                 latent_layer:  Union[str, int] = -1):
        self.model = model

        # split the model if a latent_layer is provided
        if latent_layer == "last_conv":
            self.latent_layer = next(layer for layer in 
                                    model.layers[::-1] if hasattr(layer, 'filters'))
        else:
            self.latent_layer = find_layer(model, latent_layer)

        latent_space_projection = tf.keras.Model(model.input, self.latent_layer.output,
                                                name="features_extractor")
        
        super().__init__(space_projection=latent_space_projection)
