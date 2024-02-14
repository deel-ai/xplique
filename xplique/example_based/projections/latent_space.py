"""
Custom, a projection from example based module
"""

import tensorflow as tf

from ...commons import find_layer
from ...types import Callable, Union

from .base import Projection
from .commons import model_splitting


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

    def __init__(self, model: Callable, latent_layer: Union[str, int] = -1):
        features_extractor, _ = model_splitting(model, latent_layer)
        super().__init__(space_projection=features_extractor)
        # TODO test if gpu is used for the projection
