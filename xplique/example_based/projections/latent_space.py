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
    device
        Device to use for the projection, if None, use the default device.
        Only used for PyTorch models. Ignored for TensorFlow models.
    """

    def __init__(self,
                 model: Union[tf.keras.Model, 'torch.nn.Module'],
                 latent_layer: Union[str, int] = -1,
                 device: Union["torch.device", str] = None,
                 ):
        features_extractor, _ = model_splitting(model, latent_layer=latent_layer, device=device)

        mappable = isinstance(model, tf.keras.Model)
        super().__init__(space_projection=features_extractor, mappable=mappable)
