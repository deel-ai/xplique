"""
Custom, a projection from example based module
"""

import tensorflow as tf

from ...types import Union

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
        It will be splitted if a `latent_layer` is provided.
        Otherwise, it should be a `tf.keras.Model`.
        It is recommended to split it manually and provide the first part of the model directly.
    latent_layer
        Layer used to split the `model`.

        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        To separate after the last convolution, `"last_conv"` can be used.
        Otherwise, `-1` could be used for the last layer before softmax.
    device
        Device to use for the projection, if None, use the default device.
        Only used for PyTorch models. Ignored for TensorFlow models.
    mappable
        Used only if not `latent_layer` is provided. Thus if the model is already splitted.
        If the model can be placed in a `tf.data.Dataset` mapping function.
        It is not the case for wrapped PyTorch models.
        If you encounter errors in the `project_dataset` method, you can set it to `False`.
    """

    def __init__(self,
                 model: Union[tf.keras.Model, 'torch.nn.Module'],
                 latent_layer: Union[str, int] = -1,
                 device: Union["torch.device", str] = None,
                 mappable: bool = True,
                 ):
        if latent_layer is None:
            assert isinstance(model, tf.keras.Model),\
                "If no latent_layer is provided, the model should be a tf.keras.Model."
            features_extractor = model
        else:
            features_extractor, _ = model_splitting(model, latent_layer=latent_layer, device=device)
            mappable = isinstance(model, tf.keras.Model)

        super().__init__(
            space_projection=features_extractor,
            mappable=mappable,
            requires_targets=False
        )
