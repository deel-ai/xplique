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

    @classmethod
    def from_splitted_model(cls,
                            features_extractor: tf.keras.Model,
                            mappable=True):
        """
        Create LatentSpaceProjection from a splitted model.
        The projection will project the inputs in the latent space,
        which corresponds to the output of the `features_extractor`.

        Parameters
        ----------
        features_extractor
            The feature extraction part of the model. Mapping inputs to the latent space.
        mappable
            If the model can be placed in a `tf.data.Dataset` mapping function.
            It is not the case for wrapped PyTorch models.
            If you encounter errors in the `project_dataset` method, you can set it to `False`.
        """
        # pylint: disable=fixme
        # TODO: test
        assert isinstance(features_extractor, tf.keras.Model),\
            f"features_extractor should be a tf.keras.Model, got {type(features_extractor)}"\
            f" instead. If you have a PyTorch model, you can use the `TorchWrapper`."
        super().__init__(space_projection=features_extractor, mappable=mappable)
