"""
Attribution, a projection from example based module
"""

import warnings

import tensorflow as tf

from xplique.types import Optional

from ...attributions import Saliency
from ...attributions.base import BlackBoxExplainer
from ...types import Optional, Union
from .base import Projection
from .commons import model_splitting, target_free_classification_operator


class AttributionProjection(Projection):
    """
    Projection build on an attribution function to provide local projections.
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

        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        The method as described in the paper apply the separation on the last convolutional layer.
        To do so, the `"last_conv"` parameter will extract it.
        Otherwise, `-1` could be used for the last layer before softmax.
    attribution_method
        Class of the attribution method to use for projection.
        It should inherit from `xplique.attributions.base.BlackBoxExplainer`.
        Ignored if a projection is given.
    attribution_kwargs
        Parameters to be passed at the construction of the `attribution_method`.
    """

    def __init__(
        self,
        model: Union[tf.keras.Model, "torch.nn.Module"],
        attribution_method: BlackBoxExplainer = Saliency,
        latent_layer: Optional[Union[str, int]] = None,
        **attribution_kwargs,
    ):
        self.attribution_method = attribution_method

        if latent_layer is None:
            # no split
            self.latent_layer = None
            space_projection = None
            self.predictor = model
        else:
            # split the model if a latent_layer is provided
            space_projection, self.predictor = model_splitting(model, latent_layer)

        # change default operator
        if "operator" not in attribution_kwargs or attribution_kwargs["operator"] is None:
            warnings.warn(
                "No operator provided, using standard classification operator. "
                + "For non-classification tasks, please specify an operator."
            )
            attribution_kwargs["operator"] = target_free_classification_operator

        # compute attributions
        get_weights = self.attribution_method(self.predictor, **attribution_kwargs)

        # set methods
        super().__init__(
            get_weights=get_weights,
            space_projection=space_projection,
            mappable=False,
            requires_targets=True,
        )
