"""
Attribution, a projection from example based module
"""
import warnings

import tensorflow as tf
from xplique.types import Optional

from ...commons import get_gradient_functions
from ...types import Union, Optional, OperatorSignature

from .base import Projection
from .commons import model_splitting, target_free_classification_operator


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
        It can be splitted manually outside of the projection and provided as two models:
        the `feature_extractor` and the `predictor`. In this case, `model` should be `None`.
        It is recommended to split it manually.
    latent_layer
        Layer used to split the model, the first part will be used for projection and
        the second to compute the attributions. By default, the model is not split.
        For such split, the `model` should be a `tf.keras.Model`.
        Ignored if `model` is `None`, hence if a splitted model is provided through:
        the `feature_extractor` and the `predictor`.

        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        The method as described in the paper apply the separation on the last convolutional layer.
        To do so, the `"last_conv"` parameter will extract it.
        Otherwise, `-1` could be used for the last layer before softmax.
    operator
        Operator to use to compute the explanation, if None use standard predictions.
        The default operator is the classification operator with online targets computations.
        For more information, refer to the Attribution documentation.
    device
        Device to use for the projection, if None, use the default device.
        Only used for PyTorch models. Ignored for TensorFlow models.
    features_extractor
        The feature extraction part of the model. Mapping inputs to the latent space.
        Used to provided the first part of a splitted model.
        It cannot be provided if a `model` is provided. It should be provided with a `predictor`.
    predictor
        The prediction part of the model. Mapping the latent space to the outputs.
        Used to provided the second part of a splitted model.
        It cannot be provided if a `model` is provided.
        It should be provided with a `features_extractor`.
    mappable
        If the model parts can be placed in a `tf.data.Dataset` mapping function.
        It is not the case for wrapped PyTorch models.
        If you encounter errors in the `project_dataset` method, you can set it to `False`.
        Used only for a splitted model. Thgus if `model` is `None`.
    """
    def __init__(
        self,
        model: Optional[Union[tf.keras.Model, 'torch.nn.Module']] = None,
        latent_layer: Optional[Union[str, int]] = None,
        operator: Optional[OperatorSignature] = None,
        device: Union["torch.device", str] = None,
        features_extractor: Optional[tf.keras.Model] = None,
        predictor: Optional[tf.keras.Model] = None,
        mappable: bool = True,
    ):
        if model is None:
            assert features_extractor is not None and predictor is not None,\
                "If no model is provided, the features_extractor and predictor should be provided."

            assert isinstance(features_extractor, tf.keras.Model)\
                and isinstance(predictor, tf.keras.Model),\
                "The features_extractor and predictor should be tf.keras.Model."\
                + "The xplique.wrappers.TorchWrapper can be used for PyTorch models."
        else:
            assert features_extractor is None and predictor is None,\
                "If a model is provided, the features_extractor and predictor cannot be provided."

            if latent_layer is None:
                # no split
                self.latent_layer = None
                features_extractor = None
                predictor = model
            else:
                # split the model if a latent_layer is provided
                features_extractor, predictor = model_splitting(model,
                                                                latent_layer=latent_layer,
                                                                device=device)

            mappable = isinstance(model, tf.keras.Model)

        if operator is None:
            warnings.warn("No operator provided, using standard classification operator. "\
                          + "For non-classification tasks, please specify an operator.")
            operator = target_free_classification_operator

        # the weights are given by the gradient of the operator based on the predictor
        gradients, _ = get_gradient_functions(predictor, operator)
        get_weights = lambda inputs, targets: gradients(predictor, inputs, targets)

        # set methods
        super().__init__(
            get_weights=get_weights,
            space_projection=features_extractor,
            mappable=mappable,
            requires_targets=True
        )
