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


def _target_free_classification_operator(model: Callable,
                                         inputs: tf.Tensor,
                                         targets: Optional[tf.Tensor]) -> tf.Tensor:  # TODO: test, and use in attribution projection
    """
    Compute predictions scores, only for the label class, for a batch of samples.
    It has the same behavior as `Tasks.CLASSIFICATION` operator
    but computes targets at the same time if not provided.
    Targets are a mask with 1 on the predicted class and 0 elsewhere.
    This operator should only be used for classification tasks.


    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    scores
        Predictions scores computed, only for the label class.
    """
    predictions = model(inputs)

    targets = tf.cond(
        pred=tf.constant(targets is None, dtype=tf.bool),
        true_fn=lambda: tf.one_hot(tf.argmax(predictions, axis=-1), predictions.shape[-1]),
        false_fn=lambda: targets,
    )

    scores = tf.reduce_sum(predictions * targets, axis=-1)
    return scores


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
    operator  # TODO: make a larger description.
        Operator to use to compute the explanation, if None use standard predictions.
    device
        Device to use for the projection, if None, use the default device.
        Only used for PyTorch models. Ignored for TensorFlow models.
    """

    def __init__(
        self,
        model: Callable,
        latent_layer: Optional[Union[str, int]] = None,
        operator: Optional[OperatorSignature] = None,
        device: Union["torch.device", str] = None,
    ):
        if latent_layer is None:
            # no split
            self.latent_layer = None
            space_projection = None
            self.predictor = model
        else:
            # split the model if a latent_layer is provided
            space_projection, self.predictor = model_splitting(model,
                                                               latent_layer=latent_layer,
                                                               device=device)
        
        if operator is None:
            warnings.warn("No operator provided, using standard classification operator."\
                          + "For non-classification tasks, please specify an operator.")
            operator = _target_free_classification_operator
        
        # the weights are given by the gradient of the operator based on the predictor
        gradients, _ = get_gradient_functions(self.predictor, operator)
        get_weights = lambda inputs, targets: gradients(self.predictor, inputs, targets)  # TODO check usage of gpu

        mappable = isinstance(model, tf.keras.Model)

        # set methods
        super().__init__(get_weights, space_projection, mappable=mappable)

    @classmethod
    def from_splitted_model(cls,
                            features_extractor: tf.keras.Model,
                            predictor: tf.keras.Model,
                            operator: Optional[OperatorSignature] = None,
                            mappable=True):  # TODO: test
        """
        Create LatentSpaceProjection from a splitted model.
        The projection will project the inputs in the latent space,
        which corresponds to the output of the `features_extractor`.

        Parameters
        ----------
        features_extractor
            The feature extraction part of the model. Mapping inputs to the latent space.
        predictor
            The prediction part of the model. Mapping the latent space to the outputs.
        operator
            Operator to use to compute the explanation, if None use standard predictions.
        mappable
            If the model can be placed in a `tf.data.Dataset` mapping function.
            It is not the case for wrapped PyTorch models.
            If you encounter errors in the `project_dataset` method, you can set it to `False`.
        """
        assert isinstance(features_extractor, tf.keras.Model),\
            f"features_extractor should be a tf.keras.Model, got {type(features_extractor)}"\
            f" instead. If you have a PyTorch model, you can use the `TorchWrapper`."
        assert isinstance(predictor, tf.keras.Model),\
            f"predictor should be a tf.keras.Model, got {type(predictor)}"\
            f" instead. If you have a PyTorch model, you can use the `TorchWrapper`."
        
        # the weights are given by the gradient of the operator based on the predictor
        gradients, _ = get_gradient_functions(predictor, operator)
        get_weights = lambda inputs, targets: gradients(predictor, inputs, targets)  # TODO check usage of gpu

        super().__init__(get_weights=get_weights,
                         space_projection=features_extractor,
                         mappable=mappable)