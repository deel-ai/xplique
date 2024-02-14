"""
Attribution, a projection from example based module
"""
import warnings

import tensorflow as tf
import numpy as np
from xplique.types import Optional

from ...attributions.base import BlackBoxExplainer
from ...attributions import Saliency
from ...types import Callable, Union, Optional

from .base import Projection
from .commons import model_splitting


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

        Layer to target for the outputs (e.g logits or after softmax).
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
        model: Callable,
        method: BlackBoxExplainer = Saliency,
        latent_layer: Optional[Union[str, int]] = None,
        **attribution_kwargs
    ):
        self.method = method

        if latent_layer is None:
            # no split
            self.latent_layer = None
            space_projection = None
            self.predictor = model
        else:
            # split the model if a latent_layer is provided
            space_projection, self.predictor = model_splitting(model, latent_layer)
        
        # compute attributions
        get_weights = self.method(self.predictor, **attribution_kwargs)

        # set methods
        super().__init__(get_weights, space_projection)

    def get_input_weights(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """
        For visualization purpose (and only), we may be interested to project weights
        from the projected space to the input space.
        This is applied only if their is a difference in dimension.
        We assume here that we are treating images and an upsampling is applied.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
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
        weights = tf.cond(
            pred=tf.shape(weights).shape[0] < 4,
            true_fn=lambda: weights,
            false_fn=channel_mean_fn,
        )

        # resizing
        resize_fn = lambda: tf.image.resize(
            weights, inputs.shape[1:-1], method="bicubic"
        )
        input_weights = tf.cond(
            pred=projected_inputs.shape == inputs.shape,
            true_fn=lambda: weights,
            false_fn=resize_fn,
        )
        return input_weights

    def project_dataset(
        self,
        cases_dataset: tf.data.Dataset,
        targets_dataset: tf.data.Dataset,
    ) -> tf.data.Dataset:
        """
        Apply the projection to a dataset without `Dataset.map`.
        Because attribution methods create a `tf.data.Dataset` for batching,
        however doing so inside a `Dataset.map` is not recommended.

        Parameters
        ----------
        cases_dataset
            Dataset of samples to be projected.
        targets_dataset
            Dataset of targets for the samples.

        Returns
        -------
        projected_dataset
            The projected dataset.
        """
        # TODO see if a warning is needed

        projected_cases_dataset = []
        batch_size = None

        # iteratively project the dataset
        for inputs, targets in tf.data.Dataset.zip((cases_dataset, targets_dataset)):
            if batch_size is None:
                batch_size = inputs.shape[0]  # TODO check if there is a smarter way to do this
            projected_cases_dataset.append(self.project(inputs, targets))
        
        projected_cases_dataset = tf.concat(projected_cases_dataset, axis=0)
        projected_cases_dataset = tf.data.Dataset.from_tensor_slices(projected_cases_dataset)
        projected_cases_dataset = projected_cases_dataset.batch(batch_size)
        
        return projected_cases_dataset