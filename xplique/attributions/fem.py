"""
Module related to FEM method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer
from ..commons import find_layer, tensor_sanitize, Tasks
from ..types import Tuple, Union, Optional, OperatorSignature


class FEM(WhiteBoxExplainer):
    """
    Gradient-free, class-agnostic explanation built from activations.

    Only for Convolutional Networks.

    Ref. Fuad, Martin, Giot, Bourqui, Benois-Pineau, Zemmari, Features understanding in 3D CNNs
    for actions recognition in video, IPTA 2020.
    https://doi.org/10.1109/IPTA50016.2020.9286629

    The method thresholds each channel of a convolutional feature map with a k-sigma rule and
    aggregates the binary masks weighted by channel means.

    Parameters
    ----------
    model
        The model from which we want to obtain explanations.
    output_layer
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        Default to the last layer.
        It is recommended to use the layer before Softmax.
    batch_size
        Number of inputs to explain at once, if None compute all at once.
    operator
        Operator to use to compute the explanation, if None use standard predictions.
        Not used by FEM but kept for API consistency.
    conv_layer
        Layer to target for FEM. If an int is provided it will be interpreted as a layer index.
        If a string is provided it will look for the layer name. Defaults to last conv layer.
    k
        Number of standard deviations used in the threshold (k-sigma rule).
    """

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 conv_layer: Optional[Union[str, int]] = None,
                 k: float = 2.0):
        super().__init__(model, output_layer, batch_size, operator)

        self.k = tf.cast(k, tf.float32)

        # find the layer to apply FEM
        if conv_layer is not None:
            self.conv_layer = find_layer(model, conv_layer)
        else:
            for layer in model.layers[::-1]:
                if hasattr(layer, 'filters'):
                    self.conv_layer = layer
                    break
            else:
                raise ValueError(
                    "FEM requires a convolutional layer. Please pass `conv_layer` explicitly."
                )

        self.model = tf.keras.Model(model.input, self.conv_layer.output)

    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute FEM maps for a batch of samples.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape (N, H, W, C).
            More information in the documentation.
        targets
            Not used. Present for API symmetry.

        Returns
        -------
        explanations
            FEM maps resized to the spatial resolution of the inputs.
        """
        # pylint: disable=E1101
        if isinstance(inputs, tf.data.Dataset):
            inputs, _ = tensor_sanitize(inputs, targets)
        else:
            inputs = tf.cast(inputs, tf.float32)

        fem_maps = None
        batch_size = self.batch_size if self.batch_size is not None else len(inputs)
        for x_batch in tf.data.Dataset.from_tensor_slices(inputs).batch(batch_size):
            batch_feature_maps = self.model(x_batch)
            batch_weights, batch_masks = FEM._compute_weights(batch_feature_maps, self.k)
            batch_fem = FEM._apply_weights(batch_weights, batch_masks)

            fem_maps = batch_fem if fem_maps is None else tf.concat([fem_maps, batch_fem], axis=0)

        # FEM is based on the last convolutional layer, we resize explanations to input size
        new_size = inputs.shape[1:-1]
        fem_maps = tf.map_fn(
            fn=lambda fmap: tf.image.resize(
                fmap, new_size, method=tf.image.ResizeMethod.BICUBIC),
            elems=tf.expand_dims(fem_maps, axis=-1)
        )

        return fem_maps

    @staticmethod
    @tf.function
    def _compute_weights(feature_maps: tf.Tensor,
                         k: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute channel weights and binary masks using the k-sigma threshold.

        Parameters
        ----------
        feature_maps
            Activations for the target convolution layer.
        k
            Number of standard deviations used in the threshold.

        Returns
        -------
        weights
            Channel means used as weights.
        binary_mask
            Binary masks after k-sigma thresholding.
        """
        spatial_axes = tf.range(1, tf.rank(feature_maps) - 1)
        means = tf.reduce_mean(feature_maps, axis=spatial_axes, keepdims=True)
        stds = tf.math.reduce_std(feature_maps, axis=spatial_axes, keepdims=True)
        threshold = means + k * stds
        binary_mask = tf.cast(feature_maps >= threshold, feature_maps.dtype)
        return means, binary_mask

    @staticmethod
    @tf.function
    def _apply_weights(weights: tf.Tensor,
                       binary_masks: tf.Tensor) -> tf.Tensor:
        """
        Aggregate binary masks with the provided weights.

        Parameters
        ----------
        weights
            Channel weights.
        binary_masks
            Binary masks after thresholding.

        Returns
        -------
        weighted_feature_maps
        """
        weighted_feature_maps = tf.reduce_sum(tf.multiply(binary_masks, weights), axis=-1)
        return weighted_feature_maps
