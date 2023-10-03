"""
Module related to Grad-CAM method
"""

import tensorflow as tf
import numpy as np

from .base import WhiteBoxExplainer, sanitize_input_output
from ..commons import find_layer, Tasks
from ..types import Tuple, Union, Optional, OperatorSignature


class GradCAM(WhiteBoxExplainer):
    """
    Used to compute the Grad-CAM visualization method.

    Only for Convolutional Networks.

    Ref. Selvaraju & al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization (2016).
    https://arxiv.org/abs/1610.02391

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    output_layer
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        Default to the last layer.
        It is recommended to use the layer before Softmax.
    batch_size
        Number of inputs to explain at once, if None compute all at once.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    conv_layer
        Layer to target for Grad-CAM algorithm.
        If an int is provided it will be interpreted as a layer index.
        If a string is provided it will look for the layer name.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 output_layer: Optional[Union[str, int]] = None,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 conv_layer: Optional[Union[str, int]] = None):
        super().__init__(model, output_layer, batch_size, operator)

        # find the layer to apply grad-cam
        if conv_layer is not None:
            self.conv_layer = find_layer(model, conv_layer)
        else:
            # no conv_layer specified, assuming default procedure : the last conv layer
            self.conv_layer = next(
                layer for layer in model.layers[::-1] if hasattr(layer, 'filters'))

        self.model = tf.keras.Model(model.input, [self.conv_layer.output, self.model.output])

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute and resize explanations to match inputs shape.
        Accept Tensor, numpy array or tf.data.Dataset (in that case targets is None)

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, output_size).
            More information in the documentation.

        Returns
        -------
        grad_cam
            Grad-CAM explanations, same shape as the inputs except for the channels.
        """
        # pylint: disable=E1101
        grad_cams = None
        batch_size = self.batch_size if self.batch_size is not None else len(inputs)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, targets)).batch(
                batch_size):
            batch_feature_maps, batch_gradients = GradCAM._gradient(self.model, x_batch, y_batch)
            batch_weights = self._compute_weights(batch_gradients, batch_feature_maps)
            batch_grad_cams = GradCAM._apply_weights(batch_weights, batch_feature_maps)

            grad_cams = batch_grad_cams if grad_cams is None else tf.concat(
                [grad_cams, batch_grad_cams], axis=0)

        # as Grad-CAM is based on the last convolutional layer, the explanation output has the
        # same dimensions as this layer, we need to resize the size of the explanations to match
        # the size of the inputs
        new_size = inputs.shape[1:-1]
        grad_cams = tf.map_fn(
            fn=lambda g_cam: tf.image.resize(g_cam, new_size, method=tf.image.ResizeMethod.BICUBIC),
            elems=tf.expand_dims(grad_cams, axis=-1)
        )

        return grad_cams

    @staticmethod
    @tf.function
    def _gradient(model: tf.keras.Model,
                  inputs: tf.Tensor,
                  targets: tf.Tensor) -> Tuple[tf.Tensor,
                                              tf.Tensor]:
        """
        Compute the gradient with respect to the conv_layer.

        Parameters
        ----------
        model
            The model from which we want to obtain explanations.
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C). More information in the
            documentation (API Description).
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time.
            More information in the documentation (API Description).

        Returns
        -------
        feature_maps
            Activations for the target convolution layer.
        feature_maps_gradients
            Gradients for the target convolution layer.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            feature_maps, predictions = model(inputs)
            score = tf.reduce_sum(tf.multiply(predictions, targets), axis=-1)

        feature_maps_gradients = tape.gradient(score, feature_maps)

        return feature_maps, feature_maps_gradients

    @staticmethod
    @tf.function
    def _compute_weights(feature_maps_gradients: tf.Tensor,
                         feature_maps: tf.Tensor) -> tf.Tensor:
        """
        Compute the weights according to Grad-CAM procedure.

        Parameters
        ----------
        feature_maps_gradients
            Gradients for the target convolution layer.
        feature_maps
            Activations for the target convolution layer. Not used for Grad-CAM.

        Returns
        -------
        weights
            Weights for each feature maps.
        """
        # pylint: disable=unused-argument
        weights = tf.reduce_mean(feature_maps_gradients, axis=(1, 2), keepdims=True)
        return weights

    @staticmethod
    @tf.function
    def _apply_weights(weights: tf.Tensor,
                       feature_maps: tf.Tensor) -> tf.Tensor:
        """
        Apply the weights to the feature maps, sum them and follow it by a ReLU.

        Parameters
        ----------
        weights
            Weights for each feature maps.
        feature_maps
            Activations for the target convolution layer.

        Returns
        -------
        weighted_feature_maps
        """
        weighted_feature_maps = tf.reduce_sum(tf.multiply(feature_maps, weights), axis=-1)
        weighted_feature_maps = tf.nn.relu(weighted_feature_maps)

        return weighted_feature_maps
