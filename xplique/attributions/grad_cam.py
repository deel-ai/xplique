"""
Module related to Grad-CAM method
"""

import tensorflow as tf
from tensorflow.keras.models import Model  # pylint: disable=import-error

from .base import BaseExplanation
from ..utils import sanitize_input_output


class GradCAM(BaseExplanation):
    """
    Used to compute the Grad-CAM visualization method.

    Ref. Selvaraju & al., Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization (2016).
    https://arxiv.org/abs/1610.02391

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    output_layer_index : int, optional
        Index of the output layer, default to the last layer, it is recommended to use the layer
        before Softmax (often '-2').
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    conv_layer : int or string or None
        Layer to target for Grad-CAM algorithm, if int, will be be interpreted as layer index,
        if string will look for the layer name.
    """

    def __init__(self, model, output_layer_index=-1, batch_size=32, conv_layer=None):
        super().__init__(model, output_layer_index, batch_size)

        # find the layer to apply grad-cam
        if isinstance(conv_layer, int):
            self.conv_layer = model.layers[conv_layer]
        elif isinstance(conv_layer, str):
            self.conv_layer = model.get_layer(conv_layer)
        else:
            # No conv_layer specified, assuming default procedure : the last conv layer
            self.conv_layer = next(
                layer for layer in model.layers[::-1] if hasattr(layer, 'filters'))

        self.model = Model(model.input, [self.conv_layer.output, self.target_layer.output])

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute Grad-CAM and resize explanations to match inputs shape.

        Parameters
        ----------
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        labels : ndarray(N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        grad_cam : ndarray (N, W, H)
            Grad-CAM explanations, same shape as the inputs except for the channels.
        """
        grad_cams = None
        batch_size = self.batch_size if self.batch_size is not None else len(inputs)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(
                batch_size):
            batch_feature_maps, batch_gradients = GradCAM._gradient(self.model, x_batch, y_batch)
            batch_weights = self._compute_weights(batch_gradients, batch_feature_maps)
            batch_grad_cams = GradCAM._apply_weights(batch_weights, batch_feature_maps)

            grad_cams = batch_grad_cams if grad_cams is None else tf.concat(
                [grad_cams, batch_grad_cams], axis=0)

        # as Grad-CAM is based on the last convolutionnal layer, the explanation output has the
        # same dimensions as this layer, we need to resize the size of the explanations to match
        # the size of the inputs
        input_shape = self.model.input.shape[1:3]
        grad_cams = tf.image.resize(grad_cams[..., tf.newaxis], (*input_shape,))

        return grad_cams[..., 0]

    @staticmethod
    @tf.function
    def _gradient(model, inputs, labels):
        """
        Compute the gradient with respect to the conv_layer.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing explanations.
        inputs : ndarray (N, W, H, C)
            Batch of input samples , with N number of samples, W & H the sample dimensions,
            and C the number of channels.
        labels : ndarray (N, L)
            One hot encoded labels to compute for each sample, with N the number of samples, and L
            the number of classes.

        Returns
        -------
        feature_maps : tf.Tensor (N, ConvWidth, ConvHeight, Filters)
            Activations for the target convolution layer.
        feature_maps_gradients : tf.Tensor (N, ConvWidth, ConvHeight, Filters)
            Gradients for the target convolution layer.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            feature_maps, predictions = model(inputs)
            score = tf.reduce_sum(tf.multiply(predictions, labels), axis=-1)

        feature_maps_gradients = tape.gradient(score, feature_maps)

        return feature_maps, feature_maps_gradients

    @staticmethod
    @tf.function
    def _compute_weights(feature_maps_gradients, feature_maps):
        """
        Compute the weights according to Grad-CAM procedure.

        Parameters
        ----------
        feature_maps_gradients : tf.Tensor (N, ConvWidth, ConvHeight, Filters)
            Gradients for the target convolution layer.
        feature_maps : tf.Tensor (N, CW, CH, Filters)
            Activations for the target convolution layer. Not used for Grad-CAM.

        Returns
        -------
        weights : tf.Tensor (N, 1, 1, Filters)
            Weights for each feature maps.
        """
        # pylint: disable=unused-argument
        weights = tf.reduce_mean(feature_maps_gradients, axis=(1, 2))
        weights = tf.reshape(weights, (weights.shape[0], 1, 1, weights.shape[-1]))

        return weights

    @staticmethod
    @tf.function
    def _apply_weights(weights, feature_maps):
        """
        Apply the weights to the feature maps and sum them, and follow it by a ReLU.

        Parameters
        ----------
        weights : tf.Tensor (N, 1, 1, Filters)
            Weights for each feature maps.
        feature_maps : tf.Tensor (N, ConvWidth, ConvHeight, Filters)
            Activations for the target convolution layer.

        Returns
        -------
        weighted_feature_maps : tf.Tensor (N, ConvWidth, ConvHeight)
        """
        weighted_feature_maps = tf.reduce_sum(tf.multiply(feature_maps, weights), axis=-1)
        weighted_feature_maps = tf.nn.relu(weighted_feature_maps)

        return weighted_feature_maps
