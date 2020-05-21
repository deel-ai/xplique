"""
Module related to Grad-CAM method
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model  # pylint: disable=import-error

from .base import BaseExplanation


class GradCAM(BaseExplanation):
    """
    Used to compute the different possible variants of the Grad-CAM visualization method.

    Ref. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization (2016).
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
            # assuming default procedure : the last conv layer
            self.conv_layer = next(
                layer for layer in model.layers[::-1] if hasattr(layer, 'filters'))

        self.model = Model(model.input, [self.conv_layer.output, self.target_layer.output])

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
        inputs = tf.cast(inputs, tf.float32)
        labels = tf.cast(labels, tf.float32)

        grad_cams = GradCAM.compute(self.model, inputs, labels, self.batch_size)

        # as Grad-CAM is based on the last convolutionnal layer, the explanation output has the
        # same dimensions as this layer, we need to resize the size of the explanations to match
        # the size of the inputs
        input_shape = self.model.input.shape[1:3]
        grad_cams = np.array(
            [cv2.resize(np.array(grad_cam), (*input_shape,)) for grad_cam in grad_cams])

        return grad_cams

    @staticmethod
    def compute(model, inputs, labels, batch_size):
        """
        Compute the Grad-CAM explanations of the given samples.

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
        grad_cam : ndarray (N, CW, CH)
            Explanation computed, with CW & CH the dimensions of the conv layer.
        """
        grad_cams = None
        batch_size = batch_size if batch_size is not None else len(inputs)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(
                batch_size):
            batch_grad_cams = GradCAM.gradient(model, x_batch, y_batch)

            grad_cams = batch_grad_cams if grad_cams is None else tf.concat(
                [grad_cams, batch_grad_cams], axis=0)

        return grad_cams

    @staticmethod
    @tf.function
    def gradient(model, inputs, labels):
        """
        Apply a single pass procedure to compute Grad-CAM.

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
        grad_cam : tf.Tensor (N, CW, CH)
            Explanation computed, with CW & CH the dimensions of the conv layer.
        """
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            feature_map_activations, predictions = model(inputs)
            score = tf.reduce_sum(tf.multiply(predictions, labels), axis=-1)

        feature_map_gradients = tape.gradient(score, feature_map_activations)
        feature_map_weights = tf.reduce_mean(feature_map_gradients, axis=(1, 2))
        # reshape to apply a gradient weight to each feature map
        feature_map_weights = tf.reshape(feature_map_weights,
                                         (feature_map_weights.shape[0], 1, 1,
                                          feature_map_weights.shape[-1]))

        grad_cams = tf.reduce_sum(
            tf.multiply(feature_map_activations, feature_map_weights),
            axis=-1)
        grad_cams = tf.nn.relu(grad_cams)

        return grad_cams
