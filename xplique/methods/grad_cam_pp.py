"""
Module related to Grad-CAM++ method
"""

import cv2
import numpy as np
import tensorflow as tf

from .grad_cam import GradCAM


class GradCAMPP(GradCAM):
    """
    Used to compute the Grad-CAM++ visualization method.

    Ref. Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks (2017).
    https://arxiv.org/abs/1710.11063

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
        Layer to target for Grad-CAM++ algorithm, if int, will be be interpreted as layer index,
        if string will look for the layer name.
    """

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = tf.constant(1e-4)

    def explain(self, inputs, labels):
        """
        Compute Grad-CAM++ and resize explanations to match inputs shape.

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
        grad_cam_pp : ndarray (N, W, H)
            Grad-CAM++ explanations, same shape as the inputs except for the channels.
        """
        inputs = tf.cast(inputs, tf.float32)
        labels = tf.cast(labels, tf.float32)

        grad_cams_pp = GradCAMPP.compute(self.model, inputs, labels, self.batch_size)

        input_shape = self.model.input.shape[1:3]
        grad_cams_pp = np.array(
            [cv2.resize(np.array(grad_cam_pp), (*input_shape,)) for grad_cam_pp in grad_cams_pp])

        return grad_cams_pp

    @staticmethod
    def compute(model, inputs, labels, batch_size):
        """
        Compute the Grad-CAM++ explanations of the given samples.

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
        grad_cam : tf.Tensor (N, ConvWidth, ConvHeight)
            Explanation computed, with CW & CH the dimensions of the conv layer.
        """
        grad_cams_pp = None
        batch_size = batch_size if batch_size is not None else len(inputs)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(
                batch_size):
            batch_activations, batch_gradients = GradCAM.gradient(model, x_batch, y_batch)
            batch_weights = GradCAMPP.compute_weights(batch_activations, batch_gradients)
            batch_grad_cams_pp = GradCAM.apply_weights(batch_weights, batch_activations)

            grad_cams_pp = batch_grad_cams_pp if grad_cams_pp is None else tf.concat(
                [grad_cams_pp, batch_grad_cams_pp], axis=0)

        return grad_cams_pp

    @staticmethod
    @tf.function
    def compute_weights(feature_maps, feature_maps_gradients):
        """
        Compute the weights according to Grad-CAM procedure.

        Parameters
        ----------
        feature_maps : tf.Tensor (N, ConvWidth, ConvHeight, Filters)
            Activations for the target convolution layer.
        feature_maps_gradients : tf.Tensor (N, ConvWidth, ConvHeight, Filters)
            Gradients for the target convolution layer.

        Returns
        -------
        weights : tf.Tensor (N, 1, 1, Filters)
            Weights for each feature maps.
        """

        feature_maps_gradients_square = tf.pow(feature_maps_gradients, 2)
        feature_maps_gradients_cube = tf.pow(feature_maps_gradients, 3)

        feature_map_avg = tf.reduce_mean(feature_maps, axis=(1, 2))
        feature_map_avg = tf.reshape(feature_map_avg,
                                     (feature_map_avg.shape[0], 1, 1,
                                      feature_map_avg.shape[-1]))

        nominator = feature_maps_gradients_square
        denominator = 2.0 * feature_maps_gradients_square + \
                      feature_maps_gradients_cube * feature_map_avg
        denominator += tf.cast(denominator == 0, tf.float32) * GradCAMPP.EPSILON

        feature_map_alphas = nominator / denominator * tf.nn.relu(feature_maps_gradients)
        weights = tf.reduce_mean(feature_map_alphas, axis=(1, 2))
        weights = tf.reshape(weights,
                             (weights.shape[0], 1, 1,
                              weights.shape[-1]))

        return weights
