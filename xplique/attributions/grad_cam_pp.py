# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module related to Grad-CAM++ method
"""
import tensorflow as tf

from .grad_cam import GradCAM


class GradCAMPP(GradCAM):
    """
    Used to compute the Grad-CAM++ visualization method.

    Only for Convolutional Networks.

    Ref. Chattopadhyay & al., Grad-CAM++: Improved Visual Explanations for Deep Convolutional
    Networks (2017).
    https://arxiv.org/abs/1710.11063

    Parameters
    ----------
    model
        Model used for computing explanations.
    output_layer
        Layer to target for the output (e.g logits or after softmax), if int, will be be interpreted
        as layer index, if string will look for the layer name. Default to the last layer, it is
        recommended to use the layer before Softmax.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    conv_layer
        Layer to target for Grad-CAM++ algorithm, if int, will be be interpreted as layer index,
        if string will look for the layer name.
    """

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = tf.constant(1e-4)

    @staticmethod
    @tf.function
    def _compute_weights(
        feature_maps_gradients: tf.Tensor, feature_maps: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the weights according to Grad-CAM++ procedure.

        Parameters
        ----------
        feature_maps_gradients
            Gradients for the target convolution layer.
        feature_maps
            Activations for the target convolution layer.

        Returns
        -------
        weights
            Weights for each feature maps.
        """
        feature_maps_gradients_square = tf.pow(feature_maps_gradients, 2)
        feature_maps_gradients_cube = tf.pow(feature_maps_gradients, 3)

        feature_map_avg = tf.reduce_mean(feature_maps, axis=(1, 2), keepdims=True)

        nominator = feature_maps_gradients_square
        denominator = (
            2.0 * feature_maps_gradients_square
            + feature_maps_gradients_cube * feature_map_avg
        )
        denominator += tf.cast(denominator == 0, tf.float32) * GradCAMPP.EPSILON

        feature_map_alphas = (
            nominator / denominator * tf.nn.relu(feature_maps_gradients)
        )
        weights = tf.reduce_mean(feature_map_alphas, axis=(1, 2))
        weights = tf.reshape(weights, (weights.shape[0], 1, 1, weights.shape[-1]))

        return weights
