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
Module related to Grad-CAM method
"""
import numpy as np
import tensorflow as tf

from ..commons import find_layer
from ..types import Optional
from ..types import Tuple
from ..types import Union
from .base import sanitize_input_output
from .base import WhiteBoxExplainer


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
        Model used for computing explanations.
    output_layer
        Layer to target for the output (e.g logits or after softmax), if int, will be be interpreted
        as layer index, if string will look for the layer name. Default to the last layer, it is
        recommended to use the layer before Softmax.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    conv_layer
        Layer to target for Grad-CAM algorithm, if int, will be be interpreted as layer index,
        if string will look for the layer name.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        output_layer: Optional[Union[str, int]] = -1,
        batch_size: Optional[int] = 32,
        conv_layer: Optional[Union[str, int]] = None,
    ):
        super().__init__(model, output_layer, batch_size)

        # find the layer to apply grad-cam
        if conv_layer is not None:
            self.conv_layer = find_layer(model, conv_layer)
        else:
            # no conv_layer specified, assuming default procedure : the last conv layer
            self.conv_layer = next(
                layer for layer in model.layers[::-1] if hasattr(layer, "filters")
            )

        self.model = tf.keras.Model(
            model.input, [self.conv_layer.output, self.model.output]
        )

    @sanitize_input_output
    def explain(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> tf.Tensor:
        """
        Compute and resize explanations to match inputs shape.
        Accept Tensor, numpy array or tf.data.Dataset (in that case targets is None)

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        grad_cam
            Grad-CAM explanations, same shape as the inputs except for the channels.
        """
        grad_cams = None
        batch_size = self.batch_size if self.batch_size is not None else len(inputs)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ).batch(batch_size):
            batch_feature_maps, batch_gradients = GradCAM._gradient(
                self.model, x_batch, y_batch
            )
            batch_weights = self._compute_weights(batch_gradients, batch_feature_maps)
            batch_grad_cams = GradCAM._apply_weights(batch_weights, batch_feature_maps)

            grad_cams = (
                batch_grad_cams
                if grad_cams is None
                else tf.concat([grad_cams, batch_grad_cams], axis=0)
            )

        # as Grad-CAM is based on the last convolutionnal layer, the explanation output has the
        # same dimensions as this layer, we need to resize the size of the explanations to match
        # the size of the inputs
        input_shape: Tuple[int, int] = inputs.shape[1:3]
        grad_cams = tf.image.resize(grad_cams[..., tf.newaxis], (*input_shape,))

        return grad_cams[..., 0]

    @staticmethod
    @tf.function
    def _gradient(
        model: tf.keras.Model, inputs: tf.Tensor, labels: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the gradient with respect to the conv_layer.

        Parameters
        ----------
        model
            Model used for computing explanations.
        inputs
            Input samples to be explained.
        labels
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

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
            score = tf.reduce_sum(tf.multiply(predictions, labels), axis=-1)

        feature_maps_gradients = tape.gradient(score, feature_maps)

        return feature_maps, feature_maps_gradients

    @staticmethod
    @tf.function
    def _compute_weights(
        feature_maps_gradients: tf.Tensor, feature_maps: tf.Tensor
    ) -> tf.Tensor:
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
    def _apply_weights(weights: tf.Tensor, feature_maps: tf.Tensor) -> tf.Tensor:
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
        weighted_feature_maps = tf.reduce_sum(
            tf.multiply(feature_maps, weights), axis=-1
        )
        weighted_feature_maps = tf.nn.relu(weighted_feature_maps)

        return weighted_feature_maps
