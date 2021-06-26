"""
Module related to Occlusion sensitivity method
"""

from math import ceil

import numpy as np
import tensorflow as tf

from .base import BlackBoxExplainer
from ..utils import sanitize_input_output, repeat_labels, batch_predictions_one_hot
from ..types import Callable, Tuple, Union, Optional


class Occlusion(BlackBoxExplainer):
    """
    Used to compute the Occlusion sensitivity method, sweep a patch that occludes pixels over the
    images and use the variations of the model prediction to deduce critical areas.

    Ref. Zeiler & al., Visualizing and Understanding Convolutional Networks (2013).
    https://arxiv.org/abs/1311.2901

    Parameters
    ----------
    model
        Model used for computing explanations.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    patch_size
        Size of the patches to apply, if integer then assume an hypercube.
    patch_stride
        Stride between two patches, if integer then assume an hypercube.
    occlusion_value
        Value used as occlusion.
    """

    def __init__(self,
                 model: Callable,
                 batch_size: Optional[int] = 32,
                 patch_size: Union[int, Tuple[int, int]] = (3, 3),
                 patch_stride: Union[int, Tuple[int, int]] = (3, 3),
                 occlusion_value: float = 0.5):
        super().__init__(model, batch_size)

        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.patch_stride = patch_stride if isinstance(patch_stride, tuple) \
                                         else (patch_stride, patch_stride)
        self.occlusion_value = occlusion_value

    @sanitize_input_output
    def explain(self,
                inputs: tf.Tensor,
                labels: tf.Tensor) -> tf.Tensor:
        """
        Compute Occlusion sensitivity for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        labels
            One-hot encoded labels, one for each sample.

        Returns
        -------
        explanations
            Occlusion sensitivity, same shape as the inputs, except for the channels.
        """
        sensitivity = None
        batch_size = self.batch_size or len(inputs)

        masks = Occlusion._get_masks((*inputs.shape[1:],), self.patch_size, self.patch_stride)
        baseline_scores = batch_predictions_one_hot(self.model, inputs, labels, batch_size)

        for x_batch, y_batch, baseline in tf.data.Dataset.from_tensor_slices(
                (inputs, labels, baseline_scores)).batch(batch_size):

            occluded_inputs = Occlusion._apply_masks(x_batch, masks, self.occlusion_value)
            repeated_labels = repeat_labels(y_batch, masks.shape[0])

            batch_scores = batch_predictions_one_hot(self.model, occluded_inputs,
                                                     repeated_labels, batch_size)
            batch_sensitivity = Occlusion._compute_sensitivity(baseline, batch_scores, masks)

            sensitivity = batch_sensitivity if sensitivity is None else \
                tf.concat([sensitivity, batch_sensitivity], axis=0)

        return sensitivity

    @staticmethod
    def _get_masks(input_shape: Tuple[int, int, int],
                   patch_size: Tuple[int, int],
                   patch_stride: Tuple[int, int]) -> tf.Tensor:
        """
        Create all the possible patches for the given configuration.

        Parameters
        ----------
        input_shape
            Desired shape, dimension of one sample.
        patch_size
            Size of the patches to apply.
        patch_stride
            Stride between two patches.

        Returns
        -------
        occlusion_masks
            The boolean occlusion masks, same shape as the inputs, with 1 as occluded.
        """
        masks = []

        x_anchors = [x * patch_stride[0] for x in
                     range(0, ceil((input_shape[0] - patch_size[0] + 1) / patch_stride[0]))]
        y_anchors = [y * patch_stride[1] for y in
                     range(0, ceil((input_shape[1] - patch_size[1] + 1) / patch_stride[1]))]

        for x_anchor in x_anchors:
            for y_anchor in y_anchors:
                mask = np.zeros(input_shape, dtype=bool)
                mask[x_anchor:x_anchor + patch_size[0], y_anchor:y_anchor + patch_size[1]] = 1
                masks.append(mask)

        return tf.cast(masks, dtype=tf.bool)

    @staticmethod
    @tf.function
    def _apply_masks(inputs: tf.Tensor,
                     masks: tf.Tensor,
                     occlusion_value: float) -> tf.Tensor:
        """
        Given input samples and an occlusion mask template, apply it for every sample.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        masks
            The boolean occlusion masks, with 1 as occluded.
        occlusion_value
            Value used as occlusion.

        Returns
        -------
        occluded_inputs
            All the occluded combinations for each inputs.
        """
        occluded_inputs = tf.expand_dims(inputs, axis=1)
        occluded_inputs = tf.repeat(occluded_inputs, repeats=masks.shape[0], axis=1)
        occluded_inputs = occluded_inputs * tf.cast(tf.logical_not(masks), tf.float32) + tf.cast(
            masks, tf.float32) * occlusion_value

        occluded_inputs = tf.reshape(occluded_inputs, (-1, *occluded_inputs.shape[2:]))

        return occluded_inputs

    @staticmethod
    @tf.function
    def _compute_sensitivity(baseline_scores: tf.Tensor,
                             occluded_scores: tf.Tensor,
                             masks: tf.Tensor) -> tf.Tensor:
        """
        Compute the sensitivity score given the score of the occluded inputs

        Parameters
        ----------
        baseline_scores
            Scores obtained with the original inputs (not occluded)
        occluded_scores
            Scores of the occluded combinations for the class of
            interest.
        masks
            The boolean occlusion masks, with 1 as occluded.

        Returns
        -------
        sensitivity
            Value reflecting the effect of each occlusion patchs on the output
        """
        baseline_scores = tf.expand_dims(baseline_scores, axis=-1)
        occluded_scores = tf.reshape(occluded_scores, (-1, masks.shape[0]))

        score_delta = baseline_scores - occluded_scores
        score_delta = tf.reshape(score_delta, (*score_delta.shape, 1, 1, 1))

        sensitivity = score_delta * tf.cast(masks, tf.float32)
        sensitivity = tf.reduce_sum(sensitivity, axis=1)

        return sensitivity
