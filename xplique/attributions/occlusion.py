"""
Module related to Occlusion sensitivity method
"""

from math import ceil

import numpy as np
import tensorflow as tf

from .base import BlackBoxExplainer
from ..utils import sanitize_input_output, repeat_labels


class Occlusion(BlackBoxExplainer):
    """
    Used to compute the Occlusion sensitivity method, sweep a patch that occludes pixels over the
    images and use the variations of the model prediction to deduce critical areas.

    Ref. Zeiler & al., Visualizing and Understanding Convolutional Networks (2013).
    https://arxiv.org/abs/1311.2901

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    patch_size : tuple (int, int) or int, optional
        Size of the patches to apply, if integer then assume an hypercube.
    patch_stride : tuple (int, int) or int, optional
        Stride between two patches, if integer then assume an hypercube.
    occlusion_value : float, optional
        Value used as occlusion.
    """

    def __init__(self, model, batch_size=32, patch_size=(3, 3),
                 patch_stride=(3, 3), occlusion_value=0.5):
        super().__init__(model, batch_size)

        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.patch_stride = patch_stride if isinstance(patch_stride, tuple) \
                                         else (patch_stride, patch_stride)
        self.occlusion_value = occlusion_value

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute Occlusion sensitivity for a batch of samples.

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
        explanations : ndarray (N, W, H)
            Occlusion sensitivity, same shape as the inputs, except for the channels.
        """
        sensitivity = None
        batch_size = self.batch_size or len(inputs)

        masks = Occlusion._get_masks((*inputs.shape[1:],), self.patch_size, self.patch_stride)
        baseline_scores = tf.reduce_sum(self.model.predict(inputs, batch_size=batch_size) * labels,
                                        axis=-1)

        for x_batch, y_batch, baseline in tf.data.Dataset.from_tensor_slices(
                (inputs, labels, baseline_scores)).batch(batch_size):

            occluded_inputs = Occlusion._apply_masks(x_batch, masks, self.occlusion_value)
            repeated_labels = repeat_labels(y_batch, len(masks))

            batch_scores = BlackBoxExplainer._batch_predictions(self.model, occluded_inputs,
                                                                 repeated_labels, batch_size)
            batch_sensitivity = Occlusion._compute_sensitivity(baseline, batch_scores, masks)

            sensitivity = batch_sensitivity if sensitivity is None else \
                tf.concat([sensitivity, batch_sensitivity], axis=0)

        return sensitivity

    @staticmethod
    def _get_masks(input_shape, patch_size, patch_stride):
        """
        Create all the possible patches for the given configuration.

        Parameters
        ----------
        input_shape : tuple
            Desired shape, dimension of one sample.
        patch_size : tuple (int, int), optional
            Size of the patches to apply.
        patch_stride : tuple (int, int), optional
            Stride between two patches.

        Returns
        -------
        occlusion_masks : tf.tensor (N, W, H, C)
            The boolean occlusion masks, with 1 as occluded.
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
    def _apply_masks(inputs, masks, occlusion_value):
        """
        Given input samples and an occlusion mask template, apply it for every sample.

        Parameters
        ----------
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        masks : tf.tensor (M, W, H, C)
            The boolean occlusion masks, with 1 as occluded.
        occlusion_value : float, optional
            Value used as occlusion.

        Returns
        -------
        occluded_inputs : tf.tensor (N * M, W, H, C)
            All the occluded combinations for each inputs.
        repeated_labels : tf.tensor (N * M, L)
            Unchanged label for each occluded inputs.
        """
        occluded_inputs = tf.expand_dims(inputs, axis=1)
        occluded_inputs = tf.repeat(occluded_inputs, repeats=len(masks), axis=1)
        occluded_inputs = occluded_inputs * tf.cast(tf.logical_not(masks), tf.float32) + tf.cast(
            masks, tf.float32) * occlusion_value

        occluded_inputs = tf.reshape(occluded_inputs, (-1, *occluded_inputs.shape[2:]))

        return occluded_inputs

    @staticmethod
    @tf.function
    def _compute_sensitivity(baseline_scores, occluded_scores, masks):
        """
        Compute the sensitivity score given the score of the occluded inputs

        Parameters
        ----------
        baseline_scores : tf.tensor (N)
            Score obtained with the original inputs (not occluded)
        occluded_scores : tensor (N * M)
            The score of the occluded combinations for the class of interest.
        masks : tf.tensor (N, W, H, C)
            The boolean occlusion masks, with 1 as occluded.

        Returns
        -------
        sensitivity : tf.tensor (N, W, H, C)
            Value reflecting the effect of each occlusion patchs on the output
        """
        baseline_scores = tf.expand_dims(baseline_scores, axis=-1)
        occluded_scores = tf.reshape(occluded_scores, (-1, len(masks)))

        score_delta = baseline_scores - occluded_scores
        score_delta = tf.reshape(score_delta, (*score_delta.shape, 1, 1, 1))

        sensitivity = score_delta * tf.cast(masks, tf.float32)
        sensitivity = tf.reduce_sum(sensitivity, axis=1)

        return sensitivity
