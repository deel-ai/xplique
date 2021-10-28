"""
Module related to Occlusion sensitivity method
"""

from math import ceil

import numpy as np
import tensorflow as tf

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_tensor
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
        Number of masked samples to process at once, if None process all at once.
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
                 patch_size: Union[int, Tuple[int, int]] = 3,
                 patch_stride: Union[int, Tuple[int, int]] = 3,
                 occlusion_value: float = 0.5):
        super().__init__(model, batch_size)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.occlusion_value = occlusion_value

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute Occlusion sensitivity for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            Occlusion sensitivity, same shape as the inputs, except for the channels.
        """

        # check if data is images
        is_image = len(inputs.shape) > 2

        if is_image:
            if not isinstance(self.patch_size, tuple):
                self.patch_size = (self.patch_size, self.patch_size)
            if not isinstance(self.patch_stride, tuple):
                self.patch_stride = (self.patch_stride, self.patch_stride)

        occlusion_maps = None
        batch_size = self.batch_size or len(inputs)

        masks = Occlusion._get_masks((*inputs.shape[1:],), self.patch_size, self.patch_stride)
        base_scores = self.batch_inference_function(self.model, inputs, targets, batch_size)

        # since the number of masks is often very large, we process the entries one by one
        for single_input, single_target, single_base_score in zip(inputs, targets, base_scores):

            occlusion_map = tf.zeros(masks.shape[1:])

            for batch_masks in batch_tensor(masks, batch_size):

                occluded_inputs = Occlusion._apply_masks(single_input, batch_masks,
                                                         self.occlusion_value)
                repeated_targets = repeat_labels(single_target[tf.newaxis, :], len(batch_masks))

                batch_scores = self.inference_function(self.model,
                                                       occluded_inputs,
                                                       repeated_targets)

                batch_sensitivity = Occlusion._compute_sensitivity(
                    single_base_score, batch_scores, batch_masks
                )

                occlusion_map += batch_sensitivity

            occlusion_maps = occlusion_map if occlusion_maps is None else \
                tf.concat([occlusion_maps, occlusion_map], axis=0)

        return occlusion_maps

    @staticmethod
    def _get_masks(input_shape: Union[Tuple[int, int, int], Tuple[int, int], Tuple[int]],
                   patch_size: Union[int, Tuple[int, int]],
                   patch_stride: Union[int, Tuple[int, int]]) -> tf.Tensor:
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

        # check if we have tabular data
        is_tabular = len(input_shape) == 1

        if is_tabular:
            x_anchors = [x * patch_stride for x in
                         range(0, ceil((input_shape[0] - patch_size + 1) / patch_stride))]

            for x_anchor in x_anchors:
                mask = np.zeros(input_shape, dtype=bool)
                mask[x_anchor:x_anchor + patch_size] = 1
                masks.append(mask)
        else:
            x_anchors = [x * patch_stride[0] for x in
                         range(0, ceil((input_shape[0] - patch_size[0] + 1) / patch_stride[0]))]
            y_anchors = [y * patch_stride[1] for y in
                         range(0, ceil((input_shape[1] - patch_size[1] + 1) / patch_stride[1]))]

            for x_anchor in x_anchors:
                for y_anchor in y_anchors:
                    mask = np.zeros(input_shape[:2], dtype=bool)
                    mask[x_anchor:x_anchor + patch_size[0], y_anchor:y_anchor + patch_size[1]] = 1
                    masks.append(mask)

        return tf.cast(masks, dtype=tf.bool)

    @staticmethod
    @tf.function
    def _apply_masks(current_input: tf.Tensor,
                     masks: tf.Tensor,
                     occlusion_value: float) -> tf.Tensor:
        """
        Given input samples and an occlusion mask template, apply it for every sample.

        Parameters
        ----------
        current_input
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
        occluded_inputs = tf.expand_dims(current_input, axis=0)
        occluded_inputs = tf.repeat(occluded_inputs, repeats=len(masks), axis=0)

        # check if current input shape is (W, H, C)
        has_channels = len(current_input.shape) > 2
        if has_channels:
            masks = tf.expand_dims(masks, axis=-1)
            masks = tf.repeat(masks, repeats=current_input.shape[-1], axis=-1)

        occluded_inputs = occluded_inputs * tf.cast(tf.logical_not(masks), tf.float32)
        occluded_inputs += tf.cast(masks, tf.float32) * occlusion_value

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
        occluded_scores = tf.reshape(occluded_scores, (-1, len(masks)))

        score_delta = baseline_scores - occluded_scores
        # reshape the delta score to fit masks
        score_delta = tf.reshape(score_delta, (*score_delta.shape, *(1,) * len(masks.shape[1:])))

        sensitivity = score_delta * tf.cast(masks, tf.float32)
        sensitivity = tf.reduce_sum(sensitivity, axis=1)

        return sensitivity
