"""
Module related to RISE method
"""

import tensorflow as tf
import numpy as np

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_predictions_one_hot
from ..types import Callable, Tuple, Optional, Union


class Rise(BlackBoxExplainer):
    """
    Used to compute the RISE method, by probing the model with randomly masked versions of
    the input image and obtaining the corresponding outputs to deduce critical areas.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/abs/1806.07421

    Parameters
    ----------
    model
        Model used for computing explanations.
    batch_size
        Number of masked samples to explain at once, if None process all at once.
    nb_samples
        Number of masks generated for Monte Carlo sampling.
    granularity
        Size of the grid used to generate the scaled-down masks. Masks are then rescale to
        input_size + scaled-down size and cropped to input_size.
    preservation_probability
        Probability of preservation for each pixel (or the percentage of non-masked pixels in
        each masks), also the expectation value of the mask.
    """

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = tf.constant(1e-4)

    def __init__(self,
                 model: Callable,
                 batch_size: Optional[int] = 32,
                 nb_samples: int = 4000,
                 granularity: int = 7,
                 preservation_probability: float = .5):
        super().__init__(model, batch_size)

        self.nb_samples = nb_samples
        self.granularity = granularity
        self.preservation_probability = preservation_probability

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute RISE for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            RISE maps, same shape as the inputs, except for the channels.
        """
        rise_maps = None
        batch_size = self.batch_size or self.nb_samples

        masks = Rise._get_masks((*inputs.shape[1:],), self.nb_samples, self.granularity,
                self.preservation_probability)

        for inp, target in tf.data.Dataset.from_tensor_slices(
            (inputs, targets)
        ):
            weighted_scores = None
            target = tf.expand_dims(target, axis=0)
            for batch_masks in tf.data.Dataset.from_tensor_slices(
                (masks)
            ).batch(batch_size):

                masked_input = Rise._apply_masks(inp, batch_masks)
                repeated_targets = repeat_labels(target, len(batch_masks))

                predictions = batch_predictions_one_hot(self.model, masked_input,
                                                    repeated_targets, len(batch_masks))

                batch_weighted_scores = Rise._compute_importance(predictions, batch_masks)

                weighted_scores = batch_weighted_scores if weighted_scores is None else \
                    weighted_scores + batch_weighted_scores

            # ponderate by the presence of each pixels, we could use a mean reducer to make it
            # faster, but only if the number of sample is large enough (as the sampling is iid)
            scores = weighted_scores / (tf.squeeze(tf.reduce_sum(masks, axis=0)) + Rise.EPSILON)
            rise_maps = scores if rise_maps is None else tf.concat([rise_maps, scores], axis=0)

        return rise_maps

    @staticmethod
    @tf.function
    def _get_masks(input_shape: Tuple[int, int],
                   nb_samples: int,
                   granularity: int,
                   preservation_probability: float) -> tf.Tensor:
        """
        Random mask generation. Following the paper, we start by generating random mask in a
        lower dimension. Then, we use bilinear interpolation to upsample the masks and take a
        random crop of the size of the inputs.

        Parameters
        ----------
        input_shape
            Shape of an input sample.
        nb_samples
            Number of masks generated for Monte Carlo sampling.
        granularity
            Size of the grid used to generate the scaled-down masks. Masks are then rescale to
            input_size + scaled-down size and cropped to input_size.
        preservation_probability
            Probability of preservation for each pixel (or the percentage of non-masked pixels in
            each masks), also the expectation value of the mask.

        Returns
        -------
        masks
            The interpolated masks, with continuous values.
        """
        downsampled_shape = (input_shape[0] // granularity, input_shape[1] // granularity)
        upsampled_shape   = (input_shape[0] + downsampled_shape[0], input_shape[1] +
                             downsampled_shape[1])

        downsampled_masks = tf.random.uniform((nb_samples, *downsampled_shape, 1), 0, 1)
        downsampled_masks = downsampled_masks < preservation_probability
        downsampled_masks = tf.cast(downsampled_masks, tf.float32)

        upsampled_masks = tf.image.resize(downsampled_masks, upsampled_shape)

        masks = tf.image.random_crop(upsampled_masks, (nb_samples, *input_shape[:-1], 1))

        return masks

    @staticmethod
    @tf.function
    def _apply_masks(current_input: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
        """
        Given input samples and masks, apply it for every sample and repeat the labels.

        Parameters
        ----------
        current_input
            Input samples to be explained.
        masks
            Masks with continuous value randomly generated.

        Returns
        -------
        occluded_inputs
            All the occluded combinations for each inputs.
        """
        occluded_inputs = tf.expand_dims(current_input, axis=0)
        occluded_inputs = tf.repeat(occluded_inputs, repeats=len(masks), axis=0)

        occluded_inputs = occluded_inputs * masks

        return occluded_inputs

    @staticmethod
    @tf.function
    def _compute_importance(occluded_scores: tf.Tensor,
                            masks: tf.Tensor) -> tf.Tensor:
        """
        Compute the importance of each pixels for each prediction according to the mask used.

        Parameters
        ----------
        occluded_scores
            The score of the occluded combinations for the class of interest.
        masks
            The continuous occlusion masks, with 1 as preserved.

        Returns
        -------
        scores
            Value reflecting the contribution of each pixels on the output.
        """
        # group by input and expand
        occluded_scores = tf.reshape(occluded_scores, (-1, len(masks)))
        occluded_scores = tf.reshape(occluded_scores, (*occluded_scores.shape, 1, 1))
        # removing the channel dimension (we don't deal with input anymore)
        masks = tf.squeeze(masks, axis=-1)
        # weight each pixels according to his preservation
        weighted_scores = occluded_scores * tf.expand_dims(masks, axis=0)

        weighted_scores = tf.reduce_sum(weighted_scores, axis=1)

        return weighted_scores
