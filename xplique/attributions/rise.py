"""
Module related to RISE method
"""

import tensorflow as tf
import numpy as np

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_tensor
from ..types import Callable, Optional, Union, Tuple


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
    grid_size
        Size of the grid used to generate the scaled-down masks. Masks are then rescale to
        and cropped to input_size.
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
                 grid_size: int = 7,
                 preservation_probability: float = .5):
        super().__init__(model, batch_size)

        self.nb_samples = nb_samples
        self.grid_size = grid_size
        self.preservation_probability = preservation_probability
        self.binary_masks = Rise._get_masks(self.nb_samples, self.grid_size,
                                            self.preservation_probability)

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

        # since the number of masks is often very large, we process the entries one by one
        for single_input, single_target in zip(inputs, targets):

            rise_nominator   = tf.zeros((*single_input.shape[:-1], 1))
            rise_denominator = tf.zeros((*single_input.shape[:-1], 1))

            # we iterate on the binary masks since they are cheap in memory
            for batch_masks in batch_tensor(self.binary_masks, batch_size):
                # the upsampling/cropping phase is performed on the batched masks
                masked_inputs, masks_upsampled = Rise._apply_masks(single_input, batch_masks)
                repeated_targets = repeat_labels(single_target[tf.newaxis, :], len(batch_masks))

                predictions = self.inference_function(self.model, masked_inputs, repeated_targets)

                rise_nominator += tf.reduce_sum(tf.reshape(predictions, (-1, 1, 1, 1))
                                                * masks_upsampled, 0)
                rise_denominator += tf.reduce_sum(masks_upsampled, 0)

            rise_map = rise_nominator / (rise_denominator + Rise.EPSILON)
            rise_map = rise_map[tf.newaxis, :, :, 0]

            rise_maps = rise_map if rise_maps is None else tf.concat([rise_maps, rise_map], axis=0)

        return rise_maps


    @staticmethod
    @tf.function
    def _get_masks(nb_samples: int,
                   grid_size: int,
                   preservation_probability: float) -> tf.Tensor:
        """
        Random mask generation.
        Start by generating random mask in a lower dimension. Then,a bilinear interpolation to
        upsample the masks and take a random crop of the size of the inputs.

        Parameters
        ----------
        input_shape
            Shape of an input sample.
        nb_samples
            Number of masks generated for Monte Carlo sampling.
        grid_size
            Size of the grid used to generate the scaled-down masks.
        preservation_probability
            Probability of preservation for each pixel (or the percentage of non-masked pixels in
            each masks), also the expectation value of the mask.

        Returns
        -------
        binary_masks
            The downsampled binary masks.
        """
        downsampled_shape = (grid_size, grid_size)
        downsampled_masks = tf.random.uniform((nb_samples, *downsampled_shape, 1), 0, 1)

        binary_masks = downsampled_masks < preservation_probability

        return binary_masks

    @staticmethod
    @tf.function
    def _apply_masks(
        single_input: tf.Tensor,
        binary_masks: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Given input samples and masks, apply it for every sample and repeat the labels.

        Parameters
        ----------
        current_input
            Input samples to be explained.
        binary_masks
            Binary downsampled masks randomly generated.

        Returns
        -------
        masked_input
            All the masked combinations of the input (for each masks).
        masks
            Masks after the upsampling / cropping operation
        """
        # the upsampled size is defined as (h+1)(H/h) = H(1 + 1 / h)
        upsampled_size = single_input.shape[0] * (1.0 + 1.0 / binary_masks.shape[1])
        upsampled_size = tf.cast(upsampled_size, tf.int32)

        upsampled_masks = tf.image.resize(tf.cast(binary_masks, tf.float32),
                                          (upsampled_size, upsampled_size))

        masks = tf.image.random_crop(upsampled_masks, (len(binary_masks),
                                                       *single_input.shape[:-1], 1))

        masked_input = tf.expand_dims(single_input, 0) * masks

        return masked_input, masks
