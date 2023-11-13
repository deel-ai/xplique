"""
Module related to RISE method
"""

import tensorflow as tf
import numpy as np

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_tensor, Tasks
from ..types import Callable, Optional, Union, Tuple, OperatorSignature


class Rise(BlackBoxExplainer):
    """
    Used to compute the RISE method, by probing the model with randomly masked versions of
    the input image and obtaining the corresponding outputs to deduce critical areas.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/abs/1806.07421

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    batch_size
        Number of pertubed samples to explain at once.
        Default to 32.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    nb_samples
        Number of masks generated for Monte Carlo sampling.
    grid_size
        Size of the grid used to generate the scaled-down masks. Masks are then rescale to
        and cropped to input_size. Can be a tuple for different cutting depending on the dimension.
        Ignored for tabular data.
    preservation_probability
        Probability of preservation for each pixel (or the percentage of non-masked pixels in
        each masks), also the expectation value of the mask.
    mask_value
        Value used as when applying masks.
    """

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = 1e-4

    def __init__(self,
                 model: Callable,
                 batch_size: Optional[int] = 32,
                 operator: Optional[Union[Tasks, str, OperatorSignature]] = None,
                 nb_samples: int = 4000,
                 grid_size: Union[int, Tuple[int]] = 7,
                 preservation_probability: float = .5,
                 mask_value: float = 0.0):
        super().__init__(model, batch_size, operator)

        self.nb_samples = nb_samples
        self.grid_size = grid_size
        self.preservation_probability = preservation_probability
        self.mask_value = mask_value

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute RISE for a batch of samples.

        Parameters
        ----------
        inputs
            Dataset, Tensor or Array. Input samples to be explained.
            If Dataset, targets should not be provided (included in Dataset).
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
            More information in the documentation.
        targets
            Tensor or Array. One-hot encoding of the model's output from which an explanation
            is desired. One encoding per input and only one output at a time. Therefore,
            the expected shape is (N, output_size).
            More information in the documentation.

        Returns
        -------
        explanations
            RISE maps, same shape as the inputs, except for the channels.
        """
        binary_masks = Rise._get_masks(tuple(inputs.shape), self.nb_samples, self.grid_size,
                                       self.preservation_probability)

        rise_maps = None
        batch_size = self.batch_size or self.nb_samples

        # since the number of masks is often very large, we process the entries one by one
        for single_input, single_target in zip(inputs, targets):

            rise_nominator   = tf.zeros((*single_input.shape[:-1], 1))
            rise_denominator = tf.zeros((*single_input.shape[:-1], 1))

            # we iterate on the binary masks since they are cheap in memory
            for batch_masks in batch_tensor(binary_masks, batch_size):
                # the upsampling/cropping phase is performed on the batched masks
                masked_inputs, masks_upsampled = Rise._apply_masks(
                    single_input, batch_masks, self.mask_value)
                repeated_targets = repeat_labels(single_target[tf.newaxis, :], len(batch_masks))

                predictions = self.inference_function(self.model, masked_inputs, repeated_targets)

                while len(predictions.shape) < len(masks_upsampled.shape):
                    predictions = tf.expand_dims(predictions, axis=-1)

                rise_nominator += tf.reduce_sum(predictions * masks_upsampled, 0)
                rise_denominator += tf.reduce_sum(masks_upsampled, 0)

            rise_map = rise_nominator / (rise_denominator + Rise.EPSILON)
            rise_map = rise_map[tf.newaxis]

            rise_maps = rise_map if rise_maps is None else tf.concat([rise_maps, rise_map], axis=0)

        return rise_maps


    @staticmethod
    @tf.function
    def _get_masks(input_shape: Tuple[int],
                   nb_samples: int,
                   grid_size: Union[int, Tuple[int]],
                   preservation_probability: float) -> tf.Tensor:
        """
        Random mask generation.
        Start by generating random mask in a lower dimension. Then,a bilinear interpolation to
        upsample the masks and take a random crop of the size of the inputs.

        Parameters
        ----------
        input_shape
            Shape of an input sample.
            Expected shape among (N, W), (N, T, W), (N, H, W, C).
        nb_samples
            Number of masks generated for Monte Carlo sampling.
        grid_size
            Size of the grid used to generate the scaled-down masks.
            Can be a tuple for non-square grid.
            Ignored for tabular data.
        preservation_probability
            Probability of preservation for each pixel (or the percentage of non-masked pixels in
            each masks), also the expectation value of the mask.

        Returns
        -------
        binary_masks
            The downsampled binary masks.
        """
        if len(input_shape) == 2:  # tabular data, grid size is ignored
            mask_shape = (nb_samples, input_shape[1])

        elif len(input_shape) == 3:  # time series data
            if not isinstance(grid_size, tuple):
                downsampled_shape = (grid_size, input_shape[2])
            else:
                assert grid_size[1] == input_shape[2],\
                    "To apply Rise to time series data, the second dimension of grid size " +\
                    f"{grid_size} should match the third dimension of input shape {input_shape}."
                downsampled_shape = grid_size
            mask_shape = (nb_samples, *downsampled_shape)

        elif len(input_shape) == 4:  # image data
            if not isinstance(grid_size, tuple):
                downsampled_shape = (grid_size, grid_size)
            else:
                downsampled_shape = grid_size
            mask_shape = (nb_samples, *downsampled_shape, 1)

        downsampled_masks = tf.random.uniform(mask_shape, 0, 1)

        binary_masks = downsampled_masks < preservation_probability

        return binary_masks

    @staticmethod
    @tf.function
    def _apply_masks(
        single_input: tf.Tensor,
        binary_masks: tf.Tensor,
        mask_value: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Given input samples and masks, apply it for every sample and repeat the labels.

        Parameters
        ----------
        current_input
            Input samples to be explained.
        binary_masks
            Binary downsampled masks randomly generated.
        mask_value
            Value used when applying masks.

        Returns
        -------
        masked_input
            All the masked combinations of the input (for each masks).
        masks
            Masks after the upsampling / cropping operation
        """
        binary_masks = tf.cast(binary_masks, tf.float32)

        if len(single_input.shape) == 1:  # tabular data, grid size is ignored
            masks = binary_masks

        elif len(single_input.shape) == 2:  # time series
            # the upsampled size is defined as (t+1)(T/t) = T(1 + 1 / t)
            upsampled_size = tf.cast(
                (int(single_input.shape[0] * (1.0 + 1.0 / binary_masks.shape[1])),
                 int(single_input.shape[1])),
                tf.int32
            )

            upsampled_masks = tf.image.resize(tf.expand_dims(binary_masks, axis=-1),
                                                upsampled_size)[:, :, :, 0]
            masks = tf.image.random_crop(upsampled_masks,
                                            (binary_masks.shape[0], *single_input.shape))

        elif len(single_input.shape) == 3:  # image data
            # the upsampled size is defined as (h+1)(H/h) = H(1 + 1 / h)
            upsampled_size = (int(single_input.shape[0] * (1.0 + 1.0 / binary_masks.shape[1])),
                              int(single_input.shape[1] * (1.0 + 1.0 / binary_masks.shape[2])),)

            upsampled_size = tf.cast(upsampled_size, tf.int32)
            upsampled_masks = tf.image.resize(binary_masks, upsampled_size)

            masks = tf.image.random_crop(upsampled_masks,
                                         (binary_masks.shape[0], *single_input.shape[:-1], 1))

        masked_input = masks * tf.expand_dims(single_input, 0) + (1 - masks) * mask_value

        return masked_input, masks
