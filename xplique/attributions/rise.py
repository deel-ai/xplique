"""
Module related to RISE method
"""

import tensorflow as tf

from .base import BaseExplanation
from ..utils import sanitize_input_output, repeat_labels


class Rise(BaseExplanation):
    """
    Used to compute the RISE method, by probing the model with randomly masked versions of
    the input image and obtaining the corresponding outputs to deduce critical areas.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/abs/1806.07421

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    output_layer_index : int, optional
        Index of the output layer, default to the last layer, it is recommended to use the
        softmax layer.
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    nb_samples : int, optional
        Number of masks generated for Monte Carlo sampling.
    granularity : int, optional
        Size of the grid used to generate the scaled-down masks. Masks are then rescale to
        input_size + scaled-down size and cropped to input_size.
    preservation_probability : float, optional
        Probability of preservation for each pixel (or the percentage of non-masked pixels in
        each masks), also the expectation value of the mask.
    """

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = tf.constant(1e-4)

    def __init__(self, model, output_layer_index=-1, batch_size=32, nb_samples=80, granularity=6,
                 preservation_probability=0.5):
        super().__init__(model, output_layer_index, batch_size)

        self.nb_samples = nb_samples
        self.granularity = granularity
        self.preservation_probability = preservation_probability

    @sanitize_input_output
    def explain(self, inputs, labels):
        """
        Compute RISE for a batch of samples.

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
            RISE maps, same shape as the inputs, except for the channels.
        """
        return Rise.compute(self.model,
                            inputs,
                            labels,
                            self.batch_size,
                            self.nb_samples,
                            self.granularity,
                            self.preservation_probability)

    @staticmethod
    def compute(model, inputs, labels, batch_size, nb_samples,
                granularity, preservation_probability):
        """
        Compute Occlusion sensitivity for a batch of samples.

        Parameters
        ----------
        model : tf.keras.Model
            Model used for computing explanations.
        batch_size : int
            Number of samples to explain at once, if None compute all at once.
        nb_samples : int
            Number of masks generated for Monte Carlo sampling.
        granularity : int
            Size of the grid used to generate the scaled-down masks. Masks are then rescale to
            input_size + scaled-down size and cropped to input_size.
        preservation_probability : float
            Probability of preservation for each pixel (or the percentage of non-masked pixels in
            each masks), also the expectation value of the mask.

        Returns
        -------
        rise_maps : tf.Tensor (N, W, H)
            RISE maps, same shape as the inputs, except for the channels.
        """
        rise_maps = None
        batch_size = batch_size or len(inputs)

        masks = Rise.get_masks((*inputs.shape[1:],), nb_samples, granularity,
                               preservation_probability)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices(
                (inputs, labels)).batch(batch_size):

            masked_inputs = Rise.apply_masks(x_batch, masks)
            repeated_labels = repeat_labels(y_batch, nb_samples)

            predictions = BaseExplanation._batch_predictions(model, masked_inputs,
                                                             repeated_labels, batch_size)
            scores = Rise.compute_importance(predictions, masks)

            rise_maps = scores if rise_maps is None else tf.concat([rise_maps, scores], axis=0)

        return rise_maps

    @staticmethod
    @tf.function
    def get_masks(input_shape, nb_samples, granularity, preservation_probability):
        """
        Random mask generation. Following the paper, we start by generating random mask in a
        lower dimension. Then, we use bilinear interpolation to upsample the masks and take a
        random crop of the size of the inputs.

        Parameters
        ----------
        input_shape : tuple
            Shape of an input sample.
        nb_samples : int, optional
            Number of masks generated for Monte Carlo sampling.
        granularity : int
            Size of the grid used to generate the scaled-down masks. Masks are then rescale to
            input_size + scaled-down size and cropped to input_size.
        preservation_probability : float
            Probability of preservation for each pixel (or the percentage of non-masked pixels in
            each masks), also the expectation value of the mask.

        Returns
        -------
        masks : tf.tensor (N, W, H, 1)
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
    def apply_masks(inputs, masks):
        """
        Given input samples and masks, apply it for every sample and repeat the labels

        Parameters
        ----------
        inputs : tf.tensor (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        masks : tf.tensor (M, W, H, 1)
            Masks with continuous value randomly generated

        Returns
        -------
        occluded_inputs : tf.tensor (N * M, W, H, C)
            All the occluded combinations for each inputs.
        """
        occluded_inputs = tf.expand_dims(inputs, axis=1)
        occluded_inputs = tf.repeat(occluded_inputs, repeats=len(masks), axis=1)

        occluded_inputs = occluded_inputs * masks

        occluded_inputs = tf.reshape(occluded_inputs, (-1, *occluded_inputs.shape[2:]))

        return occluded_inputs

    @staticmethod
    @tf.function
    def compute_importance(occluded_scores, masks):
        """
        Compute the importance of each pixels for each prediction according to the mask used.

        Parameters
        ----------
        occluded_scores : tensor (N * M, W, H, C)
            The score of the occluded combinations for the class of interest.
        masks : tf.tensor (M, W, H, 1)
            The continuous occlusion masks, with 1 as preserved.

        Returns
        -------
        scores : tf.tensor (N, W, H, C)
            Value reflecting the contribution of each pixels on the output.
        """
        # group by input and expand
        occluded_scores = tf.reshape(occluded_scores, (-1, len(masks)))
        occluded_scores = tf.reshape(occluded_scores, (*occluded_scores.shape, 1, 1))
        # removing the channel dimension (we don't deal with input anymore)
        masks = tf.squeeze(masks, axis=-1)
        # weight each pixels according to his preservation
        weighted_scores = occluded_scores * tf.expand_dims(masks, axis=0)

        # ponderate by the presence of each pixels, we could use a mean reducer to make it
        # faster, but only if the number of sample is large enough (as the sampling is iid)
        scores = tf.reduce_sum(weighted_scores, axis=1) / (tf.reduce_sum(masks, axis=0) +
                                                            Rise.EPSILON)

        return scores
