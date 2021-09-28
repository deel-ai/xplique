"""
Module related to RISE method
"""

import tensorflow as tf
import numpy as np

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_predictions_one_hot, batch_tensor, inference_batching
from ..types import Callable, Optional, Union, Tuple


class DRise(BlackBoxExplainer):
    """
    Used to compute the D-RISE method, by probing the model with randomly masked versions of
    the input image and obtaining the corresponding outputs to deduce critical areas.

    Ref. Petsiuk & al., RISE: Randomized Input Sampling for Explanation of Black-box Models (2018).
    https://arxiv.org/abs/1806.07421

    Ref. Petsiuk & al., Black-box Explanation of Object Detectors via Saliency Maps (2021).
    https://arxiv.org/pdf/2006.03204.pdf

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
        self.binary_masks = DRise._get_masks(self.nb_samples, self.grid_size,
                                            self.preservation_probability)

    @staticmethod
    def inference(model, inputs, targets):
        predictions=model(inputs)
        numClasses=targets.shape[1]-5
        Lt, Pt, Vt = tf.split(targets[0], [4, 1, numClasses])
        SdtDjp=[]
        for Dj in predictions:
            Lj, Pj, Vj = tf.split(Dj[0], [4, 1, numClasses])
            Oj=tf.constant([1.0])
            iou=DRise._bbox_iou(Lt,Lj)
            cosineSim=tf.tensordot(Pt*Vt,Pj*Vj,1)/(tf.norm(Pt*Vt)*tf.norm(Pj*Vj))
            Sdtdj=iou*Oj[0]*cosineSim
            SdtDjp=tf.concat([SdtDjp,[Sdtdj]], axis=0)
        scores=tf.reduce_max(SdtDjp)
        return scores

#    @sanitize_input_output
#    def explain(self,
#                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
#                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
#        """
#        Compute D-RISE for a batch of samples.
#
#        Parameters
#        ----------
#        inputs
#            Input samples to be explained.
#        targets
#            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.
#
#        Returns
#        -------
#        explanations
#            RISE maps, same shape as the inputs, except for the channels.
#        """
#        rise_maps = None
#        targets=tf.expand_dims(targets,axis=0) 
#        batch_size = self.batch_size or len(inputs)#
#
#        masks = DRise._get_masks((*inputs.shape[1:],), self.nb_samples, self.grid_size,
#                               self.preservation_probability)
#
#        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices(
#                (inputs, targets)).batch(batch_size):
#
#            masked_inputs = DRise._apply_masks(x_batch, masks)           
#            repeated_targets = repeat_labels(y_batch, self.nb_samples)
#
#            predictions = inference_batching(DRise.inference,self.model, masked_inputs,
#                                                    repeated_targets,batch_size)
#            scores = DRise._compute_importance(predictions, masks)
#
#            rise_maps = scores if rise_maps is None else tf.concat([rise_maps, scores], axis=0)
#
#        return rise_maps

    @sanitize_input_output
    def explain(self,
                inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> tf.Tensor:
        """
        Compute D-RISE for a batch of samples.

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
        targets=tf.expand_dims(targets,axis=0) 

        # since the number of masks is often very large, we process the entries one by one
        for single_input, single_target in zip(inputs, targets):

#        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices(
#                (inputs, targets)).batch(batch_size):
            rise_nominator   = tf.zeros((*single_input.shape[:-1], 1))
            rise_denominator = tf.zeros((*single_input.shape[:-1], 1))

            # we iterate on the binary masks since they are cheap in memory
            for batch_masks in batch_tensor(self.binary_masks, batch_size):
                # the upsampling/cropping phase is performed on the batched masks
                masked_inputs, masks_upsampled = DRise._apply_masks(single_input, batch_masks)           
                repeated_targets = repeat_labels(single_target[tf.newaxis, :], len(batch_masks))

                predictions = inference_batching(DRise.inference,self.model, masked_inputs,
                                                    repeated_targets,batch_size)
#                scores = DRise._compute_importance(predictions, masks)
                rise_nominator += tf.reduce_sum(tf.reshape(predictions, (-1, 1, 1, 1))
                                                * masks_upsampled, 0)
                rise_denominator += tf.reduce_sum(masks_upsampled, 0)

            rise_map = rise_nominator / (rise_denominator + DRise.EPSILON)
            rise_map = rise_map[tf.newaxis, :, :, 0]

            rise_maps = rise_map if rise_maps is None else tf.concat([rise_maps, rise_map], axis=0)

#            rise_maps = scores if rise_maps is None else tf.concat([rise_maps, scores], axis=0)

        return rise_maps


    @staticmethod
    @tf.function
    def _get_masks(nb_samples: int,
                   grid_size: int,
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
    def _apply_masks(single_input: tf.Tensor, binary_masks: tf.Tensor) -> Tuple[tf.Tensor,
                                                                          tf.Tensor]:
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

#    @staticmethod
#    @tf.function
#    def _compute_importance(occluded_scores: tf.Tensor,
#                            masks: tf.Tensor) -> tf.Tensor:
#        """
#        Compute the importance of each pixels for each prediction according to the mask used.
#
#        Parameters
#        ----------
#        occluded_scores
#            The score of the occluded combinations for the class of interest.
#        masks
#            The continuous occlusion masks, with 1 as preserved.#
#
#        Returns
#        -------
#        scores
#            Value reflecting the contribution of each pixels on the output.
#        """
#        # group by input and expand
#        occluded_scores = tf.reshape(occluded_scores, (-1, len(masks)))
#        occluded_scores = tf.reshape(occluded_scores, (*occluded_scores.shape, 1, 1))
#        # removing the channel dimension (we don't deal with input anymore)
#        masks = tf.squeeze(masks, axis=-1)
#        # weight each pixels according to his preservation
#        weighted_scores = occluded_scores * tf.expand_dims(masks, axis=0)
#
#        # ponderate by the presence of each pixels, we could use a mean reducer to make it
#        # faster, but only if the number of sample is large enough (as the sampling is iid)
#        scores = tf.reduce_sum(weighted_scores, axis=1) / (tf.reduce_sum(masks, axis=0) +
#                                                            DRise.EPSILON)
#
#        return scores

    @staticmethod
    @tf.function
    def _bbox_iou(boxes1: tf.function, 
                  boxes2: tf.function) -> tf.Tensor:
        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = tf.concat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = tf.concat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area + DRise.EPSILON

        return 1.0 * inter_area / union_area
