"""
Module related to RISE method
"""

import tensorflow as tf
import numpy as np

from .base import BlackBoxExplainer, sanitize_input_output
from ..commons import repeat_labels, batch_predictions_one_hot,inference_batching
from ..types import Callable, Tuple, Optional, Union

def print_tf(message,o):
    print(message,type(o))
    print(o)
        
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
        Number of samples to explain at once, if None compute all at once.
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

    def foo(model, inputs, targets):
        #print("FOO.targets",len(targets))
        predictions=model(inputs)
        #print("FOO.predictions",len(predictions))
        numClasses=targets.shape[1]-5
        #print("FOO.numClasses",numClasses)
        Lt, Pt, Vt = tf.split(targets[0], [4, 1, numClasses])
        SdtDjp=[]
        for Dj in predictions:
            Lj, Pj, Vj = tf.split(Dj[0], [4, 1, numClasses])
            Oj=tf.constant([1.0])
            iou=DRise._bbox_iou(Lt,Lj)
#            print("PjVj",Pj,Vj,Pj*Vj)
            cosineSim=tf.tensordot(Pt*Vt,Pj*Vj,1)/(tf.norm(Pt*Vt)*tf.norm(Pj*Vj))
            Sdtdj=iou*Oj[0]*cosineSim
            SdtDjp=tf.concat([SdtDjp,[Sdtdj]], axis=0)
        #print("FOO.SdtDjp",SdtDjp)
        scores=tf.reduce_max(SdtDjp)
        #print("FOO.scores",scores)
        return scores

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
        #print("E.inputs.shape",inputs.shape)
        targets=tf.expand_dims(targets,axis=0) 
        #print("E.targets.shape",targets.shape)
        batch_size = self.batch_size or len(inputs)
        #print("E.batch_size",batch_size)

        masks = DRise._get_masks((*inputs.shape[1:],), self.nb_samples, self.granularity,
                               self.preservation_probability)
        #print ("E.masks.shape",masks.shape)

        for x_batch, y_batch in tf.data.Dataset.from_tensor_slices(
                (inputs, targets)).batch(batch_size):

            masked_inputs = DRise._apply_masks(x_batch, masks)           
            #print("E.masked_inputs.shape",masked_inputs.shape)
            repeated_targets = repeat_labels(y_batch, self.nb_samples)

            #numClasses=y_batch.shape[1]-5   
            #print("E.numClasses",numClasses)

            #print("go Go GO !")
            predictions=inference_batching(DRise.foo,self.model, masked_inputs,
                                                    repeated_targets,batch_size)
            #print("E.predictions",predictions)
            #print("YESSSS !")

            scores = DRise._compute_importance(predictions, masks)

            rise_maps = scores if rise_maps is None else tf.concat([rise_maps, scores], axis=0)
        #print("/!\\ STOP /!\\")
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
        #print ("E.input_shape",input_shape)
        masks = tf.image.random_crop(upsampled_masks, (nb_samples, *input_shape[:-1], 1))

        return masks

    @staticmethod
    @tf.function
    def _apply_masks(inputs: tf.Tensor, masks: tf.Tensor) -> tf.Tensor:
        """
        Given input samples and masks, apply it for every sample and repeat the labels.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        masks
            Masks with continuous value randomly generated.

        Returns
        -------
        occluded_inputs
            All the occluded combinations for each inputs.
        """
        occluded_inputs = tf.expand_dims(inputs, axis=1)
        occluded_inputs = tf.repeat(occluded_inputs, repeats=len(masks), axis=1)

        occluded_inputs = occluded_inputs * masks

        occluded_inputs = tf.reshape(occluded_inputs, (-1, *occluded_inputs.shape[2:]))

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

        # ponderate by the presence of each pixels, we could use a mean reducer to make it
        # faster, but only if the number of sample is large enough (as the sampling is iid)
        scores = tf.reduce_sum(weighted_scores, axis=1) / (tf.reduce_sum(masks, axis=0) +
                                                            DRise.EPSILON)

        return scores

    @staticmethod
#    @tf.function
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
