"""
Module related to Object detector method
"""

from typing import Iterable, Tuple, Union, Optional
import abc

import tensorflow as tf
import numpy as np

from xplique.attributions.base import BlackBoxExplainer
from xplique.commons import operator_batching


class IouCalculator:
    """
    Used to compute the Intersection Over Union (IOU).
    """
    @abc.abstractmethod
    def intersect(self, objects_a: tf.Tensor, objects_b: tf.Tensor) -> tf.Tensor:
        """
        Compute the intersection between two batched objects (e.g boxes, segmentation masks...)

        Parameters
        ----------
        objects_a
            First batch of objects to compare.
        objects_b
            Second batch of objects to compare.

        Returns
        -------
        score
            Real value between [0,1] corresponding to the intersection of the 2 objects.
        """
        raise NotImplementedError()


class SegmentationIouCalculator(IouCalculator):
    """
    Compute segmentation masks IOU.
    """
    def intersect(self, masks_a: tf.Tensor, masks_b: tf.Tensor) -> tf.Tensor:
        """
        Compute the intersection between two batched segmentation masks.
        Each segmentation is a boolean mask on the whole image

        Parameters
        ----------
        masks_a
            First batch of segmentation masks.
        masks_b
            Second batch of segmentation masks.

        Returns
        -------
        iou_score
            The IOU score between the first and second batch of masks.
        """
        # pylint: disable=W0221,W0237

        axis = np.arange(1, tf.rank(masks_a))

        inter_area = tf.reduce_sum(tf.cast(tf.logical_and(masks_a, masks_b),
                                    dtype=tf.float32), axis=axis)
        union_area = tf.reduce_sum(tf.cast(tf.logical_or(masks_a, masks_b),
                                    dtype=tf.float32), axis=axis)

        iou_score = inter_area / tf.maximum(union_area, 1.0)

        return iou_score



class BoxIouCalculator(IouCalculator):
    """
    Used to compute the Bounding Box IOU.
    """
    EPSILON = tf.constant(1e-4)

    def intersect(self, boxes_a: tf.Tensor, boxes_b: tf.Tensor) -> tf.Tensor:
        """
        Compute the intersection between two batched bounding boxes.
        Each bounding box is defined by (x1, y1, x2, y2) respectively (left, bottom, right, top).

        Parameters
        ----------
        boxes_a
            First batch of bounding boxes.
        boxes_b
            Second batch of bounding boxes.

        Returns
        -------
        iou_score
           The IOU score between the first and second batch of bounding boxes.
        """
        # pylint: disable=W0221,W0237

        # determine the intersection rectangle
        left   = tf.maximum(boxes_a[..., 0], boxes_b[..., 0])
        bottom = tf.maximum(boxes_a[..., 1], boxes_b[..., 1])
        right  = tf.minimum(boxes_a[..., 2], boxes_b[..., 2])
        top    = tf.minimum(boxes_a[..., 3], boxes_b[..., 3])

        intersection_area = tf.math.maximum(right - left, 0) * tf.math.maximum(top - bottom, 0)

        # determine the areas of the prediction and ground-truth rectangles
        a_area = (boxes_a[..., 2] - boxes_a[..., 0]) * (boxes_a[..., 3] - boxes_a[..., 1])
        b_area = (boxes_b[..., 2] - boxes_b[..., 0]) * (boxes_b[..., 3] - boxes_b[..., 1])

        union_area = a_area + b_area - intersection_area

        iou_score = intersection_area / (union_area + BoxIouCalculator.EPSILON)

        return iou_score


class IObjectFormater:
    """
    Generic class to format the model prediction
    """
    def format_objects(self, predictions) -> Iterable[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        """
        Format the model prediction of a given image to have the prediction of the following format:
        objects, proba_detection, one_hots_classifications

        Parameters
        ----------
        predictions
            prediction of the model of a given image

        Returns
        -------
        object
            bounding box or mask component of the prediction
        proba
            existence probability component of the prediction
        classification
            classification component of the prediction
        """
        raise NotImplementedError()


class ImageObjectDetectorScoreCalculator:
    """
    Class to compute batch score
    """
    def __init__(self, object_formater: IObjectFormater, iou_calculator: IouCalculator):
        self.object_formater = object_formater
        self.iou_calculator = iou_calculator

        self.batch_score = operator_batching(self.score)

    def score(self, model, inp, object_ref) -> tf.Tensor:
        """
        Compute the matching score between prediction and a given object

        Parameters
        ----------
        model
            the model used for the object detection
        inp
            the batched image
        object_ref
            the object target to compare with the prediction of the model

        Returns
        -------
        score
            for each image, the matching score between the object of reference and
            the prediction of the model
        """
        objects = model(inp)
        score_values = []
        for obj, obj_ref in zip(objects, object_ref):
            if obj is None or obj.shape[0] == 0:
                score_values.append(tf.constant(0.0, dtype=inp.dtype))
            else:
                current_boxes, proba_detection, classification = \
                            self.object_formater.format_objects(obj)

                if len(tf.shape(obj_ref)) == 1:
                    obj_ref = tf.expand_dims(obj_ref, axis=0)

                obj_ref = self.object_formater.format_objects(obj_ref)

                scores = []
                size = tf.shape(current_boxes)[0]
                for boxes_ref, proba_ref, class_ref in zip(*obj_ref):
                    boxes_ref = tf.repeat(tf.expand_dims(boxes_ref, axis=0), repeats=size, axis=0)
                    proba_ref = tf.repeat(tf.expand_dims(proba_ref, axis=0), repeats=size, axis=0)
                    class_ref = tf.repeat(tf.expand_dims(class_ref, axis=0), repeats=size, axis=0)

                    iou = self.iou_calculator.intersect(boxes_ref, current_boxes)
                    classification_similarity = tf.reduce_sum(class_ref * classification, axis=1) \
                            / (tf.norm(classification, axis=1) * tf.norm(class_ref, axis=1))

                    current_score = iou * tf.squeeze(proba_detection, axis=1) \
                                        * classification_similarity
                    current_score = tf.reduce_max(current_score)
                    scores.append(current_score)

                score_value = tf.reduce_max(tf.stack(scores))
                score_values.append(score_value)

        score_values = tf.stack(score_values)

        return score_values


class ImageObjectDetectorExplainer(BlackBoxExplainer):
    """
    Used to define method as an object detector one
    """

    def __init__(self, explainer: BlackBoxExplainer, object_detector_formater: IObjectFormater,
                 iou_calculator: IouCalculator):
        """
        Constructor

        Parameters
        ----------
        explainer
            the black box explainer used to explain the object detector model
        object_detector_formater
            the formater of the object detector model used to format the prediction
            of the right format
        iou_calculator
            the iou calculator used to compare two objects.
        """
        super().__init__(explainer.model, explainer.batch_size)
        self.explainer = explainer
        self.score_calculator = ImageObjectDetectorScoreCalculator(object_detector_formater,
                                                                    iou_calculator)
        self.explainer.inference_function = self.score_calculator.score
        self.explainer.batch_inference_function = self.score_calculator.batch_score

    def explain(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.array],
                targets: Optional[Union[tf.Tensor, np.array]] = None) -> tf.Tensor:
        if len(tf.shape(targets)) == 1:
            targets = tf.expand_dims(targets, axis=0)

        return self.explainer.explain(inputs, targets)


class BoundingBoxesExplainer(ImageObjectDetectorExplainer, IObjectFormater):
    """
    For a given black box explainer, this class allows to find explications of an object detector
    model. The object model detector shall return a list (length of the size of the batch)
    containing a tensor of 2 dimensions.
    The first dimension of the tensor is the number of bounding boxes found in the image
    The second dimension is:
    [x1_box, y1_box, x2_box, y2_box, probability_detection, ones_hot_classif_result]

    This work is a generalisation of the following article at any kind of black box explainer and
    also can be used for other kind of object detector (like segmentation)

    Ref. Petsiuk & al., Black-box Explanation of Object Detectors via Saliency Maps (2021).
    https://arxiv.org/pdf/2006.03204.pdf
    """

    def __init__(self, explainer: BlackBoxExplainer):
        super().__init__(explainer, self, BoxIouCalculator())

    def format_objects(self, predictions) -> Iterable[Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
        boxes, proba_detection, one_hots_classifications = tf.split(predictions,
                                                    [4, 1, tf.shape(predictions[0])[0] - 5], 1)
        return boxes, proba_detection, one_hots_classifications
