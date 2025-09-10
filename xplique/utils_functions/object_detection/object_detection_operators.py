"""
Operator for object detection
"""
from typing import Tuple
import tensorflow as tf

_EPSILON = 1e-4


def _box_iou(boxes_a: tf.Tensor, boxes_b: tf.Tensor) -> tf.Tensor:
    """
    Compute the intersection between two batched bounding boxes.
    Each bounding box is defined by (x1, y1, x2, y2) respectively (left, bottom, right, top).
    With left < right and bottom < top

    Parameters
    ----------
    boxes_a
        First batch of bounding boxes.
    boxes_b
        Second batch of bounding boxes.

    Returns
    -------
    iou_score
        The IOU score between the two batches of bounding boxes.
    """

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

    iou_score = intersection_area / (union_area + _EPSILON)

    return iou_score


def _format_objects(predictions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Format bounding boxes prediction for object detection operator.
    It takes a batch of bounding boxes predictions of the model and divide it between
    boxes_coordinates, proba_detection, and one_hots_classifications.

    Parameters
    ----------
    predictions
        Batch of bounding boxes predictions of shape (nb_boxes, (4 + 1 + nc)).
        (4 + 1 + nc) means: [boxes_coordinates, proba_detection, one_hots_classifications].
        Where nc is the number of classes.

    Returns
    -------
    boxes_coordinates
        A Tensor of shape (nb_boxes, 4) encoding the boxes coordinates.
    proba_detection
        A Tensor of shape (nb_boxes, 1) encoding the detection probabilities.
    one_hots_classifications
        A Tensor of shape (nb_boxes, nc) encoding the class predictions.
    """
    boxes_coordinates, proba_detection, one_hots_classifications = \
        tf.split(predictions, [4, 1, tf.shape(predictions[0])[0] - 5], 1)
    return boxes_coordinates, proba_detection, one_hots_classifications
