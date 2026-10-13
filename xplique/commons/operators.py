"""
Custom tensorflow operator for Attributions
"""

import tensorflow as tf
from deprecated import deprecated

from ..types import Callable, Optional
from ..utils_functions.object_detection import _EPSILON, _box_iou, _format_objects


@tf.function
def predictions_operator(model: Callable, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """
    Compute predictions scores, only for the label class, for a batch of samples.

    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

    Returns
    -------
    scores
        Predictions scores computed, only for the label class.
    """
    scores = tf.reduce_sum(model(inputs) * targets, axis=-1)
    return scores


@tf.function
@deprecated(version="1.0.0", reason="Gradient-based explanations are zeros with this operator.")
def regression_operator(model: Callable, inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
    """
    Compute the the mean absolute error between model prediction and the target.
    Target should the model prediction on non-perturbed input.
    This operator can be used to compute attributions for all outputs of a regression model.

    Parameters
    ----------
    model
        Model used for computing predictions.
    inputs
        Input samples to be explained.
    targets
        Model prediction on non-perturbed inputs.

    Returns
    -------
    scores
        MAE between model prediction and targets.
    """
    scores = tf.reduce_mean(tf.abs(model(inputs) - targets), axis=-1)
    return scores


@tf.function
def semantic_segmentation_operator(model, inputs, targets):
    """
    Explain the class of a zone of interest.

    Parameters
    ----------
    model
        Model used for computing predictions.
        The model outputs should be between 0 and 1, otherwise, applying a softmax is recommended.
    inputs
        Input samples to be explained.
        Expected shape of (n, h, w, c_in), with c_in the number of channels of the input.
    targets
        Tensor, a mask indicating the zone and class to explain.
        It contains the model predictions limited to a certain zone and channel.
        The zone indicates the zone of interest and the channel the class of interest.
        For more detail and examples please refer to the documentation.
        https://deel-ai.github.io/xplique/latest/api/attributions/semantic_segmentation/
        Expected shape of (n, h, w, c_out), with c_out the number of classes.
        `targets` can also be designed to explain the border of a zone of interest.

    Returns
    -------
    scores
        Segmentation scores computed.
    """
    # compute absolute difference between prediction and target on targets zone
    scores = model(inputs) * targets

    # take mean over the zone and channel of interest
    return tf.reduce_sum(scores, axis=(1, 2, 3)) / tf.reduce_sum(
        tf.cast(tf.not_equal(targets, 0), tf.float32), axis=(1, 2, 3)
    )


@tf.function
def object_detection_operator(
    model: Callable,
    inputs: tf.Tensor,
    targets: tf.Tensor,
    intersection_score_fn: Optional[Callable] = _box_iou,
    include_detection_probability: Optional[bool] = True,
    include_classification_score: Optional[bool] = True,
) -> tf.Tensor:
    """
    Compute the object detection scores for a batch of samples.

    For a given image, there are two possibilities:
        - One box per image is provided: Then, in the case of perturbation-based methods,
        the model makes prediction on the perturbed image and choose the most similar predicted box.
        This similarity is computed following the DRise method.
        In the case of gradient-based methods, the gradient is computed from the same score.
        - Several boxes are provided for one image: The largest matching score for each
        reference box is computed, then these scores are averaged and attributed.

    Therefore, to explain each box separately, the easiest way is to call the attribution method
    with a batch of the same image tiled to match the number of predicted box.
    In this case, inputs and targets shapes should be: (nb_boxes, H, W, C) and (nb_boxes, (5 + nc)).

    This work is a generalization of the following article at any kind of attribution method.
    Ref. Petsiuk & al., Black-box Explanation of Object Detectors via Saliency Maps (2021).
    https://arxiv.org/pdf/2006.03204.pdf

    Parameters
    ----------
    model
        Model used for computing object detection prediction.
        The model should have input and output shapes of (N, H, W, C) and (N, nb_boxes, (4+1+nc)).
        Black-box methods can use postprocessed predictions, including NMS. Pre-NMS
        outputs provide more candidates for matching. Gradient-based methods require
        a prediction path connected to inputs; formatting does not restore lost gradients.
    inputs
        Batched input samples to be explained. Expected shape (N, H, W, C).
        More information in the documentation.
    targets
        Specify the reference box or boxes to explain for each input. Select them once
        on the unperturbed inputs, optionally after NMS, and keep them fixed.
        It should be of shape (N, (4 + 1 + nc)) or (N, nb_boxes, (4 + 1 + nc)),
        with nc the number of classes,
        N the number of samples in the batch (it should match `inputs`),
        and nb_boxes the number of boxes to explain simultaneously.

        (4 + 1 + nc) means: [XYXY_coordinates, detection_confidence, class_scores].
        Predictions and references must share coordinate conventions and class indexing.
        Class scores may be soft probabilities or one-hot labels; one-hot labels do not
        expose soft class gradients. Detection confidence is adapter-dependent.

        In the case the nb_boxes dimension is not 1,
        several boxes will be explained at the same time.
        The operator returns the mean of the best matching scores, not independently
        computed attribution maps.
    intersection_score_fn
        Function that computes the intersection score between two bounding boxes coordinates.
        This function is batched. The default value is `_box_iou` computing IOU scores.
    include_detection_probability
        Boolean encoding if the box objectness (or detection probability)
        should be included in DRise score.
    include_classification_score
        Boolean encoding if the class associated to the box should be included in DRise score.

    Returns
    -------
    scores
        Object detection scores computed following DRise definition:
        intersection_score * proba_detection * classification_similarity
        Shape (N,). An image with no valid predictions or references receives zero.

    Notes
    -----
    All-zero rows in predictions and references are ignored as padding. Detection
    wrappers from the optional ``xplique-adapters`` package return lists by default;
    set ``output_as_tensor = True`` for this operator. PyTorch detectors additionally
    use ``xplique.wrappers.TorchWrapper`` to bridge the shared attribution API.
    """

    def batch_loop(args):
        # function to loop on for `tf.map_fn`
        obj, obj_ref = args

        # Reshape instead of branching so TensorFlow retains a statically known rank.
        obj_ref = tf.reshape(obj_ref, (-1, tf.shape(obj_ref)[-1]))
        obj_ref = tf.ensure_shape(obj_ref, [None, None])

        # Tensor-mode detection outputs are zero-padded to equalize batch shapes.
        obj = tf.boolean_mask(obj, tf.reduce_any(tf.not_equal(obj, 0), axis=-1))
        obj_ref = tf.boolean_mask(obj_ref, tf.reduce_any(tf.not_equal(obj_ref, 0), axis=-1))

        def compute_score():
            # compute predicted boxes for a given image
            # (nb_box_pred, 4), (nb_box_pred, 1), (nb_box_pred, nb_classes)
            current_boxes, proba_detection, classification = _format_objects(obj)
            size = tf.shape(current_boxes)[0]

            # DRise consider the reference objectness to be 1
            # (nb_box_ref, 4), _, (nb_box_ref, nb_classes)
            boxes_refs, _, class_refs = _format_objects(obj_ref)

            # (nb_box_ref, nb_box_pred, 4)
            boxes_refs = tf.repeat(tf.expand_dims(boxes_refs, axis=1), repeats=size, axis=1)

            # (nb_box_ref, nb_box_pred)
            intersection_score = intersection_score_fn(boxes_refs, current_boxes)

            # (nb_box_pred,)
            detection_probability = tf.squeeze(proba_detection, axis=1)

            # set detection probability to 1 if it should be included
            detection_probability = tf.cond(
                tf.cast(include_detection_probability, tf.bool),
                true_fn=lambda: detection_probability,
                false_fn=lambda: tf.ones_like(detection_probability),
            )

            # (nb_box_ref, nb_box_pred, nb_classes)
            class_refs = tf.repeat(tf.expand_dims(class_refs, axis=1), repeats=size, axis=1)

            # (nb_box_ref, nb_box_pred)
            classification_score = tf.reduce_sum(class_refs * classification, axis=-1) / (
                tf.norm(classification, axis=-1) * tf.norm(class_refs, axis=-1) + _EPSILON
            )

            # set classification score to 1 if it should be included
            classification_score = tf.cond(
                tf.cast(include_classification_score, tf.bool),
                true_fn=lambda: classification_score,
                false_fn=lambda: tf.ones_like(classification_score),
            )

            # Compute score as defined in DRise for all possible pair of boxes
            # (nb_box_ref, nb_box_pred)
            boxes_pairwise_scores = (
                intersection_score * detection_probability * classification_score
            )

            # select for a reference box the most similar predicted box score
            # (nb_box_ref,)
            ref_boxes_scores = tf.reduce_max(boxes_pairwise_scores, axis=1)

            # get an attribution for several boxes in the same time
            return tf.reduce_mean(ref_boxes_scores)

        return tf.cond(
            tf.logical_and(tf.shape(obj)[0] > 0, tf.shape(obj_ref)[0] > 0),
            compute_score,
            lambda: tf.constant(0.0, dtype=inputs.dtype),
        )

    objects = model(inputs)
    return tf.map_fn(batch_loop, (objects, targets), fn_output_signature=tf.float32)
