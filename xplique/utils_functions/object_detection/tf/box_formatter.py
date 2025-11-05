from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf

from xplique.utils_functions.object_detection.base.box_formatter import (
    BaseBoxFormatter,
)
from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat,
    BoxType,
)
from xplique.utils_functions.object_detection.tf.box_manager import (
    TfBoxCoordinatesTranslator,
)
from xplique.utils_functions.object_detection.tf.multi_box_tensor import MultiBoxTensor


class TfBaseBoxFormatter(BaseBoxFormatter, ABC):
    """
    TensorFlow implementation of the BaseBoxFormatter interface.

    Provides TensorFlow-specific box formatting and coordinate translation
    for object detection predictions. Handles conversion between different
    box formats and normalization states using TensorFlow operations.
    """

    def __init__(self,
                 input_box_type: BoxType,
                 output_box_type: BoxType = BoxType(
                     BoxFormat.XYXY, is_normalized=True)) -> None:
        """
        Initialize the TensorFlow box formatter.

        Parameters
        ----------
        input_box_type
            The format and normalization of input boxes.
        output_box_type
            The desired format and normalization for output boxes.
        """
        super().__init__(input_box_type=input_box_type, output_box_type=output_box_type)
        self.box_translator = TfBoxCoordinatesTranslator(self.input_box_type, self.output_box_type)

    @abstractmethod
    def forward(self, predictions):
        """
        Transform model predictions to MultiBoxTensor format.

        This abstract method must be implemented by subclasses to handle
        framework-specific prediction formats.

        Parameters
        ----------
        predictions
            Model predictions in framework-specific format.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    def format_predictions(self, predictions, input_image_size=None,
                           output_image_size=None) -> MultiBoxTensor:
        """
        Format detection predictions into unified MultiBoxTensor representation.

        Translates box coordinates to the desired format and concatenates with
        scores and class probabilities into a single tensor.

        Parameters
        ----------
        predictions
            Dictionary with 'boxes', 'scores', and 'probas' keys.
        input_image_size
            Optional size of the input image for denormalization.
        output_image_size
            Optional target size for coordinate scaling.

        Returns
        -------
        formatted_predictions
            MultiBoxTensor containing formatted predictions with shape (N, 4+1+num_classes).
        """
        boxes = predictions['boxes']
        boxes = self.box_translator.translate(boxes, input_image_size, output_image_size)
        probas = predictions['probas']
        scores = predictions['scores']
        return MultiBoxTensor(tf.concat([boxes,  # boxes coordinates
                                         scores,  # detection probability
                                         probas],  # class logits predictions for the given box
                                        axis=1))


class RetinaNetProcessedBoxFormatter(TfBaseBoxFormatter):
    """
    Box formatter for RetinaNet detection model predictions.

    Handles RetinaNet-specific output format with boxes, confidence scores,
    and class predictions, converting them to unified Xplique format.
    """

    def __init__(self, nb_classes: int,
                 input_box_type: BoxType = BoxType(BoxFormat.XYWH, is_normalized=False),
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),
                 input_image_size: tuple = None,
                 output_image_size: tuple = None,
                 **kwargs) -> None:
        """
        Initialize the RetinaNet box formatter.

        Parameters
        ----------
        nb_classes
            Number of object classes in the dataset.
        input_box_type
            Format of boxes from RetinaNet (default XYWH, unnormalized).
        output_box_type
            Desired output format (default XYXY, normalized).
        input_image_size
            Size of input image for coordinate conversion.
        output_image_size
            Target size for output coordinates.
        **kwargs
            Additional arguments passed to parent class.
        """
        super().__init__(input_box_type, output_box_type)

        self.nb_classes = nb_classes
        self.input_image_size = input_image_size
        self.output_image_size = output_image_size

    def forward(self, predictions) -> List[MultiBoxTensor]:
        """
        Process RetinaNet predictions handling both single and multi-batch inputs.

        Converts class IDs to one-hot encoding and formats boxes with scores
        and probabilities.

        Parameters
        ----------
        predictions
            Dictionary with 'boxes', 'confidence', and 'classes' keys.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        def process_single_batch(batch_idx):
            boxes = predictions['boxes'][batch_idx]
            scores = predictions['confidence'][batch_idx]
            classes = predictions['classes'][batch_idx]

            labels_one_hot = tf.one_hot(tf.cast(classes, tf.int32), depth=self.nb_classes)
            scores_expanded = scores[:, tf.newaxis]

            pred_dict = {
                'boxes': boxes,
                'scores': scores_expanded,
                'probas': labels_one_hot
            }
            return self.format_predictions(pred_dict, self.input_image_size, self.output_image_size)

        results = []
        for batch_idx in range(predictions['boxes'].shape[0]):
            formatted = process_single_batch(batch_idx)
            results.append(formatted)
        return results


class TfSsdBoxFormatter(TfBaseBoxFormatter):
    """
    Box formatter for SSD (Single Shot Detector) model predictions.

    Handles SSD-specific output format with coordinate swapping from
    (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax) format,
    and supports both processed and raw detection outputs.
    """

    def __init__(self, nb_classes: int,
                 input_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),
                 output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True)) -> None:
        """
        Initialize the SSD box formatter.

        Parameters
        ----------
        nb_classes
            Number of object classes in the dataset.
        input_box_type
            Format of boxes from SSD (default XYXY, normalized).
        output_box_type
            Desired output format (default XYXY, normalized).
        """
        super().__init__(input_box_type, output_box_type)
        self.nb_classes = nb_classes

    def forward_no_gradient_debug(self, predictions):
        """
        Process SSD predictions without gradient support (debug version).

        Converts detection boxes, scores, and classes to unified format with
        coordinate swapping. Uses eager execution for debugging.

        Parameters
        ----------
        predictions
            Dictionary with 'detection_boxes', 'detection_scores',
            and 'detection_classes' keys.

        Returns
        -------
        result
            Stacked tensor of formatted predictions.
        """
        results = []
        for boxes, scores, classes in zip(
                predictions['detection_boxes'], predictions['detection_scores'], predictions['detection_classes']):
            labels_one_hot = tf.one_hot(tf.cast(classes, tf.int32), depth=self.nb_classes)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': scores,
                'probas': labels_one_hot
            }
            formatted = self.format_predictions(pred_dict)
            results.append(formatted)
        result = tf.stack(results, axis=0)
        return result

    def forward_no_gradient(self, predictions):
        """
        Process SSD predictions without gradient support using tf.map_fn.

        Uses tf.map_fn for efficient batch processing without gradient tracking.
        Converts boxes to standard coordinate order and applies one-hot encoding.

        Parameters
        ----------
        predictions
            Dictionary with 'detection_boxes', 'detection_scores',
            and 'detection_classes' keys.

        Returns
        -------
        results
            Tensor of formatted predictions processed via map_fn.
        """
        def process_single_prediction(args):
            boxes, scores, classes = args
            labels_one_hot = tf.one_hot(tf.cast(classes, tf.int32), depth=self.nb_classes)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': scores,
                'probas': labels_one_hot
            }
            formatted = self.format_predictions(pred_dict)
            return formatted

        # Stack the inputs for tf.map_fn
        stacked_inputs = (
            predictions['detection_boxes'],
            predictions['detection_scores'],
            predictions['detection_classes']
        )

        # Use tf.map_fn to process each batch element
        results = tf.map_fn(
            process_single_prediction,
            stacked_inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            parallel_iterations=10
        )
        return results

    def forward(self, predictions) -> tf.Tensor:
        """
        Process raw SSD predictions with gradient support.

        Applies softmax to raw scores, swaps coordinate order, and formats
        boxes with computed detection scores and class probabilities.
        Supports gradient computation for attribution methods.

        Parameters
        ----------
        predictions
            Dictionary with 'raw_detection_boxes' and
            'raw_detection_scores' keys.

        Returns
        -------
        results
            Tensor of formatted predictions with gradient support.
        """
        def process_single_prediction(args):
            boxes, scores = args
            probas = tf.nn.softmax(scores, axis=-1)
            detection_scores = tf.reduce_max(probas, axis=-1, keepdims=True)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': detection_scores,
                'probas': probas
            }
            formatted = self.format_predictions(pred_dict)
            return formatted

        # Stack the inputs for tf.map_fn
        stacked_inputs = (
            predictions['raw_detection_boxes'],    # (1, 1917, 4)
            predictions['raw_detection_scores']    # (1, 1917, 91)
        )

        # Use tf.map_fn to process each batch element
        results = tf.map_fn(
            process_single_prediction,
            stacked_inputs,
            fn_output_signature=tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            parallel_iterations=10
        )
        return results

    # Forward with raw results
    def forward_debug(self, predictions):
        """
        Process raw SSD predictions for debugging without tf.map_fn.

        Similar to forward() but uses Python loop instead of tf.map_fn
        for easier debugging and inspection of intermediate results.

        Parameters
        ----------
        predictions
            Dictionary with 'raw_detection_boxes' and
            'raw_detection_scores' keys.

        Returns
        -------
        results
            Stacked tensor of formatted predictions.
        """
        def process_single_prediction(args):
            boxes, scores = args
            probas = tf.nn.softmax(scores, axis=-1)
            detection_scores = tf.reduce_max(probas, axis=-1, keepdims=True)
            # swap coordinates from (ymin, xmin, ymax, xmax) to (xmin, ymin, xmax, ymax)
            boxes = tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)
            pred_dict = {
                'boxes': boxes,
                'scores': detection_scores,
                'probas': probas
            }
            formatted = self.format_predictions(pred_dict)
            return formatted

        # Stack the inputs for tf.map_fn
        stacked_inputs = (
            predictions['raw_detection_boxes'],    # (1, 1917, 4)
            predictions['raw_detection_scores']    # (1, 1917, 91)
        )

        results = []
        for args in zip(*stacked_inputs):
            formatted = process_single_prediction(args)
            results.append(formatted)
        results = tf.stack(results, axis=0)
        return results


class YoloGradientsTfBoxFormatter(TfBaseBoxFormatter):
    """
    Box formatter for YOLO model predictions with gradient support.

    Handles YOLO-specific center-based box format (CXCYWH) and extracts
    class probabilities and scores from concatenated prediction tensors.
    Designed to work with gradient-based attribution methods.
    """

    def __init__(self) -> None:
        """
        Initialize the YOLO box formatter with center-based normalized format.
        """
        super().__init__(input_box_type=BoxType(BoxFormat.CXCYWH, is_normalized=True))

    def forward(self, predictions) -> MultiBoxTensor:
        """
        Process YOLO predictions with automatic batch handling.

        Transposes predictions, separates boxes from class predictions,
        and computes objectness scores from class probabilities.
        Handles both single and batched inputs.

        Parameters
        ----------
        predictions
            Tensor of shape (batch, features, num_boxes) or
            (1, features, num_boxes) containing YOLO predictions.

        Returns
        -------
        formatted
            MultiBoxTensor with formatted box predictions.
        """
        if predictions.shape[0] == 1:
            predictions = tf.squeeze(predictions, axis=0)
        predictions = tf.transpose(predictions)
        boxes = predictions[:, :4]
        probas = predictions[:, 4:]  # sigmoid result
        scores = tf.reduce_max(probas, axis=-1)
        scores = tf.expand_dims(scores, axis=-1)
        pred_dict = {
            'boxes': boxes,
            'scores': scores,
            'probas': probas
        }
        formatted = self.format_predictions(pred_dict)  # needs boxes, scores, probas
        return formatted
