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
