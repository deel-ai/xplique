"""
TensorFlow implementation of box formatter for object detection predictions.

This module provides TensorFlow-specific box formatting and coordinate translation
for object detection model outputs, handling conversion between different box formats
and normalization states.
"""

from abc import ABC, abstractmethod

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
from xplique.utils_functions.object_detection.tf.multi_box_tensor import TfMultiBoxTensor


class TfBaseBoxFormatter(BaseBoxFormatter, ABC):
    """
    TensorFlow implementation of the BaseBoxFormatter interface.

    Provides TensorFlow-specific box formatting and coordinate translation
    for object detection predictions. Handles conversion between different
    box formats and normalization states using TensorFlow operations.
    """

    def __init__(
        self,
        input_box_type: BoxType,
        output_box_type: BoxType = BoxType(BoxFormat.XYXY, is_normalized=True),
    ) -> None:
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

    def format_predictions(self, predictions, image_size=None) -> TfMultiBoxTensor:
        """
        Format detection predictions into unified MultiBoxTensor representation.

        Translates box coordinates to the desired format and concatenates with
        scores and class probabilities into a single tensor.

        Parameters
        ----------
        predictions
            Dictionary with 'boxes', 'scores', and 'probas' keys.
        image_size
            Optional size of the image for coordinate conversion.

        Returns
        -------
        formatted_predictions
            MultiBoxTensor containing formatted predictions with shape (N, 4+1+num_classes).
        """
        boxes = predictions["boxes"]
        boxes = self.box_translator.translate(boxes, image_size=image_size)
        probas = predictions["probas"]
        scores = predictions["scores"]
        return TfMultiBoxTensor(
            tf.concat(
                [
                    boxes,  # boxes coordinates
                    scores,  # detection probability
                    probas,
                ],  # class logits predictions for the given box
                axis=1,
            )
        )
