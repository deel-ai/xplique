"""
TensorFlow model wrappers for object detection models with box formatting.
"""

from typing import Any, List, Union

import tensorflow as tf

from xplique.utils_functions.object_detection.tf.box_formatter import TfBaseBoxFormatter
from xplique.utils_functions.object_detection.tf.multi_box_tensor import MultiBoxTensor


class TfBoxesModelWrapper(tf.keras.Model):
    """
    Wrapper for TensorFlow object detection models with box formatting capabilities.

    This class wraps an object detection model and applies a box formatter to its outputs.
    It can return predictions either as a list of formatted boxes (one per image) or as
    a single stacked tensor.
    """

    def __init__(self, model: Any, box_formatter: TfBaseBoxFormatter) -> None:
        """
        Initialize the TensorFlow box model wrapper.

        Parameters
        ----------
        model
            TensorFlow object detection model to wrap.
        box_formatter
            Formatter to process and convert model predictions to Xplique format.
        """
        super().__init__()
        self.model = model
        self.box_formatter = box_formatter
        self.output_as_list = True

    def set_output_as_list(self) -> None:
        """
        Configure the wrapper to return predictions as a list.

        Sets the output format to return a list of predictions, one entry per image in the batch.
        """
        self.output_as_list = True

    def set_output_as_tensor(self) -> None:
        """
        Configure the wrapper to return predictions as a stacked tensor.

        Sets the output format to return predictions as a single stacked tensor instead of a list.
        """
        self.output_as_list = False

    def call(self, x, **kwargs) -> Union[tf.Tensor, List[MultiBoxTensor]]:
        """
        Forward pass through the wrapped model with box formatting.

        Processes input through the object detection model and formats the predictions
        using the box formatter. Returns either a list or stacked tensor based on the
        output_as_list flag.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, height, width, channels).
        **kwargs
            Additional keyword arguments to pass to the model.

        Returns
        -------
        predictions
            If output_as_list is True: List of MultiBoxTensor objects, one per image.
            If output_as_list is False: Stacked tensor of formatted predictions with shape
            (batch_size, ...).
        """
        predictions = self.model(x, **kwargs)
        list_of_predictions = self.box_formatter(predictions)
        if self.output_as_list:
            return list_of_predictions
        concatenated = tf.stack(list_of_predictions, axis=0)
        return concatenated
