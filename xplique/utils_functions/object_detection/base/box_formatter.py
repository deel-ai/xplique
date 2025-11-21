from abc import ABC, abstractmethod
from typing import Any

from .box_manager import BoxFormat, BoxType
from .multi_box_tensor import MultiBoxTensor


class BaseBoxFormatter(ABC):
    """
    Abstract base class for formatting object detection predictions.

    This class provides a common interface for converting model predictions into
    a standardized format with configurable box coordinate representations.

    Parameters
    ----------
    input_box_type
        The box type (format and normalization) of the input predictions.
    output_box_type
        The desired box type (format and normalization) for the output.
        Default is XYXY format with normalized coordinates.
    """

    def __init__(
            self,
            input_box_type: BoxType,
            output_box_type: BoxType = BoxType(
                BoxFormat.XYXY, is_normalized=True)) -> None:
        super().__init__()
        self.input_box_type = input_box_type
        self.output_box_type = output_box_type

    def __call__(self, predictions: Any) -> MultiBoxTensor:
        """
        Callable interface for formatting predictions.

        Parameters
        ----------
        predictions
            Raw model predictions to format.

        Returns
        -------
        formatted_predictions
            Formatted predictions in the standardized format.
        """
        return self.forward(predictions)

    @abstractmethod
    def forward(self, predictions: Any) -> MultiBoxTensor:
        """
        Forward pass to format predictions.

        This method must be implemented by framework-specific subclasses.

        Parameters
        ----------
        predictions
            Raw model predictions to format.

        Returns
        -------
        formatted_predictions
            Formatted predictions in the standardized format.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    @abstractmethod
    def format_predictions(self, predictions: Any) -> MultiBoxTensor:
        """
        Format predictions into the standardized MultiBoxTensor format.

        This method must be implemented by framework-specific subclasses.

        Parameters
        ----------
        predictions
            Dictionary or tensor containing boxes, scores, and class probabilities.

        Returns
        -------
        MultiBoxTensor
            Formatted predictions with boxes, scores, and class probabilities.
        """
        raise NotImplementedError("This method should be implemented in the subclass")
