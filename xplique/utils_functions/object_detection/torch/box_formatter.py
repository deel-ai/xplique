"""
PyTorch box formatters for converting object detection model outputs to Xplique format.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import nn

from xplique.utils_functions.object_detection.base.box_formatter import (
    BaseBoxFormatter,
)
from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat,
    BoxType,
)
from xplique.utils_functions.object_detection.torch.box_manager import (
    TorchBoxCoordinatesTranslator,
)
from xplique.utils_functions.object_detection.torch.multi_box_tensor import (
    MultiBoxTensor,
)


class TorchBaseBoxFormatter(BaseBoxFormatter, nn.Module, ABC):
    """
    Abstract base class for PyTorch-based box formatters.

    This class combines BaseBoxFormatter functionality with PyTorch nn.Module
    capabilities, enabling gradient computation through box formatting operations.
    """

    def __init__(self,
                 input_box_type: BoxType,
                 output_box_type: BoxType = BoxType(
                     BoxFormat.XYXY, is_normalized=True)) -> None:
        """
        Initialize the PyTorch box formatter.

        Parameters
        ----------
        input_box_type
            Format of input bounding boxes (coordinate system and normalization).
        output_box_type
            Desired output format for bounding boxes. Defaults to normalized XYXY.
        """
        super().__init__(input_box_type=input_box_type, output_box_type=output_box_type)
        self.box_translator = TorchBoxCoordinatesTranslator(
            self.input_box_type, self.output_box_type)

    @abstractmethod
    def forward(self, predictions: Any) -> List[MultiBoxTensor]:
        """
        Transform model predictions into Xplique MultiBoxTensor format.

        Parameters
        ----------
        predictions
            Raw predictions from the object detection model.

        Returns
        -------
        formatted_predictions
            List of MultiBoxTensor objects, one per image in the batch.
        """
        raise NotImplementedError("This method should be implemented in the subclass")

    # needs boxes, scores, probas
    def format_predictions(
            self, predictions: Dict[str, torch.Tensor]) -> MultiBoxTensor:
        """
        Convert prediction dictionary to MultiBoxTensor format.

        Parameters
        ----------
        predictions
            Dictionary containing 'boxes', 'scores', and 'probas' keys.

        Returns
        -------
        formatted_tensor
            MultiBoxTensor with concatenated boxes, scores, and class probabilities.
        """
        boxes = predictions['boxes']
        boxes = self.box_translator.translate(boxes)
        probas = predictions['probas']
        scores = predictions['scores']
        return MultiBoxTensor(torch.cat([boxes,  # boxes coordinates
                                         scores,  # detection probability
                                         probas],  # class logits predictions for the given box
                                        dim=1))
