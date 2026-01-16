"""PyTorch wrapper for object detection models with box formatting capabilities."""
from abc import ABC
from typing import Any, List, Union

import torch

from xplique.utils_functions.object_detection.torch.box_formatter import (
    MultiBoxTensor,
    TorchBaseBoxFormatter,
)


class TorchBoxesModelWrapper(torch.nn.Module, ABC):
    """
    Wrapper for PyTorch object detection models with box formatting capabilities.

    This class wraps an object detection model and applies a box formatter to its outputs.
    It can return predictions either as a list of formatted boxes (one per image) or as
    a single stacked tensor.
    """

    def __init__(self, model: Any, box_formatter: TorchBaseBoxFormatter) -> None:
        """
        Initialize the PyTorch box model wrapper.

        Parameters
        ----------
        model
            PyTorch object detection model to wrap.
        box_formatter
            Formatter to process and convert model predictions to Xplique format.
        """
        super().__init__()
        self.model = model
        self.box_formatter = box_formatter
        self.output_as_list = True

    def __call__(
            self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[MultiBoxTensor]]:
        return self.forward(x, **kwargs)

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

    def forward(self, x: torch.Tensor, **kwargs) -> Union[torch.Tensor, List[MultiBoxTensor]]:
        """
        Forward pass through the wrapped model with box formatting.

        Processes input through the object detection model and formats the predictions
        using the box formatter. Returns either a list or stacked tensor based on the
        output_as_list flag.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, channels, height, width).
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
        concatenated = torch.stack(list_of_predictions, dim=0)
        return concatenated
