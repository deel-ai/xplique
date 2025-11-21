from abc import ABC
from typing import Any, List, Union

import torch

from xplique.utils_functions.object_detection.torch.box_formatter import (
    DetrBoxFormatter,
    MultiBoxTensor,
    TorchvisionBoxFormatter,
    TorchBaseBoxFormatter,
    YoloRawBoxFormatter,
    YoloResultBoxFormatter,
)
from xplique.wrappers.pytorch import TorchWrapper


class TorchBoxesModelWrapper(torch.nn.Module, ABC):
    """
    Wrapper for PyTorch object detection models with box formatting capabilities.

    This class wraps an object detection model and applies a box formatter to its outputs.
    It can return predictions either as a list of formatted boxes (one per image) or as
    a single stacked tensor.
    """

    def __init__(
            self, model: Any, box_formatter: TorchBaseBoxFormatter) -> None:
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
            self, x: torch.Tensor) -> Union[torch.Tensor, List[MultiBoxTensor]]:
        return self.forward(x)

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

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[MultiBoxTensor]]:
        """
        Forward pass through the wrapped model with box formatting.

        Processes input through the object detection model and formats the predictions
        using the box formatter. Returns either a list or stacked tensor based on the
        output_as_list flag.

        Parameters
        ----------
        x
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        predictions
            If output_as_list is True: List of MultiBoxTensor objects, one per image.
            If output_as_list is False: Stacked tensor of formatted predictions with shape
            (batch_size, ...).
        """
        predictions = self.model(x)
        list_of_predictions = self.box_formatter(predictions)
        if self.output_as_list:
            return list_of_predictions
        else:
            concatenated = torch.stack(list_of_predictions, dim=0)
            return concatenated


# class XpliqueTorchBoxesModelWrapper(torch.nn.Module, ABC):
#     """
#     Wrapper for PyTorch models with Xplique-specific integration for box formatting.

#     This class combines PyTorch model wrapping with box formatting, automatically
#     handling device management and channel ordering for Xplique compatibility.
#     """

#     def __init__(
#             self,
#             model: Any,
#             box_formatter: TorchBaseBoxFormatter,
#             device: str = 'cpu',
#             is_channel_first: bool = True) -> None:
#         """
#         Initialize the Xplique-integrated PyTorch box model wrapper.

#         Parameters
#         ----------
#         model
#             PyTorch object detection model to wrap.
#         box_formatter
#             Formatter to process and convert model predictions to Xplique format.
#         device
#             Device to run the model on ('cpu' or 'cuda').
#         is_channel_first
#             Whether input tensors have channels-first format (NCHW vs NHWC).
#         """
#         super().__init__()
#         self.model = TorchWrapper(model.eval(), device=device, is_channel_first=is_channel_first)
#         self.box_formatter = box_formatter

#     def __call__(
#             self, x: torch.Tensor) -> Union[torch.Tensor, List[MultiBoxTensor]]:
#         return self.forward(x)

#     def forward(self, x: torch.Tensor) -> Union[torch.Tensor, List[MultiBoxTensor]]:
#         """
#         Forward pass through the Xplique-wrapped model with box formatting.

#         Parameters
#         ----------
#         x
#             Input tensor of shape (batch_size, channels, height, width) or
#             (batch_size, height, width, channels) depending on is_channel_first.

#         Returns
#         -------
#         predictions
#             Stacked tensor of formatted predictions with shape (batch_size, ...).
#         """
#         predictions = self.model(x)
#         return torch.stack([self.box_formatter(pred) for pred in predictions], dim=0)


class YoloResultBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for YOLO models that return Results objects.

    This class extends TorchBoxesModelWrapper with YOLO-specific box formatting,
    automatically using YoloResultBoxFormatter for predictions processing.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the YOLO result box model wrapper.

        Parameters
        ----------
        model
            PyTorch YOLO object detection model that returns Results objects.
        """
        box_formatter = YoloResultBoxFormatter()
        super().__init__(model, box_formatter=box_formatter)


class YoloRawBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for YOLO models that return raw tensor outputs.

    This class extends TorchBoxesModelWrapper with YOLO-specific box formatting,
    automatically using YoloRawBoxFormatter for predictions processing.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the YOLO raw output box model wrapper.

        Parameters
        ----------
        model
            PyTorch YOLO object detection model that returns raw tensors.
        """
        box_formatter = YoloRawBoxFormatter()
        super().__init__(model, box_formatter=box_formatter)


class TorchvisionBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for Torchvision object detection models.

    This class extends TorchBoxesModelWrapper with Torchvision-specific box formatting,
    automatically using TorchvisionBoxFormatter for predictions processing.
    """

    def __init__(self, model: Any, nb_classes: int) -> None:
        """
        Initialize the Torchvision box model wrapper.

        Parameters
        ----------
        model
            Torchvision object detection model (e.g., FCOS, RetinaNet).
        nb_classes
            Total number of object classes in the detection model.
        """
        box_formatter = TorchvisionBoxFormatter(nb_classes=nb_classes)
        super().__init__(model, box_formatter=box_formatter)


class DetrBoxesModelWrapper(TorchBoxesModelWrapper):
    """
    Specialized wrapper for DETR object detection models.

    This class extends TorchBoxesModelWrapper with DETR-specific box formatting,
    automatically using DetrBoxFormatter for predictions processing.
    """

    def __init__(self, model: Any) -> None:
        """
        Initialize the DETR box model wrapper.

        Parameters
        ----------
        model
            DETR object detection model.
        """
        box_formatter = DetrBoxFormatter()
        super().__init__(model, box_formatter=box_formatter)
