"""
PyTorch-specific box management utilities for coordinate transformations.
"""

from typing import Optional, Tuple

import torch

from xplique.utils_functions.object_detection.base.box_manager import (
    BoxFormat,
    BoxManager,
    BoxType,
)


class TorchBoxManager(BoxManager):
    """
    PyTorch implementation of box management for bounding box operations.

    This class provides static methods for converting between different bounding box
    coordinate formats and handling normalization/denormalization using PyTorch tensors.
    """

    def __init__(self, format: BoxFormat, normalized: bool) -> None:
        """
        Initialize the PyTorch box manager.

        Parameters
        ----------
        format
            Coordinate format of the boxes (XYXY, CXCYWH, or XYWH).
        normalized
            Whether box coordinates are normalized to [0,1] range.
        """
        self.format = format
        self.normalized = normalized

    @staticmethod
    def normalize_boxes(
            raw_boxes: torch.Tensor,
            image_source_size: torch.Size) -> torch.Tensor:
        """
        Normalize bounding box coordinates to [0,1] range based on image size.

        Parameters
        ----------
        raw_boxes
            Boxes in pixel coordinates with shape (N, 4).
        image_source_size
            Image dimensions as (width, height).

        Returns
        -------
        normalized_boxes
            Normalized boxes with coordinates in [0,1] range.
        """
        [sx, sy] = image_source_size
        normalized_boxes = raw_boxes.clone()
        normalized_boxes[:, [0, 2]] /= sx
        normalized_boxes[:, [1, 3]] /= sy
        return normalized_boxes

    @staticmethod
    def box_cxcywh_to_xyxy(normalized_boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from CXCYWH to corner format XYXY.

        Parameters
        ----------
        normalized_boxes
            Boxes in CXCYWH format with shape (N, 4).

        Returns
        -------
        xyxy_boxes
            Boxes in XYXY format with shape (N, 4).
        """
        x_c, y_c, w, h = normalized_boxes.unbind(1)  # extract the columns
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def box_xyxy_to_cxcywh(xyxy_boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from XYXY to CXCYWH format.

        Parameters
        ----------
        xyxy_boxes
            Boxes in XYXY format with shape (N, 4).

        Returns
        -------
        cxcywh_boxes
            Boxes in CXCYWH format with shape (N, 4).
        """
        x1, y1, x2, y2 = xyxy_boxes.unbind(1)
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + 0.5 * w
        y_c = y1 + 0.5 * h
        b = [x_c, y_c, w, h]
        return torch.stack(b, dim=1)

    @staticmethod
    def box_xywh_to_xyxy(normalized_boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes XYWH to XYXY format.

        Parameters
        ----------
        normalized_boxes
            Boxes in XYWH format with shape (N, 4).

        Returns
        -------
        xyxy_boxes
            Boxes in XYXY format with shape (N, 4).
        """
        x, y, w, h = normalized_boxes.unbind(1)  # extract the columns
        b = [x, y, x + w, y + h]
        return torch.stack(b, dim=1)

    @staticmethod
    def denormalize_boxes(
            boxes: torch.Tensor, size: torch.Size) -> torch.Tensor:
        """
        Convert normalized boxes [0,1] to pixel coordinates.

        Parameters
        ----------
        boxes
            Boxes in normalized coordinates [0,1]
        size
            Image size (width, height)

        Returns
        -------
        denormalized_boxes
            Boxes in pixel coordinates
        """
        img_w, img_h = size
        denormalized_boxes = boxes * torch.tensor(
            [img_w, img_h, img_w, img_h], device=boxes.device
        )
        return denormalized_boxes

    @staticmethod
    def to_numpy_tuple(*tensors) -> Tuple:
        """
        Convert one or more PyTorch tensors to tuple of numpy arrays.
        Handles GPU tensors by moving to CPU first.
        Always returns a tuple, even if a single tensor is provided.

        Parameters
        ----------
        *tensors
            Variable number of tensors to convert

        Returns
        -------
        numpy_arrays
            Tuple of numpy arrays
        """
        return tuple(t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
                     for t in tensors)

    @staticmethod
    def probas_argmax(proba: torch.Tensor) -> int:
        """
        Get the class ID from a probability tensor.

        Parameters
        ----------
        proba
            Probability tensor for a single detection

        Returns
        -------
        class_id
            Class ID as Python int
        """
        return proba.argmax().item()


class TorchBoxCoordinatesTranslator:
    """
    Translates bounding boxes between different coordinate formats for PyTorch tensors.

    This class handles conversions between box formats (XYXY, CXCYWH, XYWH) and
    manages normalization/denormalization of coordinates.
    """

    def __init__(
            self, input_box_type: BoxType, output_box_type: BoxType) -> None:
        """
        Initialize the box coordinates translator.

        Parameters
        ----------
        input_box_type
            Format specification of input boxes.
        output_box_type
            Desired format specification for output boxes.
        """
        self.input_box_type = input_box_type
        self.output_box_type = output_box_type

    def translate(
            self,
            box: torch.Tensor,
            image_size: Optional[torch.Size] = None) -> torch.Tensor:
        """
        Translate boxes from input format to output format.

        This method performs a multi-step conversion:
        1. Normalize input boxes if needed
        2. Convert to XYXY intermediate format
        3. Convert to output format
        4. Denormalize if needed

        Parameters
        ----------
        box
            Bounding boxes in input format with shape (N, 4).
        image_size
            Image dimensions as (width, height). Required if input or output
            boxes are not normalized.

        Returns
        -------
        translated_boxes
            Boxes in output format with shape (N, 4).

        Raises
        ------
        ValueError
            If image_size is None when required for non-normalized boxes.
        """
        if not isinstance(box, torch.Tensor):
            box = torch.tensor(box)

        # normalize the input box if needed
        if not self.input_box_type.is_normalized:
            if image_size is None:
                raise ValueError("Image size must be provided for non-normalized boxes.")
            box = TorchBoxManager.normalize_boxes(box, image_size)

        # convert the input box to XYXY format if needed
        if self.input_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TorchBoxManager.box_cxcywh_to_xyxy(box)
        elif self.input_box_type.format.value == BoxFormat.XYWH.value:
            box = TorchBoxManager.box_xywh_to_xyxy(box)

        # now convert to the output format
        if self.output_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TorchBoxManager.box_xyxy_to_cxcywh(box)
        elif self.output_box_type.format.value == BoxFormat.XYWH.value:
            box = TorchBoxManager.box_xyxy_to_xywh(box)  # TODO: add this method

        # denormalize the box to the output image size if needed
        if not self.output_box_type.is_normalized:
            box = TorchBoxManager.denormalize_boxes(box, image_size)

        return box
