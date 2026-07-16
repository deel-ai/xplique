"""
PyTorch-specific box management utilities for coordinate transformations.
"""

from typing import Optional, Tuple

import torch

from xplique.utils_functions.object_detection.base.box_manager import (
    BaseBoxCoordinatesTranslator,
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

    @staticmethod
    def _as_floating(boxes: torch.Tensor) -> torch.Tensor:
        """Return boxes in a floating dtype suitable for coordinate arithmetic."""
        boxes = torch.as_tensor(boxes)
        if not boxes.is_floating_point():
            boxes = boxes.to(torch.float32)
        return boxes

    @staticmethod
    def _coordinate_scale(boxes: torch.Tensor, size: torch.Size) -> torch.Tensor:
        """Build a scale that leaves prediction fields after coordinates unchanged."""
        if boxes.shape[-1] < 4:
            raise ValueError("Boxes must contain at least four coordinate columns.")
        size = torch.as_tensor(size, dtype=boxes.dtype, device=boxes.device).reshape(-1)
        if size.numel() != 2:
            raise ValueError("Image size must contain width and height.")
        trailing_scale = torch.ones(boxes.shape[-1] - 4, dtype=boxes.dtype, device=boxes.device)
        return torch.cat([size.repeat(2), trailing_scale])

    @staticmethod
    def normalize_boxes(raw_boxes: torch.Tensor, image_source_size: torch.Size) -> torch.Tensor:
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
        raw_boxes = TorchBoxManager._as_floating(raw_boxes)
        image_source_size = torch.as_tensor(image_source_size, device=raw_boxes.device).reshape(-1)
        if image_source_size.numel() != 2 or torch.any(image_source_size == 0):
            raise ValueError("Image width and height must be greater than zero for normalization.")
        return raw_boxes / TorchBoxManager._coordinate_scale(raw_boxes, image_source_size)

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
        normalized_boxes = TorchBoxManager._as_floating(normalized_boxes)
        x_c, y_c, w, h = normalized_boxes[..., :4].unbind(-1)  # extract the columns
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.cat([torch.stack(b, dim=-1), normalized_boxes[..., 4:]], dim=-1)

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
        xyxy_boxes = TorchBoxManager._as_floating(xyxy_boxes)
        x1, y1, x2, y2 = xyxy_boxes[..., :4].unbind(-1)
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + 0.5 * w
        y_c = y1 + 0.5 * h
        b = [x_c, y_c, w, h]
        return torch.cat([torch.stack(b, dim=-1), xyxy_boxes[..., 4:]], dim=-1)

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
        normalized_boxes = TorchBoxManager._as_floating(normalized_boxes)
        x, y, w, h = normalized_boxes[..., :4].unbind(-1)
        b = [x, y, x + w, y + h]
        return torch.cat([torch.stack(b, dim=-1), normalized_boxes[..., 4:]], dim=-1)

    @staticmethod
    def box_xyxy_to_xywh(xyxy_boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from XYXY to XYWH format.

        Parameters
        ----------
        xyxy_boxes
            Boxes in XYXY format with shape (N, 4).

        Returns
        -------
        xywh_boxes
            Boxes in XYWH format with shape (N, 4).
        """
        xyxy_boxes = TorchBoxManager._as_floating(xyxy_boxes)
        x_min, y_min, x_max, y_max = xyxy_boxes[..., :4].unbind(-1)
        w = x_max - x_min
        h = y_max - y_min
        b = [x_min, y_min, w, h]
        return torch.cat([torch.stack(b, dim=-1), xyxy_boxes[..., 4:]], dim=-1)

    @staticmethod
    def denormalize_boxes(boxes: torch.Tensor, size: torch.Size) -> torch.Tensor:
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
        boxes = TorchBoxManager._as_floating(boxes)
        return boxes * TorchBoxManager._coordinate_scale(boxes, size)

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
        return tuple(
            t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t for t in tensors
        )

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


class TorchBoxCoordinatesTranslator(BaseBoxCoordinatesTranslator):
    """
    Translates bounding boxes between different coordinate formats for PyTorch tensors.

    This class handles conversions between box formats (XYXY, CXCYWH, XYWH) and
    manages normalization/denormalization of coordinates.
    """

    def __init__(self, input_box_type: BoxType, output_box_type: BoxType) -> None:
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
        self._box_manager = TorchBoxManager()

    @property
    def box_manager(self) -> TorchBoxManager:
        return self._box_manager

    def translate(self, box: torch.Tensor, image_size: Optional[torch.Size] = None) -> torch.Tensor:
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
        box = TorchBoxManager._as_floating(box)

        # Early return if input and output formats are identical
        if (
            self.input_box_type.format == self.output_box_type.format
            and self.input_box_type.is_normalized == self.output_box_type.is_normalized
        ):
            return box

        # normalize the input box if needed
        if not self.input_box_type.is_normalized:
            if image_size is None:
                raise ValueError("Input image size must be provided for non-normalized boxes.")
            box = TorchBoxManager.normalize_boxes(box, image_size)

        # convert the input box to XYXY format if needed
        if self.input_box_type.format is BoxFormat.CXCYWH:
            box = TorchBoxManager.box_cxcywh_to_xyxy(box)
        elif self.input_box_type.format is BoxFormat.XYWH:
            box = TorchBoxManager.box_xywh_to_xyxy(box)

        # now convert to the output format
        if self.output_box_type.format is BoxFormat.CXCYWH:
            box = TorchBoxManager.box_xyxy_to_cxcywh(box)
        elif self.output_box_type.format is BoxFormat.XYWH:
            box = TorchBoxManager.box_xyxy_to_xywh(box)

        # denormalize the box to the output image size if needed
        if not self.output_box_type.is_normalized:
            if image_size is None:
                raise ValueError("Output image size must be provided for non-normalized boxes.")
            box = TorchBoxManager.denormalize_boxes(box, image_size)

        return box
