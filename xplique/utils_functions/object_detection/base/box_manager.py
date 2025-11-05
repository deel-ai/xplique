"""
Base classes for managing bounding box operations across different frameworks.
"""
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np


class BoxFormat(Enum):
    """
    Enumeration of supported bounding box coordinate formats.

    Attributes
    ----------
    CXCYWH
        Center x, center y, width, height format.
    XYWH
        Top-left x, top-left y, width, height format.
    XYXY
        Top-left x, top-left y, bottom-right x, bottom-right y format.
    """
    CXCYWH = "CXCYWH"
    XYWH = "XYWH"
    XYXY = "XYXY"


@dataclass
class BoxType:
    """
    Data class representing the type and coordinate system of bounding boxes.

    Attributes
    ----------
    format
        The coordinate format of the boxes (CXCYWH, XYWH, or XYXY).
    is_normalized
        Whether coordinates are normalized to [0, 1] range or in pixel units.
    """
    format: BoxFormat
    is_normalized: bool


class BoxManager(ABC):
    """
    Abstract base class for managing bounding box operations.

    This class defines the interface for box management operations across different
    frameworks (NumPy, TensorFlow, PyTorch). Subclasses should implement framework-specific
    box transformations and conversions.
    """


class NumpyBoxManager(BoxManager):
    """
    NumPy-based implementation of box management operations.

    This class provides methods for converting between different bounding box formats,
    normalizing/denormalizing coordinates, and performing transformations using NumPy.
    """

    def __init__(self, box_format: BoxFormat, normalized: bool) -> None:
        """
        Initialize the NumPy box manager.

        Parameters
        ----------
        box_format
            The coordinate format of the boxes.
        normalized
            Whether the box coordinates are normalized to [0, 1].
        """
        self.format = box_format
        self.normalized = normalized

    @staticmethod
    def normalize_boxes(
            raw_boxes: np.ndarray,
            image_source_size: Tuple[int, int]) -> np.ndarray:
        """
        Normalize bounding box coordinates from pixel values to [0, 1] range.

        Divides x-coordinates by image width and y-coordinates by image height
        to convert from absolute pixel values to normalized coordinates.

        Parameters
        ----------
        raw_boxes
            Boxes in pixel coordinates of shape (N, 4+).
        image_source_size
            Image dimensions as (width, height).

        Returns
        -------
        normalized_boxes
            Normalized boxes with coordinates in [0, 1] range, same shape as input.
        """
        sx, sy = image_source_size
        normalized_boxes = raw_boxes.copy()
        normalized_boxes[:, [0, 2]] /= sx
        normalized_boxes[:, [1, 3]] /= sy
        return normalized_boxes

    @staticmethod
    def box_cxcywh_to_xyxy(normalized_boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from CXCYWH to XYXY format.

        Transforms from (center_x, center_y, width, height) format to
        (x_min, y_min, x_max, y_max) format by computing the corner coordinates
        from the center point and dimensions.

        Parameters
        ----------
        normalized_boxes
            Boxes in CXCYWH format of shape (N, 4).

        Returns
        -------
        boxes
            Boxes in XYXY format of shape (N, 4).
        """
        x_c = normalized_boxes[:, 0]
        y_c = normalized_boxes[:, 1]
        w = normalized_boxes[:, 2]
        h = normalized_boxes[:, 3]
        b = np.stack([
            x_c - 0.5 * w,
            y_c - 0.5 * h,
            x_c + 0.5 * w,
            y_c + 0.5 * h
        ], axis=1)
        return b

    @staticmethod
    def box_xywh_to_xyxy(normalized_boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from XYWH to XYXY format.

        Transforms from (x_min, y_min, width, height) format to
        (x_min, y_min, x_max, y_max) format by computing the bottom-right
        corner from the top-left corner and dimensions.

        Parameters
        ----------
        normalized_boxes
            Boxes in XYWH format of shape (N, 4).

        Returns
        -------
        boxes
            Boxes in XYXY format of shape (N, 4).
        """
        x = normalized_boxes[:, 0]
        y = normalized_boxes[:, 1]
        w = normalized_boxes[:, 2]
        h = normalized_boxes[:, 3]
        b = np.stack([
            x,
            y,
            x + w,
            y + h
        ], axis=1)
        return b

    @staticmethod
    def denormalize_boxes(
            boxes: np.ndarray,
            size: Tuple[int, int]) -> np.ndarray:
        """
        Convert normalized boxes from [0, 1] range to pixel coordinates.

        Multiplies x-coordinates by image width and y-coordinates by image height
        to convert from normalized coordinates to absolute pixel values.

        Parameters
        ----------
        boxes
            Boxes in normalized coordinates [0, 1] of shape (N, 4+).
        size
            Image dimensions as (width, height).

        Returns
        -------
        denormalized_boxes
            Boxes in pixel coordinates, same shape as input.
        """
        img_w, img_h = size
        denormalized_boxes = boxes * np.array([img_w, img_h, img_w, img_h])
        return denormalized_boxes
