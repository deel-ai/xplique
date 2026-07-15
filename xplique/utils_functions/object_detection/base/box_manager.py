"""
Base classes for managing bounding box operations across different frameworks.
"""

from abc import ABC, abstractmethod
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

    @staticmethod
    @abstractmethod
    def to_numpy_tuple(*arrays) -> Tuple:
        """Convert framework tensors/arrays to a tuple of NumPy arrays."""

    @staticmethod
    @abstractmethod
    def probas_argmax(proba) -> int:
        """Return the class ID with the highest probability as a Python int."""


class BaseBoxCoordinatesTranslator(ABC):
    """
    Abstract base class for box coordinates translators.

    Defines the common interface for all framework-specific translators.
    Each subclass must expose a ``box_manager`` instance providing
    framework-specific utility operations (e.g. tensor-to-numpy conversion).
    """

    @property
    @abstractmethod
    def box_manager(self) -> "BoxManager":
        """Framework-specific BoxManager instance for utility operations."""

    @abstractmethod
    def translate(self, box, image_size=None):
        """Translate boxes from input format/scale to output format/scale."""


class NumpyBoxManager(BoxManager):
    """
    NumPy-based implementation of box management operations.

    This class provides methods for converting between different bounding box formats,
    normalizing/denormalizing coordinates, and performing transformations using NumPy.
    """

    @staticmethod
    def _as_floating(boxes: np.ndarray) -> np.ndarray:
        """Return boxes in a floating dtype suitable for coordinate arithmetic."""
        boxes = np.asarray(boxes)
        if not np.issubdtype(boxes.dtype, np.floating):
            boxes = boxes.astype(np.float32)
        return boxes

    @staticmethod
    def _coordinate_scale(boxes: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Build a scale that leaves prediction fields after coordinates unchanged."""
        if boxes.shape[-1] < 4:
            raise ValueError("Boxes must contain at least four coordinate columns.")
        size = np.asarray(size, dtype=boxes.dtype).reshape(-1)
        if size.size != 2:
            raise ValueError("Image size must contain width and height.")
        return np.concatenate([np.tile(size, 2), np.ones(boxes.shape[-1] - 4, dtype=boxes.dtype)])

    @staticmethod
    def normalize_boxes(raw_boxes: np.ndarray, image_source_size: Tuple[int, int]) -> np.ndarray:
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
        raw_boxes = NumpyBoxManager._as_floating(raw_boxes)
        image_source_size = np.asarray(image_source_size).reshape(-1)
        if image_source_size.size != 2 or np.any(image_source_size == 0):
            raise ValueError("Image width and height must be greater than zero for normalization.")
        return raw_boxes / NumpyBoxManager._coordinate_scale(raw_boxes, image_source_size)

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
        normalized_boxes = NumpyBoxManager._as_floating(normalized_boxes)
        coordinates = normalized_boxes[..., :4]
        x_c = coordinates[..., 0]
        y_c = coordinates[..., 1]
        w = coordinates[..., 2]
        h = coordinates[..., 3]
        b = np.stack([x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h], axis=-1)
        return np.concatenate([b, normalized_boxes[..., 4:]], axis=-1)

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
        normalized_boxes = NumpyBoxManager._as_floating(normalized_boxes)
        coordinates = normalized_boxes[..., :4]
        x = coordinates[..., 0]
        y = coordinates[..., 1]
        w = coordinates[..., 2]
        h = coordinates[..., 3]
        b = np.stack([x, y, x + w, y + h], axis=-1)
        return np.concatenate([b, normalized_boxes[..., 4:]], axis=-1)

    @staticmethod
    def box_xyxy_to_cxcywh(xyxy_boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from XYXY to CXCYWH format.

        Transforms from (x_min, y_min, x_max, y_max) format to
        (center_x, center_y, width, height) format.

        Parameters
        ----------
        xyxy_boxes
            Boxes in XYXY format of shape (N, 4).

        Returns
        -------
        boxes
            Boxes in CXCYWH format of shape (N, 4).
        """
        xyxy_boxes = NumpyBoxManager._as_floating(xyxy_boxes)
        coordinates = xyxy_boxes[..., :4]
        x_min = coordinates[..., 0]
        y_min = coordinates[..., 1]
        x_max = coordinates[..., 2]
        y_max = coordinates[..., 3]
        w = x_max - x_min
        h = y_max - y_min
        x_c = x_min + 0.5 * w
        y_c = y_min + 0.5 * h
        b = np.stack([x_c, y_c, w, h], axis=-1)
        return np.concatenate([b, xyxy_boxes[..., 4:]], axis=-1)

    @staticmethod
    def box_xyxy_to_xywh(xyxy_boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from XYXY to XYWH format.

        Transforms from (x_min, y_min, x_max, y_max) format to
        (x_min, y_min, width, height) format.

        Parameters
        ----------
        xyxy_boxes
            Boxes in XYXY format of shape (N, 4).

        Returns
        -------
        boxes
            Boxes in XYWH format of shape (N, 4).
        """
        xyxy_boxes = NumpyBoxManager._as_floating(xyxy_boxes)
        coordinates = xyxy_boxes[..., :4]
        x_min = coordinates[..., 0]
        y_min = coordinates[..., 1]
        x_max = coordinates[..., 2]
        y_max = coordinates[..., 3]
        w = x_max - x_min
        h = y_max - y_min
        b = np.stack([x_min, y_min, w, h], axis=-1)
        return np.concatenate([b, xyxy_boxes[..., 4:]], axis=-1)

    @staticmethod
    def denormalize_boxes(boxes: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
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
        boxes = NumpyBoxManager._as_floating(boxes)
        return boxes * NumpyBoxManager._coordinate_scale(boxes, size)

    @staticmethod
    def to_numpy_tuple(*arrays) -> Tuple:
        """
        Convert one or more numpy arrays to tuple of numpy arrays.
        Always returns a tuple, even if a single array is provided.

        Parameters
        ----------
        *arrays
            Variable number of arrays to convert

        Returns
        -------
        numpy_arrays
            Tuple of numpy arrays
        """
        return tuple(t for t in arrays)

    @staticmethod
    def probas_argmax(proba: np.ndarray) -> int:
        """
        Get the class ID from a probability array.

        Parameters
        ----------
        proba
            Probability array for a single detection

        Returns
        -------
        class_id
            Class ID as Python int
        """
        return int(proba.argmax())


class NumpyBoxCoordinatesTranslator(BaseBoxCoordinatesTranslator):
    """
    Translates bounding boxes between different coordinate formats using NumPy.

    Mirrors TorchBoxCoordinatesTranslator and TfBoxCoordinatesTranslator for
    use when the MultiBoxTensor is backed by plain NumPy arrays.
    """

    def __init__(self, input_box_type: BoxType, output_box_type: BoxType) -> None:
        """
        Initialize the NumPy box coordinates translator.

        Parameters
        ----------
        input_box_type
            Format specification of input boxes.
        output_box_type
            Desired format specification for output boxes.
        """
        self.input_box_type = input_box_type
        self.output_box_type = output_box_type
        self._box_manager = NumpyBoxManager()

    def translate(self, box: np.ndarray, image_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Translate boxes from input format to output format.

        Performs a multi-step conversion:
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
        box = NumpyBoxManager._as_floating(box)

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
            box = NumpyBoxManager.normalize_boxes(box, image_size)

        # convert the input box to XYXY format if needed
        if self.input_box_type.format.value == BoxFormat.CXCYWH.value:
            box = NumpyBoxManager.box_cxcywh_to_xyxy(box)
        elif self.input_box_type.format.value == BoxFormat.XYWH.value:
            box = NumpyBoxManager.box_xywh_to_xyxy(box)

        # convert to the output format
        if self.output_box_type.format.value == BoxFormat.CXCYWH.value:
            box = NumpyBoxManager.box_xyxy_to_cxcywh(box)
        elif self.output_box_type.format.value == BoxFormat.XYWH.value:
            box = NumpyBoxManager.box_xyxy_to_xywh(box)

        # denormalize if needed
        if not self.output_box_type.is_normalized:
            if image_size is None:
                raise ValueError("Output image size must be provided for non-normalized boxes.")
            box = NumpyBoxManager.denormalize_boxes(box, image_size)

        return box

    @property
    def box_manager(self) -> NumpyBoxManager:
        return self._box_manager
