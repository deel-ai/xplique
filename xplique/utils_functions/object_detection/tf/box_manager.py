"""
TensorFlow implementation for bounding box management operations.
"""

from typing import Optional, Tuple

import tensorflow as tf

from xplique.utils_functions.object_detection.base.box_manager import (
    BaseBoxCoordinatesTranslator,
    BoxFormat,
    BoxManager,
    BoxType,
)


class TfBoxManager(BoxManager):
    """
    TensorFlow implementation of box management operations.

    Provides TensorFlow-specific methods for converting between different
    bounding box formats, normalizing/denormalizing coordinates, and performing
    transformations using TensorFlow operations with gradient support.
    """

    @staticmethod
    def _as_floating(boxes: tf.Tensor) -> tf.Tensor:
        """Return boxes in a floating dtype suitable for coordinate arithmetic."""
        boxes = tf.convert_to_tensor(boxes)
        if not boxes.dtype.is_floating:
            boxes = tf.cast(boxes, tf.float32)
        return boxes

    @staticmethod
    def _coordinate_scale(boxes: tf.Tensor, size: tf.Tensor) -> tf.Tensor:
        """Build a scale that leaves prediction fields after coordinates unchanged."""
        size = tf.reshape(tf.cast(tf.convert_to_tensor(size), boxes.dtype), [-1])
        tf.debugging.assert_equal(
            tf.size(size), 2, message="Image size must contain width and height."
        )
        tf.debugging.assert_greater_equal(
            tf.shape(boxes)[-1], 4, message="Boxes must contain at least four coordinate columns."
        )
        return tf.concat(
            [
                tf.tile(size, [2]),
                tf.ones(tf.reshape(tf.shape(boxes)[-1] - 4, [1]), dtype=boxes.dtype),
            ],
            axis=0,
        )

    @staticmethod
    def normalize_boxes(raw_boxes: tf.Tensor, image_source_size: tf.Tensor) -> tf.Tensor:
        """
        Normalize bounding box coordinates from pixel values to [0, 1] range.

        Parameters
        ----------
        raw_boxes
            Boxes in pixel coordinates of shape (N, 4+).
        image_source_size
            Image dimensions as tensor (width, height).

        Returns
        -------
        normalized_boxes
            Normalized boxes with coordinates in [0, 1] range, same shape as input.
        """
        raw_boxes = TfBoxManager._as_floating(raw_boxes)
        size = tf.cast(tf.convert_to_tensor(image_source_size), raw_boxes.dtype)
        tf.debugging.assert_positive(
            size, message="Image width and height must be greater than zero for normalization."
        )
        return raw_boxes / TfBoxManager._coordinate_scale(raw_boxes, size)

    @staticmethod
    def box_cxcywh_to_xyxy(normalized_boxes: tf.Tensor) -> tf.Tensor:
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
        normalized_boxes = TfBoxManager._as_floating(normalized_boxes)
        x_c, y_c, w, h = tf.unstack(normalized_boxes[..., :4], axis=-1)
        x_min = x_c - 0.5 * w
        y_min = y_c - 0.5 * h
        x_max = x_c + 0.5 * w
        y_max = y_c + 0.5 * h
        b = tf.stack([x_min, y_min, x_max, y_max], axis=-1)
        return tf.concat([b, normalized_boxes[..., 4:]], axis=-1)

    @staticmethod
    def box_xywh_to_xyxy(normalized_boxes: tf.Tensor) -> tf.Tensor:
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
        normalized_boxes = TfBoxManager._as_floating(normalized_boxes)
        x, y, w, h = tf.unstack(normalized_boxes[..., :4], axis=-1)  # extract the columns
        b = [x, y, x + w, y + h]
        return tf.concat([tf.stack(b, axis=-1), normalized_boxes[..., 4:]], axis=-1)

    @staticmethod
    def box_xyxy_to_cxcywh(xyxy_boxes: tf.Tensor) -> tf.Tensor:
        """
        Convert boxes from XYXY to CXCYWH format.

        Transforms from (x_min, y_min, x_max, y_max) format to
        (center_x, center_y, width, height) format by computing the center
        point and dimensions from the corner coordinates.

        Parameters
        ----------
        xyxy_boxes
            Boxes in XYXY format of shape (N, 4).

        Returns
        -------
        boxes
            Boxes in CXCYWH format of shape (N, 4).
        """
        xyxy_boxes = TfBoxManager._as_floating(xyxy_boxes)
        x_min, y_min, x_max, y_max = tf.unstack(xyxy_boxes[..., :4], axis=-1)
        w = x_max - x_min
        h = y_max - y_min
        x_c = x_min + 0.5 * w
        y_c = y_min + 0.5 * h
        b = [x_c, y_c, w, h]
        return tf.concat([tf.stack(b, axis=-1), xyxy_boxes[..., 4:]], axis=-1)

    @staticmethod
    def box_xyxy_to_xywh(xyxy_boxes: tf.Tensor) -> tf.Tensor:
        """
        Convert boxes from XYXY to XYWH format.

        Transforms from (x_min, y_min, x_max, y_max) format to
        (x_min, y_min, width, height) format by computing the dimensions
        from the corner coordinates.

        Parameters
        ----------
        xyxy_boxes
            Boxes in XYXY format of shape (N, 4).

        Returns
        -------
        boxes
            Boxes in XYWH format of shape (N, 4).
        """
        xyxy_boxes = TfBoxManager._as_floating(xyxy_boxes)
        x_min, y_min, x_max, y_max = tf.unstack(xyxy_boxes[..., :4], axis=-1)
        w = x_max - x_min
        h = y_max - y_min
        b = [x_min, y_min, w, h]
        return tf.concat([tf.stack(b, axis=-1), xyxy_boxes[..., 4:]], axis=-1)

    @staticmethod
    def denormalize_boxes(boxes: tf.Tensor, size: tf.Tensor) -> tf.Tensor:
        """
        Convert normalized boxes from [0, 1] range to pixel coordinates.

        Multiplies x-coordinates by image width and y-coordinates by image height
        to convert from normalized coordinates to absolute pixel values.

        Parameters
        ----------
        boxes
            Boxes in normalized coordinates [0, 1] of shape (N, 4+).
        size
            Image dimensions as tensor (width, height).

        Returns
        -------
        denormalized_boxes
            Boxes in pixel coordinates, same shape as input.
        """
        boxes = TfBoxManager._as_floating(boxes)
        return boxes * TfBoxManager._coordinate_scale(boxes, size)

    @staticmethod
    def to_numpy_tuple(*tensors) -> Tuple:
        """
        Convert one or more TensorFlow tensors to tuple of NumPy arrays.

        Parameters
        ----------
        *tensors
            Variable number of tensors or arrays to convert.

        Returns
        -------
        arrays
            Tuple of NumPy arrays corresponding to input tensors.
        """
        return tuple(t.numpy() if isinstance(t, tf.Tensor) else t for t in tensors)

    @staticmethod
    def probas_argmax(proba: tf.Tensor) -> int:
        """
        Get the class ID with highest probability from a probability tensor.

        Finds the index of the maximum probability value and converts it to a
        Python integer for use as a class identifier.

        Parameters
        ----------
        proba
            Probability tensor for a single detection of shape (num_classes,).

        Returns
        -------
        class_id
            Class ID as Python int corresponding to highest probability.
        """
        return int(tf.argmax(proba).numpy())


class TfBoxCoordinatesTranslator(BaseBoxCoordinatesTranslator):
    """
    Translates bounding boxes between different coordinate formats and scales.

    Handles the full pipeline of box coordinate transformations including:
    - Normalization/denormalization
    - Format conversion (CXCYWH, XYWH, XYXY)
    - Image size scaling

    All operations use TensorFlow to maintain gradient flow for attribution methods.
    """

    def __init__(self, input_box_type: BoxType, output_box_type: BoxType) -> None:
        """
        Initialize the box coordinates translator.

        Parameters
        ----------
        input_box_type
            The format and normalization of input boxes.
        output_box_type
            The desired format and normalization for output boxes.
        """
        self.input_box_type = input_box_type
        self.output_box_type = output_box_type
        self._box_manager = TfBoxManager()

    @property
    def box_manager(self) -> TfBoxManager:
        return self._box_manager

    def translate(
        self,
        box: tf.Tensor,
        image_size: Optional[tf.TensorShape] = None,
    ) -> tf.Tensor:
        """
        Translate box coordinates from input format/scale to output format/scale.

        Performs a complete transformation pipeline:
        1. Normalize boxes if input is in pixel coordinates
        2. Convert to XYXY format as intermediate representation
        3. Convert from XYXY to desired output format
        4. Denormalize boxes if output should be in pixel coordinates

        Parameters
        ----------
        box
            Box tensor of shape (N, 4) to translate.
        image_size
            Image dimensions as (width, height). Required if input or output
            boxes are not normalized.

        Returns
        -------
        translated_box
            Translated box tensor in the desired format and scale.

        Raises
        ------
        ValueError
            If image size is required but not provided.
        """
        box = TfBoxManager._as_floating(box)

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
            box = TfBoxManager.normalize_boxes(box, image_size)

        # convert the input box to XYXY format if needed
        if self.input_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TfBoxManager.box_cxcywh_to_xyxy(box)
        elif self.input_box_type.format.value == BoxFormat.XYWH.value:
            box = TfBoxManager.box_xywh_to_xyxy(box)

        # now convert to the output format
        if self.output_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TfBoxManager.box_xyxy_to_cxcywh(box)
        elif self.output_box_type.format.value == BoxFormat.XYWH.value:
            box = TfBoxManager.box_xyxy_to_xywh(box)

        # denormalize the box to the output image size if needed
        if not self.output_box_type.is_normalized:
            if image_size is None:
                raise ValueError("Output image size must be provided for non-normalized boxes.")
            box = TfBoxManager.denormalize_boxes(box, image_size)

        return box
