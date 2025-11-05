"""
TensorFlow implementation for bounding box management operations.
"""
from typing import Optional, Tuple

import tensorflow as tf

from xplique.utils_functions.object_detection.base.box_manager import (
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

    def __init__(self, box_format: BoxFormat, normalized: bool) -> None:
        """
        Initialize the TensorFlow box manager.

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
        sx, sy = image_source_size
        normalized_boxes = tf.identity(raw_boxes)
        normalized_boxes = tf.concat(
            [
                normalized_boxes[:, 0:1] / sx,
                normalized_boxes[:, 1:2] / sy,
                normalized_boxes[:, 2:3] / sx,
                normalized_boxes[:, 3:4] / sy,
            ],
            axis=1,
        )
        return normalized_boxes

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
        x_c, y_c, w, h = tf.unstack(normalized_boxes, axis=1)
        x_min = x_c - 0.5 * w
        y_min = y_c - 0.5 * h
        x_max = x_c + 0.5 * w
        y_max = y_c + 0.5 * h
        b = tf.stack([x_min, y_min, x_max, y_max], axis=1)
        return b

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
        x1, y1, x2, y2 = tf.unstack(xyxy_boxes, axis=1)
        w = x2 - x1
        h = y2 - y1
        x_c = x1 + 0.5 * w
        y_c = y1 + 0.5 * h
        b = [x_c, y_c, w, h]
        return tf.stack(b, axis=1)

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
        x, y, w, h = tf.unstack(normalized_boxes, axis=1)  # extract the columns
        b = [x, y, x + w, y + h]
        # logging.debug("Converted from XYWH to XYXY: %s", b[0][0])
        return tf.stack(b, axis=1)

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
        # Ensure size is a tensor and cast to float32 for compatibility
        size = tf.cast(tf.convert_to_tensor(size), tf.float32)
        # Concatenate size with itself to create [width, height, width, height]
        scale = tf.concat([size, size], axis=0)
        denormalized_boxes = boxes * scale
        return denormalized_boxes

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
        return tuple(t.numpy() if isinstance(t, tf.Tensor) else t
                     for t in tensors)

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


class TfBoxCoordinatesTranslator:
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

    def translate(self, box: tf.Tensor,
                  input_image_size: Optional[tf.TensorShape] = None,
                  output_image_size: Optional[tf.TensorShape] = None) -> tf.Tensor:
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
        input_image_size
            Size of input image for normalizing pixel coordinates.
        output_image_size
            Target image size for denormalizing output coordinates.

        Returns
        -------
        translated_box
            Translated box tensor in the desired format and scale.

        Raises
        ------
        ValueError
            If image size is required but not provided.
        """
        if not isinstance(box, tf.Tensor):
            box = tf.convert_to_tensor(box)

        # normalize the input box if needed
        if not self.input_box_type.is_normalized:
            if input_image_size is None:
                raise ValueError("Input image size must be provided for non-normalized boxes.")
            box = TfBoxManager.normalize_boxes(box, input_image_size)

        # convert the input box to XYXY format if needed
        if self.input_box_type.format.value == BoxFormat.CXCYWH.value:
            box = TfBoxManager.box_cxcywh_to_xyxy(box)
        elif self.input_box_type.format.value == BoxFormat.XYWH.value:
            box = TfBoxManager.box_xywh_to_xyxy(box)

        # now convert to the output format
        if self.output_box_type.format.value == BoxFormat.CXCYWH.value:
            # pylint: disable=fixme
            box = TfBoxManager.box_xyxy_to_cxcywh(box)  # TODO: add this method
        elif self.output_box_type.format.value == BoxFormat.XYWH.value:
            # pylint: disable=fixme
            box = TfBoxManager.box_xyxy_to_xywh(box)  # TODO: add this method

        # denormalize the box to the output image size if needed
        if not self.output_box_type.is_normalized:
            if output_image_size is None:
                raise ValueError("Output image size must be provided for non-normalized boxes.")
            box = TfBoxManager.denormalize_boxes(box, output_image_size)

        return box
