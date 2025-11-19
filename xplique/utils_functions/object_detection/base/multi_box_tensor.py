"""
Protocol for multi-box tensor representations in object detection.

This module defines the MultiBoxTensor Protocol, which provides a common interface
for tensors containing multiple detection boxes across different frameworks (TensorFlow, PyTorch).
"""

from typing import Any, Protocol, runtime_checkable

from xplique.commons.prediction_types import StructuredPrediction

# pylint: disable=unnecessary-ellipsis


@runtime_checkable
class BaseMultiBoxTensor(StructuredPrediction, Protocol):
    """
    Protocol for tensors containing multiple detection boxes.

    Tensor with shape (N, C) where:
    - N is the number of boxes
    - C is the encoding of a bounding box prediction, C = 4 + 1 + nb_classes
      - 4 coordinates (box coordinates)
      - 1 score (objectness/detection confidence)
      - nb_classes (soft class predictions or one-hot encoded class predictions)

    Example: [100, 85] for 100 boxes with 80 classes (COCO dataset: 4 + 1 + 80 = 85)

    This is a Protocol (structural type) rather than an ABC to avoid metaclass conflicts
    with framework-specific tensor types (torch.Tensor, tf.Tensor, etc.)
    """

    def boxes(self) -> Any:
        """
        Return box coordinates tensor with shape (N, 4).

        Returns
        -------
        tensor
            Box coordinates in the format [x1, y1, x2, y2].
        """
        ...

    def scores(self) -> Any:
        """
        Return detection scores tensor with shape (N,).

        Returns
        -------
        tensor
            Objectness or detection confidence scores.
        """
        ...

    def probas(self) -> Any:
        """
        Return class probabilities tensor with shape (N, nb_classes).

        Returns
        -------
        tensor
            Class probabilities or one-hot encoded class predictions.
        """
        ...
