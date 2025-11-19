"""
Protocols and types for model prediction outputs.

This module defines common interfaces for different types of model predictions
(object detection, classification, etc.) to enable polymorphic handling without
runtime type checking.
"""
# pylint: disable=unnecessary-ellipsis

from typing import Protocol, runtime_checkable


@runtime_checkable
class StructuredPrediction(Protocol):
    """
    Protocol for unified handling of structured model predictions.

    This protocol defines a common interface that both object detection outputs
    (MultiBoxTensor) and classifier outputs (ClassifierTensor) implement, allowing
    code to work with either type without isinstance checks.

    The protocol includes:
    - to_batched_tensor(): Ensures output has batch dimension for attribution methods
    - filter(): Filters predictions based on class_id and/or confidence threshold
    """

    def to_batched_tensor(self):
        """
        Convert prediction to batched tensor format.

        Ensures the output has a batch dimension, which is required by
        attribution methods. For single predictions, adds a batch dimension.
        For already-batched predictions, returns as-is.

        Returns
        -------
        tensor
            Tensor with batch dimension: (batch_size, ...)
        """
        ...

    def filter(self, class_id=None, confidence=None):
        """
        Filter predictions by class ID and/or confidence threshold.

        For object detection, this filters bounding boxes by class and score.
        For classifiers, this is typically a no-op returning self.

        Parameters
        ----------
        class_id
            Optional class ID to filter by (for object detection)
        confidence
            Optional minimum confidence threshold (for object detection)

        Returns
        -------
        StructuredPrediction
            Filtered predictions (or self for classifiers)
        """
        ...
