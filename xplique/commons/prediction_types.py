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
    - filter(): Filters predictions based on confidence threshold (OD only, no-op for classifiers)
    - to_attribution_target(): Builds the attribution target (OD: returns filtered boxes;
      classifiers: builds a one-hot vector for the requested class_id)
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
        For classifiers, this is a no-op returning self (there are no boxes to filter).

        Parameters
        ----------
        class_id
            Optional class ID to filter by (for object detection only)
        confidence
            Optional minimum confidence threshold (for object detection only)

        Returns
        -------
        StructuredPrediction
            Filtered predictions (or self for classifiers)
        """
        ...

    def to_attribution_target(self, class_id=None):
        """
        Build the attribution target for the requested class.

        For object detection, the filtered boxes (from filter()) are already the
        correct target — this returns self unchanged.
        For classifiers, a one-hot vector is constructed for ``class_id`` since
        the actual logit/probability values are not used by the attribution method
        (only the class index and output shape matter).

        Parameters
        ----------
        class_id
            Class to target. For classifiers, builds one_hot(class_id, num_classes).
            For object detection, ignored (boxes carry the class information).
            If None, returns self unchanged for both types.

        Returns
        -------
        StructuredPrediction
            Attribution target ready for to_batched_tensor().
        """
        ...
