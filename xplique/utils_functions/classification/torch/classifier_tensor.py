"""
PyTorch implementation of ClassifierTensor for classification predictions.

This module provides a PyTorch tensor subclass for classification predictions
with a unified format. Due to metaclass conflicts with torch.Tensor, this class
cannot explicitly inherit from the StructuredPrediction protocol but implements its
interface via structural typing (duck typing).
"""

import torch


class ClassifierTensor(torch.Tensor):
    """
    Tensor representation for classification predictions.

    This class extends torch.Tensor to represent classification model outputs
    (logits or probabilities) with a shape of (num_classes,) for single predictions
    or (batch_size, num_classes) for batched predictions.

    Note: This class implements the StructuredPrediction protocol (see
    xplique.commons.prediction_types.StructuredPrediction) via structural typing.
    The class complies with the protocol by implementing:
    - to_batched_tensor(): Adds batch dimension if needed
    - filter(class_id, confidence): No-op for classifiers (returns self)
    """

    def to_batched_tensor(self) -> torch.Tensor:
        """
        Ensure tensor has batch dimension.

        For classifiers, if the tensor is 1D (single prediction), adds a batch
        dimension. If already 2D or higher, returns as-is.

        Returns
        -------
        batched_tensor
            Tensor with batch dimension: (1, num_classes) or (batch, num_classes)
        """
        if len(self.shape) == 1:
            return torch.unsqueeze(self, 0)
        return self

    # pylint: disable=unused-argument
    def filter(self, class_id=None, confidence=None):
        """
        Filter predictions (no-op for classifiers).

        Classifiers don't have multiple detections to filter, so this method
        simply returns self for interface compatibility.

        Parameters
        ----------
        class_id
            Ignored for classifiers
        confidence
            Ignored for classifiers

        Returns
        -------
        self
            Returns self unchanged
        """
        return self
