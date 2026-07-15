"""
PyTorch implementation of ClassifierTensor for classification predictions.

This module provides a PyTorch tensor subclass for classification predictions
with a unified format. Due to metaclass conflicts with torch.Tensor, this class
cannot explicitly inherit from the StructuredPrediction protocol but implements its
interface via structural typing (duck typing).
"""

from numbers import Integral

import torch

from xplique.commons.prediction_types import StructuredPrediction


class TorchClassifierTensor(torch.Tensor):
    """
    Tensor representation for classification predictions.

    This class extends torch.Tensor to represent classification model outputs
    (logits or probabilities) with a shape of (num_classes,) for single predictions
    or (batch_size, num_classes) for batched predictions.

    Note: This class implements the StructuredPrediction protocol (see
    xplique.commons.prediction_types.StructuredPrediction) via structural typing.
    The class complies with the protocol by implementing:
    - to_batched_tensor(): Adds batch dimension if needed
    - filter(class_id, confidence): Creates a one-hot target for a selected class
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Delegate torch operations while preserving Tensor subclass semantics.

        PyTorch does not expose an equivalent of TensorFlow's ``__tf_tensor__``
        conversion hook, so this class intentionally subclasses ``torch.Tensor``
        to keep autograd and tensor operations available on formatted outputs.
        Delegating here keeps the default PyTorch subclass behavior explicit.
        """
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def from_predictions(cls, predictions):
        """Wrap raw classifier predictions unless they are already formatted."""
        if isinstance(predictions, cls):
            return predictions
        return cls(predictions)

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

    def filter(self, class_id=None, confidence=None):
        """
        Build a target for a selected classification class.

        Classifiers do not have detections to filter. When ``class_id`` is
        provided, return a one-hot target with the same batch shape as the
        predictions. ``confidence`` is ignored for classifiers.

        Parameters
        ----------
        class_id
            Class to target.
        confidence
            Ignored for classifiers

        Returns
        -------
        filtered_tensor
            One-hot target for ``class_id``, or self when no class is selected.
        """
        if class_id is None:
            return self

        if not isinstance(class_id, Integral) or isinstance(class_id, bool):
            raise ValueError("class_id must be an integer.")

        class_id = int(class_id)
        num_classes = self.shape[-1]
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"class_id must be in [0, {num_classes}).")

        target = torch.zeros_like(self)
        target[..., class_id] = 1
        # Ensure subclass identity is preserved regardless of upstream torch.zeros_like semantics.
        if not isinstance(target, type(self)):
            target = target.as_subclass(type(self))
        return target


# Verify structural compliance with StructuredPrediction protocol at import time.
# TorchClassifierTensor cannot explicitly inherit from StructuredPrediction due to a
# metaclass conflict between torch.Tensor (_TensorMeta) and Protocol (_ProtocolMeta).
assert issubclass(TorchClassifierTensor, StructuredPrediction), (
    "TorchClassifierTensor must structurally satisfy the StructuredPrediction protocol"
)
