"""
PyTorch implementation of MultiBoxTensor for object detection predictions.

This module provides a PyTorch tensor subclass for multi-box detection predictions
with a unified format.
"""

from typing import Optional

import torch


class MultiBoxTensor(torch.Tensor):
    """
    Tensor representation for multiple bounding box predictions with class probabilities.

    This class extends torch.Tensor to represent object detection predictions with shape
    (B, C) where B is the number of boxes and C encodes box coordinates, objectness score,
    and class predictions. The encoding is: 4 coordinates + 1 objectness + nb_classes.
    For example, (9, 85) represents 9 boxes with 80 classes (COCO dataset).
    
    Methods:
    - boxes(): Extract box coordinates
    - scores(): Extract objectness scores
    - probas(): Extract class probabilities
    - filter(class_id, accuracy): Filters boxes by class/score
    - to_batched_tensor(): Adds batch dimension
    """

    def __format__(self, format_spec: str) -> str:
        """
        Format the tensor as a string.

        For scalar tensors, extracts and formats the single value. Otherwise uses
        default tensor formatting.

        Parameters
        ----------
        format_spec
            Format specification string.

        Returns
        -------
        formatted_str
            Formatted string representation of the tensor.
        """
        if self.numel() == 1:
            scalar_value = self.item()
            return format(scalar_value, format_spec)
        return super().__format__(format_spec)

    def boxes(self) -> torch.Tensor:
        """
        Extract box coordinates from the predictions.

        Returns
        -------
        boxes
            Tensor of shape (B, 4) containing box coordinates for each detection.
        """
        return self[:, :4]

    def scores(self) -> torch.Tensor:
        """
        Extract objectness scores from the predictions.

        Returns
        -------
        scores
            Tensor of shape (B,) containing confidence scores for each detection.
        """
        return self[:, 4]

    def probas(self) -> torch.Tensor:
        """
        Extract class probabilities from the predictions.

        Returns
        -------
        probas
            Tensor of shape (B, num_classes) containing class probabilities for each box.
        """
        return self[:, 5:]

    def filter(
            self,
            class_id: Optional[int] = None,
            accuracy: Optional[float] = None) -> 'MultiBoxTensor':
        """
        Filter detections by class ID and/or confidence threshold.

        Parameters
        ----------
        class_id
            If provided, keep only detections of this class.
        accuracy
            If provided, keep only detections with score >= this threshold.

        Returns
        -------
        filtered_tensor
            Filtered MultiBoxTensor containing only detections matching the criteria.
        """
        if class_id is None and accuracy is None:
            return self
        probas = self.probas()
        class_ids = probas.argmax(dim=-1)
        scores = self.scores()
        if class_id is not None and accuracy is not None:
            keep = (class_ids == class_id) & (scores >= accuracy)
        elif class_id is None:
            keep = scores > accuracy
        else:
            keep = class_ids == class_id
        return self[keep, :]

    def to_batched_tensor(self) -> torch.Tensor:
        """
        Add batch dimension to MultiBoxTensor.

        Converts (num_boxes, features) -> (1, num_boxes, features)

        Returns
        -------
        batched_tensor
            torch.Tensor with batch dimension added
        """
        return torch.unsqueeze(self, dim=0)
