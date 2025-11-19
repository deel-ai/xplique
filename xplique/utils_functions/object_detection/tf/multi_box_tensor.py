"""
TensorFlow implementation of MultiBoxTensor for object detection predictions.

This module provides a TensorFlow wrapper for multi-box detection predictions
with a unified format compatible with the MultiBoxTensor protocol.
"""

from typing import Any, Optional

import tensorflow as tf

from xplique.commons.prediction_types import StructuredPrediction


class MultiBoxTensor(StructuredPrediction):
    """
    TensorFlow wrapper for multi-box detection predictions with unified format.

    Encapsulates a tensor with shape (N, C) where:
    - N is the number of detected boxes
    - C = 4 + 1 + nb_classes encoding: 4 box coordinates, 1 objectness score,
      nb_classes class predictions (soft probabilities or one-hot encoding)

    This class provides convenient access to boxes, scores, and class probabilities,
    along with filtering capabilities and TensorFlow integration support.
    """
    # Ex: [9, 85] for 9 boxes, 80 classes (COCO dataset)

    def __init__(self, tensor: tf.Tensor) -> None:
        """
        Initialize the MultiBoxTensor.

        Parameters
        ----------
        tensor
            TensorFlow tensor of shape (N, C) containing box predictions.
        """
        self.tensor = tensor

    def __len__(self) -> int:
        # Get the number of features per box.
        return self.tensor.shape[1]

    @property
    def shape(self):
        """Get the shape of the underlying tensor."""
        return self.tensor.shape

    @property
    def dtype(self):
        """Get the data type of the underlying tensor."""
        return self.tensor.dtype

    def numpy(self) -> Any:
        """Convert the tensor to a NumPy array."""
        return self.tensor.numpy()

    def __tensor__(self) -> tf.Tensor:
        # Get the underlying TensorFlow tensor.
        return self.tensor

    def __tf_tensor__(self, dtype: Optional[Any] = None,
                      name: Optional[str] = None) -> tf.Tensor:
        # Critical: This makes tf.stack() work by converting to tensor
        return tf.convert_to_tensor(self.tensor, dtype=dtype, name=name)

    def __getitem__(self, key: Any) -> Any:
        # Support indexing operations
        return self.tensor[key]

    def boxes(self) -> tf.Tensor:
        """
        Extract bounding box coordinates from predictions.

        Returns
        -------
        boxes
            Tensor of shape (N, 4) containing box coordinates.
        """
        return self[:, :4]

    def scores(self) -> tf.Tensor:
        """
        Extract objectness scores from predictions.

        Returns
        -------
        scores
            Tensor of shape (N,) containing detection confidence scores.
        """
        return self[:, 4]

    def probas(self) -> tf.Tensor:
        """
        Extract class probability predictions from predictions.

        Returns
        -------
        probas
            Tensor of shape (N, num_classes) containing class probabilities.
        """
        return self[:, 5:]

    def filter(self, class_id: Optional[int] = None,
               confidence: Optional[float] = None) -> 'MultiBoxTensor':
        """
        Filter boxes by class ID and/or detection confidence threshold.

        Filters the detections based on predicted class and confidence score.
        If both filters are provided, boxes must satisfy both conditions.

        Parameters
        ----------
        class_id
            Optional class ID to filter for specific object class.
        confidence
            Optional minimum confidence score threshold.

        Returns
        -------
        filtered_tensor
            New MultiBoxTensor containing only filtered detections.
        """
        if class_id is None and confidence is None:
            return self
        probas = self.probas()
        class_ids = tf.argmax(probas, axis=-1)
        scores = self.scores()
        if class_id is not None and confidence is not None:
            keep = (class_ids == class_id) & (scores >= confidence)
        elif class_id is None:
            keep = scores > confidence
        else:
            keep = class_ids == class_id
        filtered_tensor = tf.boolean_mask(self.tensor, keep)
        return MultiBoxTensor(filtered_tensor)

    def to_batched_tensor(self) -> tf.Tensor:
        """
        Add batch dimension to MultiBoxTensor.

        Converts (num_boxes, features) -> (1, num_boxes, features)

        Returns
        -------
        tensor
            tf.Tensor with batch dimension added
        """
        return tf.expand_dims(self.tensor, axis=0)
