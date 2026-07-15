"""
TensorFlow implementation of ClassifierTensor for classification predictions.

This module provides a TensorFlow wrapper for classification predictions
with a unified format compatible with the StructuredPrediction protocol.
"""

from numbers import Integral

import tensorflow as tf

from xplique.commons.prediction_types import StructuredPrediction


class TfClassifierTensor(StructuredPrediction):
    """
    TensorFlow wrapper for classification predictions.

    This class wraps TensorFlow tensors from classification models to provide
    the same interface as MultiBoxTensor, allowing polymorphic handling of
    both object detection and classification predictions.

    Parameters
    ----------
    tensor
        TensorFlow tensor containing classifier predictions (logits or probabilities)
    """

    def __init__(self, tensor: tf.Tensor):
        self.tensor = tensor

    @classmethod
    def from_predictions(cls, predictions):
        """Wrap raw classifier predictions unless they are already formatted."""
        if isinstance(predictions, cls):
            return predictions
        return cls(predictions)

    @property
    def shape(self):
        """Return the shape of the underlying tensor for compatibility."""
        return self.tensor.shape

    def __len__(self):
        """
        Return the number of classes (dimension 1 of the tensor).

        For batched tensors with shape (batch, num_classes), returns num_classes.
        For single predictions with shape (num_classes,), returns num_classes.

        Returns
        -------
        length
            Number of classes in the classification output.
        """
        if len(self.tensor.shape) == 1:
            return int(self.tensor.shape[0])
        return int(self.tensor.shape[1])

    def __tf_tensor__(self, dtype=None, name=None):
        """
        Convert to TensorFlow tensor for use in TF operations.

        This method enables ClassifierTensor to be used directly in TensorFlow
        operations like tf.stack(), tf.expand_dims(), etc. without explicitly
        accessing the .tensor attribute.

        Parameters
        ----------
        dtype
            Optional dtype to convert to.
        name
            Optional name for the operation.

        Returns
        -------
        tensor
            The underlying TensorFlow tensor.
        """
        return tf.convert_to_tensor(self.tensor, dtype=dtype, name=name)

    def to_batched_tensor(self) -> tf.Tensor:
        """
        Ensure tensor has batch dimension.

        For classifiers, if the tensor is 1D (single prediction), adds a batch
        dimension. If already 2D or higher, returns as-is.

        Returns
        -------
        batched_tensor
            Tensor with batch dimension: (1, num_classes) or (batch, num_classes)
        """
        if len(self.tensor.shape) == 1:
            return tf.expand_dims(self.tensor, axis=0)
        return self.tensor

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
        num_classes = self.tensor.shape[-1]
        if class_id < 0 or (num_classes is not None and class_id >= num_classes):
            raise ValueError(f"class_id must be in [0, {num_classes}).")

        target = tf.one_hot(class_id, tf.shape(self.tensor)[-1], dtype=self.tensor.dtype)
        target = tf.broadcast_to(target, tf.shape(self.tensor))
        return TfClassifierTensor(target)
