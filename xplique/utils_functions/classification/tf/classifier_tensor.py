"""
TensorFlow implementation of ClassifierTensor for classification predictions.

This module provides a TensorFlow wrapper for classification predictions
with a unified format compatible with the StructuredPrediction protocol.
"""

import tensorflow as tf


class ClassifierTensor:
    """
    TensorFlow wrapper for classification predictions.

    This class wraps TensorFlow tensors from classification models to provide
    the same interface as MultiBoxTensor, allowing polymorphic handling of
    both object detection and classification predictions.

    Note: This class implements the StructuredPrediction protocol (see
    xplique.commons.prediction_types.StructuredPrediction) via structural typing.
    The class complies with the protocol by implementing:
    - to_batched_tensor(): Adds batch dimension if needed
    - filter(class_id, confidence): No-op for classifiers (returns self)

    Parameters
    ----------
    tensor
        TensorFlow tensor containing classifier predictions (logits or probabilities)
    """

    def __init__(self, tensor: tf.Tensor):
        self.tensor = tensor

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
