"""
TensorFlow implementation of ClassifierTensor for classification predictions.

This module provides a TensorFlow wrapper for classification predictions
with a unified format compatible with the StructuredPrediction protocol.
"""

import warnings
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
        tensor = tf.convert_to_tensor(tensor)
        if tensor.shape.rank not in (1, 2):
            raise ValueError("Classifier predictions must have rank 1 or 2.")
        self.tensor = tensor

    @classmethod
    def from_predictions(cls, predictions):
        """Wrap raw classifier predictions unless they are already formatted.

        Raises
        ------
        ValueError
            If predictions rank is not 1 or 2.
        """
        if isinstance(predictions, cls):
            return predictions
        return cls(predictions)

    @property
    def shape(self):
        """Return the shape of the underlying tensor for compatibility."""
        return self.tensor.shape

    @staticmethod
    def _dim_as_int(tensor: tf.Tensor, axis: int) -> int:
        """Return static dimension when available, else evaluate dynamic size eagerly."""
        dim = tensor.shape[axis]
        if dim is not None:
            return int(dim)
        return int(tf.shape(tensor)[axis])

    @property
    def num_classes(self) -> int:
        """Number of classes in the prediction tensor."""
        return self._dim_as_int(self.tensor, -1)

    @property
    def batch_size(self) -> int:
        """Batch size of predictions (1 for rank-1 single predictions)."""
        if self.tensor.shape.rank == 1:
            return 1
        return self._dim_as_int(self.tensor, 0)

    @property
    def is_empty(self) -> bool:
        """Whether there are no predictions to explain."""
        return self.batch_size == 0 or self.num_classes == 0

    def __len__(self):
        """
        Deprecated length of the classifier tensor.

        Use ``num_classes``, ``batch_size``, or ``is_empty`` instead.

        Returns
        -------
        length
            Number of classes in the classification output.
        """
        warnings.warn(
            "len(TfClassifierTensor) is ambiguous and deprecated; use "
            "num_classes, batch_size, or is_empty instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_classes

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
        if self.tensor.shape.rank == 1:
            return tf.convert_to_tensor(tf.expand_dims(self.tensor, axis=0))
        return tf.convert_to_tensor(self.tensor)

    def filter(self, class_id=None, confidence=None):
        """
        No-op for classifiers.

        Classifiers do not have multiple detections to filter. Returns self
        unchanged for interface compatibility with object detection types.
        """
        return self

    def to_attribution_target(self, class_id=None):
        """
        Build a one-hot attribution target for a selected class.

        The actual logit/probability values are not used by the attribution
        method — only the class index and output shape matter. ``class_id``
        selects which output neuron to differentiate through.

        Parameters
        ----------
        class_id
            Class to target. If None, returns self (raw model output as target).
        confidence
            Ignored for classifiers.

        Returns
        -------
        target
            One-hot tensor for ``class_id``, or self when class_id is None.
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
