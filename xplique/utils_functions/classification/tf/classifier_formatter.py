"""
TensorFlow implementation of classifier formatter for classification predictions.

This module provides a formatter that converts TensorFlow tensors into ClassifierTensor
instances for standardized classification prediction handling.
"""

from xplique.utils_functions.classification.base.classifier_formatter import (
    BaseClassifierFormatter
)
from .classifier_tensor import ClassifierTensor


class TfClassifierFormatter(BaseClassifierFormatter):
    """
    TensorFlow-specific formatter for classification predictions.

    Converts TensorFlow tensor outputs into ClassifierTensor instances,
    ensuring a consistent interface for classification model predictions.
    """

    def forward(self, predictions):
        if isinstance(predictions, ClassifierTensor):
            return predictions
        return ClassifierTensor(predictions)
