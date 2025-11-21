from xplique.utils_functions.classification.base.classifier_formatter import (
    BaseClassifierFormatter
)
from .classifier_tensor import ClassifierTensor


class TfClassifierFormatter(BaseClassifierFormatter):

    def forward(self, predictions):
        if isinstance(predictions, ClassifierTensor):
            return predictions
        return ClassifierTensor(predictions)
