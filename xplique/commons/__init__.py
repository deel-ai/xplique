"""
Utility classes and functions
"""

from .data_conversion import tensor_sanitize, numpy_sanitize
from .model_override import guided_relu_policy, deconv_relu_policy, override_relu_gradient, \
                            find_layer
from .tf_operations import repeat_labels, batch_gradient, batch_predictions_one_hot, \
    batch_tensor, predictions_one_hot
