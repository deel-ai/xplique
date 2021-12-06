"""
Utility classes and functions
"""
from .callable_operations import batch_predictions_one_hot_callable
from .callable_operations import predictions_one_hot_callable
from .data_conversion import numpy_sanitize
from .data_conversion import tensor_sanitize
from .model_override import deconv_relu_policy
from .model_override import find_layer
from .model_override import guided_relu_policy
from .model_override import open_relu_policy
from .model_override import override_relu_gradient
from .tf_operations import batch_gradient
from .tf_operations import batch_predictions_one_hot
from .tf_operations import batch_tensor
from .tf_operations import predictions_one_hot
from .tf_operations import repeat_labels
