"""
Utility classes and functions
"""

from .data_conversion import tensor_sanitize, numpy_sanitize
from .model_override import guided_relu_policy, deconv_relu_policy, override_relu_gradient, \
                            find_layer, open_relu_policy
from .tf_operations import repeat_labels, batch_tensor
from .callable_operations import predictions_one_hot_callable
from .operators_operations import (Tasks, get_operator, check_operator, operator_batching,
                                   get_inference_function, get_gradient_functions)
from .exceptions import no_gradients_available, raise_invalid_operator
from .forgrad import forgrad
