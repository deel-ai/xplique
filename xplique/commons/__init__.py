"""
Utility classes and functions
"""

from .callable_operations import predictions_one_hot_callable
from .data_conversion import numpy_sanitize, sanitize_inputs_targets, tensor_sanitize
from .exceptions import no_gradients_available, raise_invalid_operator
from .forgrad import forgrad
from .model_override import (
    deconv_relu_policy,
    find_layer,
    guided_relu_policy,
    open_relu_policy,
    override_relu_gradient,
)
from .operators_operations import (
    Tasks,
    check_operator,
    get_gradient_functions,
    get_inference_function,
    get_operator,
    operator_batching,
    unwatch_layer,
    watch_layer,
)
from .tf_operations import batch_tensor, get_device, repeat_labels
