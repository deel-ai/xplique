"""
Utility classes and functions
"""

from .data_conversion import tensor_sanitize, numpy_sanitize
from .model_override import guided_relu_policy, deconv_relu_policy, override_relu_gradient, \
                            find_layer, open_relu_policy
from .tf_operations import repeat_labels, batch_tensor
from .callable_operations import predictions_one_hot_callable, \
    batch_predictions_one_hot_callable
from .operators import predictions_operator, get_gradient_of_operator, operator_batching, \
                       batch_predictions, gradients_predictions, batch_gradients_predictions
from .exceptions import no_gradients_available, raise_invalid_operator
