"""
Utility classes and functions
"""

from .data_conversion import tensor_sanitize, numpy_sanitize, sanitize_inputs_targets
from .model_override import guided_relu_policy, deconv_relu_policy, override_relu_gradient, \
                            find_layer, open_relu_policy
from .tf_operations import repeat_labels, batch_tensor
from .callable_operations import predictions_one_hot_callable, \
    batch_predictions_one_hot_callable
from .operators import predictions_operator, get_gradient_of_operator, operator_batching, \
                       batch_predictions, gradients_predictions, batch_gradients_predictions, \
                       check_operator
from .exceptions import no_gradients_available, raise_invalid_operator
from .tf_dataset_operations import are_dataset_first_elems_equal, dataset_gather, sanitize_dataset,\
                                   is_not_shuffled, batch_size_matches
