"""
Override relu gradients policy
"""

import tensorflow as tf
from tensorflow.keras.layers import Activation # pylint: disable=E0611
from tensorflow.keras.models import clone_model # pylint: disable=E0611

from ..types import Tuple, Callable, Union


@tf.custom_gradient
def guided_relu(inputs: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
    """
    Guided relu activation function.
    Act like a relu during forward pass, but allows only positive gradients with positive
    activation to pass through during backprop.

    Parameters
    ----------
    inputs
        Input tensor

    Returns
    -------
    output
        Tensor, output or relu transformation.
    grad_func
        Gradient function for guided relu.
    """

    def grad_func(grads):
        gate_activation = tf.cast(inputs > 0.0, tf.float32)
        return tf.nn.relu(grads) * gate_activation

    return tf.nn.relu(inputs), grad_func


@tf.custom_gradient
def deconv_relu(inputs: tf.Tensor) -> Tuple[tf.Tensor, Callable]:
    """
    DeconvNet activation function.
    Act like a relu during forward pass, but allows only positive gradients to pass through
    during backprop.

    Parameters
    ----------
    inputs
        Input tensor

    Returns
    -------
    output
        Tensor, output or relu transformation.
    grad_func
        Gradient function for DeconvNet relu.
    """

    def grad_func(grads):
        return tf.nn.relu(grads)

    return tf.nn.relu(inputs), grad_func


def is_relu(layer: tf.keras.layers.Layer) -> bool:
    """
    Check if a layer is a ReLU layer

    Parameters
    ----------
    layer
        Layer to check.

    Returns
    -------
    is_relu : bool
        True if the layer is a relu activation.
    """
    return isinstance(layer, tf.keras.layers.ReLU)


def has_relu_activation(layer: tf.keras.layers.Layer) -> bool:
    """
    Check if a layer has a ReLU activation.

    Parameters
    ----------
    layer
        Layer to check.

    Returns
    -------
    has_relu
        True if the layer has a relu activation.
    """
    if not hasattr(layer, 'activation'):
        return False
    return layer.activation in [tf.nn.relu, tf.keras.activations.relu]


def override_relu_gradient(model: tf.keras.Model, relu_policy: Callable) -> tf.keras.Model:
    """
    Given a model, commute all original ReLU by a new given ReLU policy.

    Parameters
    ----------
    model
        Model to commute.
    relu_policy
        Function wrapped with custom_gradient, defining the ReLU backprop behaviour.

    Returns
    -------
    model_commuted
    """
    cloned_model = clone_model(model)
    cloned_model.set_weights(model.get_weights())

    for layer_id in range(len(cloned_model.layers)): # pylint: disable=C0200
        layer = cloned_model.layers[layer_id]
        if has_relu_activation(layer):
            layer.activation = relu_policy
        elif is_relu(layer):
            cloned_model.layers[layer_id] = Activation(relu_policy)

    return cloned_model


def find_layer(model: tf.keras.Model, layer: Union[str, int]) -> tf.keras.layers.Layer:
    """
    Find a layer in a model either by his name or by his index.

    Parameters
    ----------
    model
        Model on which to search.
    layer
        Layer name or layer index

    Returns
    -------
    layer
        Layer found
    """
    if isinstance(layer, str):
        return model.get_layer(layer)
    if isinstance(layer, int):
        return model.layers[layer]
    raise ValueError(f"Could not find any layer {layer}.")
