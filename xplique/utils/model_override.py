"""
Override relu gradients policy
"""

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import clone_model


@tf.custom_gradient
def guided_relu(inputs):
    """
    Guided relu activation function.
    Act like a relu during forward pass, but allows only positive gradients with positive
    activation to pass through during backprop.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor

    Returns
    -------
    output : tf.Tensor
        Tensor, output or relu transformation.
    grad_func : function
        Gradient function for guided relu.
    """

    def grad_func(grads):
        gate_activation = tf.cast(inputs > 0.0, tf.float32)
        return tf.nn.relu(grads) * gate_activation

    return tf.nn.relu(inputs), grad_func


@tf.custom_gradient
def deconv_relu(inputs):
    """
    DeconvNet activation function.
    Act like a relu during forward pass, but allows only positive gradients to pass through
    during backprop.

    Parameters
    ----------
    inputs : tf.Tensor
        Input tensor

    Returns
    -------
    output : tf.Tensor
        Tensor, output or relu transformation.
    grad_func : function
        Gradient function for DeconvNet relu.
    """

    def grad_func(grads):
        return tf.nn.relu(grads)

    return tf.nn.relu(inputs), grad_func


def is_relu(layer):
    """
    Check if a layer is a ReLU layer

    Parameters
    ----------
    layer : tf.keras.layers.Layers

    Returns
    -------
    is_relu : bool
    """
    return isinstance(layer, tf.keras.layers.ReLU)


def has_relu_activation(layer):
    """
    Check if a layer has a ReLU activation.

    Parameters
    ----------
    layer : tf.keras.layers.Layers

    Returns
    -------
    has_relu: bool
    """
    if not hasattr(layer, 'activation'):
        return False
    return layer.activation in [tf.nn.relu, tf.keras.activations.relu]


def override_relu_gradient(model, relu_policy):
    """
    Given a model, commute all original ReLU by a new given ReLU policy.

    Parameters
    ----------
    model : tf.keras.model
        Model to commute.
    relu_policy : tf.custom_gradient
        Function wrapped with custom_gradient, defining the ReLU backprop behaviour.

    Returns
    -------
    model_commuted : tf.keras.model
    """
    cloned_model = clone_model(model)
    cloned_model.set_weights(model.get_weights())

    for layer_id in range(len(cloned_model.layers)):
        layer = cloned_model.layers[layer_id]
        if has_relu_activation(layer):
            layer.activation = relu_policy
        elif is_relu(layer):
            cloned_model.layers[layer_id] = Activation(relu_policy)

    return cloned_model


def find_layer(model, layer):
    """
    Find a layer in a model either by his name or by his index.

    Parameters
    ----------
    model : tf.keras.Model
        Model on which to search.
    layer : str or int
        Layer name or layer index

    Returns
    -------
    layer : tf.keras.Layer
        Layer found
    """
    if isinstance(layer, str):
        return model.get_layer(layer)
    if isinstance(layer, int):
        return model.layers[layer]
    raise ValueError(f"Could not find any layer {layer}.")
