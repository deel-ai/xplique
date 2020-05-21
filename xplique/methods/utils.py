"""
Utils related to differents methods
"""

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import clone_model


def sanitize_input_output(explanation_method):
    """
    Wrap a method explanation function to ensure tf.Tensor as inputs,
    and numpy as output

    explanation_method : function
        Function to wrap, should return an tf.tensor.
    """

    def sanitize(self, inputs, labels, *args):
        inputs = tf.cast(inputs, tf.float32)
        labels = tf.cast(labels, tf.float32)

        explanations = explanation_method(self, inputs, labels, *args)

        return explanations.numpy()

    return sanitize


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
    if isinstance(layer, tf.keras.layers.ReLU):
        return True
    return False


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

    def clone_func(layer):
        if is_relu(layer):
            return Activation(relu_policy)

        if has_relu_activation(layer):
            layer.activation = relu_policy

        return layer

    model_commuted = clone_model(model, clone_function=clone_func)

    return model_commuted
