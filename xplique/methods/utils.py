"""
Utils related to differents methods
"""

import tensorflow as tf


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
