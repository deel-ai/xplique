"""
Module related to the Testing of CAVs
"""

import tensorflow as tf
import numpy as np


class Tcav:
    """
    Used to Test a Concept Activation Vector, using the sign of the directional derivative of a
    concept vector relative to a class.

    Ref. Kim & al., Interpretability Beyond Feature Attribution: Quantitative Testing with Concept
    Activation Vectors (TCAV) (2018).
    https://arxiv.org/abs/1711.11279

    Parameters
    ----------
    model : tf.keras.Model
        Model to extract concept from.
    target_layer : int or string
        Index of the target layer or name of the layer.
    cav : ndarray
        Concept Activation Vector, see CAV module.
    batch_size : int, optional
        Batch size during the predictions.
    """

    def __init__(self, model, target_layer, cav, batch_size=64):
        self.model = model
        self.cav = tf.cast(cav, tf.float32)
        self.batch_size = batch_size

        # configure model bottleneck
        target_layer = model.get_layer(target_layer).output if isinstance(target_layer, str) else \
            model.layers[target_layer].output
        self.multi_head = tf.keras.Model(model.input, [target_layer, model.output])

    def score(self, inputs, label):
        """
        Compute and return the Concept Activation Vector (CAV) associated to the dataset and the
        layer targeted.

        Parameters
        ----------
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        label : int
            Index of the class to test.

        Returns
        -------
        tcav : ndarray
            Vector of the same shape as the layer output
        """

        directional_derivatives = None
        label = tf.cast(label, tf.int32)

        for x_batch in tf.data.Dataset.from_tensor_slices(inputs).batch(self.batch_size):
            batch_dd = Tcav.directional_derivative(self.multi_head,
                                                   x_batch, label,
                                                   self.cav)
            directional_derivatives = batch_dd if directional_derivatives is None else \
                tf.concat([directional_derivatives, batch_dd], axis=0)

        # tcav is the number of positive directional derivatives
        tcav = np.sum(directional_derivatives > 0.0) / len(directional_derivatives)

        return tcav

    __call__ = score

    @staticmethod
    @tf.function
    def directional_derivative(multi_head_model, inputs, label, cav):
        """
        Compute the gradient of the label relative to the activations of the CAV layer.

        Parameters
        ----------
        multi_head_model : tf.keras.Model
            Model reconfigured, first output is the activations of the CAV layer, and the second
            output is the prediction layer.
        inputs : ndarray (N, W, H, C)
            Input samples, with N number of samples, W & H the sample dimensions, and C the
            number of channels.
        label : int
            Index of the class to test.
        cav : tf.tensor
            Concept Activation Vector, same shape as the activations output.
        Returns
        -------
        directional_derivative : tf.tensor
            Directional derivative values of each samples.
        """
        with tf.GradientTape() as tape:
            tape.watch(inputs)

            activations, y_pred = multi_head_model(inputs)
            score = y_pred[:, label]

        gradients = tape.gradient(score, activations)

        # compute the directional derivatives in terms of partial derivatives
        axis_to_reduce = tf.range(1, tf.rank(gradients))
        directional_derivative = tf.reduce_sum(gradients * cav, axis=axis_to_reduce)

        return directional_derivative
