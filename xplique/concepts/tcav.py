"""
Module related to the Testing of CAVs
"""

import tensorflow as tf
import numpy as np

from ..commons import find_layer, batch_tensor
from ..types import Union, Optional


class Tcav:
    """
    Used to Test a Concept Activation Vector, using the sign of the directional derivative of a
    concept vector relative to a class.

    Ref. Kim & al., Interpretability Beyond Feature Attribution: Quantitative Testing with Concept
    Activation Vectors (TCAV) (2018).
    https://arxiv.org/abs/1711.11279

    Parameters
    ----------
    model
        Model to extract concept from.
    target_layer
        Index of the target layer or name of the layer.
    batch_size
        Batch size during the predictions.
    """

    def __init__(self,
                 model: tf.keras.Model,
                 target_layer: Union[str, int],
                 batch_size: Optional[int] = 64):
        self.model = model
        self.batch_size = batch_size

        # configure model bottleneck
        target_layer = find_layer(model, target_layer)
        self.multi_head = tf.keras.Model(model.input, [target_layer.output, model.output])

    def score(self,
              inputs: tf.Tensor,
              label: int,
              cav: tf.Tensor) -> float:
        """
        Compute and return the TCAV score of the CAV associated to class tested.

        Parameters
        ----------
        inputs
            Input sample on which to test the influence of the concept.
        label
            Index of the class to test.
        cav
            Concept Activation Vector, see CAV module.

        Returns
        -------
        tcav
            Percentage of sample for which increasing the concept has a positive impact on the
            class logit.
        """

        directional_derivatives = None
        label = tf.cast(label, tf.int32)
        cav = tf.cast(cav, tf.float32)

        batch_size = self.batch_size or len(inputs)

        for x_batch in batch_tensor(inputs, batch_size):
            batch_dd = Tcav.directional_derivative(self.multi_head,
                                                   x_batch, label,
                                                   cav)
            directional_derivatives = batch_dd if directional_derivatives is None else \
                tf.concat([directional_derivatives, batch_dd], axis=0)

        # tcav is the number of positive directional derivatives
        tcav = np.mean(directional_derivatives > 0.0)

        return tcav

    __call__ = score

    @staticmethod
    @tf.function
    def directional_derivative(multi_head_model: tf.keras.Model,
                               inputs: tf.Tensor,
                               label: int,
                               cav: tf.Tensor) -> tf.Tensor:
        """
        Compute the gradient of the label relative to the activations of the CAV layer.

        Parameters
        ----------
        multi_head_model
            Model reconfigured, first output is the activations of the CAV layer, and the second
            output is the prediction layer.
        inputs
            Input sample on which to test the influence of the concept.
        label
            Index of the class to test.
        cav
            Concept Activation Vector, same shape as the activations output.

        Returns
        -------
        directional_derivative
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
