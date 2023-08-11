"""
Custom, a projection from example based module
"""

import tensorflow as tf
import numpy as np

from ...types import Callable, Union

from .base import Projection


class CustomProjection(Projection):
    """
    Base class used by `NaturalExampleBasedExplainer` to projet samples to a meaningfull space
    for the model to explain.

    Projection have two parts a `space_projection` and `weights`, to apply a projection,
    the samples are first projected to a new space and then weighted.
    Either the `space_projection` or the `weights` could be `None` but,
    if both are, the projection is an identity function.

    At least one of the two part should include the model in the computation
    for distance between projected elements to make sense for the model.

    Note that the cost of this projection should be limited
    as it will be applied to all samples of the train dataset.

    Parameters
    ----------
    weights
        Either a Tensor or a Callable.
        - In the case of a Tensor, weights are applied in the projected space
        (after `space_projection`).
        Hence weights should have the same shape as a `projected_input`.
        - In the case of a Callable, the function should return the weights when called,
        as a way to get the weights (a Tensor)
        It is pertinent in the case on weights dependent on the inputs, i.e. local weighting.

        Example of `get_weights()` function:
        ```
        def get_weights_example(projected_inputs: Union(tf.Tensor, np.ndarray),
                                targets: Union(tf.Tensor, np.ndarray) = None):
            '''
            Example of function to get weights,
            projected_inputs are the elements for which weights are comlputed.
            targets are optionnal additionnal parameters for weights computation.
            '''
            weights = ...  # do some magic with inputs and targets, it should use the model.
            return weights
        ```
    space_projection
        Callable that take samples and return a Tensor in the projected sapce.
        An example of projected space is the latent space of a model.
        In this case, the model should be splitted and the
    """

    def __init__(
        self,
        weights: Union[Callable, tf.Tensor, np.ndarray] = None,
        space_projection: Callable = None,
    ):
        # Set weights or
        if weights is None or hasattr(weights, "__call__"):
            # weights is already a function or there is no weights
            get_weights = weights
        elif isinstance(weights, (tf.Tensor, np.ndarray)):
            # weights is a tensor
            if isinstance(weights, np.ndarray):
                weights = tf.convert_to_tensor(weights, dtype=tf.float32)

            # define a function that returns the weights
            def get_weights(inputs, _ = None):
                nweights = tf.expand_dims(weights, axis=0)
                return tf.repeat(nweights, tf.shape(inputs)[0], axis=0)

        else:
            raise TypeError(
                "`weights` should be a tensor or a `Callable`,"
                + f"not a {type(weights)}"
            )

        # Set space_projection
        if space_projection is not None and not hasattr(space_projection, "__call__"):
            raise TypeError(
                "`space_projection` should be a `Callable`,"
                + f"not a {type(space_projection)}"
            )

        super().__init__(get_weights, space_projection)
