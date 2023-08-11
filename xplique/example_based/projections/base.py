"""
Base projection for similar examples in example based module
"""

from abc import ABC

import tensorflow as tf
import numpy as np

from ...commons import sanitize_inputs_targets
from ...types import Callable, Union, Optional


class Projection(ABC):
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
    get_weights
        Callable, a function that return the weights (Tensor) for a given input (Tensor).
        Weights should have the same shape as the input (possible difference on channels).

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
        An example of projected space is the latent space of a model. See `LatentSpaceProjection`
    """

    def __init__(self, get_weights: Callable = None, space_projection: Callable = None):
        assert get_weights is not None or space_projection is not None, (
            "At least one of `get_weights` and `space_projection`"
            + "should not be `None`."
        )

        # set get weights
        if get_weights is None:
            # no weights
            get_weights = lambda inputs, _: tf.ones(tf.shape(inputs))
        if not hasattr(get_weights, "__call__"):
            raise TypeError(
                f"`get_weights` should be  `Callable`, not a {type(get_weights)}"
            )
        self.get_weights = get_weights

        # set space_projection
        if space_projection is None:
            space_projection = lambda inputs: inputs
        if not hasattr(space_projection, "__call__"):
            raise TypeError(
                f"`space_projection` should be a `Callable`, not a {type(space_projection)}"
            )
        self.space_projection = space_projection

    def get_input_weights(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """
        Depending on the projection, we may not be able to visualize weights
        as they are after the space projection. In this case, this method should be overwritten,
        as in `AttributionProjection` that applies an upsampling.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Additional parameter for `self.get_weights` function.

        Returns
        -------
        input_weights
            Tensor with the same dimension as `inputs` modulo the channels.
            They are an upsampled version of the actual weights used in the projection.
        """
        projected_inputs = self.space_projection(inputs)
        assert tf.reduce_all(tf.equal(projected_inputs, inputs)), (
            "Weights cannot be interpreted in the input space"
            + "if `space_projection()` is not an identity."
            + "Either remove 'weights' from the returns or"
            + "make your own projection and overwrite `get_input_weights`."
        )

        weights = self.get_weights(projected_inputs, targets)

        return weights

    @sanitize_inputs_targets
    def project(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """
        Project samples in a space meaningful for the model,
        either by weights the inputs, projecting in a latent space or both.
        This function should be called at the init and for each explanation.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Additional parameter for `self.get_weights` function.

        Returns
        -------
        projected_samples
            The samples projected in the new space.
        """
        projected_inputs = self.space_projection(inputs)
        weights = self.get_weights(projected_inputs, targets)

        return tf.multiply(weights, projected_inputs)

    def __call__(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """project alias"""
        return self.project(inputs, targets)
