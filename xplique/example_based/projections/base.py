"""
Base projection for similar examples in example based module
"""

from abc import ABC

import tensorflow as tf
import numpy as np

from ...commons import sanitize_inputs_targets, get_device
from ...types import Callable, Union, Optional


class Projection(ABC):
    """
    Base class used by `NaturalExampleBasedExplainer` to project samples to a meaningful space
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
        Either a Tensor or a Callable.
        - In the case of a Tensor, weights are applied in the projected space.
        - In the case of a callable, a function is expected.
        It should take inputs and targets as parameters and return the weights (Tensor).
        Weights should have the same shape as the input (possible difference on channels).
        The inputs of `get_weights()` correspond to the projected inputs.

        Example of `get_weights()` function:
        ```
        def get_weights_example(projected_inputs: Union(tf.Tensor, np.ndarray),
                                targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
            '''
            Example of function to get weights,
            projected_inputs are the elements for which weights are computed.
            targets are optional additional parameters for weights computation.
            '''
            weights = ...  # do some magic with inputs and targets, it should use the model.
            return weights
        ```
    space_projection
        Callable that take samples and return a Tensor in the projected space.
        An example of projected space is the latent space of a model. See `LatentSpaceProjection`
    device
        Device to use for the projection, if None, use the default device.
    """

    def __init__(self,
                 get_weights: Optional[Union[Callable, tf.Tensor, np.ndarray]] = None,
                 space_projection: Optional[Callable] = None,
                 device: Optional[str] = None):
        assert get_weights is not None or space_projection is not None, (
            "At least one of `get_weights` and `space_projection`"
            + "should not be `None`."
        )

        # set get_weights
        if get_weights is None:
            # no weights
            self.get_weights = lambda inputs, _: tf.ones(tf.shape(inputs))
        elif isinstance(get_weights, (tf.Tensor, np.ndarray)):
            # weights is a tensor
            if isinstance(get_weights, np.ndarray):
                weights = tf.convert_to_tensor(get_weights, dtype=tf.float32)

            # define a function that returns the weights
            def get_weights(inputs, _ = None):
                nweights = tf.expand_dims(weights, axis=0)
                return tf.repeat(nweights, tf.shape(inputs)[0], axis=0)
            self.get_weights = get_weights
        elif hasattr(get_weights, "__call__"):
            # weights is a function
            self.get_weights = get_weights
        else:
            raise TypeError(
                f"`get_weights` should be `Callable` or a Tensor, not a {type(get_weights)}"
            )
        
        # set space_projection
        if space_projection is None:
            self.space_projection = lambda inputs: inputs
        elif hasattr(space_projection, "__call__"):
            self.space_projection = space_projection
        else:
            raise TypeError(
                f"`space_projection` should be a `Callable`, not a {type(space_projection)}"
            )

        # set device
        self.device = get_device(device)

    def get_input_weights(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """
        Depending on the projection, we may not be able to visualize weights
        as they are after the space projection. In this case, this method should be overwritten,
        as in `AttributionProjection` that applies an up-sampling.

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
            They are an up-sampled version of the actual weights used in the projection.
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
        with tf.device(self.device):
            projected_inputs = self.space_projection(inputs)
            weights = self.get_weights(projected_inputs, targets)
            weighted_projected_inputs =  tf.multiply(weights, projected_inputs)
        return weighted_projected_inputs

    def __call__(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ):
        """project alias"""
        return self.project(inputs, targets)

    def project_dataset(
        self,
        cases_dataset: tf.data.Dataset,
        targets_dataset: Optional[tf.data.Dataset] = None,
    ) -> Optional[tf.data.Dataset]:
        """
        Apply the projection to a dataset through `Dataset.map`

        Parameters
        ----------
        cases_dataset
            Dataset of samples to be projected.
        targets_dataset
            Dataset of targets for the samples.

        Returns
        -------
        projected_dataset
            The projected dataset.
        """
        # project dataset, note that projection is done at iteration time
        if targets_dataset is None:
            projected_cases_dataset = cases_dataset.map(self.project)
        else:
            # in case targets are provided, we zip the datasets and project them together
            projected_cases_dataset = tf.data.Dataset.zip(
                (cases_dataset, targets_dataset)
            ).map(
                lambda x, y: self.project(x, y)
            )
        
        return projected_cases_dataset
