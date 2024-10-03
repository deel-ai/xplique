"""
Base projection for similar examples in example based module
"""

import warnings

import tensorflow as tf
import numpy as np

from ...commons import sanitize_inputs_targets, get_device
from ...types import Callable, Union, Optional


class Projection():
    """
    Base class used by `BaseExampleMethod` to project samples to a meaningful space
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
    mappable
        If True, the projection can be applied to a dataset through `Dataset.map`.
        Otherwise, the dataset projection will be done through a loop.
    """

    def __init__(self,
                 get_weights: Optional[Union[Callable, tf.Tensor, np.ndarray]] = None,
                 space_projection: Optional[Callable] = None,
                 device: Optional[str] = None,
                 mappable: bool = True,
                 requires_targets: bool = False):
        if get_weights is not None or space_projection is not None:
            warnings.warn(
                "At least one of `get_weights` and `space_projection`"
                + "should not be `None`. Otherwise the projection is an identity function."
        )

        self.mappable = mappable
        self.requires_targets = requires_targets

        # set get_weights
        if get_weights is None:
            # no weights
            self.get_weights = lambda inputs, _: tf.ones(tf.shape(inputs))
        elif isinstance(get_weights, (tf.Tensor, np.ndarray)):
            # weights is a tensor
            if isinstance(get_weights, np.ndarray):
                weights = tf.convert_to_tensor(get_weights, dtype=tf.float32)
            else:
                weights = get_weights

            # define a function that returns the weights
            self.get_weights = lambda inputs, _: tf.repeat(tf.expand_dims(weights, axis=0),
                                                           tf.shape(inputs)[0],
                                                           axis=0)
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
        elif isinstance(space_projection, tf.python.eager.def_function.Function):
            self.space_projection = space_projection
        elif hasattr(space_projection, "__call__"):
            self.mappable = False
            self.space_projection = space_projection
        else:
            raise TypeError(
                f"`space_projection` should be a `Callable`, not a {type(space_projection)}"
            )

        # set device
        self.device = get_device(device)

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
        if self.requires_targets and targets_dataset is None:
            warnings.warn(
                "The projection requires `targets` but `targets_dataset` is not provided. "\
                +"`targets` will be computed online, assuming a classification setting. "\
                +"Hence, online `targets` will be the predicted class one-hot-encoding. "\
                +"If this is not the expected behavior, please provide a `targets_dataset`.")

        if self.mappable:
            return self._map_project_dataset(cases_dataset, targets_dataset)
        return self._loop_project_dataset(cases_dataset, targets_dataset)

    def _map_project_dataset(
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
            ).map(self.project)

        return projected_cases_dataset

    def _loop_project_dataset(
        self,
        cases_dataset: tf.data.Dataset,
        targets_dataset: tf.data.Dataset,
    ) -> tf.data.Dataset:
        """
        Apply the projection to a dataset without `Dataset.map`.
        Because some projections are not compatible with a `tf.data.Dataset.map`.
        For example, the attribution methods, because they create a `tf.data.Dataset` for batching,
        however doing so inside a `Dataset.map` is not recommended.

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
        projected_cases_dataset = []
        batch_size = tf.shape(next(iter(cases_dataset)))[0].numpy()

        # iteratively project the dataset
        if targets_dataset is None:
            for inputs in cases_dataset:
                projected_cases_dataset.append(self.project(inputs, None))
        else:
            # in case targets are provided, we zip the datasets and project them together
            for inputs, targets in tf.data.Dataset.zip((cases_dataset, targets_dataset)):
                projected_cases_dataset.append(self.project(inputs, targets))

        projected_cases_dataset = tf.concat(projected_cases_dataset, axis=0)
        projected_cases_dataset = tf.data.Dataset.from_tensor_slices(projected_cases_dataset)
        projected_cases_dataset = projected_cases_dataset.batch(batch_size)

        return projected_cases_dataset
