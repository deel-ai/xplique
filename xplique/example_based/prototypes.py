"""
Base model for prototypes
"""

import math

import time

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Type, Union

from ..commons import sanitize_inputs_targets
from ..commons import sanitize_dataset, dataset_gather
from .search_methods import ProtoGreedySearch
from .projections import Projection
from .base_example_method import BaseExampleMethod

from .search_methods.base import _sanitize_returns


class Prototypes(BaseExampleMethod):
    """
    Base class for prototypes.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection. See `projection` for detail.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other dataset should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distance make sense for the model.
        It should not be `None`, otherwise,
        all examples could be computed only with the `search_method`.

        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optional parameters to orientated the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    case_returns
        String or list of string with the elements to return in `self.explain()`.
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).
    search_method_kwargs
        Parameters to be passed at the construction of the `search_method`.
    """

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        search_method: Type[ProtoGreedySearch] = ProtoGreedySearch,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        **search_method_kwargs,
    ):
        assert (
            projection is not None
        ), "`BaseExampleMethod` without `projection` is a `BaseSearchMethod`."

        # set attributes
        batch_size = self.__initialize_cases_dataset(
            cases_dataset, labels_dataset, targets_dataset, batch_size
        )

        self.k = k
        self.set_returns(case_returns)

        assert hasattr(projection, "__call__"), "projection should be a callable."

        # check projection type
        if isinstance(projection, Projection):
            self.projection = projection
        elif hasattr(projection, "__call__"):
            self.projection = Projection(get_weights=None, space_projection=projection)
        else:
            raise AttributeError(
                "projection should be a `Projection` or a `Callable`, not a"
                + f"{type(projection)}"
            )

        # project dataset
        projected_cases_dataset = self.projection.project_dataset(self.cases_dataset,
                                                                  self.targets_dataset)

        # set `search_returns` if not provided and overwrite it otherwise
        search_method_kwargs["search_returns"] = ["indices", "distances"]

        # initiate search_method
        self.search_method = search_method(
            cases_dataset=projected_cases_dataset,
            labels_dataset=labels_dataset,
            k=k,
            batch_size=batch_size,
            **search_method_kwargs,
        )
  
    def get_global_prototypes(self):
        """
        Return all the prototypes computed by the search method, 
        which consist of a global explanation of the dataset.

        Returns:
            prototype_indices : Tensor
                prototype indices. 
            prototype_weights : Tensor    
                prototype weights.
        """
        return self.search_method.prototype_indices, self.search_method.prototype_weights
    
    def __initialize_cases_dataset(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]],
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]],
        batch_size: Optional[int],
    ) -> int:
        """
        Factorization of `__init__()` method for dataset related attributes.

        Parameters
        ----------
        cases_dataset
            The dataset used to train the model, examples are extracted from the dataset.
        labels_dataset
            Labels associated to the examples in the dataset.
            Indices should match with cases_dataset.
        targets_dataset
            Targets associated to the cases_dataset for dataset projection.
            See `projection` for detail.
        batch_size
            Number of sample treated simultaneously when using the datasets.
            Ignored if `tf.data.Dataset` are provided (those are supposed to be batched).

        Returns
        -------
        batch_size
            Number of sample treated simultaneously when using the datasets.
            Extracted from the datasets in case they are `tf.data.Dataset`.
            Otherwise, the input value.
        """
        # at least one dataset provided
        if isinstance(cases_dataset, tf.data.Dataset):
            # set batch size (ignore provided argument) and cardinality
            if isinstance(cases_dataset.element_spec, tuple):
                batch_size = tf.shape(next(iter(cases_dataset))[0])[0].numpy()
            else:
                batch_size = tf.shape(next(iter(cases_dataset)))[0].numpy()

            cardinality = cases_dataset.cardinality().numpy()
        else:
            # if case_dataset is not a `tf.data.Dataset`, then neither should the other.
            assert not isinstance(labels_dataset, tf.data.Dataset)
            assert not isinstance(targets_dataset, tf.data.Dataset)
            # set batch size and cardinality
            batch_size = min(batch_size, len(cases_dataset))
            cardinality = math.ceil(len(cases_dataset) / batch_size)

        # verify cardinality and create datasets from the tensors
        self.cases_dataset = sanitize_dataset(
            cases_dataset, batch_size, cardinality
        )
        self.labels_dataset = sanitize_dataset(
            labels_dataset, batch_size, cardinality
        )
        self.targets_dataset = sanitize_dataset(
            targets_dataset, batch_size, cardinality
        )

        # if the provided `cases_dataset` has several columns
        if isinstance(self.cases_dataset.element_spec, tuple):
            # switch case on the number of columns of `cases_dataset`
            if len(self.cases_dataset.element_spec) == 2:
                assert self.labels_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels."
                    + "Hence, `labels_dataset` should be empty."
                )
                self.labels_dataset = self.cases_dataset.map(lambda x, y: y)
                self.cases_dataset = self.cases_dataset.map(lambda x, y: x)

            elif len(self.cases_dataset.element_spec) == 3:
                assert self.labels_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels."
                    + "Hence, `labels_dataset` should be empty."
                )
                assert self.targets_dataset is None, (
                    "The second column of `cases_dataset` is assumed to be the labels."
                    + "Hence, `labels_dataset` should be empty."
                )
                self.targets_dataset = self.cases_dataset.map(lambda x, y, t: t)
                self.labels_dataset = self.cases_dataset.map(lambda x, y, t: y)
                self.cases_dataset = self.cases_dataset.map(lambda x, y, t: x)
            else:
                raise AttributeError(
                    "`cases_dataset` cannot possess more than 3 columns,"
                    + f"{len(self.cases_dataset.element_spec)} were detected."
                )

        self.cases_dataset = self.cases_dataset.prefetch(tf.data.AUTOTUNE)
        if self.labels_dataset is not None:
            self.labels_dataset = self.labels_dataset.prefetch(tf.data.AUTOTUNE)
        if self.targets_dataset is not None:
            self.targets_dataset = self.targets_dataset.prefetch(tf.data.AUTOTUNE)

        return batch_size

