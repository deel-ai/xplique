"""
Base model for prototypes
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Type, Union

from ..commons.tf_dataset_operations import dataset_gather

from .search_methods import ProtoGreedySearch, MMDCriticSearch, ProtoDashSearch
from .projections import Projection
from .base_example_method import BaseExampleMethod


class Prototypes(BaseExampleMethod, ABC):
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
        For decision explanations, the number of closest prototypes to return. Used in `explain`.
        Default is 1, which means that only the closest prototype is returned.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distance make sense for the model.
        The output of the projection should be a two dimensional tensor. (nb_samples, nb_features).
        `projection` should not be `None`, otherwise,
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
        Ignored if `tf.data.Dataset` are provided (these are supposed to be batched).
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    nb_prototypes : int
        For general explanations, the number of prototypes to select.
        If `class_wise` is True, it will correspond to the number of prototypes per class.    
    kernel_type : str, optional
        The kernel type. It can be 'local' or 'global', by default 'local'.
        When it is local, the distances are calculated only within the classes.
    kernel_fn : Callable, optional
        Kernel function, by default the rbf kernel.
        This function must only use TensorFlow operations.
    gamma : float, optional
        Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=duplicate-code

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = None,
        nb_prototypes: int = 1,
        kernel_type: str = 'local',
        kernel_fn: callable = None,
        gamma: float = None
    ):
        # set common example-based parameters
        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=k,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # set prototypes parameters
        self.distance = distance
        self.nb_prototypes = nb_prototypes
        self.kernel_type = kernel_type
        self.kernel_fn = kernel_fn
        self.gamma = gamma

        # initiate search_method
        self.search_method = self.search_method_class(
            cases_dataset=self.projected_cases_dataset,
            labels_dataset=self.labels_dataset,
            k=self.k,
            search_returns=self._search_returns,
            batch_size=self.batch_size,
            distance=self.distance,
            nb_prototypes=self.nb_prototypes,
            kernel_type=self.kernel_type,
            kernel_fn=self.kernel_fn,
            gamma=self.gamma
        )

    @property
    @abstractmethod
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        raise NotImplementedError

    def get_global_prototypes(self) -> Dict[str, tf.Tensor]:
        """
        Provide the global prototypes computed at the initialization.
        Prototypes and their labels are extracted from the indices.
        The weights of the prototypes and their indices are also returned. 

        Returns
        -------
        prototypes_dict : Dict[str, tf.Tensor]
            A dictionary with the following
            - 'prototypes': The prototypes found by the method.
            - 'prototype_labels': The labels of the prototypes.
            - 'prototype_weights': The weights of the prototypes.
            - 'prototype_indices': The indices of the prototypes.
        """
        # (nb_prototypes,)
        indices = self.search_method.prototypes_indices
        batch_indices = indices // self.batch_size
        elem_indices = indices % self.batch_size

        # (nb_prototypes, 2)
        batch_elem_indices = tf.stack([batch_indices, elem_indices], axis=1)

        # (1, nb_prototypes, 2)
        batch_elem_indices = tf.expand_dims(batch_elem_indices, axis=0)

        # (nb_prototypes, ...)
        prototypes = dataset_gather(self.cases_dataset, batch_elem_indices)[0]

        # (nb_prototypes,)
        labels = dataset_gather(self.labels_dataset, batch_elem_indices)[0]

        # (nb_prototypes,)
        weights = self.search_method.prototypes_weights

        return {
            "prototypes": prototypes,
            "prototypes_labels": labels,
            "prototypes_weights": weights,
            "prototypes_indices": indices,
        }


class ProtoGreedy(Prototypes):
    # pylint: disable=missing-class-docstring
    @property
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        return ProtoGreedySearch


class MMDCritic(Prototypes):
    # pylint: disable=missing-class-docstring
    @property
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        return MMDCriticSearch


class ProtoDash(Prototypes):
    # pylint: disable=missing-class-docstring
    @property
    def search_method_class(self) -> Type[ProtoGreedySearch]:
        return ProtoDashSearch
