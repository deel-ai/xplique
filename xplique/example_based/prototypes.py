"""
Base model for prototypes
"""

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np

from ..types import Callable, Dict, List, Optional, Type, Union, DatasetOrTensor

from .datasets_operations.tf_dataset_operations import dataset_gather

from .search_methods import ProtoGreedySearch, MMDCriticSearch, ProtoDashSearch
from .search_methods import KNN, ORDER
from .projections import Projection
from .base_example_method import BaseExampleMethod


class Prototypes(BaseExampleMethod, ABC):
    """
    Base class for prototypes.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from this dataset.
        All datasets (cases, labels, and targets) should be of the same type.
        Supported types are: `tf.data.Dataset`, `torch.utils.data.DataLoader`,
        `tf.Tensor`, `np.ndarray`, `torch.Tensor`.
        For datasets with multiple columns, the first column is assumed to be the cases.
        While the second column is assumed to be the labels, and the third the targets.
        Warning: datasets tend to reshuffle at each iteration, ensure the datasets are
        not reshuffle as we use index in the dataset.
    labels_dataset
        Labels associated with the examples in the `cases_dataset`.
        It should have the same type as `cases_dataset`.
    targets_dataset
        Targets associated with the `cases_dataset` for dataset projection,
        oftentimes the one-hot encoding of a model's predictions. See `projection` for detail.
        It should have the same type as `cases_dataset`.
        It is not be necessary for all projections.
        Furthermore, projections which requires it compute it internally by default.
    nb_global_prototypes
        Number of prototypes to select to explain the dataset or the model.
        They define the number of elements returned by the `get_global_prototypes` method.
        They have a huge impact on the computation time of the method.
    nb_local_prototypes
        Number of prototypes to select to explain the decision of the model on given inputs.
        They define the number of elements returned by the `explain` method.
        (Calling this method do not make sens if `projection` is `None`.)
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space should be a space where distance make sense for the model.
        The output of the projection should be a two dimensional tensor. (nb_samples, nb_features).
        If `projection` is `None`, the model is not explained and prototypes represent the dataset.

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
        In the case of prototypes, the indices returned by local search are
        the indices of the prototypes in the list of prototypes.
        To obtain the indices of the prototypes in the dataset, use `get_global_prototypes`.
    batch_size
        Number of samples treated simultaneously for projection and search.
        Ignored if `tf.data.Dataset` are provided (these are supposed to be batched).
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable.
        By default a distance function based on the kernel_fn is used.
    kernel_fn : Callable, optional
        Kernel function, by default the rbf kernel.
        This function must only use TensorFlow operations.
    gamma : float, optional
        Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
    """
    # pylint: disable=duplicate-code

    def __init__(
        self,
        cases_dataset: DatasetOrTensor,
        labels_dataset: Optional[DatasetOrTensor] = None,
        targets_dataset: Optional[DatasetOrTensor] = None,
        nb_global_prototypes: int = 1,
        nb_local_prototypes: int = 1,
        projection: Union[Projection, Callable] = None,
        case_returns: Union[List[str], str] = "examples",
        batch_size: Optional[int] = None,
        distance: Optional[Union[int, str, Callable]] = None,
        kernel_fn: callable = None,
        gamma: float = None
    ):
        # set common example-based parameters
        super().__init__(
            cases_dataset=cases_dataset,
            labels_dataset=labels_dataset,
            targets_dataset=targets_dataset,
            k=nb_local_prototypes,
            projection=projection,
            case_returns=case_returns,
            batch_size=batch_size,
        )

        # initiate search_method and search global prototypes
        self.global_prototypes_search_method = self.search_method_class(
            cases_dataset=self.projected_cases_dataset,
            batch_size=self.batch_size,
            nb_prototypes=nb_global_prototypes,
            kernel_fn=kernel_fn,
            gamma=gamma
        )

        # get global prototypes through the indices found by the search method
        self.get_global_prototypes()

        # set knn for local explanations
        self.search_method = KNN(
            cases_dataset=self.global_prototypes_search_method.prototypes,
            search_returns=self._search_returns,
            k=self.k,
            batch_size=self.batch_size,
            distance=self.global_prototypes_search_method._get_distance_fn(distance),
            order=ORDER.ASCENDING,
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
        # pylint: disable=access-member-before-definition
        if not hasattr(self, "prototypes") or self.prototypes is None:
            assert self.global_prototypes_search_method is not None, (
                "global_prototypes_search_method is not initialized"
            )
            assert self.global_prototypes_search_method.prototypes_indices is not None, (
                "prototypes_indices are not initialized"
            )

            # (nb_prototypes, 2)
            self.prototypes_indices = self.global_prototypes_search_method.prototypes_indices
            indices = self.prototypes_indices[tf.newaxis, ...]

            # (nb_prototypes, ...)
            self.prototypes = dataset_gather(self.cases_dataset, indices)[0]

            # (nb_prototypes,)
            if self.labels_dataset is not None:
                self.prototypes_labels = dataset_gather(self.labels_dataset, indices)[0]
            else:
                self.prototypes_labels = None

            # (nb_prototypes,)
            self.prototypes_weights = self.global_prototypes_search_method.prototypes_weights

        return {
            "prototypes": self.prototypes,
            "prototypes_labels": self.prototypes_labels,
            "prototypes_weights": self.prototypes_weights,
            "prototypes_indices": self.prototypes_indices,
        }

    def format_search_output(
        self,
        search_output: Dict[str, tf.Tensor],
        inputs: Union[tf.Tensor, np.ndarray],
    ):
        """
        Format the output of the `search_method` to match the expected returns in `self.returns`.

        Parameters
        ----------
        search_output
            Dictionary with the required outputs from the `search_method`.
        inputs
            Tensor or Array. Input samples to be explained.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).

        Returns
        -------
        return_dict
            Dictionary with listed elements in `self.returns`.
            The elements that can be returned are defined with the `_returns_possibilities`
            static attribute of the class.
        """
        # initialize return dictionary
        return_dict = {}

        # indices in the list of prototypes
        # (n, k)
        flatten_indices = search_output["indices"][:, :, 0] * self.batch_size\
                          + search_output["indices"][:, :, 1]
        flatten_indices = tf.reshape(flatten_indices, [-1])

        # add examples and weights
        if "examples" in self.returns:  #  or "weights" in self.returns:
            # (n * k, ...)
            examples = tf.gather(params=self.prototypes, indices=flatten_indices)
            # (n, k, ...)
            examples = tf.reshape(examples, (inputs.shape[0], self.k) + examples.shape[1:])
            if "include_inputs" in self.returns:
                # include inputs
                inputs = tf.expand_dims(inputs, axis=1)
                examples = tf.concat([inputs, examples], axis=1)
            return_dict["examples"] = examples

        # add indices, distances, and labels
        if "indices" in self.returns:
            # convert indices in the list of prototypes to indices in the dataset
            # (n * k, 2)
            indices = tf.gather(params=self.prototypes_indices, indices=flatten_indices)
            # (n, k, 2)
            return_dict["indices"] = tf.reshape(indices, (inputs.shape[0], self.k, 2))
        if "distances" in self.returns:
            return_dict["distances"] = search_output["distances"]
        if "labels" in self.returns:
            assert (
                self.prototypes_labels is not None
            ), "The method cannot return labels without a label dataset."

            # (n * k,)
            labels = tf.gather(params=self.prototypes_labels, indices=flatten_indices)
            # (n, k)
            return_dict["labels"] = tf.reshape(labels, (inputs.shape[0], self.k))

        return return_dict


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
