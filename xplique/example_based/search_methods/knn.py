"""
KNN online search method in example-based module
"""
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from ...commons import dataset_gather, sanitize_dataset
from ...types import Callable, List, Union, Optional, Tuple

from .base import BaseSearchMethod, ORDER
from .common import get_distance_function

class BaseKNN(BaseSearchMethod):
    """
    Base class for the KNN search methods. It is an abstract class that should be inherited by a specific KNN method.

    Parameters
    ----------
    cases_dataset
        The dataset containing the examples to search in.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        It should be a subset of `self._returns_possibilities`.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    order
        The order of the distances, either `ORDER.ASCENDING` or `ORDER.DESCENDING`. Default is `ORDER.ASCENDING`.
        ASCENDING means that the smallest distances are the best, DESCENDING means that the biggest distances are
        the best.
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        order: ORDER = ORDER.ASCENDING,
    ):
        super().__init__(
            cases_dataset=cases_dataset,
            k=k,
            search_returns=search_returns,
            batch_size=batch_size,
        )
        # set order
        assert isinstance(order, ORDER), f"order should be an instance of ORDER and not {type(order)}"
        self.order = order
        # fill value
        self.fill_value = np.inf if self.order == ORDER.ASCENDING else -np.inf
   
    @abstractmethod
    def kneighbors(self, inputs: Union[tf.Tensor, np.ndarray], targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the k-nearest neighbors to each tensor of `inputs` in `self.cases_dataset`.
        Here `self.cases_dataset` is a `tf.data.Dataset`, hence, computations are done by batches.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples on which knn are computed.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Tensor or Array. Target of the samples to be explained.

        Returns
        -------
        best_distances
            Tensor of distances between the knn and the inputs with dimension (n, k).
            The n inputs times their k-nearest neighbors.
        best_indices
            Tensor of indices of the knn in `self.cases_dataset` with dimension (n, k, 2).
            Where, n represent the number of inputs and k the number of corresponding examples.
            The index of each element is encoded by two values,
            the batch index and the index of the element in the batch.
            Those indices can be used through `xplique.commons.tf_dataset_operation.dataset_gather`.
        """
        raise NotImplementedError

    def find_examples(self, inputs: Union[tf.Tensor, np.ndarray], targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> dict:
        """
        Search the samples to return as examples. Called by the explain methods.
        It may also return the indices corresponding to the samples,
        based on `return_indices` value.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Assumed to have been already projected.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        targets
            Tensor or Array. Target of the samples to be explained.

        Returns
        -------
        return_dict
            Dictionary containing the elements to return which are specified in `self.returns`.
        """
        # compute neighbors
        examples_distances, examples_indices = self.kneighbors(inputs, targets)

        # build the return dict
        return_dict = self._build_return_dict(inputs, examples_distances, examples_indices)

        return return_dict

    def _build_return_dict(self, inputs, examples_distances, examples_indices) -> dict:
        """
        Build the return dict based on the `self.returns` values. It builds the return dict with the value in the
        subset of ['examples', 'include_inputs', 'indices', 'distances'] which is commonly shared.
        """
        # Set values in return dict
        return_dict = {}
        if "examples" in self.returns:
            return_dict["examples"] = dataset_gather(self.cases_dataset, examples_indices)
            if "include_inputs" in self.returns:
                inputs = tf.expand_dims(inputs, axis=1)
                return_dict["examples"] = tf.concat(
                    [inputs, return_dict["examples"]], axis=1
                )
        if "indices" in self.returns:
            return_dict["indices"] = examples_indices
        if "distances" in self.returns:
            return_dict["distances"] = examples_distances

        return return_dict

class KNN(BaseKNN):
    """
    KNN method to search examples. Based on `sklearn.neighbors.NearestNeighbors`.
    The kneighbors method is implemented in a batched way to handle large datasets.

    Parameters
    ----------
    cases_dataset
        The dataset containing the examples to search in.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        It should be a subset of `self._returns_possibilities`.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    order
        The order of the distances, either `ORDER.ASCENDING` or `ORDER.DESCENDING`. Default is `ORDER.ASCENDING`.
        ASCENDING means that the smallest distances are the best, DESCENDING means that the biggest distances are
        the best.
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
        order: ORDER = ORDER.ASCENDING,
    ): # pylint: disable=R0801
        super().__init__(
            cases_dataset=cases_dataset,
            k=k,
            search_returns=search_returns,
            batch_size=batch_size,
            order=order,
        )

        # set distance function
        self.distance_fn = get_distance_function(distance)

    @tf.function
    def _crossed_distances_fn(self, x1, x2) -> tf.Tensor:
        """
        Element-wise distance computation between two tensors.
        It has been vectorized to handle batches of inputs and cases.

        Parameters
        ----------
        x1
            Tensor. Input samples of shape (n, ...).
        x2
            Tensor. Cases samples of shape (m, ...).

        Returns
        -------
        distances
            Tensor of distances between the inputs and the cases with dimension (n, m).
        """
        n = x1.shape[0]
        m = x2.shape[0]
        x2 = tf.expand_dims(x2, axis=0)
        x2 = tf.repeat(x2, n, axis=0)
        # reshape for broadcasting
        x1 = tf.reshape(x1, (n, 1, -1))
        x2 = tf.reshape(x2, (n, m, -1))
        def compute_distance(args):
            a, b = args
            return self.distance_fn(a, b)
        args = (x1, x2)
        # Use vectorized_map to apply compute_distance element-wise
        distances = tf.vectorized_map(compute_distance, args)
        return distances

    def kneighbors(self, inputs: Union[tf.Tensor, np.ndarray], _ = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the k-neareast neighbors to each tensor of `inputs` in `self.cases_dataset`.
        Here `self.cases_dataset` is a `tf.data.Dataset`, hence, computations are done by batches.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples on which knn are computed.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Tensor or Array. Target of the samples to be explained.

        Returns
        -------
        best_distances
            Tensor of distances between the knn and the inputs with dimension (n, k).
            The n inputs times their k-nearest neighbors.
        best_indices
            Tensor of indices of the knn in `self.cases_dataset` with dimension (n, k, 2).
            Where, n represent the number of inputs and k the number of corresponding examples.
            The index of each element is encoded by two values,
            the batch index and the index of the element in the batch.
            Those indices can be used through `xplique.commons.tf_dataset_operation.dataset_gather`.
        """
        nb_inputs = tf.shape(inputs)[0]

        # initialize
        # (n, k, 2)
        best_indices = tf.Variable(tf.fill((nb_inputs, self.k, 2), -1))
        # (n, k)
        best_distances = tf.Variable(tf.fill((nb_inputs, self.k), self.fill_value))
        # (n, bs)
        batch_indices = tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32), axis=0)
        batch_indices = tf.tile(batch_indices, multiples=(nb_inputs, 1))

        # iterate on batches
        # for batch_index, (cases, cases_targets) in enumerate(zip(self.cases_dataset, self.targets_dataset)):
        for batch_index, cases in enumerate(self.cases_dataset):
            # add new elements
            # (n, current_bs, 2)
            indices = batch_indices[:, : tf.shape(cases)[0]]
            new_indices = tf.stack(
                [tf.fill(indices.shape, tf.cast(batch_index, tf.int32)), indices], axis=-1
            )

            # compute distances
            # (n, current_bs)
            distances = self._crossed_distances_fn(inputs, cases)

            # (n, k+curent_bs, 2)
            concatenated_indices = tf.concat([best_indices, new_indices], axis=1)
            # (n, k+curent_bs)
            concatenated_distances = tf.concat([best_distances, distances], axis=1)

            # sort all
            # (n, k)
            sort_order = tf.argsort(
                concatenated_distances, axis=1, direction=self.order.name.upper()
            )[:, : self.k]

            best_indices.assign(
                tf.gather(concatenated_indices, sort_order, axis=1, batch_dims=1)
            )
            best_distances.assign(
                tf.gather(concatenated_distances, sort_order, axis=1, batch_dims=1)
            )

        return best_distances, best_indices

class FilterKNN(BaseKNN):
    """
    KNN method to search examples. Based on `sklearn.neighbors.NearestNeighbors`.
    The kneighbors method is implemented in a batched way to handle large datasets.
    In addition, a filter function is used to select the elements to compute the distances, thus reducing the
    computational cost of the distance computation (worth if the computation of the filter is low and the matrix
    of distances is sparse).

    Parameters
    ----------
    cases_dataset
        The dataset containing the examples to search in.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets are expected to be the one-hot encoding of the model's predictions for the samples in cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        It should be a subset of `self._returns_possibilities`.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    order
        The order of the distances, either `ORDER.ASCENDING` or `ORDER.DESCENDING`. Default is `ORDER.ASCENDING`.
        ASCENDING means that the smallest distances are the best, DESCENDING means that the biggest distances are
        the best.
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    filter_fn
        A Callable that takes as inputs the inputs, their targets, the cases and their targets and
        returns a boolean mask of shape (n, m) where n is the number of inputs and m the number of cases.
        This boolean mask is used to choose between which inputs and cases to compute the distances. 
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
        order: ORDER = ORDER.ASCENDING,
        filter_fn: Optional[Callable] = None,
    ): # pylint: disable=R0801
        super().__init__(
            cases_dataset=cases_dataset,
            k=k,
            search_returns=search_returns,
            batch_size=batch_size,
            order=order,
        )

        # set distance function
        if hasattr(distance, "__call__"):
            self.distance_fn = distance
        else:
            self.distance_fn = lambda x1, x2, m:\
                tf.where(m, get_distance_function(distance)(x1, x2), self.fill_value)

        # TODO: Assertion on the function signature
        if filter_fn is None:
            filter_fn = lambda x, z, y, t: tf.ones((tf.shape(x)[0], tf.shape(z)[0]), dtype=tf.bool)
        self.filter_fn = filter_fn

        # set targets_dataset
        if targets_dataset is not None:
            self.targets_dataset = sanitize_dataset(targets_dataset, self.batch_size)
        else:
            # make an iterable of None
            self.targets_dataset = [None]*len(cases_dataset)

    @tf.function
    def _crossed_distances_fn(self, x1, x2, mask):
        """
        Element-wise distance computation between two tensors with a mask.
        It has been vectorized to handle batches of inputs and cases.

        Parameters
        ----------
        x1
            Tensor. Input samples of shape (n, ...).
        x2
            Tensor. Cases samples of shape (m, ...).
        mask
            Tensor. Boolean mask of shape (n, m). It is used to filter the elements for which the distance is computed.

        Returns
        -------
        distances
            Tensor of distances between the inputs and the cases with dimension (n, m).
        """
        n = x1.shape[0]
        m = x2.shape[0]
        x2 = tf.expand_dims(x2, axis=0)
        x2 = tf.repeat(x2, n, axis=0)
        # reshape for broadcasting
        x1 = tf.reshape(x1, (n, 1, -1))
        x2 = tf.reshape(x2, (n, m, -1))
        def compute_distance(args):
            a, b, mask = args
            return self.distance_fn(a, b, mask)
        args = (x1, x2, mask)
        # Use vectorized_map to apply compute_distance element-wise
        distances = tf.vectorized_map(compute_distance, args)
        return distances

    def kneighbors(self, inputs: Union[tf.Tensor, np.ndarray], targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the k-neareast neighbors to each tensor of `inputs` in `self.cases_dataset`.
        Here `self.cases_dataset` is a `tf.data.Dataset`, hence, computations are done by batches.
        In addition, a filter function is used to select the elements to compute the distances, thus reducing the
        computational cost of the distance computation (worth if the computation of the filter is low and the matrix
        of distances is sparse).

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples on which knn are computed.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.
        targets
            Tensor or Array. Target of the samples to be explained.

        Returns
        -------
        best_distances
            Tensor of distances between the knn and the inputs with dimension (n, k).
            The n inputs times their k-nearest neighbors.
        best_indices
            Tensor of indices of the knn in `self.cases_dataset` with dimension (n, k, 2).
            Where, n represent the number of inputs and k the number of corresponding examples.
            The index of each element is encoded by two values,
            the batch index and the index of the element in the batch.
            Those indices can be used through `xplique.commons.tf_dataset_operation.dataset_gather`.
        """
        nb_inputs = tf.shape(inputs)[0]

        # initialiaze
        # (n, k, 2)
        best_indices = tf.Variable(tf.fill((nb_inputs, self.k, 2), -1))
        # (n, k)
        best_distances = tf.Variable(tf.fill((nb_inputs, self.k), self.fill_value))
        # (n, bs)
        batch_indices = tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32), axis=0)
        batch_indices = tf.tile(batch_indices, multiples=(nb_inputs, 1))

        # iterate on batches
        for batch_index, (cases, cases_targets) in enumerate(zip(self.cases_dataset, self.targets_dataset)):
            # add new elements
            # (n, current_bs, 2)
            indices = batch_indices[:, : tf.shape(cases)[0]]
            new_indices = tf.stack(
                [tf.fill(indices.shape, tf.cast(batch_index, tf.int32)), indices], axis=-1
            )

            # get filter masks
            # (n, current_bs)
            filter_mask = self.filter_fn(inputs, cases, targets, cases_targets)

            # compute distances
            # (n, current_bs)
            distances = self._crossed_distances_fn(inputs, cases, mask=filter_mask)

            # (n, k+curent_bs, 2)
            concatenated_indices = tf.concat([best_indices, new_indices], axis=1)
            # (n, k+curent_bs)
            concatenated_distances = tf.concat([best_distances, distances], axis=1)

            # sort all
            # (n, k)
            sort_order = tf.argsort(
                concatenated_distances, axis=1, direction=self.order.name.upper()
            )[:, : self.k]

            best_indices.assign(
                tf.gather(concatenated_indices, sort_order, axis=1, batch_dims=1)
            )
            best_distances.assign(
                tf.gather(concatenated_distances, sort_order, axis=1, batch_dims=1)
            )

        return best_distances, best_indices
    