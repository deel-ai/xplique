"""
KNN online search method in example-based module
"""
import math
from abc import abstractmethod

import numpy as np
import tensorflow as tf

from ...commons import dataset_gather, sanitize_dataset
from ...types import Callable, List, Union, Optional, Tuple

from .base import BaseSearchMethod, ORDER

class BaseKNN(BaseSearchMethod):
    """
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        order: ORDER = ORDER.ASCENDING,
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        possibilities: Optional[List[str]] = None,
    ):
        super().__init__(
            cases_dataset, k, search_returns, batch_size, targets_dataset, possibilities
        )

        assert isinstance(order, ORDER), f"order should be an instance of ORDER and not {type(order)}"
        self.order = order
        # fill value
        self.fill_value = np.inf if self.order == ORDER.ASCENDING else -np.inf
   
    @abstractmethod
    def kneighbors(self, inputs: Union[tf.Tensor, np.ndarray], targets: Optional[Union[tf.Tensor, np.ndarray]] = None) -> Tuple[tf.Tensor, tf.Tensor]:
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
            Tensor or Array. Target samples to be explained.

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

    def find_examples(self, inputs: Union[tf.Tensor, np.ndarray], targets: Optional[Union[tf.Tensor, np.ndarray]] = None):
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
        """
        # compute neighbors
        examples_distances, examples_indices = self.kneighbors(inputs, targets)

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

        # Return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        return return_dict

class KNN(BaseKNN):
    """
    KNN method to search examples. Based on `sklearn.neighbors.NearestNeighbors`.
    Basically a wrapper of `NearestNeighbors` to match the `BaseSearchMethod` API.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    distance
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Their documentation (https://www.tensorflow.org/api_docs/python/tf/norm) say:
        "Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number
        yielding the corresponding p-norm." We also added 'cosine'.
    """
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
        order: ORDER = ORDER.ASCENDING,
        targets_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        possibilities: Optional[List[str]] = None,
    ): # pylint: disable=R0801
        super().__init__(
            cases_dataset, k, search_returns, batch_size, order, targets_dataset, possibilities
        )

        if hasattr(distance, "__call__"):
            self.distance_fn = distance
        elif distance in ["fro", "euclidean", 1, 2, np.inf] or isinstance(
            distance, int
        ):
            self.distance_fn = lambda x1, x2: tf.norm(x1 - x2, ord=distance, axis=-1)
        else:
            raise AttributeError(
                "The distance parameter is expected to be either a Callable or in"
                + " ['fro', 'euclidean', 1, 2, np.inf] "
                +f"but {type(distance)} was received."
            )

    @tf.function
    def _crossed_distances_fn(self, x1, x2):
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
    TODO: Change the class description
    KNN method to search examples. Based on `sklearn.neighbors.NearestNeighbors`.
    Basically a wrapper of `NearestNeighbors` to match the `BaseSearchMethod` API.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    distance
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Their documentation (https://www.tensorflow.org/api_docs/python/tf/norm) say:
        "Supported values are 'fro', 'euclidean', 1, 2, np.inf and any positive real number
        yielding the corresponding p-norm." We also added 'cosine'.
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
        filter_fn: Optional[Callable] = None,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
        order: ORDER = ORDER.ASCENDING,
        possibilities: Optional[List[str]] = None,
    ): # pylint: disable=R0801
        super().__init__(
            cases_dataset, k, search_returns, batch_size, order, targets_dataset, possibilities
        )
        
        if hasattr(distance, "__call__"):
            self.distance_fn = distance
        elif distance in ["fro", "euclidean", 1, 2, np.inf] or isinstance(
            distance, int
        ):
            self.distance_fn = lambda x1, x2, m: tf.where(m, tf.norm(x1 - x2, ord=distance, axis=-1), self.fill_value)
        else:
            raise AttributeError(
                "The distance parameter is expected to be either a Callable or in"
                + " ['fro', 'euclidean', 1, 2, np.inf] "
                +f"but {type(distance)} was received."
            )

        # TODO: Assertion on the function signature
        if filter_fn is None:
            filter_fn = lambda x, z, y, t: tf.ones((tf.shape(x)[0], tf.shape(z)[0]), dtype=tf.bool)
        self.filter_fn = filter_fn

    @tf.function
    def _crossed_distances_fn(self, x1, x2, mask):
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

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples on which knn are computed.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
            More information in the documentation.

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
    