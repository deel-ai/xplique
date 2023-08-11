"""
KNN online search method in example-based module
"""

import numpy as np
import tensorflow as tf

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .base import BaseSearchMethod
from ..projections import Projection


class KNN(BaseSearchMethod):
    """
    KNN method to search examples. Based on `sklearn.neighbors.NearestNeighbors`.
    Basically a wrapper of `NearestNeighbors` to match the `BaseSearchMethod` API.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection. See `projection` for detail.
    k
        The number of examples to retrieve.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space sould be a space where distance make sense for the model.
        It should not be `None`, otherwise,
        all examples could be computed only with the `search_method`.

        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optionnal parameters to orientated the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
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
        targets_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
    ): # pylint: disable=R0801
        super().__init__(
            cases_dataset, targets_dataset, k, projection, search_returns, batch_size
        )

        if hasattr(distance, "__call__"):
            self.distance_fn = distance
        elif distance in ["fro", "euclidean", 1, 2, np.inf] or isinstance(
            distance, int
        ):
            self.distance_fn = lambda x1, x2: tf.norm(x1 - x2, ord=distance)
        else:
            raise AttributeError(
                "The distance parameter is expected to be either a Callable or in"
                + " ['fro', 'euclidean', 'cosine', 1, 2, np.inf] ",
                +f"but {distance} was received.",
            )

        self.distance_fn_over_all_x2 = lambda x1, x2: tf.map_fn(
            fn=lambda x2: self.distance_fn(x1, x2),
            elems=x2,
        )

        # Computes crossed distances between two tensors x1(shape=(n1, ...)) and x2(shape=(n2, ...))
        # The result is a distance matrix of size (n1, n2)
        self.crossed_distances_fn = lambda x1, x2: tf.vectorized_map(
            fn=lambda a1: self.distance_fn_over_all_x2(a1, x2),
            elems=x1
        )

    def kneighbors(self, inputs: Union[tf.Tensor, np.ndarray]) -> Tuple[tf.Tensor, tf.Tensor]:
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
        best_distances = tf.Variable(tf.fill((nb_inputs, self.k), np.inf))
        # (n, bs)
        batch_indices = tf.expand_dims(tf.range(self.batch_size), axis=0)
        batch_indices = tf.tile(batch_indices, multiples=(nb_inputs, 1))

        # iterate on batches
        for batch_index, (cases, targets) in enumerate(
            zip(self.cases_dataset, self.targets_dataset)
        ):
            # project batch of dataset cases
            if self.projection is not None:
                projected_cases = self.projection.project(cases, targets)
            else:
                projected_cases = cases

            # add new elements
            # (n, current_bs, 2)
            indices = batch_indices[:, : tf.shape(projected_cases)[0]]
            new_indices = tf.stack(
                [tf.fill(indices.shape, batch_index), indices], axis=-1
            )

            # compute distances
            # (n, current_bs)
            distances = self.crossed_distances_fn(inputs, projected_cases)

            # (n, k+curent_bs, 2)
            concatenated_indices = tf.concat([best_indices, new_indices], axis=1)
            # (n, k+curent_bs)
            concatenated_distances = tf.concat([best_distances, distances], axis=1)

            # sort all
            # (n, k)
            sort_order = tf.argsort(
                concatenated_distances, axis=1, direction="ASCENDING"
            )[:, : self.k]

            best_indices.assign(
                tf.gather(concatenated_indices, sort_order, axis=1, batch_dims=1)
            )
            best_distances.assign(
                tf.gather(concatenated_distances, sort_order, axis=1, batch_dims=1)
            )

        return best_distances, best_indices

    def find_examples(self, inputs: Union[tf.Tensor, np.ndarray]):
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
        examples_distances, examples_indices = self.kneighbors(inputs)

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
