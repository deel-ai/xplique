"""
Define the KLEOR search method.
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from ...types import Callable, List, Optional, Tuple, Union
from ..datasets_operations.tf_dataset_operations import dataset_gather
from .base import ORDER
from .knn import FilterKNN


class BaseKLEORSearch(FilterKNN, ABC):
    """
    Base class for the KLEOR search methods.
    In those methods, one should first retrieve the Nearest Unlike Neighbor (NUN)
    which is the closest example to the query that has a different prediction than the query.
    Then, the method search for the K-Nearest Neighbors (KNN)
    of the NUN that have the same prediction as the query.

    Depending on the KLEOR method some additional condition for the search are added.
    See the specific KLEOR method for more details.

    Parameters
    ----------
    cases_dataset
        The dataset used to search the examples.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    targets_dataset
        Targets are expected to be the one-hot encoding of the model's predictions
        for the samples in cases_dataset.
        `tf.data.Dataset` are assumed to be batched as tensorflow provide no method to verify it.
        Batch size and cardinality of other datasets should match `cases_dataset`.
        Be careful, `tf.data.Dataset` are often reshuffled at each iteration, be sure that it is not
        the case for your dataset, otherwise, examples will not make sense.
    k
        The number of examples to retrieve per input.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        It should be a subset of `self._returns_possibilities`.
    batch_size
        Number of samples treated simultaneously.
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    """

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        k: int = 1,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
    ):
        super().__init__(
            cases_dataset=cases_dataset,
            targets_dataset=targets_dataset,
            k=k,
            search_returns=search_returns,
            batch_size=batch_size,
            distance=distance,
            order=ORDER.ASCENDING,
            filter_fn=self._filter_fn,
        )

        # search method for the Nearest Unlike Neighbors
        self.search_nuns = FilterKNN(
            cases_dataset=cases_dataset,
            targets_dataset=targets_dataset,
            k=1,
            search_returns=["indices", "distances"],
            batch_size=batch_size,
            distance=distance,
            order=ORDER.ASCENDING,
            filter_fn=self._filter_fn_nun,
        )

    def find_examples(
        self,
        inputs: Union[tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> dict:
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
        examples_distances, examples_indices, nuns, nuns_indices, nuns_sf_distances = (
            self.kneighbors(inputs, targets)
        )

        # build return dict
        return_dict = self._build_return_dict(inputs, examples_distances, examples_indices)

        # add the nuns if needed
        if "nuns" in self.returns:
            return_dict["nuns"] = nuns

        if "dist_to_nuns" in self.returns:
            return_dict["dist_to_nuns"] = nuns_sf_distances

        if "nuns_indices" in self.returns:
            return_dict["nuns_indices"] = nuns_indices

        return return_dict

    def _filter_fn(self, _, __, targets, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases
        for which the prediction is the same as the predicted label on the inputs.
        """
        # get the labels predicted by the model
        # (n, )
        predicted_labels = tf.argmax(targets, axis=-1)
        label_targets = tf.argmax(cases_targets, axis=-1)
        # for each input, if the target label is the same as the cases label
        # the mask as a True value and False otherwise
        mask = tf.equal(tf.expand_dims(predicted_labels, axis=1), label_targets)
        return mask

    def _filter_fn_nun(self, _, __, targets, cases_targets) -> tf.Tensor:
        """
        Filter function to mask the cases for which the label is different from the predicted
        label on the inputs.
        """
        # get the labels predicted by the model
        # (n, )
        predicted_labels = tf.argmax(targets, axis=-1)
        label_targets = tf.argmax(cases_targets, axis=-1)  # (bs,)
        # for each input, if the target label is the same as the predicted label
        # the mask as a False value and True otherwise
        mask = tf.not_equal(tf.expand_dims(predicted_labels, axis=1), label_targets)  # (n, bs)
        return mask

    def _get_nuns(
        self, inputs: Union[tf.Tensor, np.ndarray], targets: Union[tf.Tensor, np.ndarray]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Get the Nearest Unlike Neighbors and their distance to the related input.
        """
        nuns_dict = self.search_nuns(inputs, targets)
        nuns_indices, nuns_distances = nuns_dict["indices"], nuns_dict["distances"]
        nuns = dataset_gather(self.cases_dataset, nuns_indices)
        return nuns, nuns_indices, nuns_distances

    def kneighbors(
        self, inputs: Union[tf.Tensor, np.ndarray], targets: Union[tf.Tensor, np.ndarray]
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the k SF to each tensor of `inputs` in `self.cases_dataset`.
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
        input_sf_distances
            Tensor of distances between the SFs and the inputs with dimension (n, k).
            The n inputs times their k-SF.
        sf_indices
            Tensor of indices of the SFs in `self.cases_dataset` with dimension (n, k, 2).
            Where, n represent the number of inputs and k the number of corresponding SFs.
            The index of each element is encoded by two values,
            the batch index and the index of the element in the batch.
            Those indices can be used through `xplique.commons.tf_dataset_operation.dataset_gather`.
        nuns
            Tensor of Nearest Unlike Neighbors with dimension (n, 1, ...).
            The n inputs times their NUN.
        nuns_indices
            Tensor of indices of the NUN in `self.cases_dataset` with dimension (n, 1, 2).
            Where, n represent the number of inputs.
            The index of each element is encoded by two values,
            the batch index and the index of the element in the batch.
            Those indices can be used through `xplique.commons.tf_dataset_operation.dataset_gather`.
        nun_sf_distances
            Tensor of distances between the SFs and the NUN with dimension (n, k).
            The n NUNs times the k-SF.
        """
        # pylint: disable=signature-differs
        # pylint: disable=duplicate-code
        # get the Nearest Unlike Neighbors and their distance to the related input
        nuns, nuns_indices, nuns_input_distances = self._get_nuns(inputs, targets)

        # initialize the search for the KLEOR semi-factual methods
        sf_indices, input_sf_distances, nun_sf_distances, batch_indices = self._initialize_search(
            inputs
        )

        # iterate on batches
        for batch_index, (cases, cases_targets) in enumerate(
            zip(self.cases_dataset, self.targets_dataset)
        ):
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
            b_nun_sf_distances = self._crossed_distances_fn(nuns, cases, mask=filter_mask)
            b_input_sf_distances = self._crossed_distances_fn(inputs, cases, mask=filter_mask)

            # additional filtering
            b_nun_sf_distances, b_input_sf_distances = self._additional_filtering(
                b_nun_sf_distances, b_input_sf_distances, nuns_input_distances
            )
            # concatenate distances and indices
            # (n, k+curent_bs, 2)
            concatenated_indices = tf.concat([sf_indices, new_indices], axis=1)
            # (n, k+curent_bs)
            concatenated_nun_sf_distances = tf.concat(
                [nun_sf_distances, b_nun_sf_distances], axis=1
            )
            concatenated_input_sf_distances = tf.concat(
                [input_sf_distances, b_input_sf_distances], axis=1
            )

            # sort according to the smallest distances between sf and nun
            # (n, k)
            sort_order = tf.argsort(
                concatenated_nun_sf_distances, axis=1, direction=self.order.name.upper()
            )[:, : self.k]

            sf_indices.assign(tf.gather(concatenated_indices, sort_order, axis=1, batch_dims=1))
            nun_sf_distances.assign(
                tf.gather(concatenated_nun_sf_distances, sort_order, axis=1, batch_dims=1)
            )
            input_sf_distances.assign(
                tf.gather(concatenated_input_sf_distances, sort_order, axis=1, batch_dims=1)
            )

        return input_sf_distances, sf_indices, nuns, nuns_indices, nun_sf_distances

    def _initialize_search(
        self, inputs: Union[tf.Tensor, np.ndarray]
    ) -> Tuple[tf.Variable, tf.Variable, tf.Variable, tf.Tensor]:
        """
        Initialize the search for the KLEOR semi-factual methods.
        """
        nb_inputs = tf.shape(inputs)[0]

        # sf_indices shape (n, k, 2)
        sf_indices = tf.Variable(tf.fill((nb_inputs, self.k, 2), -1))
        # (n, k)
        input_sf_distances = tf.Variable(tf.fill((nb_inputs, self.k), self.fill_value))
        nun_sf_distances = tf.Variable(tf.fill((nb_inputs, self.k), self.fill_value))
        # (n, bs)
        batch_indices = tf.expand_dims(tf.range(self.batch_size, dtype=tf.int32), axis=0)
        batch_indices = tf.tile(batch_indices, multiples=(nb_inputs, 1))
        return sf_indices, input_sf_distances, nun_sf_distances, batch_indices

    @abstractmethod
    def _additional_filtering(
        self,
        nun_sf_distances: tf.Tensor,
        input_sf_distances: tf.Tensor,
        nuns_input_distances: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Additional filtering to apply to the distances.
        """
        raise NotImplementedError


class KLEORSimMissSearch(BaseKLEORSearch):
    """
    The KLEORSimMiss method search for Semi-Factuals examples
    by searching for the Nearest Unlike Neighbor (NUN) of the query.
    The NUN is the closest example to the query that has a different prediction than the query.
    Then, the method search for the K-Nearest Neighbors (KNN)
    of the NUN that have the same prediction as the query.
    """

    def _additional_filtering(
        self,
        nun_sf_distances: tf.Tensor,
        input_sf_distances: tf.Tensor,
        nuns_input_distances: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        No additional filtering for the KLEORSimMiss method.
        """
        return nun_sf_distances, input_sf_distances


class KLEORGlobalSimSearch(BaseKLEORSearch):
    """
    The KLEORGlobalSim method search for Semi-Factuals examples
    by searching for the Nearest Unlike Neighbor (NUN) of the query.
    The NUN is the closest example to the query that has a different prediction than the query.
    Then, the method search for the K-Nearest Neighbors (KNN)
    of the NUN that have the same prediction as the query.

    In addition, for a SF candidate to be considered,
    the SF should be closer to the query than the NUN
    (i.e. the SF should be 'between' the input and its NUN).
    This condition is added to the search.
    """

    def _additional_filtering(
        self,
        nun_sf_distances: tf.Tensor,
        input_sf_distances: tf.Tensor,
        nuns_input_distances: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Filter the distances to keep only the SF that are 'between' the input and its NUN.

        Parameters
        ----------
        nun_sf_distances
            Distances between the SF and the NUN.
        input_sf_distances
            Distances between the SF and the input.
        nuns_input_distances
            Distances between the input and the NUN.

        Returns
        -------
        nun_sf_distances
            Filtered distances between the SF and the NUN.
        input_sf_distances
            Filtered distances between the SF and the input.
        """
        # filter non acceptable cases, i.e. cases for which the distance to the input is greater
        # than the distance between the input and its nun
        # (n, current_bs)
        mask = tf.less(input_sf_distances, nuns_input_distances)
        nun_sf_distances = tf.where(mask, nun_sf_distances, self.fill_value)
        input_sf_distances = tf.where(mask, input_sf_distances, self.fill_value)
        return nun_sf_distances, input_sf_distances
