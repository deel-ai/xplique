"""
MMDCritic search method in example-based module
"""

import numpy as np
import tensorflow as tf

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .proto_greedy_search import ProtoGreedySearch
from ..projections import Projection


class MMDCriticSearch(ProtoGreedySearch):
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

    def compute_objectives(self, selection_indices, selection_cases, selection_labels, selection_weights, selection_selection_kernel, candidates_indices, candidates_cases, candidates_labels, candidates_selection_kernel):
        """
        Compute the objective function and corresponding weights for a given set of selected prototypes and a candidate.

        Here, we have a special case of protogreedy where we give equal weights to all prototypes, 
        the objective here is simplified to speed up processing
        
        Find argmax_{c} F(S ∪ c) - F(S)
        ≡
        Find argmax_{c} F(S ∪ c)
        ≡
        Find argmax_{c} (sum1 - sum2) where: sum1 = (2 / n) * ∑[i=1 to n] κ(x_i, c) 
                                            sum2 = 1/(|S|+1) [2 * ∑[j=1 to |S|] * κ(x_j, c) + κ(c, c)]
            
        Parameters
        ----------
        selection_indices : Tensor
            Indices corresponding to the selected prototypes.
        selection_cases : Tensor
            Cases corresponding to the selected prototypes.
        selection_labels : Tensor
            Labels corresponding to the selected prototypes.
        selection_weights : Tensor
            Weights corresponding to the selected prototypes.
        selection_selection_kernel : Tensor
            Kernel matrix computed from the selected prototypes.
        candidates_indices : Tensor
            Indices corresponding to the candidate prototypes.
        candidates_cases : Tensor
            Cases corresponding to the candidate prototypes.
        candidates_labels : Tensor
            Labels corresponding to the candidate prototypes.
        candidates_selection_kernel : Tensor
            Kernel matrix between the candidates and the selected prototypes.

        Returns
        -------
        objectives
            Tensor that contains the computed objective values for each candidate.
        objectives_weights
            Tensor that contains the computed objective weights for each candidate.
        """  

        nb_candidates = candidates_indices.shape[0]
        nb_selection = selection_indices.shape[0]

        sum1 = 2 * tf.gather(self.col_means, candidates_indices)

        if nb_selection == 0:
            sum2 = tf.abs(tf.gather(self.diag, candidates_indices))
        else:
            temp = tf.transpose(candidates_selection_kernel, perm=[1, 0])
            sum2 = tf.reduce_sum(temp, axis=0) * 2 + tf.gather(self.diag, candidates_indices)
            sum2 /= (nb_selection + 1)

        objectives = sum1 - sum2
        objectives_weights = tf.ones(shape=(nb_candidates, nb_selection+1), dtype=tf.float32) / tf.cast(nb_selection+1, dtype=tf.float32)

        return objectives, objectives_weights
