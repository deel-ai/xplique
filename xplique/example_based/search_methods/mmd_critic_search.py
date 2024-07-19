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
    MMDCritic method to search prototypes.

    References:
    .. [#] `Been Kim, Rajiv Khanna, Oluwasanmi Koyejo, 
        "Examples are not enough, learn to criticize! criticism for interpretability"
        <https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf>`_

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    labels_dataset
        Labels associated to the examples in the dataset. Indices should match with cases_dataset.
    k
        The number of examples to retrieve.
    search_returns
        String or list of string with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for detail.
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    distance
        Distance function for examples search. It can be an integer, a string in
        {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable,
        by default "euclidean".
    nb_prototypes : int
            Number of prototypes to find.    
    kernel_type : str, optional
        The kernel type. It can be 'local' or 'global', by default 'local'.
        When it is local, the distances are calculated only within the classes.
    kernel_fn : Callable, optional
        Kernel function, by default the rbf kernel.
        This function must only use TensorFlow operations.
    gamma : float, optional
        Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
    """

    def compute_objectives(self, selection_indices, selection_cases, selection_weights, selection_selection_kernel, candidates_indices, candidates_selection_kernel):
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
        selection_weights : Tensor
            Weights corresponding to the selected prototypes.
        selection_selection_kernel : Tensor
            Kernel matrix computed from the selected prototypes.
        candidates_indices : Tensor
            Indices corresponding to the candidate prototypes.
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
