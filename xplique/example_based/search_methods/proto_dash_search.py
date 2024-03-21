"""
ProtoDash search method in example-based module
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize
import tensorflow as tf

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .proto_greedy_search import ProtoGreedySearch
from ..projections import Projection

class Optimizer():
    """
    Class to solve the quadratic problem:
    F(S) ≡ max_{w:supp(w)∈ S, w ≥ 0} l(w), 
    where l(w) = w^T * μ_p - 1/2 * w^T * K * w

    Parameters
    ----------
    initial_weights : Tensor
        Initial weight vector.
    min_weight : float, optional
        Lower bound on weight. Default is 0.
    max_weight : float, optional
        Upper bound on weight. Default is 10000.
    """

    def __init__(
            self, 
            initial_weights: Union[tf.Tensor, np.ndarray], 
            min_weight: float = 0, 
            max_weight: float = 10000
    ):
        self.initial_weights = initial_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.bounds = [(min_weight, max_weight)] * initial_weights.shape[0]
        self.objective_fn = lambda w, u, K: - (w @ u - 0.5 * w @ K @ w)

    def optimize(self, u, K):
        """
        Perform optimization to find the optimal values of the weight vector (w) 
        and the corresponding objective function value.

        Parameters
        ----------
        u : Tensor
            Mean similarity of each prototype.
        K : Tensor
            The kernel matrix.

        Returns
        -------
        best_weights : Tensor
            The optimal value of the weight vector (w).
        best_objective : Tensor
            The value of the objective function corresponding to the best_weights.
        """

        u = u.numpy()
        K = K.numpy()

        result = minimize(self.objective_fn, self.initial_weights, args=(u, K), method='SLSQP', bounds=self.bounds, options={'disp': False})

        # Get the best weights
        best_weights = result.x
        best_weights = tf.expand_dims(tf.convert_to_tensor(best_weights, dtype=tf.float32), axis=0)

        # Get the best objective
        best_objective = -result.fun
        best_objective = tf.expand_dims(tf.convert_to_tensor(best_objective, dtype=tf.float32), axis=0)

        assert tf.reduce_all(best_weights >= 0)

        return best_weights, best_objective


class ProtoDashSearch(ProtoGreedySearch):
    """
    Protodash method for searching prototypes.

    References:
    .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
        "ProtoDash: Fast Interpretable Prototype Selection"
        <https://arxiv.org/abs/1707.01212>`_

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
    nb_prototypes : int
            Number of prototypes to find.    
    find_prototypes_kwargs
        Additional parameters passed to `find_prototypes` function.
    """

    def find_prototypes(self, nb_prototypes, kernel_type: str = 'local', kernel_fn: callable = rbf_kernel, use_optimizer: bool = False):
        """
        Search for prototypes and their corresponding weights.

        Parameters
        ----------
        nb_prototypes : int
            Number of prototypes to find.
        nb_prototypes : int
            Number of prototypes to find.
        kernel_type : str, optional
            The kernel type. It can be 'local' or 'global', by default 'local'.
            When it is local, the distances are calculated only within the classes.
        kernel_fn : Callable, optional
            Kernel function or kernel matrix, by default rbf_kernel.
        use_optimizer : bool, optional
            Flag indicating whether to use an optimizer for prototype selection, by default False.

        Returns
        -------
        prototype_indices : Tensor
            The indices of the selected prototypes.
        prototype_weights : 
            The normalized weights of the selected prototypes.
        """

        self.use_optimizer = use_optimizer

        return super().find_prototypes(nb_prototypes, kernel_type, kernel_fn)

    def update_selection_weights(self, selection_indices, selection_weights, selection_selection_kernel, best_indice, best_weights, best_objective):
        """
        Update the selection weights based on the given parameters.
        Pursuant to Lemma IV.4:
        If best_gradient ≤ 0, then
        ζ(S∪{best_sample_index}) = ζ(S) and specifically, w_{best_sample_index} = 0. 
        Otherwise, the stationarity and complementary slackness KKT conditions
        entails that w_{best_sample_index} = best_gradient / κ(best_sample_index, best_sample_index)

        Parameters
        ----------
        selected_indices : Tensor
            Indices corresponding to the selected prototypes.
        selected_weights : Tensor
            Weights corresponding to the selected prototypes.
        selection_selection_kernel : Tensor
            Kernel matrix computed from the selected prototypes.
        best_indice : int
            The index of the selected prototype with the highest objective function value.
        best_weights : Tensor
            The weights corresponding to the optimal solution of the objective function for each candidate.
        best_objective : float
            The computed objective function value.

        Returns
        -------
        selection_weights : Tensor
            Updated weights corresponding to the selected prototypes.
        """

        if best_objective <= 0:
            selection_weights = tf.concat([selection_weights, [0]], axis=0)
        else:       
            u = tf.expand_dims(tf.gather(self.col_means, selection_indices), axis=1)
            K = selection_selection_kernel

            if self.use_optimizer:
                initial_weights = tf.concat([selection_weights, [best_objective / tf.gather(self.diag, best_indice)]], axis=0)
                opt = Optimizer(initial_weights)
                selection_weights, _ = opt.optimize(u, K)
                selection_weights = tf.squeeze(selection_weights, axis=0)
            else:
                # We added epsilon to the diagonal of K to ensure that K is invertible
                K_inv = tf.linalg.inv(K + ProtoDashSearch.EPSILON * tf.eye(K.shape[-1]))
                selection_weights = tf.linalg.matmul(K_inv, u)
                selection_weights = tf.maximum(selection_weights, 0)            
                selection_weights = tf.squeeze(selection_weights, axis=1)

        return selection_weights

    def compute_objectives(self, selection_indices, selection_cases, selection_labels, selection_weights, selection_selection_kernel, candidates_indices, candidates_cases, candidates_labels, candidates_selection_kernel):
        """
        Compute the objective function and corresponding weights for a given set of selected prototypes and a candidate.
        Calculate the gradient of l(w) = w^T * μ_p - 1/2 * w^T * K * w 
        w.r.t w, on the optimal weight point ζ^(S)
        g = ∇l(ζ^(S)) = μ_p - K * ζ^(S)
        g is computed for each candidate c

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
        
        u = tf.gather(self.col_means, candidates_indices)

        if selection_indices.shape[0] == 0:
            # S = ∅ and ζ^(∅) = 0, g = ∇l(ζ^(∅)) = μ_p            
            objectives = u
        else:
            u = tf.expand_dims(u, axis=1)
            K = candidates_selection_kernel

            objectives = u - tf.matmul(K, tf.expand_dims(selection_weights, axis=1))
            objectives = tf.squeeze(objectives, axis=1)

        return objectives, None
