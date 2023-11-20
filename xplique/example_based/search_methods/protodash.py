"""
Protodash search method in example-based module
"""

import numpy as np
from scipy.optimize import minimize
import sklearn
import tensorflow as tf

from sklearn.metrics.pairwise import rbf_kernel

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .protogreedy import Protogreedy, Optimiser


class Protodash(Protogreedy):
    """
    Protodash method for searching prototypes.

    References:
    .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
        "ProtoDash: Fast Interpretable Prototype Selection"
        <https://arxiv.org/abs/1707.01212>`_

    Parameters
    ----------
    cases_dataset : Union[tf.data.Dataset, tf.Tensor, np.ndarray]
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods, it is the train dataset.
    labels_dataset : Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]]
        Targets associated with the cases_dataset for dataset projection.
        See `projection` for details.
    targets_dataset : Union[tf.data.Dataset, tf.Tensor, np.ndarray]
        The dataset used for projecting samples from the input space to the search space.
        The search space should be a space where distances make sense for the model.
        It should not be `None`, otherwise, all examples could be computed only with the `search_method`.
    k : int, optional
        The number of examples to retrieve, by default 1.
    projection : Union[Projection, Callable], optional
        Projection or Callable that projects samples from the input space to the search space.
        The search space should be a space where distance makes sense for the model.
        It should not be `None`, otherwise, all examples could be computed only with the `search_method`.
        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optional parameters to orientate the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    search_returns : Optional[Union[List[str], str]], optional
        String or list of strings with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for details, by default None.
    batch_size : Optional[int], optional
        Number of samples treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`, by default 32.
    distance : Union[int, str, Callable], optional
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Supported values are 'fro', 'euclidean', 'cosine', 1, 2, np.inf,
        and any positive real number yielding the corresponding p-norm, by default "euclidean".
    kernel : Union[Callable, tf.Tensor, np.ndarray], optional
        Kernel function or kernel matrix, by default rbf_kernel.
    kernel_type : str, optional
        The kernel type. It can be 'local' or 'global', by default 'local'.
        When it is local, the distances are calculated only within the classes.
    use_optimiser : bool, optional
        Flag indicating whether to use an optimizer for prototype selection, by default False.
    """

    def compute_objective(self, S, Sw, c):
        """
        Compute the objective function and corresponding weights for a given set of selected prototypes and a candidate.

        Calculate the gradient of l(w) = w^T * μ_p - 1/2 * w^T * K * w 
        w.r.t w, on the optimal weight point ζ^(S)
        g = ∇l(ζ^(S)) = μ_p - K * ζ^(S)
        g is computed for each candidate c

        Parameters
        ----------
        S : Tensor
            Indices of currently selected prototypes.
        Sw : Tensor
            Weights corresponding to the selected prototypes.
        c : Tensor
            Indices of the candidate prototype to be considered.

        Returns
        -------
        Tuple
            A tuple containing:
            - objective : Tensor
                The computed objective function value.
            - objective_weights : Tensor
                The weights corresponding to the optimal solution of the objective function.
        """
        u = tf.gather(self.colmean, c)

        if S.shape[0] == 0:

            # S = ∅ and ζ^(∅) = 0, g = ∇l(ζ^(∅)) = μ_p            
            objective = u

        else:

            u = tf.expand_dims(u, axis=1)
            K = tf.gather(tf.gather(self.kernel_matrix, c), S, axis=1)
            Sw = tf.expand_dims(Sw, axis=1)

            objective = u - tf.matmul(K, Sw)
            objective = tf.squeeze(objective, axis=1)

        return objective, None 


    def update_selection(self, selected_indices, selected_weights, objective, objective_weights, objective_argmax, best_sample_index):
        """
        Update the set of selected prototypes and their corresponding weights based on the optimization results.

        Pursuant to Lemma IV.4
        If best_gradient ≤ 0, then
        ζ(S∪{best_sample_index}) = ζ(S) and specifically, w_{best_sample_index} = 0. 
        Otherwise, the stationarity and complementary slackness KKT conditions
        entails that w_{best_sample_index} = best_gradient / κ(best_sample_index, best_sample_index)
        
        Parameters
        ----------
        selected_indices : Tensor
            Indices of currently selected prototypes.
        selected_weights : Tensor
            Weights corresponding to the selected prototypes.
        objective : Tensor
            The computed objective function values for each candidate.
        objective_weights : Tensor
            The weights corresponding to the optimal solution of the objective function for each candidate.
        objective_argmax : Tensor
            The index of the candidate with the highest objective function value.
        best_sample_index : Tensor
            The index of the selected prototype with the highest objective function value.

        Returns
        -------
        Tuple
            A tuple containing:
            - selected_indices : Tensor
                Updated indices of selected prototypes.
            - selected_weights : Tensor
                Updated weights corresponding to the selected prototypes.
        """
        # update selected_indices 
        selected_indices = tf.concat([selected_indices, [best_sample_index]], axis=0)

        best_gradient = tf.gather(objective, objective_argmax)
        if best_gradient <= 0:

            selected_weights = tf.concat([selected_weights, [0]], axis=0)

        else:
            
            u = tf.expand_dims(tf.gather(self.colmean, selected_indices), axis=1)
            K = tf.gather(tf.gather(self.kernel_matrix, selected_indices), selected_indices, axis=1)

            if self.use_optimiser:

                initial_w = tf.concat([selected_weights, [best_gradient / tf.gather(tf.linalg.diag_part(self.kernel_matrix), best_sample_index)]], axis=0)
                opt = Optimiser(initial_w)

                selected_weights, _ = opt.optimize(u, K)
                selected_weights = tf.squeeze(selected_weights, axis=0)

            else:

                # We added epsilon to the diagonal of K to ensure that K is invertible
                epsilon = 1e-6
                K_inv = tf.linalg.inv(K + epsilon * tf.eye(K.shape[-1]))

                selected_weights = tf.linalg.matmul(K_inv, u)
                selected_weights = tf.maximum(selected_weights, 0)            
                selected_weights = tf.squeeze(selected_weights, axis=1)
       
        return selected_indices, selected_weights
    



 
        

