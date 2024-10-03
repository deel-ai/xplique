"""
ProtoDash search method in example-based module
"""

import numpy as np
from scipy.optimize import minimize
import tensorflow as tf

from ...types import Union, Optional, Tuple

from .proto_greedy_search import ProtoGreedySearch


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
        # pylint: disable=invalid-name

        u = u.numpy()
        K = K.numpy()

        result = minimize(self.objective_fn, self.initial_weights, args=(u, K),
                          method='SLSQP', bounds=self.bounds, options={'disp': False})

        # Get the best weights
        best_weights = result.x
        best_weights = tf.expand_dims(tf.convert_to_tensor(best_weights, dtype=tf.float32), axis=0)

        # Get the best objective
        best_objective = -result.fun
        best_objective = tf.expand_dims(tf.convert_to_tensor(best_objective, dtype=tf.float32),
                                        axis=0)

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
    batch_size
        Number of samples treated simultaneously.
        It should match the batch size of the `cases_dataset` in the case of a `tf.data.Dataset`.
    nb_prototypes : int
            Number of prototypes to find.
    kernel_fn : Callable, optional
        Kernel function, by default the rbf kernel.
        This function must only use TensorFlow operations.
    gamma : float, optional
        Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
    exact_selection_weights_update : bool, optional
        Wether to use an exact method to update selection weights, by default False.
        Exact method is based on a scipy optimization,
        while the other is based on a tensorflow inverse operation.
    """

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        batch_size: Optional[int] = 32,
        nb_prototypes: int = 1,
        kernel_fn: callable = None,
        gamma: float = None,
        exact_selection_weights_update: bool = False,
    ):

        self.exact_selection_weights_update = exact_selection_weights_update

        super().__init__(
            cases_dataset=cases_dataset,
            batch_size=batch_size,
            nb_prototypes=nb_prototypes,
            kernel_fn=kernel_fn,
            gamma=gamma
        )

    def _update_selection_weights(self,
                                  selection_kernel_col_means: tf.Tensor,
                                  selection_selection_kernel: tf.Tensor,
                                  best_diag: tf.Tensor,
                                  best_objective: tf.Tensor
                                  ) -> tf.Tensor:
        """
        Update the selection weights based on the given parameters.
        Pursuant to Lemma IV.4:
        If best_gradient ≤ 0, then
        ζ(S∪{best_sample_index}) = ζ(S) and specifically, w_{best_sample_index} = 0. 
        Otherwise, the stationarity and complementary slackness KKT conditions
        entails that w_{best_sample_index} = best_gradient / κ(best_sample_index, best_sample_index)

        Parameters
        ----------
        selection_kernel_col_means : Tensor
            Column means of the kernel matrix computed from the selected prototypes. Shape (|S|,).
        selection_selection_kernel : Tensor
            Kernel matrix computed from the selected prototypes. Shape (|S|, |S|).
        best_diag : tf.Tensor
            The diagonal element of the kernel matrix corresponding to the lastly added prototype.
            Shape (1,).
        best_objective : tf.Tensor
            The computed objective function value of the lastly added prototype. Shape (1,).
            Used to initialize the weights for the exact weights update.

        """
        # pylint: disable=invalid-name
        nb_selected = selection_kernel_col_means.shape[0]

        if best_objective <= 0:
            self.prototypes_weights[nb_selected - 1].assign(0)
        else:
            # (|S|,)
            u = selection_kernel_col_means

            # (|S|, |S|)
            K = selection_selection_kernel

            if self.exact_selection_weights_update:
                # initialize the weights
                best_objective_diag = best_objective / best_diag
                self.prototypes_weights[nb_selected - 1].assign(best_objective_diag)

                # optimize the weights
                opt = Optimizer(self.prototypes_weights[:nb_selected])
                optimized_weights, _ = opt.optimize(u[:, tf.newaxis], K)

                # update the weights
                self.prototypes_weights[:nb_selected].assign(tf.squeeze(optimized_weights, axis=0))
            else:
                # We added epsilon to the diagonal of K to ensure that K is invertible
                # (|S|, |S|)
                K_inv = tf.linalg.inv(K + ProtoDashSearch.EPSILON * tf.eye(K.shape[-1]))

                # use w* = K^-1 * u as the optimal weights
                # (|S|,)
                selection_weights = tf.linalg.matvec(K_inv, u)
                selection_weights = tf.maximum(selection_weights, 0)

                # update the weights
                self.prototypes_weights[:nb_selected].assign(selection_weights)

    def _compute_batch_objectives(self,
                                  candidates_kernel_diag: tf.Tensor,
                                  candidates_kernel_col_means: tf.Tensor,
                                  selection_kernel_col_means: tf.Tensor,
                                  candidates_selection_kernel: tf.Tensor,
                                  selection_selection_kernel: tf.Tensor
                                  ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the objective function and corresponding weights
        for a given set of selected prototypes and a candidate.
        Calculate the gradient of l(w) = w^T * μ_p - 1/2 * w^T * K * w 
        w.r.t w, on the optimal weight point ζ^(S)
        g = ∇l(ζ^(S)) = μ_p - K * ζ^(S)
        g is computed for each candidate c

        Parameters
        ----------
        candidates_kernel_diag : Tensor
            Diagonal values of the kernel matrix between the candidates and themselves. Shape (bc,).
        candidates_kernel_col_means : Tensor
            Column means of the kernel matrix, subset for the candidates. Shape (bc,).
        selection_kernel_col_means : Tensor
            Column means of the kernel matrix, subset for the selected prototypes. Shape (|S|,).
        candidates_selection_kernel : Tensor
            Kernel matrix between the candidates and the selected prototypes. Shape (bc, |S|).
        selection_selection_kernel : Tensor
            Kernel matrix between the selected prototypes. Shape (|S|, |S|).

        Returns
        -------
        objectives
            Tensor that contains the computed objective values for each candidate. Shape (bc,).
        objectives_weights
            No weights are returned in this case. It is set to None.
            The weights are computed and updated in the `_update_selection_weights` method.
        """
        # pylint: disable=invalid-name

        if candidates_selection_kernel is None:
            # (bc,)
            # S = ∅ and ζ^(∅) = 0, g = ∇l(ζ^(∅)) = μ_p
            objectives = candidates_kernel_col_means
        else:
            # (bc,) - g = μ_p - K * ζ^(S)
            objectives = candidates_kernel_col_means - tf.linalg.matvec(candidates_selection_kernel,
                                                                        selection_kernel_col_means)

        return objectives, None
