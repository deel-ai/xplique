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
    Protodash method to search prototypes.

    References:
        .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
           "ProtoDash: Fast Interpretable Prototype Selection"
           <https://arxiv.org/abs/1707.01212>`_
    """

    def compute_objective(self, S, Sw, c):
        """    
        Calculate the gradient of l(w) = w^T * μ_p - 1/2 * w^T * K * w 
        w.r.t w, on the optimal weight point ζ^(S)
        g = ∇l(ζ^(S)) = μ_p - K * ζ^(S)
        g is computed for each candidate c
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

        # update selected_indices 
        selected_indices = tf.concat([selected_indices, [best_sample_index]], axis=0)

        """
        update selected_weights
        Pursuant to Lemma IV.4
        If best_gradient ≤ 0, then
        ζ(S∪{best_sample_index}) = ζ(S) and specifically, w_{best_sample_index} = 0. 
        Otherwise, the stationarity and complementary slackness KKT conditions
        entails that w_{best_sample_index} = best_gradient / κ(best_sample_index, best_sample_index)
        """
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
    



 
        

