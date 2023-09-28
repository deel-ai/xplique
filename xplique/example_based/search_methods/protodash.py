"""
KNN online search method in example-based module
"""

import numpy as np
from scipy.optimize import minimize
import sklearn
import tensorflow as tf

from sklearn.metrics.pairwise import rbf_kernel

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .protogreedy import Protogreedy


class Protodash(Protogreedy):


    
    def compute_objective(self, S, Sw, c):
        """    
        # Calculate the gradient of l(w) = w^T * μ - 1/2 * w^T * K * w with respect to w on the point ζ^(S)
        # g = ∇l(ζ^(S)) = μ - Kζ^(S)
        """

        u = tf.gather(self.colmean, c)

        if S.shape[0] == 0:
            
            objective = u

        else:

            u = tf.expand_dims(u, axis=1)

            K = tf.gather(tf.gather(self.kernel_matrix, c), S, axis=1)

            Sw = tf.expand_dims(Sw, axis=1)

            objective = u - tf.matmul(K, Sw)

            objective = tf.squeeze(objective, axis=1)

        return objective 
    



 
        

