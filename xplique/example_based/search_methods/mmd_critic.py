"""
MMDCritic search method in example-based module
"""

import numpy as np
import sklearn
import tensorflow as tf

from sklearn.metrics.pairwise import rbf_kernel

from xplique.example_based.projections import Projection
from xplique.types import Callable, List, Optional, Union

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .protogreedy import Protogreedy


class MMDCritic(Protogreedy):
    

    def compute_objective(self, S, Sw, c):
        
        # special case of protogreedy where we give equal weights to all prototypes, 
        # the objective here is simplified to speed up processing
        """
        Find argmax_{c} F(S ∪ c) - F(S)
        ≡
        Find argmax_{c} F(S ∪ c)
        ≡
        Find argmax_{c} (sum1 - sum2) where: sum1 = (2 / n) * ∑[i=1 to n] κ(x_i, c) 
                                                sum2 = 1/(|S|+1) [2 * ∑[j=1 to |S|] * κ(x_j, c) + κ(c, c)]
        """

        sum1 = 2 * tf.gather(self.colmean, c)

        if S.shape[0] == 0:
            sum2 = tf.abs(tf.gather(tf.linalg.diag_part(self.kernel_matrix),c))
        else:
            temp = tf.gather(tf.gather(self.kernel_matrix, S), c, axis=1)
            sum2 = tf.reduce_sum(temp, axis=0) * 2 + tf.gather(tf.linalg.diag_part(self.kernel_matrix),c)
            sum2 /= (S.shape[0] + 1)

        objective = sum1 - sum2
        objective_weights = tf.ones(shape=(c.shape[0], S.shape[0]+1), dtype=tf.float32) / tf.cast(S.shape[0]+1, dtype=tf.float32)

        return objective, objective_weights
    
    
    def compute_MMD_distance(self, Z):

        Zw = tf.ones_like(Z, dtype=tf.float32) / tf.cast(Z.shape[0], dtype=tf.float32)

        return self.compute_weighted_MMD_distance(Z, Zw)



    

 
        

