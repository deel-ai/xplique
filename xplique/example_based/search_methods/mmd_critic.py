"""
KNN online search method in example-based module
"""

import numpy as np
import sklearn
import tensorflow as tf

from sklearn.metrics.pairwise import rbf_kernel

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .protogreedy import Protogreedy


class MMDCritic(Protogreedy):

    def compute_weights(self, indices):

        w = tf.ones_like(indices, dtype=tf.float32) / tf.cast(indices.shape[0], dtype=tf.float32)

        return w
    
    def compute_MMD_distance(self, Z):
        """
        MMD2 = (1/n**2) * ∑(i, j=1 to n) k(xi, xj)
            - (2/n*m) * ∑(j=1 to m) Zw_j * ∑(i=1 to n) k(xi, zj)
            + (1/m**2) * ∑(i, j=1 to m) k(zi, zj)
        """

        Zw = tf.ones_like(Z, dtype=tf.float32) / tf.cast(Z.shape[0], dtype=tf.float32)
        return MMDCritic.compute_weighted_MMD_distance(self, Z, Zw)

 
        

