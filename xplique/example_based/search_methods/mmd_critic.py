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
    
    def compute_weights(self, selected_indices, selected_weights, candidate_indices):

        if (selected_indices.shape[0]==0):
            selected_weights = tf.constant([], dtype=tf.float32)
            candidate_weights = tf.ones_like(candidate_indices, dtype=tf.float32)

        else:
            selected_weights =  tf.ones(shape=(selected_indices.shape[0], candidate_indices.shape[0]), dtype=tf.float32) / tf.cast(selected_indices.shape[0]+1, dtype=tf.float32)
            candidate_weights = tf.ones(shape=candidate_indices.shape, dtype=tf.float32) / tf.cast(selected_indices.shape[0]+1, dtype=tf.float32)

        return selected_weights, candidate_weights
    

    def compute_MMD_distance(self, Z):

        Zw = tf.ones_like(Z, dtype=tf.float32) / tf.cast(Z.shape[0], dtype=tf.float32)

        return self.compute_weighted_MMD_distance(Z, Zw)

 
        

