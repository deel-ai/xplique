"""
KNN using sklearn library
"""

import numpy as np
import sklearn
import tensorflow as tf

from sklearn.neighbors import NearestNeighbors

from xplique.types import List, Optional, Union

from ...attributions.base import BlackBoxExplainer, sanitize_input_output
from ...types import Callable, Dict, Tuple, Union, Optional

from .base import BaseSearchMethod


class SklearnKNN(BaseSearchMethod):
    """
    KNN method to search examples. Based on sklearn.neighbors.NearestNeighbors.

    sklearn NearestNeighbors description:
    https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors
    """  
    def __init__(self,
                 search_set: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
                 k: int = 1,
                 returns: Optional[Union[List[str], str]] = None,
                 distance: Union[str, Callable] = "euclidean",
                 **nn_args):
        super().__init__(search_set, k, returns)

        self.nn_algo = NearestNeighbors(n_neighbors=k, metric=distance, **nn_args)

        # flatten dataset for knn initialization
        flattened_search_set = tf.reshape(search_set, [search_set.shape[0], -1])
        self.nn_algo.fit(flattened_search_set)

    def find_samples(self, inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray]):
        """
        ...
        """
        # flatten inputs for knn
        flattened_inputs = tf.reshape(inputs, [inputs.shape[0], -1])
        examples_distances, examples_indices = self.nn_algo.kneighbors(
            flattened_inputs, n_neighbors=self.k)

        # Set values in return dict
        return_dict = {}
        if "examples" in self.returns:
            return_dict["examples"] = tf.gather(self.search_set, examples_indices)
            if "include_inputs" in self.returns:
                inputs = tf.expand_dims(inputs, axis=1)
                return_dict["examples"] = tf.concat([inputs, return_dict["examples"]], axis=1)
        if "indices" in self.returns:
            return_dict["indices"] = examples_indices
        if "distances" in self.returns:
            return_dict["distances"] = examples_distances

        # Return a dict only different variables are returned
        if len(return_dict) == 1:
            return return_dict.values()[0]
        else:
            return return_dict