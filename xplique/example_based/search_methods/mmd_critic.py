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
    
    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        labels_dataset: Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]] = None,
        targets_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray] = None,
        k: int = 1,
        projection: Union[Projection, Callable] = None,
        search_returns: Optional[Union[List[str], str]] = None,
        batch_size: Optional[int] = 32,
        distance: Union[int, str, Callable] = "euclidean",
        kernel: Union[Callable, tf.Tensor, np.ndarray] = rbf_kernel,
        kernel_type: str = 'local',
    ): # pylint: disable=R0801
        super().__init__(
            cases_dataset, labels_dataset, targets_dataset, k, projection, search_returns, batch_size, distance, kernel, kernel_type
        )

        self.set_equal_weights = True
    
    
    def compute_MMD_distance(self, Z):

        Zw = tf.ones_like(Z, dtype=tf.float32) / tf.cast(Z.shape[0], dtype=tf.float32)

        return self.compute_weighted_MMD_distance(Z, Zw)



    

 
        

