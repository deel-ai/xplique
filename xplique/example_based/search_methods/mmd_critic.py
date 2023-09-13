"""
KNN online search method in example-based module
"""

import numpy as np
import sklearn
import tensorflow as tf

from sklearn.metrics.pairwise import rbf_kernel

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .base import BaseSearchMethod
from ..projections import Projection


class MMDCritic(BaseSearchMethod):
    """
    MMDCritic method to search prototypes.

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    targets_dataset
        Targets associated to the cases_dataset for dataset projection. See `projection` for detail.
    k
        The number of examples to retrieve.
    projection
        Projection or Callable that project samples from the input space to the search space.
        The search space sould be a space where distance make sense for the model.
        It should not be `None`, otherwise,
        all examples could be computed only with the `search_method`.

        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optionnal parameters to orientated the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
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
    """

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
            cases_dataset, labels_dataset, targets_dataset, k, projection, search_returns, batch_size
        )

        if hasattr(distance, "__call__"):
            self.distance_fn = distance
        elif distance in ["fro", "euclidean", 1, 2, np.inf] or isinstance(
            distance, int
        ):
            self.distance_fn = lambda x1, x2: tf.norm(x1 - x2, ord=distance)
        else:
            raise AttributeError(
                "The distance parameter is expected to be either a Callable or in"
                + " ['fro', 'euclidean', 'cosine', 1, 2, np.inf] ",
                +f"but {distance} was received.",
            )

        self.distance_fn_over_all_x2 = lambda x1, x2: tf.map_fn(
            fn=lambda x2: self.distance_fn(x1, x2),
            elems=x2,
        )

        # Computes crossed distances between two tensors x1(shape=(n1, ...)) and x2(shape=(n2, ...))
        # The result is a distance matrix of size (n1, n2)
        self.crossed_distances_fn = lambda x1, x2: tf.vectorized_map(
            fn=lambda a1: self.distance_fn_over_all_x2(a1, x2),
            elems=x1
        )

        if isinstance(kernel, Callable):
            self.kernel_matrix = MMDCritic.compute_kernel_matrix(self.cases_dataset, self.labels_dataset, kernel, kernel_type)
            # the distance for knn is computed using the given kernel
            if distance is None:
                def custom_distance(x,y):
                    x = np.expand_dims(x, axis=0)
                    y = np.expand_dims(y, axis=0)
                    distance = np.sqrt(kernel(x,x) - 2 * kernel(x,y) + kernel(y,y))
                    return distance
                distance = custom_distance
        elif isinstance(kernel, tf.Tensor) or isinstance(kernel, np.ndarray):
            self.kernel_matrix = kernel
            if distance is None:
                distance = "euclidean"

    def compute_kernel_matrix(X, y, kernel, kernel_type):
    
        if kernel_type == 'local':
            # only calculate distance within class. across class, distance = 0
            kernel_matrix = tf.Variable(tf.zeros((X.shape[0], X.shape[0])))
            y_unique = tf.unique(y)[0]
            for i in tf.range(y_unique[-1] + 1):
                ind = tf.where(tf.equal(y, y_unique[i]))[:, 0]
                start = tf.reduce_min(ind)
                end = tf.reduce_max(ind) + 1
                kernel_matrix[start:end, start:end].assign(kernel(X[start:end, :]))
        
        elif kernel_type == 'global':
            kernel_matrix = kernel(X)

        return kernel_matrix
    
    def compute_MMD2(kernel_matrix, Z):
   
        MMD2 = (1/tf.cast(kernel_matrix.shape[0], dtype=tf.float32) ** 2) * tf.reduce_sum(kernel_matrix) + \
        -(2/tf.cast(kernel_matrix.shape[0]*Z.shape[0], dtype=tf.float32)) * tf.reduce_sum(tf.gather(tf.reduce_sum(kernel_matrix, axis=0), Z)) + \
        (1/tf.cast(Z.shape[0], dtype=tf.float32) ** 2) * tf.reduce_sum(tf.gather(tf.gather(kernel_matrix, Z, axis=0), Z, axis=1))
        
        return MMD2
    
    def find_prototypes(self, num_prototypes):

        sample_indices = tf.range(0, self.kernel_matrix.shape[0])
        num_samples = sample_indices.shape[0]

        colsum = 2 * tf.reduce_sum(self.kernel_matrix, axis=0) / num_samples
        is_selected = tf.zeros_like(sample_indices)
        selected = tf.boolean_mask(sample_indices, is_selected > 0)
     
        for i in range(num_prototypes):
            candidate_indices = tf.boolean_mask(sample_indices, is_selected == 0)

            s1 = tf.gather(colsum, candidate_indices)

            if tf.size(selected) == 0:
                s2 = tf.abs(tf.gather(tf.linalg.diag_part(self.kernel_matrix),candidate_indices))
            else:
                s2 = tf.reduce_sum(tf.gather(tf.gather(tf.transpose(self.kernel_matrix), selected), candidate_indices, axis=1), axis=0) * 2 + tf.gather(tf.linalg.diag_part(self.kernel_matrix),candidate_indices)
                s2 /= tf.cast(tf.shape(selected)[0] + 1, dtype=tf.float32) 
   
            s = s1 - s2

            best_sample_index = tf.gather(candidate_indices, tf.argmax(s))
            is_selected = tf.tensor_scatter_nd_update(is_selected, [[best_sample_index]], [i + 1])
            selected = tf.boolean_mask(sample_indices, is_selected > 0)
         
        self.prototype_indices = tf.gather(selected, tf.boolean_mask(is_selected, is_selected > 0).numpy().argsort())

    def find_examples(self, inputs: Union[tf.Tensor, np.ndarray]):
        """
        Search the samples to return as examples. Called by the explain methods.
        It may also return the indices corresponding to the samples,
        based on `return_indices` value.

        Parameters
        ----------
        inputs
            Tensor or Array. Input samples to be explained.
            Assumed to have been already projected.
            Expected shape among (N, W), (N, T, W), (N, W, H, C).
        """
        # compute prototypes
        self.find_prototypes(self.k)
        examples_indices = self.prototype_indices
        examples_distances = None

        # Set values in return dict
        return_dict = {}
        if "examples" in self.returns:
            return_dict["examples"] = dataset_gather(self.cases_dataset, examples_indices)
            if "include_inputs" in self.returns:
                inputs = tf.expand_dims(inputs, axis=1)
                return_dict["examples"] = tf.concat(
                    [inputs, return_dict["examples"]], axis=1
                )
        if "indices" in self.returns:
            return_dict["indices"] = examples_indices
        if "distances" in self.returns:
            return_dict["distances"] = examples_distances

        # Return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        return return_dict
