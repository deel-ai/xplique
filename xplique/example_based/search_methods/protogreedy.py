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


class Protogreedy(BaseSearchMethod):
    """
    Protogreedy method to search prototypes.

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
            self.kernel_matrix = Protogreedy.compute_kernel_matrix(self.cases_dataset, self.labels_dataset, kernel, kernel_type)
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

        self.sample_indices = tf.range(0, self.kernel_matrix.shape[0])
        self.n = self.sample_indices.shape[0]
        self.colsum = tf.reduce_sum(self.kernel_matrix, axis=0)
        self.colmean = self.colsum / self.n 
        self.colmean_times_2 = 2 * self.colmean 
    
    @staticmethod
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
    

    def compute_weighted_MMD_distance(self, Z, Zw):
        """
        weighted_MMD2 = (1/n**2) * ∑(i, j=1 to n) k(xi, xj)
                    - (2/n) * ∑(j=1 to m) Zw_j * ∑(i=1 to n) k(xi, zj)
                    + ∑(i, j=1 to m) Zw_i * Zw_j * k(zi, zj)
        """

        temp = tf.gather(tf.gather(self.kernel_matrix, Z), Z, axis=1)
        temp = temp * tf.expand_dims(Zw, axis=1)
        temp = temp * tf.expand_dims(Zw, axis=0)

        weighted_mmd2 = (1/(self.n**2)) * tf.reduce_sum(self.colsum) \
            - (2/self.n) * tf.reduce_sum(Zw * tf.gather(self.colsum, Z)) \
            + tf.reduce_sum(temp)
        
        weighted_mmd = tf.sqrt(weighted_mmd2)

        return weighted_mmd
    
    
    def compute_objective(self, S, Sw, c, cw):
        """
        Find argmax_{c} F(S ∪ c, w) - F(S, w)
        ≡
        Find argmax_{c} F(S ∪ c, w)
        ≡
        Find argmax_{c} (sum1 - sum2) where sum1 = (2 / n) * cw * ∑[i=1 to n] κ(x_i, c) and 
                                     sum2 = 2 * cw * ∑[j=1 to |S|] Sw_j * κ(x_j, c) 
                                             + (cw^2) * κ(c, c)
        """

        sum1 = tf.gather(self.colmean_times_2, c) * cw

        if S.shape[0] == 0:
            sum2 = tf.abs(tf.gather(tf.linalg.diag_part(self.kernel_matrix),c)) * (cw**2)
        else:
            temp = tf.gather(tf.gather(self.kernel_matrix, S), c, axis=1)
            temp = temp * tf.expand_dims(Sw, axis=1)
            temp = temp * tf.expand_dims(cw, axis=0)
            sum2 = tf.reduce_sum(temp, axis=0) * 2 + tf.gather(tf.linalg.diag_part(self.kernel_matrix),c) * (cw**2)
        
        objective = sum1 - sum2

        return objective
    

    def compute_weights(self, indices):

        u = tf.expand_dims(tf.gather(self.colmean, indices), axis=0)

        K = tf.gather(tf.gather(self.kernel_matrix, indices), indices, axis=1)

        # Perform matrix division (u / K)
        w = tf.matmul(u, tf.linalg.inv(K))

        w = tf.squeeze(w, axis=0)

        return w
        

    def find_prototypes(self, num_prototypes):

        is_selected = tf.zeros_like(self.sample_indices)
        selected_indices = tf.boolean_mask(self.sample_indices, is_selected > 0)
        selected_weights = self.compute_weights(selected_indices)
        
        k = 0
        while selected_indices.shape[0] < num_prototypes:
            candidate_indices = tf.boolean_mask(self.sample_indices, is_selected == 0)
            candidate_weights = self.compute_weights(candidate_indices)

            objective = self.compute_objective(selected_indices, selected_weights, candidate_indices, candidate_weights)

            best_sample_index = tf.gather(candidate_indices, tf.argmax(objective))
            is_selected = tf.tensor_scatter_nd_update(is_selected, [[best_sample_index]], [k + 1])
            selected_indices = tf.boolean_mask(self.sample_indices, is_selected > 0)
            selected_weights = self.compute_weights(selected_indices)
            k += 1

            # f_selected = 2 * tf.reduce_sum(tf.gather(colmean, selected_indices)) / selected_indices.shape[0] - tf.gather(tf.gather(self.kernel_matrix, selected_indices), selected_indices, axis=1) / selected_indices.shape[0] **2
        order = tf.boolean_mask(is_selected, is_selected > 0).numpy().argsort()
        self.prototype_indices = tf.gather(selected_indices, order)
        self.prototype_weights = tf.gather(selected_weights, order)


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
        examples_weights = self.prototype_weights

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
        if "weights" in self.returns:
            return_dict["weights"] = examples_weights

        # Return a dict only different variables are returned
        if len(return_dict) == 1:
            return list(return_dict.values())[0]
        return return_dict
