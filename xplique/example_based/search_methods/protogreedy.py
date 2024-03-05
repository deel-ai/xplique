"""
Protogreedy search method in example-based module
"""

import numpy as np
from scipy.optimize import minimize
import sklearn
import tensorflow as tf

from sklearn.metrics.pairwise import rbf_kernel

from ...commons import dataset_gather
from ...types import Callable, List, Union, Optional, Tuple

from .base import BaseSearchMethod
from ..projections import Projection

class Optimizer():
    """
    Class to solve the quadratic problem:
    F(S) ≡ max_{w:supp(w)∈ S, w ≥ 0} l(w), 
    where l(w) = w^T * μ_p - 1/2 * w^T * K * w

    Parameters
    ----------
    initial_w
        Initial weight.
    minWeight
        Lower bound on weight.
    maxWeight
        Upper bound on weight.

    """

    def __init__(
            self, 
            initial_w, 
            minWeight = 0, 
            maxWeight = 10000
    ):
        self.initial_w = initial_w
        self.minWeight = minWeight
        self.maxWeight = maxWeight
        self.bounds = [(minWeight, maxWeight)] * initial_w.shape[0] # w >= minWeight and w <= maxWeight
        self.objective = self.create_objective()

    def create_objective(self):
        """
        Create the objective function to be optimized.

        Returns:
        The objective function, which is defined as the negative of l(w) to maximize.
        """
        def objective(w, u, K):
            return - (w @ u - 0.5 * w @ K @ w)  # Negative of l(w) to maximize
        return objective

    def optimize(self, u, K):
        """
        Perform optimization to find the optimal values of the weight vector (w) and the corresponding objective function value.

        Parameters
        ----------
        u : Tensor
            Mean similarity of each prototype.
        K : Tensor
            The kernel matrix.

        Returns
        -------
        Tuple
            A tuple containing:
            - optimal_w : Tensor
                The optimal value of the weight vector (w).
            - F_S : Tensor
                The value of the objective function corresponding to the optimal_w.
        """
        u = u.numpy()
        K = K.numpy()

        result = minimize(self.objective, self.initial_w, args=(u, K), method='SLSQP', bounds=self.bounds, options={'disp': False})

        # Extract the optimal value of w
        optimal_w = result.x
        optimal_w = tf.convert_to_tensor(optimal_w, dtype=tf.float32)
        optimal_w = tf.expand_dims(optimal_w, axis=0)

        # Compute F(S)
        F_S = -result.fun
        F_S = tf.convert_to_tensor(F_S, dtype=tf.float32)
        F_S = tf.expand_dims(F_S, axis=0)

        # Print the result
        # print("Optimal w:", optimal_w)
        # print("Optimal F(S):", F_S)  
        # print("Number of iterations (function evaluations):", result.nfev)

        assert tf.reduce_all(optimal_w >= 0)

        return optimal_w, F_S


class Protogreedy(BaseSearchMethod):
    """
    Protogreedy method for searching prototypes.

    References:
    .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
        "ProtoDash: Fast Interpretable Prototype Selection"
        <https://arxiv.org/abs/1707.01212>`_

    Parameters
    ----------
    cases_dataset : Union[tf.data.Dataset, tf.Tensor, np.ndarray]
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods, it is the train dataset.
    labels_dataset : Optional[Union[tf.data.Dataset, tf.Tensor, np.ndarray]]
        Targets associated with the cases_dataset for dataset projection.
        See `projection` for details.
    targets_dataset : Union[tf.data.Dataset, tf.Tensor, np.ndarray]
        The dataset used for projecting samples from the input space to the search space.
        The search space should be a space where distances make sense for the model.
        It should not be `None`, otherwise, all examples could be computed only with the `search_method`.
    k : int, optional
        The number of examples to retrieve, by default 1.
    projection : Union[Projection, Callable], optional
        Projection or Callable that projects samples from the input space to the search space.
        The search space should be a space where distance makes sense for the model.
        It should not be `None`, otherwise, all examples could be computed only with the `search_method`.
        Example of Callable:
        ```
        def custom_projection(inputs: tf.Tensor, np.ndarray, targets: tf.Tensor, np.ndarray = None):
            '''
            Example of projection,
            inputs are the elements to project.
            targets are optional parameters to orientate the projection.
            '''
            projected_inputs = # do some magic on inputs, it should use the model.
            return projected_inputs
        ```
    search_returns : Optional[Union[List[str], str]], optional
        String or list of strings with the elements to return in `self.find_examples()`.
        See `self.set_returns()` for details, by default None.
    batch_size : Optional[int], optional
        Number of samples treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`, by default 32.
    distance : Union[int, str, Callable], optional
        Either a Callable, or a value supported by `tf.norm` `ord` parameter.
        Supported values are 'fro', 'euclidean', 'cosine', 1, 2, np.inf,
        and any positive real number yielding the corresponding p-norm, by default "euclidean".
    kernel : Union[Callable, tf.Tensor, np.ndarray], optional
        Kernel function or kernel matrix, by default rbf_kernel.
    kernel_type : str, optional
        The kernel type. It can be 'local' or 'global', by default 'local'.
        When it is local, the distances are calculated only within the classes.
    use_optimizer : bool, optional
        Flag indicating whether to use an optimizer for prototype selection, by default False.
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
        use_optimizer: bool = False,
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

        self.use_optimizer = use_optimizer
        self.sample_indices = tf.range(0, self.kernel_matrix.shape[0])
        self.n = self.sample_indices.shape[0]
        self.colsum = tf.reduce_sum(self.kernel_matrix, axis=0)
        self.colmean = self.colsum / self.n 

    @staticmethod
    def compute_kernel_matrix(X, y, kernel, kernel_type):
        """
        Compute the kernel matrix to be used for finding prototypes.

        Parameters
        ----------
        X : Tensor
            Input data.
        y : Tensor
            Labels corresponding to the input data.
        kernel : Callable
            Function that computes the distance between two arrays.
        kernel_type : str
            The kernel type. It can be 'local' or 'global'.
            When it is local, the distances are calculated only within the classes.

        Returns
        -------
        Tensor
            The computed kernel matrix based on the specified kernel and kernel type.
        """
    
        if kernel_type == 'local':
            # only calculate distance within class. across class, distance = 0
            kernel_matrix = tf.Variable(tf.zeros((X.shape[0], X.shape[0])))
            y_unique = tf.unique(y)[0]
            for i in range(y_unique.shape[0]):
                ind = tf.where(tf.equal(y, y_unique[i]))[:, 0]
                start = tf.reduce_min(ind)
                end = tf.reduce_max(ind) + 1
                kernel_matrix[start:end, start:end].assign(kernel(X[start:end, :]))
        
        elif kernel_type == 'global':
            kernel_matrix = kernel(X)
        
        kernel_matrix = tf.cast(kernel_matrix, dtype=tf.float32)

        return kernel_matrix
    
    def compute_weighted_MMD_distance(self, Z, Zw):
        """
        Compute the weighted Maximum Mean Discrepancy (MMD) distance.

        The weighted MMD^2 is calculated using the formula:
            weighted_MMD2 = (1/n**2) * ∑(i, j=1 to n) k(xi, xj)
                            - (2/n) * ∑(j=1 to m) Zw_j * ∑(i=1 to n) k(xi, zj)
                            + ∑(i, j=1 to m) Zw_i * Zw_j * k(zi, zj)

        Parameters
        ----------
        Z : Tensor
            Indices of selected prototypes.
        Zw : Tensor
            Weights corresponding to the selected prototypes.

        Returns
        -------
        Tensor
            The computed weighted MMD distance.
        """
        temp = tf.gather(tf.gather(self.kernel_matrix, Z), Z, axis=1)
        temp = temp * tf.expand_dims(Zw, axis=1)
        temp = temp * tf.expand_dims(Zw, axis=0)

        weighted_MMD2 = (1/(self.n**2)) * tf.reduce_sum(self.colsum) \
            - (2/self.n) * tf.reduce_sum(Zw * tf.gather(self.colsum, Z)) \
            + tf.reduce_sum(temp)
        
        weighted_MMD = tf.sqrt(weighted_MMD2)

        return weighted_MMD
    
    
    def compute_objective(self, S, Sw, c):
        """
        Compute the objective function and corresponding weights for a given set of selected prototypes and a candidate.

        Parameters
        ----------
        S : Tensor
            Indices of currently selected prototypes.
        Sw : Tensor
            Weights corresponding to the selected prototypes.
        c : Tensor
            Indices of the candidate prototype to be considered.

        Returns
        -------
        Tuple
            A tuple containing:
            - objective : Tensor
                The computed objective function value.
            - objective_weights : Tensor
                The weights corresponding to the optimal solution of the objective function.
        """
        # Concatenate selection with each candidate
        # S = S ∪ {c}
        repeated_selected_indices = tf.tile(tf.expand_dims(S, 0), [c.shape[0], 1])
        all_indices = tf.concat([repeated_selected_indices, tf.expand_dims(c, 1)], axis=1)

        if self.use_optimizer:

            opt = Optimizer(initial_w=tf.concat([Sw, [0]], axis=0))
            
            objective_weights_list = []
            objective_list = []
            for c in range(all_indices.shape[0]):
            
                indices = tf.gather(all_indices, c, axis=0)

                u = tf.expand_dims(tf.gather(self.colmean, indices), axis=1)
                K = tf.gather(tf.gather(self.kernel_matrix, indices), indices, axis=1)

                optimal_w, F = opt.optimize(u, K)

                objective_weights_list.append(optimal_w)
                objective_list.append(F)
            
            objective_weights = tf.concat(objective_weights_list, axis=0)
            objective = tf.concat(objective_list, axis=0)

        else:

            # adjust all_indices to be used to gather from flattened kernel_matrix
            all_indices_adjusted = self.kernel_matrix.shape[1] * tf.expand_dims(all_indices, axis=2) + tf.expand_dims(all_indices, axis=1)

            u = tf.expand_dims(tf.gather(self.colmean, all_indices), axis=2)
            K = tf.gather(tf.reshape(self.kernel_matrix, [-1]), all_indices_adjusted)

            # determinant = tf.linalg.det(K)
            # is_not_invertible = tf.math.abs(determinant) <= 1e-6

            # We added epsilon to the diagonal of K to ensure that K is invertible          
            epsilon = 1e-6
            K_inv = tf.linalg.inv(K + epsilon * tf.eye(K.shape[-1]))

            optimal_w = tf.matmul(K_inv, u)
            optimal_w = tf.maximum(optimal_w, 0)

            F = tf.matmul(tf.transpose(optimal_w, [0, 2, 1]), u) - 0.5 * tf.matmul(tf.matmul(tf.transpose(optimal_w, [0, 2, 1]), K), optimal_w)

            objective = tf.squeeze(F, axis=[1,2])
            objective_weights = tf.reshape(optimal_w, (optimal_w.shape[0], -1))

        return objective, objective_weights
    
    def update_selection(self, selected_indices, selected_weights, objective, objective_weights, objective_argmax, best_sample_index):
        """
        Update the set of selected prototypes and their corresponding weights based on the optimization results.

        Parameters
        ----------
        selected_indices : Tensor
            Indices of currently selected prototypes.
        selected_weights : Tensor
            Weights corresponding to the selected prototypes.
        objective : Tensor
            The computed objective function values for each candidate.
        objective_weights : Tensor
            The weights corresponding to the optimal solution of the objective function for each candidate.
        objective_argmax : Tensor
            The index of the candidate with the highest objective function value.
        best_sample_index : Tensor
            The index of the selected prototype with the highest objective function value.

        Returns
        -------
        Tuple
            A tuple containing:
            - selected_indices : Tensor
                Updated indices of selected prototypes.
            - selected_weights : Tensor
                Updated weights corresponding to the selected prototypes.
        """
        # update selected_indices 
        selected_indices = tf.concat([selected_indices, [best_sample_index]], axis=0)

        # update selected_weights
        selected_weights = tf.gather(objective_weights, objective_argmax)
       
        return selected_indices, selected_weights
        
    def find_prototypes(self, num_prototypes):
        """
        Search for prototypes and their corresponding weights using an iterative selection process.

        Parameters
        ----------
        num_prototypes : int
            Number of prototypes to find.

        Returns
        -------
        None
            The function updates the object's state with the selected prototype indices and their normalized weights.
        """
        # Initialize a binary mask to keep track of selected samples.
        is_selected = tf.zeros_like(self.sample_indices)
        # Initialize empty lists to store selected indices and their corresponding weights.
        selected_indices = tf.constant([], dtype=tf.int32)
        selected_weights = tf.constant([], dtype=tf.float32)

        k = 0
        while k < num_prototypes:
            # Find candidate indices that have not been selected.
            candidate_indices = tf.boolean_mask(self.sample_indices, is_selected == 0)

            # Compute the objective function.
            objective,  objective_weights = self.compute_objective(selected_indices, selected_weights, candidate_indices)

            # Select the best sample index
            objective_argmax = tf.argmax(objective)
            best_sample_index = tf.gather(candidate_indices, objective_argmax)
            
            # Update the binary mask to mark the best sample as selected.
            is_selected = tf.tensor_scatter_nd_update(is_selected, [[best_sample_index]], [k + 1])

            # Update selected_indices and selected_weights
            selected_indices, selected_weights = self.update_selection(selected_indices, selected_weights, objective, objective_weights, objective_argmax, best_sample_index)

            k += 1

        self.prototype_indices = selected_indices
        self.prototype_weights = selected_weights
        # Normalize prototype_weights
        self.prototype_weights = self.prototype_weights / tf.reduce_sum(self.prototype_weights)


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
