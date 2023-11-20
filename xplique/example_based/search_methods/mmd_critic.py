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
    """
    MMDCritic method to search prototypes.

    References:
    .. [#] `Been Kim, Rajiv Khanna, Oluwasanmi Koyejo, 
        "Examples are not enough, learn to criticize! criticism for interpretability"
        <https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf>`_
    
    
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
    use_optimiser : bool, optional
        Flag indicating whether to use an optimizer for prototype selection, by default False.
    """
    
    def compute_objective(self, S, Sw, c):
        """
        Compute the objective function and corresponding weights for a given set of selected prototypes and a candidate.

        Here, we have a special case of protogreedy where we give equal weights to all prototypes, 
        the objective here is simplified to speed up processing
     
        Find argmax_{c} F(S ∪ c) - F(S)
        ≡
        Find argmax_{c} F(S ∪ c)
        ≡
        Find argmax_{c} (sum1 - sum2) where: sum1 = (2 / n) * ∑[i=1 to n] κ(x_i, c) 
                                                sum2 = 1/(|S|+1) [2 * ∑[j=1 to |S|] * κ(x_j, c) + κ(c, c)]
        
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
        """
        Compute the Maximum Mean Discrepancy (MMD) distance for a set of prototypes.

        Parameters
        ----------
        Z : Tensor
            Indices of the selected prototypes.

        Returns
        -------
        Tensor
            The computed MMD distance for the given set of prototypes.
        """

        Zw = tf.ones_like(Z, dtype=tf.float32) / tf.cast(Z.shape[0], dtype=tf.float32)

        return self.compute_weighted_MMD_distance(Z, Zw)



    

 
        

