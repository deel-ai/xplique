"""
ProtoGreedy search method in example-based module
"""

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from ...commons import dataset_gather, sanitize_dataset
from ...types import Callable, List, Union, Optional, Tuple

from .prototypes_search import PrototypesSearch
from ..projections import Projection


class ProtoGreedySearch(PrototypesSearch):
    """
    ProtoGreedy method for searching prototypes.

    References:
    .. [#] `Karthik S. Gurumoorthy, Amit Dhurandhar, Guillermo Cecchi,
        "ProtoDash: Fast Interpretable Prototype Selection"
        <https://arxiv.org/abs/1707.01212>`_

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    k
        The number of examples to retrieve.
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
    nb_prototypes : int
            Number of prototypes to find.    
    find_prototypes_kwargs
        Additional parameters passed to `find_prototypes` function.
    """

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = tf.constant(1e-6)

    def compute_objectives(self, selection_indices, selection_cases, selection_labels, selection_weights, selection_selection_kernel, candidates_indices, candidates_cases, candidates_labels, candidates_selection_kernel):
        """
        Compute the objective and its weights for each candidate.

        Parameters
        ----------
        selection_indices : Tensor
            Indices corresponding to the selected prototypes.
        selection_cases : Tensor
            Cases corresponding to the selected prototypes.
        selection_labels : Tensor
            Labels corresponding to the selected prototypes.
        selection_weights : Tensor
            Weights corresponding to the selected prototypes.
        selection_selection_kernel : Tensor
            Kernel matrix computed from the selected prototypes.
        candidates_indices : Tensor
            Indices corresponding to the candidate prototypes.
        candidates_cases : Tensor
            Cases corresponding to the candidate prototypes.
        candidates_labels : Tensor
            Labels corresponding to the candidate prototypes.
        candidates_selection_kernel : Tensor
            Kernel matrix between the candidates and the selected prototypes.

        Returns
        -------
        objectives
            Tensor that contains the computed objective values for each candidate.
        objectives_weights
            Tensor that contains the computed objective weights for each candidate.
        """  

        nb_candidates = candidates_indices.shape[0]
        nb_selection = selection_cases.shape[0]

        repeated_selection_indices = tf.tile(tf.expand_dims(selection_indices, 0), [nb_candidates, 1])
        repeated_selection_candidates_indices = tf.concat([repeated_selection_indices, tf.expand_dims(candidates_indices, 1)], axis=1)            
        u = tf.expand_dims(tf.gather(self.col_means, repeated_selection_candidates_indices), axis=2)

        if nb_selection == 0:
            K = tf.expand_dims(tf.expand_dims(tf.gather(self.diag, candidates_indices), axis=-1), axis=-1) 
        else:
            repeated_selection_selection_kernel = tf.tile(tf.expand_dims(selection_selection_kernel, 0), [nb_candidates, 1, 1])
            repeated_selection_selection_kernel = tf.pad(repeated_selection_selection_kernel, [[0, 0], [0, 1], [0, 1]])

            candidates_diag = tf.expand_dims(tf.expand_dims(tf.gather(self.diag, candidates_indices), axis=-1), axis=-1)
            candidates_diag = tf.pad(candidates_diag, [[0, 0], [nb_selection, 0], [nb_selection, 0]])

            candidates_selection_kernel = tf.expand_dims(candidates_selection_kernel, axis=-1)
            candidates_selection_kernel = tf.pad(candidates_selection_kernel, [[0, 0], [0, 1], [nb_selection, 0]])

            K = repeated_selection_selection_kernel + candidates_diag + candidates_selection_kernel + tf.transpose(candidates_selection_kernel, [0, 2, 1])

        # Compute the objective weights for each candidate in the batch         
        K_inv = tf.linalg.inv(K + ProtoGreedySearch.EPSILON * tf.eye(K.shape[-1]))
        objectives_weights = tf.matmul(K_inv, u)
        objectives_weights = tf.maximum(objectives_weights, 0)
        
        # Compute the objective for each candidate in the batch
        objectives = tf.matmul(tf.transpose(objectives_weights, [0, 2, 1]), u) - 0.5 * tf.matmul(tf.matmul(tf.transpose(objectives_weights, [0, 2, 1]), K), objectives_weights)
        objectives = tf.squeeze(objectives, axis=[1,2])

        return objectives, objectives_weights

    def update_selection_weights(self, selection_indices, selection_weights, selection_selection_kernel, best_indice, best_weights, best_objective):
        """
        Update the selection weights based on the optimization results.

        Parameters
        ----------
        selected_indices : Tensor
            Indices corresponding to the selected prototypes.
        selected_weights : Tensor
            Weights corresponding to the selected prototypes.
        selection_selection_kernel : Tensor
            Kernel matrix computed from the selected prototypes.
        best_indice : int
            The index of the selected prototype with the highest objective function value.
        best_weights : Tensor
            The weights corresponding to the optimal solution of the objective function for each candidate.
        best_objective : float
            The computed objective function value.

        Returns
        -------
        selection_weights : Tensor
            Updated weights corresponding to the selected prototypes.
        """

        selection_weights = best_weights

        return selection_weights
    
    def compute_kernel_attributes(self, kernel_type: str = 'local', kernel_fn: callable = rbf_kernel):
        """
        Compute the attributes of the class that are related to the kernel.

        Parameters
        ----------
        kernel_type : str, optional
            The kernel type. It can be 'local' or 'global', by default 'local'.
            When it is local, the distances are calculated only within the classes.
        kernel_fn : Callable, optional
            Kernel function or kernel matrix, by default rbf_kernel.

        Returns
        -------
        selection_weights : Tensor
            Updated weights corresponding to the selected prototypes.
        """
        if kernel_type in ['local', 'global']:
            self.kernel_type = kernel_type
        else:
            raise AttributeError(
                "The kernel_type parameter is expected to be in"
                + " ['local', 'global'] ",
                +f"but {kernel_type} was received.",
            )
        
        if hasattr(kernel_fn, "__call__"):
            def custom_kernel_fn(x1, x2, y1, y2):
                if self.kernel_type == 'global':
                    kernel_matrix = kernel_fn(x1,x2)
                    if isinstance(kernel_matrix, np.ndarray):
                        kernel_matrix = tf.convert_to_tensor(kernel_matrix)
                else:
                    # In the case of a local kernel, calculations are limited to within the class. 
                    # Across different classes, the kernel values are set to 0.
                    kernel_matrix = np.zeros((x1.shape[0], x2.shape[0]), dtype=np.float32)
                    y_intersect = np.intersect1d(y1, y2)
                    for i in range(y_intersect.shape[0]):
                        y1_indices = tf.where(tf.equal(y1, y_intersect[i]))[:, 0]
                        y2_indices = tf.where(tf.equal(y2, y_intersect[i]))[:, 0] 
                        sub_matrix = kernel_fn(tf.gather(x1, y1_indices), tf.gather(x2, y2_indices))             
                        kernel_matrix[tf.reshape(y1_indices, (-1, 1)), tf.reshape(y2_indices, (1, -1))] = sub_matrix
                    kernel_matrix = tf.convert_to_tensor(kernel_matrix)
                return kernel_matrix

            self.kernel_fn = custom_kernel_fn
        else:
            raise AttributeError(
                "The kernel parameter is expected to be a Callable",
                +f"but {kernel_fn} was received.",
            ) 

        # TODO: for local explanation add the ability to compute distance_fn based on the kernel  

        # Compute the sum of the columns and the diagonal values of the kernel matrix of the dataset.
        # We take advantage of the symmetry of this matrix to traverse only its lower triangle.
        col_sums = []
        diag = []
        row_sums = []
        
        for batch_col_index, (batch_col_cases, batch_col_labels) in enumerate(
            zip(self.cases_dataset, self.labels_dataset)
        ):
            batch_col_sums = tf.zeros((batch_col_cases.shape[0]))

            for batch_row_index, (batch_row_cases, batch_row_labels) in enumerate(
                zip(self.cases_dataset, self.labels_dataset)
            ):
                if batch_row_index < batch_col_index:
                    continue
                
                batch_kernel = self.kernel_fn(batch_row_cases, batch_col_cases, batch_row_labels, batch_col_labels)

                batch_col_sums = batch_col_sums + tf.reduce_sum(batch_kernel, axis=0)

                if batch_col_index == batch_row_index:        
                    if batch_col_index != 0:
                        batch_col_sums = batch_col_sums + row_sums[batch_row_index]                   
                
                    diag.append(tf.linalg.diag_part(batch_kernel))

                if batch_col_index == 0:
                    if batch_row_index == 0:
                        row_sums.append(None)
                    else:
                        row_sums.append(tf.reduce_sum(batch_kernel, axis=1))
                else:
                    row_sums[batch_row_index] += tf.reduce_sum(batch_kernel, axis=1)
                      
            col_sums.append(batch_col_sums)

        self.col_sums = tf.concat(col_sums, axis=0)
        self.n = self.col_sums.shape[0]
        self.col_means = self.col_sums / self.n
        self.diag = tf.concat(diag, axis=0)
        self.nb_features = batch_col_cases.shape[1]

    def find_prototypes(self, nb_prototypes, kernel_type: str = 'local', kernel_fn: callable = rbf_kernel):
        """
        Search for prototypes and their corresponding weights.

        Parameters
        ----------
        nb_prototypes : int
            Number of prototypes to find.
        kernel_type : str, optional
            The kernel type. It can be 'local' or 'global', by default 'local'.
            When it is local, the distances are calculated only within the classes.
        kernel_fn : Callable, optional
            Kernel function or kernel matrix, by default rbf_kernel.

        Returns
        -------
        prototype_indices : Tensor
            The indices of the selected prototypes.
        prototype_weights : 
            The normalized weights of the selected prototypes.
        """

        self.compute_kernel_attributes(kernel_type, kernel_fn)

        # Tensors to store selected indices and their corresponding cases, labels and weights.
        selection_indices = tf.constant([], dtype=tf.int32)
        selection_cases = tf.zeros((0, self.nb_features), dtype=tf.float32)
        selection_labels = tf.constant([], dtype=tf.int32)   
        selection_weights = tf.constant([], dtype=tf.float32)     
        # Tensor to store the all_candidates-selection kernel of the previous iteration.
        all_candidates_selection_kernel = tf.zeros((self.n, 0), dtype=tf.float32)
        # Tensor to store the selection-selection kernel.
        selection_selection_kernel = None
        
        k = 0
        while k < nb_prototypes:
  
            nb_selection = selection_cases.shape[0]

            # Tensor to store the all_candidates-last_selected kernel
            if nb_selection !=0:
                all_candidates_last_selected_kernel = tf.zeros((self.n), dtype=tf.float32)

            best_objective = None   
            best_indice = None
            best_case = None 
            best_label = None 
            best_weights = None
        
            for batch_index, (cases, labels) in enumerate(
                zip(self.cases_dataset, self.labels_dataset)
            ):
                batch_inside_indices = tf.range(cases.shape[0], dtype=tf.int32)
                batch_indices = batch_index * self.batch_size + batch_inside_indices
        
                # Filter the batch to keep only candidate indices.
                if nb_selection == 0:
                    candidates_indices = batch_indices
                else:
                    candidates_indices = tf.convert_to_tensor(np.setdiff1d(batch_indices, selection_indices))

                nb_candidates = candidates_indices.shape[0]

                if nb_candidates == 0: 
                    continue

                candidates_inside_indices = candidates_indices % self.batch_size
                candidates_cases = tf.gather(cases, candidates_inside_indices)
                candidates_labels = tf.gather(labels, candidates_inside_indices)

                # Compute the candidates-selection kernel for the batch
                if nb_selection == 0:
                    candidates_selection_kernel = None
                else:
                    candidates_last_selected_kernel = self.kernel_fn(candidates_cases, selection_cases[-1:, :], candidates_labels, selection_labels[-1:])
                    candidates_selection_kernel = tf.concat([tf.gather(all_candidates_selection_kernel, candidates_indices, axis=0), candidates_last_selected_kernel], axis=1)
                    all_candidates_last_selected_kernel = tf.tensor_scatter_nd_update(all_candidates_last_selected_kernel, tf.expand_dims(candidates_indices, axis=1), tf.squeeze(candidates_last_selected_kernel, axis=1))
                 
                # Compute the objectives for the batch
                objectives, objectives_weights = self.compute_objectives(selection_indices, selection_cases, selection_labels, selection_weights, selection_selection_kernel, candidates_indices, candidates_cases, candidates_labels, candidates_selection_kernel)
    
                # Select the best objective in the batch           
                objectives_argmax = tf.argmax(objectives)
                
                if (best_objective is None) or (tf.gather(objectives, objectives_argmax) > best_objective):
                    best_objective = tf.gather(objectives, objectives_argmax)        
                    best_indice = tf.squeeze(tf.gather(candidates_indices, objectives_argmax))
                    best_case = tf.gather(candidates_cases, objectives_argmax)
                    best_label = tf.gather(candidates_labels, objectives_argmax)
                    if objectives_weights is not None:
                        best_weights = tf.squeeze(tf.gather(objectives_weights, objectives_argmax))  

            # Update the all_candidates-selection kernel
            if nb_selection != 0:
                all_candidates_selection_kernel = tf.concat([all_candidates_selection_kernel, tf.expand_dims(all_candidates_last_selected_kernel, axis=1)], axis=1)
           
            # Update the selection-selection kernel
            if nb_selection == 0:
                selection_selection_kernel = tf.gather(self.diag, [[best_indice]])
            else:    
                selection_selection_kernel = tf.pad(selection_selection_kernel, [[0, 1], [0, 1]])

                best_candidate_selection_kernel = tf.gather(all_candidates_selection_kernel, [best_indice], axis=0)
                best_candidate_selection_kernel = tf.pad(best_candidate_selection_kernel, [[nb_selection, 0], [0, 1]])

                best_candidate_diag = tf.expand_dims(tf.gather(self.diag, [best_indice]), axis=-1)
                best_candidate_diag = tf.pad(best_candidate_diag, [[nb_selection, 0], [nb_selection, 0]])

                selection_selection_kernel = selection_selection_kernel + best_candidate_diag + best_candidate_selection_kernel + tf.transpose(best_candidate_selection_kernel)

            # Update selection indices, cases and labels
            selection_indices = tf.concat([selection_indices, [best_indice]], axis=0)
            selection_cases = tf.concat([selection_cases, [best_case]], axis=0)
            selection_labels = tf.concat([selection_labels, [best_label]], axis=0)

            # Update selection weights
            selection_weights = self.update_selection_weights(selection_indices, selection_weights, selection_selection_kernel, best_indice, best_weights, best_objective)
               
            k += 1

        prototype_indices = selection_indices
        prototype_weights = selection_weights

        # Normalize the weights
        prototype_weights = prototype_weights / tf.reduce_sum(prototype_weights)

        return prototype_indices, prototype_weights