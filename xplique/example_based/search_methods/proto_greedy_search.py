"""
ProtoGreedy search method in example-based module
"""

import numpy as np
import tensorflow as tf

from ...types import Callable, Union, Optional, Tuple

from ..datasets_operations.tf_dataset_operations import sanitize_dataset

from .common import get_distance_function


class ProtoGreedySearch():
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
    batch_size
        Number of sample treated simultaneously.
        It should match the batch size of the `search_set` in the case of a `tf.data.Dataset`.
    nb_prototypes : int
            Number of prototypes to find.
    kernel_fn : Callable, optional
        Kernel function, by default the rbf kernel.
        The overall method will be much faster if the provided function is a `tf.function`.
    gamma : float, optional
        Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
    """
    # pylint: disable=too-many-instance-attributes

    # Avoid zero division during procedure. (the value is not important, as if the denominator is
    # zero, then the nominator will also be zero).
    EPSILON = 1e-6

    def __init__(
        self,
        cases_dataset: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        batch_size: Optional[int] = 32,
        nb_prototypes: int = 1,
        kernel_fn: callable = None,
        gamma: float = None
    ):
        # set batch size
        if hasattr(cases_dataset, "_batch_size"):
            self.batch_size = tf.cast(cases_dataset._batch_size, tf.int32)
        else:
            self.batch_size = batch_size

        self.cases_dataset = sanitize_dataset(cases_dataset, self.batch_size)

        # set kernel function
        if kernel_fn is None:
            # define kernel fn to default rbf kernel
            self.__set_default_kernel_fn(self.cases_dataset, gamma)
        elif isinstance(kernel_fn, tf.types.experimental.PolymorphicFunction):
            # the kernel_fn was decorated with a tf.function
            self.kernel_fn = kernel_fn
        elif hasattr(kernel_fn, "__call__"):
            # the kernel_fn is a callable the output is converted to a tensor for consistency
            self.kernel_fn = lambda x1, x2: tf.convert_to_tensor(kernel_fn(x1, x2))
        else:
            raise AttributeError(
                "The kernel_fn parameter is expected to be None or a Callable"\
                +f"but {kernel_fn} was received."\
            )

        # compute the sum of the columns and the diagonal values of the kernel matrix of the dataset
        self.__set_kernel_matrix_column_means_and_diagonal()

        # compute the prototypes in the latent space
        self.find_global_prototypes(nb_prototypes)

    def _get_distance_fn(self, distance: Optional[Union[int, str, Callable]]) -> Callable:
        """
        Get the distance function for examples search.
        Function called through the Prototypes class.
        The distance function is used to search for the closest examples to the prototypes.

        Parameters
        ----------
        distance
            Distance function for examples search. It can be an integer, a string in
            {"manhattan", "euclidean", "cosine", "chebyshev", "inf"}, or a Callable.

        Returns
        -------
        Callable
            Distance function for examples search.
        """
        if distance is None:
            def kernel_induced_distance(x1, x2):
                def dist(x):
                    x = tf.expand_dims(x, axis=0)
                    return tf.sqrt(
                        self.kernel_fn(x1, x1) - 2 * self.kernel_fn(x1, x) + self.kernel_fn(x, x)
                    )
                distance = tf.map_fn(dist, x2)
                return tf.squeeze(distance)
            return kernel_induced_distance

        return get_distance_function(distance)

    def __set_default_kernel_fn(self,
                                cases_dataset: tf.data.Dataset,
                                gamma: float = None,
                                ) -> None:
        """
        Set the default kernel function.

        Parameters
        ----------
        cases_dataset : tf.data.Dataset
            The dataset used to train the model, examples are extracted from the dataset.
            The shape are extracted from the dataset, it is necessary for optimal performance,
            and to set the default gamma value.
        gamma : float, optional
            Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
        """
        cases_shape = cases_dataset.element_spec.shape
        self.nb_features = cases_shape[-1]

        # elements should be batched tabular data
        assert len(cases_shape) == 2,\
            "Prototypes' searches expects 2D data, (nb_samples, nb_features), but got "+\
            f"{cases_shape}. Please verify your projection "+\
            "if you provided a custom one. If you use a splitted model, "+\
            "make sure the output of the first part of the model is flattened."

        if gamma is None:
            if cases_dataset is None:
                raise ValueError(
                    "For the default kernel_fn, the default gamma value requires samples shape."
                )
            gamma = 1.0 / self.nb_features

        gamma = tf.constant(gamma, dtype=tf.float32)

        # created inside a function for gamma to be a constant and prevent graph retracing
        @tf.function(input_signature=[
            tf.TensorSpec(shape=cases_shape, dtype=tf.float32, name="tensor_1"),
            tf.TensorSpec(shape=cases_shape, dtype=tf.float32, name="tensor_2")
        ])
        def rbf_kernel(tensor_1: tf.Tensor, tensor_2: tf.Tensor,) -> tf.Tensor:
            """
            Compute the rbf kernel matrix between two sets of samples.

            Parameters
            ----------
            tensor_1
                The first set of samples of shape (n, d).
            tensor_2
                The second set of samples of shape (m, d).
            
            Returns
            -------
            Tensor
                The rbf kernel matrix of shape (n, m).
            """

            # (n, m, d)
            pairwise_diff = tensor_1[:, tf.newaxis, :] - tensor_2[tf.newaxis, :, :]

            # (n, m)
            pairwise_sq_dist = tf.reduce_sum(tf.square(pairwise_diff), axis=-1)
            kernel_matrix = tf.exp(-gamma * pairwise_sq_dist)

            return kernel_matrix

        self.kernel_fn = rbf_kernel

    def __set_kernel_matrix_column_means_and_diagonal(self) -> None:
        """
        Compute the sum of the columns and the diagonal values of the kernel matrix of the dataset.
        Results are stored in the object.

        Parameters
        ----------
        cases_dataset : tf.data.Dataset
            The kernel matrix is computed between the cases of this dataset.
        kernel_fn : Callable
            Kernel function to compute the kernel matrix between two sets of samples.
        """
        # Compute the sum of the columns and the diagonal values of the kernel matrix of the dataset
        # We take advantage of the symmetry of this matrix to traverse only its lower triangle
        col_sums = []
        diag = []
        row_sums = [0]  # first batch has no row sums and not computed, 0 is a placeholder
        nb_samples = 0

        for batch_col_index, batch_col_cases in enumerate(self.cases_dataset):

            batch_col_sums = tf.zeros((batch_col_cases.shape[0]), dtype=tf.float32)

            for batch_row_index, batch_row_cases in enumerate(self.cases_dataset):
                # ignore batches that are above the diagonal
                if batch_col_index > batch_row_index:
                    continue

                # Compute the kernel matrix between the two batches
                # (n_b_row, n_b_col)
                batch_kernel = self.kernel_fn(batch_row_cases, batch_col_cases)

                # increment the column sums
                # (n_b_col,)
                batch_col_sums = batch_col_sums + tf.reduce_sum(batch_kernel, axis=0)

                # current pair of batches is on the diagonal
                if batch_col_index == batch_row_index:
                    # stock the diagonal values
                    diag.append(tf.linalg.diag_part(batch_kernel))

                    # complete the column sums with the row sums when the batch is on the diagonal
                    # (n_b_col,)
                    batch_col_sums = batch_col_sums + row_sums[batch_row_index]
                    continue

                # increment the row sums
                # (n_b_row,)
                current_batch_row_sums = tf.reduce_sum(batch_kernel, axis=1)
                if batch_col_index == 0:
                    row_sums.append(current_batch_row_sums)
                else:
                    row_sums[batch_row_index] += current_batch_row_sums

            col_sums.append(batch_col_sums)
            nb_samples += batch_col_cases.shape[0]

        # pad the last batch to have the same size as the others
        col_sums[-1] = tf.pad(col_sums[-1], [[0, self.batch_size - col_sums[-1].shape[0]]])

        # (nb, b)
        self.kernel_col_means = tf.stack(col_sums, axis=0) / tf.cast(nb_samples, dtype=tf.float32)

        # pad the last batch to have the same size as the others
        diag[-1] = tf.pad(diag[-1], [[0, self.batch_size - diag[-1].shape[0]]])

        # (nb, b)
        self.kernel_diag = tf.stack(diag, axis=0)

    def _compute_batch_objectives(self,
                                  candidates_kernel_diag: tf.Tensor,
                                  candidates_kernel_col_means: tf.Tensor,
                                  selection_kernel_col_means: tf.Tensor,
                                  candidates_selection_kernel: tf.Tensor,
                                  selection_selection_kernel: tf.Tensor
                                  ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the objective function and corresponding weights
        for a given set of selected prototypes and a batch of candidates.

        Here, we have a special case of protogreedy where we give equal weights to all prototypes,
        the objective here is simplified to speed up processing.

        Find argmax_{c} F(S ∪ c) - F(S)
        ≡
        Find argmax_{c} F(S ∪ c)
        ≡
        Find argmax_{c} max_{w} (w^T mu_p) - (w^T K w) / 2

        w*, the optimal objective weights, is computed as follows: w* = K^-1 mu_p
        
        where:
        - mu_p is the column means of the kernel matrix
        - K is the kernel matrix
        
        Parameters
        ----------
        candidates_kernel_diag : Tensor
            Diagonal values of the kernel matrix between the candidates and themselves. Shape (bc,).
        candidates_kernel_col_means : Tensor
            Column means of the kernel matrix, subset for the candidates. Shape (bc,).
        selection_kernel_col_means : Tensor
            Column means of the kernel matrix, subset for the selected prototypes. Shape (|S|,).
        candidates_selection_kernel : Tensor
            Kernel matrix between the candidates and the selected prototypes. Shape (bc, |S|).
        selection_selection_kernel : Tensor
            Kernel matrix between the selected prototypes. Shape (|S|, |S|).
            
        Returns
        -------
        objectives
            Tensor that contains the computed objective values for each candidate. Shape (bc,).
        objectives_weights
            Tensor that contains the computed objective weights for each candidate.
            Shape (bc, |S|+1).
        """
        # pylint: disable=invalid-name
        # construct the kernel matrix for (S ∪ c) for each candidate (S is the selection)
        # (bc, |S| + 1, |S| + 1)
        if candidates_selection_kernel is None:
            # no selected prototypes yet, S = {}
            # (bc, 1, 1)
            K = candidates_kernel_diag[:, tf.newaxis, tf.newaxis]
        else:
            # repeat the selection-selection kernel for each candidate
            # (bc, |S|, |S|)
            selection_selection_kernel = tf.tile(
                tf.expand_dims(selection_selection_kernel, 0),
                [candidates_selection_kernel.shape[0], 1, 1]
            )

            # add candidates-selection kernel row to the selection-selection kernel matrix
            # (bc, |S| + 1, |S|)
            extended_selection_selection_kernel = tf.concat(
                [
                    selection_selection_kernel,
                    candidates_selection_kernel[:, tf.newaxis, :]
                ],
                axis=1
            )

            # create the extended column for the candidates with the diagonal values
            # (bc, |S| + 1)
            extended_candidates_selection_kernel = tf.concat(
                [
                    candidates_selection_kernel,
                    candidates_kernel_diag[:, tf.newaxis]
                ],
                axis=1
            )

            # add the extended column for the candidates to the extended selection-selection kernel
            # (bc, |S| + 1, |S| + 1)
            K = tf.concat(
                [
                    extended_selection_selection_kernel,
                    extended_candidates_selection_kernel[:, :, tf.newaxis],
                ],
                axis=2
            )

        # (bc, |S|) - extended selected kernel col means
        selection_kernel_col_means = tf.tile(
            selection_kernel_col_means[tf.newaxis, :],
            multiples=[candidates_kernel_col_means.shape[0], 1]
        )

        # (bc, |S| + 1) - mu_p
        candidates_selection_kernel_col_means = tf.concat(
            [
                selection_kernel_col_means,
                candidates_kernel_col_means[:, tf.newaxis]],
            axis=1
        )

        # compute the optimal objective weights for each candidate in the batch
        # (bc, |S| + 1, |S| + 1) - K^-1
        K_inv = tf.linalg.inv(K + ProtoGreedySearch.EPSILON * tf.eye(K.shape[-1]))

        # (bc, |S| + 1) - w* = K^-1 mu_p
        objectives_weights = tf.einsum("bsp,bp->bs", K_inv, candidates_selection_kernel_col_means)
        objectives_weights = tf.maximum(objectives_weights, 0)

        # (bc,) - (w*^T mu_p)
        weights_mu_p = tf.einsum("bp,bp->b",
                                 objectives_weights, candidates_selection_kernel_col_means)

        # (bc,) - (w*^T K w*)
        weights_K_weights = tf.einsum("bs,bsp,bp->b",
                                      objectives_weights, K, objectives_weights)

        # (bc,) - (w*^T mu_p) - (w*^T K w*) / 2
        objectives = weights_mu_p - 0.5 * weights_K_weights

        return objectives, objectives_weights

    def find_global_prototypes(self, nb_prototypes: int):
        """
        Search for global prototypes and their corresponding weights.
        Iteratively select the best prototype candidate and add it to the selection.
        The selected candidate is the one with the highest objective function value.

        The indices, weights, and cases of the selected prototypes are stored in the object.

        Parameters
        ----------
        nb_prototypes : int
            Number of global prototypes to find.
        """
        # pylint: disable=too-many-statements
        assert 0 < nb_prototypes, "`nb_prototypes` should be between at least 1."

        # initialize variables with placeholders
        # final prototypes variables
        # (np, 2) - final prototypes indices
        self.prototypes_indices = tf.Variable(tf.fill((nb_prototypes, 2), -1))
        # (np,) - final prototypes weights
        self.prototypes_weights = tf.Variable(tf.zeros((nb_prototypes,), dtype=tf.float32))
        # (np, d) - final prototypes cases
        self.prototypes = tf.Variable(tf.zeros((nb_prototypes, self.nb_features), dtype=tf.float32))

        # kernel matrix variables
        # (np, np) - kernel matrix between selected prototypes
        selection_selection_kernel = tf.Variable(tf.zeros((nb_prototypes, nb_prototypes),
                                                        dtype=tf.float32))
        # (nb, b, np) - kernel matrix between samples and selected prototypes
        samples_selection_kernel = tf.Variable(tf.zeros((*self.kernel_diag.shape, nb_prototypes)))

        # (nb, b) - mask encoding the selected prototypes
        mask_of_selected = tf.Variable(tf.fill(self.kernel_diag.shape, False))

        # (np,) - selected column means
        selection_kernel_col_means = tf.Variable(tf.zeros((nb_prototypes,), dtype=tf.float32))

        # iterate till we find all the prototypes
        for nb_selected in range(nb_prototypes):
            # initialize
            best_objective = tf.constant(-np.inf, dtype=tf.float32)

            # iterate over the batches
            for batch_index, cases in enumerate(self.cases_dataset):
                # (b,)
                candidates_batch_mask = tf.math.logical_not(mask_of_selected[batch_index])

                # last batch, pad with False
                if cases.shape[0] < self.batch_size:
                    candidates_batch_mask = tf.math.logical_and(
                        candidates_batch_mask, tf.range(self.batch_size) < cases.shape[0]
                    )

                # no candidates in the batch skipping
                if not tf.reduce_any(candidates_batch_mask):
                    continue

                # compute the kernel matrix between the last selected prototypes and the candidates
                if nb_selected > 0:
                    # (b,)
                    batch_samples_last_selection_kernel = self.kernel_fn(
                        cases, last_selected
                    )[:, 0]
                    samples_selection_kernel[batch_index, :cases.shape[0], nb_selected - 1].assign(
                        batch_samples_last_selection_kernel
                    )

                    # (b, |S|)
                    batch_candidates_selection_kernel =\
                        samples_selection_kernel[batch_index, :cases.shape[0], :nb_selected]
                    # (bc, |S|)
                    batch_candidates_selection_kernel = tf.boolean_mask(
                        tensor=batch_candidates_selection_kernel,
                        mask=candidates_batch_mask[:cases.shape[0]],
                        axis=0,
                    )

                else:
                    batch_candidates_selection_kernel = None

                # extract kernel values for the batch
                # (bc,)
                batch_candidates_kernel_diag = self.kernel_diag[batch_index][candidates_batch_mask]
                # (bc,)
                batch_candidates_kernel_col_means =\
                    self.kernel_col_means[batch_index][candidates_batch_mask]

                # compute the objectives for the batch
                # (bc,), (bc, |S| + 1)
                objectives, objectives_weights = self._compute_batch_objectives(
                    batch_candidates_kernel_diag,
                    batch_candidates_kernel_col_means,
                    selection_kernel_col_means[:nb_selected],
                    batch_candidates_selection_kernel,
                    selection_selection_kernel[:nb_selected, :nb_selected],
                )

                # select the best candidate in the batch
                objectives_argmax = tf.argmax(objectives)
                batch_best_objective = tf.gather(objectives, objectives_argmax)

                if batch_best_objective > best_objective:
                    best_objective = batch_best_objective
                    best_batch_index = batch_index
                    best_index = tf.range(self.batch_size)[candidates_batch_mask][objectives_argmax]
                    best_case = cases[best_index]
                    if objectives_weights is not None:
                        best_weights = objectives_weights[objectives_argmax]

            # update the selected prototypes
            # pylint: disable=possibly-used-before-assignment
            last_selected = best_case[tf.newaxis, :]
            mask_of_selected[best_batch_index, best_index].assign(True)
            self.prototypes_indices[nb_selected].assign([best_batch_index, best_index])
            self.prototypes[nb_selected].assign(best_case)

            # update selected-selected kernel matrix (S = S ∪ c)
            selection_selection_kernel[nb_selected, nb_selected].assign(
                self.kernel_diag[best_batch_index, best_index]
            )
            if nb_selected > 0:
                # (|S|,)
                new_selected = samples_selection_kernel[best_batch_index, best_index, :nb_selected]

                # add the new row and column to the selected-selected kernel matrix
                selection_selection_kernel[nb_selected, :nb_selected].assign(
                    new_selected
                )
                selection_selection_kernel[:nb_selected, nb_selected].assign(
                    new_selected
                )

            # update the selected column means
            selection_kernel_col_means[nb_selected].assign(
                self.kernel_col_means[best_batch_index, best_index]
            )

            # update the selected weights
            if not hasattr(self, "_update_selection_weights"):
                # pylint: disable=used-before-assignment
                self.prototypes_weights[:nb_selected + 1].assign(best_weights)
            else:
                self._update_selection_weights(
                    selection_kernel_col_means[:nb_selected + 1],
                    selection_selection_kernel[:nb_selected + 1, :nb_selected + 1],
                    self.kernel_diag[best_batch_index, best_index],
                    best_objective,
                )

        # normalize the weights
        self.prototypes_weights.assign(
            self.prototypes_weights / tf.reduce_sum(self.prototypes_weights)
        )

        # convert variables to tensors
        self.prototypes_indices = tf.convert_to_tensor(self.prototypes_indices)
        self.prototypes = tf.convert_to_tensor(self.prototypes)
        self.prototypes_weights = tf.convert_to_tensor(self.prototypes_weights)

        assert tf.reduce_sum(tf.cast(mask_of_selected, tf.int32)) == nb_prototypes,\
            "The number of prototypes found is not equal to the number of prototypes expected."
