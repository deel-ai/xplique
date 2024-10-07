"""
MMDCritic search method in example-based module
"""

import tensorflow as tf

from ...types import Tuple

from .proto_greedy_search import ProtoGreedySearch


class MMDCriticSearch(ProtoGreedySearch):
    """
    MMDCritic method to search prototypes.

    References:
    .. [#] `Been Kim, Rajiv Khanna, Oluwasanmi Koyejo, 
        "Examples are not enough, learn to criticize! criticism for interpretability"
        <https://proceedings.neurips.cc/paper_files/paper/2016/file/5680522b8e2bb01943234bce7bf84534-Paper.pdf>`_

    Parameters
    ----------
    cases_dataset
        The dataset used to train the model, examples are extracted from the dataset.
        For natural example-based methods it is the train dataset.
    batch_size
        Number of samples treated simultaneously.
        It should match the batch size of the `cases_dataset` in the case of a `tf.data.Dataset`.
    nb_prototypes : int
            Number of prototypes to find.
    kernel_fn : Callable, optional
        Kernel function, by default the rbf kernel.
        This function must only use TensorFlow operations.
    gamma : float, optional
        Parameter that determines the spread of the rbf kernel, defaults to 1.0 / n_features.
    """

    def _compute_batch_objectives(self,
                                  candidates_kernel_diag: tf.Tensor,
                                  candidates_kernel_col_means: tf.Tensor,
                                  selection_kernel_col_means: tf.Tensor,
                                  candidates_selection_kernel: tf.Tensor,
                                  selection_selection_kernel: tf.Tensor
                                  ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute the objective function and corresponding weights
        for a given set of selected prototypes and a candidate.

        Here, we have a special case of protogreedy where we give equal weights to all prototypes, 
        the objective here is simplified to speed up processing
        
        Find argmax_{c} F(S ∪ c) - F(S)
        ≡
        Find argmax_{c} F(S ∪ c)
        ≡
        Find argmax_{c} (sum1 - sum2)
        where: sum1 = (2 / n) * ∑[i=1 to n] κ(x_i, c) 
               sum2 = 1/(|S|+1) [κ(c, c) + 2 * ∑[j=1 to |S|] κ(x_j, c)]
            
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

        nb_candidates = tf.shape(candidates_kernel_diag)[0]

        # (bc,) - 2 * ∑[i=1 to n] κ(x_i, c)
        sum1 = 2 * candidates_kernel_col_means

        if candidates_selection_kernel is None:
            extended_nb_selected = 1

            # (bc,) - κ(c, c)
            sum2 = candidates_kernel_diag
        else:
            extended_nb_selected = tf.shape(selection_kernel_col_means)[0] + 1

            # (bc,) - κ(c, c) + 2 * ∑[j=1 to |S|] κ(x_j, c)
            # the second term is 0 when the selection is empty
            sum2 = candidates_kernel_diag + 2 * tf.reduce_sum(candidates_selection_kernel, axis=1)

        # (bc,) - 1/(|S|+1) [κ(c, c) + 2 * ∑[j=1 to |S|] κ(x_j, c)]
        sum2 /= tf.cast(extended_nb_selected, tf.float32)

        # (bc,)
        objectives = sum1 - sum2

        # (bc, |S|+1) - ones (the weights are normalized later)
        objectives_weights = tf.ones((nb_candidates, extended_nb_selected), dtype=tf.float32)

        return objectives, objectives_weights
