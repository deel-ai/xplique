"""
Module related to SquareGrad method
"""

import tensorflow as tf

from .gradient_statistic import GradientStatistic


class SquareGrad(GradientStatistic):
    """
    SquareGrad (or SmoothGrad^2) is an unpublished variant of classic SmoothGrad which squares
    each gradients of the noisy inputs before averaging.

    Ref. Hooker & al., A Benchmark for Interpretability Methods in Deep Neural Networks (2019).
    https://papers.nips.cc/paper/9167-a-benchmark-for-interpretability-methods-in-deep-neural-networks.pdf

    Parameters
    ----------
    model
        The model from which we want to obtain explanations
    output_layer
        Layer to target for the outputs (e.g logits or after softmax).
        If an `int` is provided it will be interpreted as a layer index.
        If a `string` is provided it will look for the layer name.

        Default to the last layer.
        It is recommended to use the layer before Softmax.
    batch_size
        Number of inputs to explain at once, if None compute all at once.
    operator
        Function g to explain, g take 3 parameters (f, x, y) and should return a scalar,
        with f the model, x the inputs and y the targets. If None, use the standard
        operator g(f, x, y) = f(x)[y].
    reducer
        String, name of the reducer to use. Either "min", "mean", "max", "sum", or `None` to ignore.
        Used only for images to obtain explanation with shape (n, h, w, 1).
    nb_samples
        Number of noisy samples generated for the smoothing procedure.
    noise
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    def _initialize_online_statistic(self):
        """
        Initialize values for the online statistic.
        """
        self._elements_counter = 0
        self._actual_square_sum = 0

    def _update_online_statistic(self, elements):
        """
        Update the running square mean by taking new elements into account.

        Parameters
        ----------
        elements
            Batch of batch of elements.
            Part of all the elements the square mean should be computed on.
            Shape: (inputs_batch_size, perturbation_batch_size, ...)
        """
        new_elements_square_sum = tf.reduce_sum(elements**2, axis=1)
        new_elements_count = elements.shape[1]

        # actualize mean
        self._actual_square_sum += new_elements_square_sum

        # actualize count
        self._elements_counter += new_elements_count

    def _get_online_statistic_final_value(self):
        """
        Return the final value of the square mean.

        Returns
        -------
        square_mean
            The square mean computed online.
        """
        return self._actual_square_sum / self._elements_counter
