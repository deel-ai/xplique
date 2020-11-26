"""
Module related to SquareGrad method
"""

import tensorflow as tf

from .smoothgrad import SmoothGrad


class SquareGrad(SmoothGrad):
    """
    SquareGrad (or SmoothGrad^2) is an unpublished variant of classic SmoothGrad which squares
    each gradients of the noisy inputs before averaging.

    Ref. Hooker & al., A Benchmark for Interpretability Methods in Deep Neural Networks (2019).
    https://papers.nips.cc/paper/9167-a-benchmark-for-interpretability-methods-in-deep-neural-networks.pdf

    Parameters
    ----------
    model : tf.keras.Model
        Model used for computing explanations.
    output_layer_index : int, optional
        Index of the output layer, default to the last layer, it is recommended to use the layer
        before Softmax (often '-2').
    batch_size : int, optional
        Number of samples to explain at once, if None compute all at once.
    nb_samples : int, optional
        Number of noisy samples generated for the smoothing procedure.
    noise : float, optional
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    @staticmethod
    @tf.function
    def reduce_gradients(gradients):
        """
        Reduce the gradients using the square of the gradients obtained on each noisy samples.

        Parameters
        ----------
        gradients : tf.tensor (N, S, W, H, C)
            Gradients to reduce for each of the S samples of each of the N samples. VarGrad use
            the variance of all the gradients.

        Returns
        -------
        reduced_gradients : tf.tensor (N, W, H, C)
            Single saliency map for each input.
        """
        return tf.math.reduce_mean(gradients**2, axis=1)
