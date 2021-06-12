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
    model
        Model used for computing explanations.
    output_layer
        Layer to target for the output (e.g logits or after softmax), if int, will be be interpreted
        as layer index, if string will look for the layer name. Default to the last layer, it is
        recommended to use the layer before Softmax.
    batch_size
        Number of samples to explain at once, if None compute all at once.
    nb_samples
        Number of noisy samples generated for the smoothing procedure.
    noise
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    @staticmethod
    @tf.function
    def _reduce_gradients(gradients: tf.Tensor) -> tf.Tensor:
        """
        Reduce the gradients using the square of the gradients obtained on each noisy samples.

        Parameters
        ----------
        gradients
            Gradients to reduce the sampling dimension for each inputs.

        Returns
        -------
        reduced_gradients
            Single saliency map for each input.
        """
        return tf.math.reduce_mean(gradients**2, axis=1)
