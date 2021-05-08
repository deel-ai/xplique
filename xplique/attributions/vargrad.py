"""
Module related to VarGrad method
"""

import tensorflow as tf

from .smoothgrad import SmoothGrad


class VarGrad(SmoothGrad):
    """
     VarGrad is a variance analog to SmoothGrad.

    Ref. Adebayo & al., Sanity check for Saliency map (2018).
    https://papers.nips.cc/paper/8160-sanity-checks-for-saliency-maps.pdf

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
    def _reduce_gradients(gradients):
        """
        Reduce the gradients using the variance obtained on each noisy samples.

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
        return tf.math.reduce_variance(gradients, axis=1)
