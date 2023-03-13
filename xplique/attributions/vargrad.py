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
    nb_samples
        Number of noisy samples generated for the smoothing procedure.
    noise
        Scalar, noise used as standard deviation of a normal law centered on zero.
    """

    @staticmethod
    @tf.function
    def _reduce_gradients(gradients: tf.Tensor) -> tf.Tensor:
        """
        Reduce the gradients using the variance obtained on each noisy samples.

        Parameters
        ----------
        gradients
            Gradients to reduce the sampling dimension for each inputs.

        Returns
        -------
        reduced_gradients
            Single saliency map for each input.
        """
        return tf.math.reduce_variance(gradients, axis=1)
