# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
