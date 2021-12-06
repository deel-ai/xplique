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
Module related to Gradient x Input method
"""
import numpy as np
import tensorflow as tf

from ..commons import batch_gradient
from ..types import Optional
from ..types import Union
from .base import sanitize_input_output
from .base import WhiteBoxExplainer


class GradientInput(WhiteBoxExplainer):
    """
    Used to compute elementwise product between the saliency maps of Simonyan et al. and the
    input (Gradient x Input).

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
    """

    @sanitize_input_output
    def explain(
        self,
        inputs: Union[tf.data.Dataset, tf.Tensor, np.ndarray],
        targets: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> tf.Tensor:
        """
        Compute gradients x inputs for a batch of samples.

        Parameters
        ----------
        inputs
            Input samples to be explained.
        targets
            One-hot encoded labels or regression target (e.g {+1, -1}), one for each sample.

        Returns
        -------
        explanations
            Gradients x Inputs, with the same shape as the inputs.
        """
        gradients = batch_gradient(self.model, inputs, targets, self.batch_size)
        gradients_inputs = tf.multiply(gradients, inputs)

        return gradients_inputs
