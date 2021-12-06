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

import tensorflow as tf

from ..utils import almost_equal
from ..utils import generate_data
from ..utils import generate_model
from xplique.attributions import SmoothGrad


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = SmoothGrad(model, -2, nb_samples=100)
        smoothed_gradients = method.explain(x, y)

        assert x.shape == smoothed_gradients.shape


def test_noisy_mask():
    """Ensure the inputs generated are not the same as the original"""
    x = tf.zeros((1, 32, 32, 3), tf.float32)
    noise = tf.random.normal((4, *x.shape[1:]), dtype=tf.float32)

    x_noisy = SmoothGrad._apply_noise(x, noise)

    assert almost_equal(x_noisy, noise)
