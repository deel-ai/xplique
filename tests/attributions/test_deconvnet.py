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

import numpy as np
import tensorflow as tf

from ..utils import generate_data
from ..utils import generate_model
from xplique.attributions import DeconvNet


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 100)
        model = generate_model(input_shape, nb_labels)

        method = DeconvNet(model, -2)
        explanations = method.explain(x, y)

        assert x.shape == explanations.shape


def test_deconv_mechanism():
    """Ensure we have a proper DeconvNet relu instead of relu"""

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(tf.nn.relu, input_shape=(4,)))
    model.add(tf.keras.layers.Lambda(lambda x: 10 * x))

    x = tf.constant(np.expand_dims([5.0, 0.0, -5.0, 10.0], axis=0))

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    normal_grads = tape.gradient(y, x).numpy()[0]

    # dy / dx = { 10 for x > 0, else 0 }
    assert np.array_equal(normal_grads, np.array([10.0, 0.0, 0.0, 10.0]))

    deconvnet_model = DeconvNet(model).model

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = deconvnet_model(x)
    deconv_grads = tape.gradient(y, x).numpy()[0]

    # dy / dx = { 10 }
    assert np.array_equal(deconv_grads, np.array([10.0, 10.0, 10.0, 10.0]))

    # ensure we didn't change the mechanism of the original model
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    normal_grads_2 = tape.gradient(y, x).numpy()[0]

    assert np.array_equal(normal_grads, normal_grads_2)
