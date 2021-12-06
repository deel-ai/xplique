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
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D

from ..utils import almost_equal
from ..utils import generate_data
from xplique.attributions import GradCAM


def _generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2), activation="relu", name="conv2d"))
    model.add(Conv2D(4, kernel_size=(2, 2), activation="relu", name="conv2d_1"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    return model


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        samples, labels = generate_data(input_shape, nb_labels, 100)
        model = _generate_model(input_shape, nb_labels)

        method = GradCAM(model, -2)
        outputs = method.explain(samples, labels)

        assert samples.shape[:3] == outputs.shape[:3]


def test_conv_layer():
    """We should target the right layer using either int, string or default procedure"""
    tf.keras.backend.clear_session()

    model = _generate_model()

    last_conv_layer = model.get_layer("conv2d_1")
    first_conv_layer = model.get_layer("conv2d")
    flatten_layer = model.get_layer("flatten")

    # default should target the last conv layer
    gc_default = GradCAM(model)
    assert gc_default.conv_layer == last_conv_layer

    # target the first conv layer
    gc_input_conv = GradCAM(model, conv_layer=0)
    assert gc_input_conv.conv_layer == first_conv_layer

    # target a random flatten layer
    gc_flatten = GradCAM(model, conv_layer="flatten")
    assert gc_flatten.conv_layer == flatten_layer


def test_weights_computation():
    """Ensure the grad-cam weights are correct"""
    activations = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.5], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, 0.0]],
        ]
    )[None, :, :, :]
    grads = np.array(
        [
            [[1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 0.0]],
            [[1.0, 0.5], [0.0, 0.0]],
            [[0.5, 0.0], [0.0, 0.0]],
        ]
    )[None, :, :, :]

    # move so that the filters F are at the end [F, W, H] -> [W, H, F]
    activations = np.moveaxis(activations, 1, 3)
    grads = np.moveaxis(grads, 1, 3)

    weights = GradCAM._compute_weights(grads, activations)
    assert almost_equal(weights[0], [4.0 / 4.0, 2.0 / 4.0, 1.5 / 4.0, 0.5 / 4.0])

    grad_cam = GradCAM._apply_weights(weights, activations)
    assert almost_equal(
        grad_cam, np.sum(activations * weights, -1)
    )  # as we have no negative value
