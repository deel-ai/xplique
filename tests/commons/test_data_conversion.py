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
from xplique.commons import numpy_sanitize
from xplique.commons import tensor_sanitize


def test_tensor_sanitize():
    """Ensure we get tf.Tensor for numpy array, tf tensor and tf.data.Dataset"""
    nb_samples = 71
    inputs_shapes = [(32, 32, 1), (32, 32, 3)]

    for shape in inputs_shapes:
        inputs_np, targets_np = generate_data(shape, 10, nb_samples)
        inputs_tf, targets_tf = tf.cast(inputs_np, tf.float32), tf.cast(
            targets_np, tf.float32
        )
        dataset = tf.data.Dataset.from_tensor_slices((inputs_np, targets_np))
        dataset_batched = tf.data.Dataset.from_tensor_slices(
            (inputs_np, targets_np)
        ).batch(10)

        for inputs, targets in [
            (inputs_np, targets_np),
            (inputs_tf, targets_tf),
            (dataset, None),
            (dataset_batched, None),
        ]:
            inputs_sanitize_tf, targets_sanitize_tf = tensor_sanitize(inputs, targets)
            inputs_sanitize_np, targets_sanitize_np = numpy_sanitize(inputs, targets)

            assert isinstance(inputs_sanitize_tf, tf.Tensor)
            assert isinstance(targets_sanitize_tf, tf.Tensor)
            assert isinstance(inputs_sanitize_np, np.ndarray)
            assert isinstance(targets_sanitize_np, np.ndarray)

            assert len(inputs_sanitize_tf) == nb_samples
            assert len(targets_sanitize_tf) == nb_samples
            assert len(inputs_sanitize_np) == nb_samples
            assert len(targets_sanitize_np) == nb_samples
