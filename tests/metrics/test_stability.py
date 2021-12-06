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

from ..utils import almost_equal
from ..utils import generate_data
from ..utils import generate_model
from xplique.metrics import AverageStability


def test_average_stability():
    # ensure we can compute the metric with consistents arguments
    input_shape, nb_labels, nb_samples = ((32, 32, 3), 10, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)
    explainer = lambda x, y: np.random.uniform(0, 1, x.shape[:-1])

    l_inf_dist = lambda phi_a, phi_b: np.max(phi_a - phi_b)

    for batch_size in [64, None]:
        for dist in ["l1", "l2", l_inf_dist]:
            score = AverageStability(
                model,
                x,
                y,
                batch_size=batch_size,
                radius=0.1,
                distance=dist,
                nb_samples=100,
            )(explainer)

            assert isinstance(score, float)
            assert score < np.prod(x.shape[:-1])


def test_perfect_stability():
    """Ensure we get perfect stability when explanation is the same"""
    input_shape, nb_labels, nb_samples = ((8, 8, 1), 2, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = generate_model(input_shape, nb_labels)

    explainer = lambda x, y: tf.ones(x.shape)

    perfect_score = AverageStability(model, x, y)(explainer)
    assert almost_equal(perfect_score, 0.0)


def test_worst_stability():
    """Ensure we get worst stability when explanation is totally different"""
    dim = 8
    input_shape, nb_labels, nb_samples = ((dim, dim, 1), 2, 20)
    x, y = generate_data(input_shape, nb_labels, nb_samples)
    model = model = generate_model(input_shape, nb_labels)

    def worst_explainer(inputs, _):
        is_original = any([np.sum(np.abs(inputs[0] - _x)) == 0.0 for _x in x])
        if is_original:
            return np.ones(inputs.shape)
        else:
            return np.zeros(inputs.shape)

    worst_score = AverageStability(model, x, y)(worst_explainer)
    assert almost_equal(worst_score, dim ** 2)
