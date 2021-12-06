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
from xplique.attributions import GradientInput
from xplique.attributions import Saliency
from xplique.metrics import MeGe


def test_best_mege():
    # ensure we get perfect score when the models are the same
    input_shape, nb_labels, nb_samples = ((8, 8, 1), 10, 80)
    x, y = generate_data(input_shape, nb_labels, nb_samples)

    model = generate_model(input_shape, nb_labels)
    learning_algorithm = lambda x_train, y_train, x_test, y_test: model

    for method in [Saliency, GradientInput]:

        metric = MeGe(learning_algorithm, x, y, k_splits=4)
        mege, _ = metric.evaluate(method)

        assert almost_equal(mege, 1.0)
