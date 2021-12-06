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

from ..utils import almost_equal
from xplique.features_visualizations.losses import cosine_similarity


def test_cosine_similarity():
    vec = np.array([10.0, 20.0, 30.0])[np.newaxis, :]
    vec_colinear = np.array([1.0, 2.0, 3.0])[np.newaxis, :]
    vec_orthogonal = np.array([0.0, 0.0, 0.0])[np.newaxis, :]
    vec_opposite = np.array([-0.01, -0.02, -0.03])[np.newaxis, :]

    # cosine_similarity(a, b) = <a,b> / (|a| + |b|)
    assert almost_equal(cosine_similarity(vec, vec_colinear)[0], 1.0)
    assert almost_equal(cosine_similarity(vec, vec_orthogonal)[0], 0.0)
    assert almost_equal(cosine_similarity(vec, vec_opposite)[0], -1.0)
