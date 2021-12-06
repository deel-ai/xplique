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
from xplique.features_visualizations.regularizers import l1_reg
from xplique.features_visualizations.regularizers import l2_reg
from xplique.features_visualizations.regularizers import l_inf_reg
from xplique.features_visualizations.regularizers import total_variation_reg


def test_lp_norm():
    vec = np.array([[[-4.0, 4.0]]])[np.newaxis, :]
    l1 = l1_reg(1.0)
    l2 = l2_reg(2.0)
    linf = l_inf_reg(10.0)

    assert almost_equal(l1(vec)[0], 4.0)
    assert almost_equal(l2(vec)[0], 8.0)
    assert almost_equal(linf(vec)[0], 40.0)


def test_tv():
    vec = tf.cast([[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]], tf.float32)[
        :, :, :, tf.newaxis
    ]
    tv = total_variation_reg(1.0)

    assert almost_equal(tv(vec)[0], 10.0)
