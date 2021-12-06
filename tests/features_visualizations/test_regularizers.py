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
