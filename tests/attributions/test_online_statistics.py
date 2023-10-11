"""
Test that online statistics give similar results to normal ones
"""

import numpy as np
import tensorflow as tf

from xplique.attributions import SmoothGrad, VarGrad, SquareGrad

from ..utils import almost_equal


def test_online_mean():
    method = SmoothGrad(lambda x: None)
    for shape in [(1, 7, 4), (5, 2, 4, 4), (5, 7, 4, 4, 3)]:
        samples = tf.reshape(tf.range(int(np.prod(shape)), dtype=tf.float32), shape)

        normal_stat = tf.reduce_mean(samples, axis=1)
        method._initialize_online_statistic()

        for i in range(int(np.ceil(shape[1] / 2))):
            sample = samples[:, 2 * i:2 * (i + 1)]
            method._update_online_statistic(sample)

        assert almost_equal(normal_stat, method._get_online_statistic_final_value())


def test_online_square_mean():
    method = SquareGrad(lambda x: None)
    for shape in [(1, 7, 4), (5, 2, 4, 4), (5, 7, 4, 4, 3)]:
        samples = tf.reshape(tf.range(int(np.prod(shape)), dtype=tf.float32), shape)

        normal_stat = tf.reduce_mean(samples**2, axis=1)
        method._initialize_online_statistic()

        for i in range(int(np.ceil(shape[1] / 2))):
            sample = samples[:, 2 * i:2 * (i + 1)]
            method._update_online_statistic(sample)

        assert almost_equal(normal_stat, method._get_online_statistic_final_value())


def test_online_variance():
    method = VarGrad(lambda x: None)
    for shape in [(1, 7, 4), (5, 2, 4, 4), (5, 7, 4, 4, 3)]:
        samples = tf.reshape(tf.range(int(np.prod(shape)), dtype=tf.float32), shape)

        normal_stat = tf.math.reduce_variance(samples, axis=1) * (shape[1] / (shape[1] - 1))
        method._initialize_online_statistic()

        for i in range(int(np.ceil(shape[1] / 2))):
            sample = samples[:, 2 * i:2 * (i + 1)]
            method._update_online_statistic(sample)

        assert almost_equal(normal_stat, method._get_online_statistic_final_value())
