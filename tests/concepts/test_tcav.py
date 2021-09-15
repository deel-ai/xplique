import numpy as np
import tensorflow as tf

from xplique.concepts import Tcav


def test_tcav_computation():
    """For a given simple example, ensure that the computation of tcav works"""

    # we create a toy example for a function f(x1, x2) -> y1, y2
    # and y1 = x1 > 0.5, y2 = x2 > 0.5
    x = np.random.random((1000, 2))
    y = np.array(x > 0.5, np.float32)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(2),
        tf.keras.layers.Lambda(lambda x: tf.cast(x > 0.5, tf.float32)),
        tf.keras.layers.Dense(2, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(x, y, batch_size=32, epochs=1)

    # by forcing the model to learn this mapping, we should have the following
    # score for the cav :
    tcav_maker = Tcav(model, 1)

    # cav [1.0, 0.0] relative to y1 = 1
    # cav [0.0, 1.0] relative to y1 = 0
    # cav [1.0, 0.0] relative to y2 = 0
    # cav [0.0, 1.0] relative to y2 = 1
    tcav_x1_y1 = tcav_maker(x, 0, [1.0, 0.0])
    tcav_x2_y1 = tcav_maker(x, 0, [0.0, 1.0])
    tcav_x1_y2 = tcav_maker(x, 1, [1.0, 0.0])
    tcav_x2_y2 = tcav_maker(x, 1, [0.0, 1.0])

    epsilon = 0.01
    assert np.abs(tcav_x1_y1 - 1.0) < epsilon
    assert np.abs(tcav_x1_y2) < epsilon
    assert np.abs(tcav_x2_y1) < epsilon
    assert np.abs(tcav_x2_y2 - 1.0) < epsilon