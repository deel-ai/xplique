import tensorflow as tf
import numpy as np

from xplique.features_visualizations import maco_image_parametrization, init_maco_buffer, maco, Objective
from ..utils import almost_equal


def dummy_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((8, 8, 3)),
        tf.keras.layers.Conv2D(4, (3, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10)
    ])
    model.compile()
    return model


def test_init_maco_buffer():
    """ Ensure we can init the magnitude and phase for any size """
    img_size_to_magnitude_size = {
        (8, 8):(8, 5),
        (16, 24):(16, 13),
        (32, 32):(32, 17),
        (128, 128):(128, 65),
        (256, 512):(256, 257),
        (1024, 2048):(1024, 1025),
    }

    for img_size, spectrum_size in img_size_to_magnitude_size.items():
        magnitude, phase = init_maco_buffer(img_size)
        assert magnitude.shape[1:] == spectrum_size
        assert phase.shape[1:] == spectrum_size


def test_maco_image_param():
    """ Ensure we can reconstruct an image from magnitude and phase """
    img_size_to_magnitude_size = {
        (8, 8):(8, 5),
        (16, 24):(16, 13),
        (32, 32):(32, 17),
        (128, 128):(128, 65),
        (256, 512):(256, 257),
        (1024, 2048):(1024, 1025),
    }

    for img_size, spectrum_size in img_size_to_magnitude_size.items():
        magnitude, phase = init_maco_buffer(img_size)
        img = maco_image_parametrization(magnitude, phase, (0, 1))
        assert img.shape[:-1] == img_size


def test_maco():
    """ Ensure the optimization process is returning a valid image """
    model = dummy_model()

    objectives = [
        Objective.neuron(model, -1, 0),
        Objective.direction(model, -1, tf.one_hot(1, 10)),
        Objective.channel(model, -3, 0),
    ]

    for objective in objectives:
        img, transparency = maco(objective, nb_steps=10, custom_shape=(8, 8), values_range=(-127, 127))
        assert img.shape == (8, 8, 3)
        assert transparency.shape == (8, 8, 3)
        assert np.min(img) >= -127
        assert np.max(img) <= 127
