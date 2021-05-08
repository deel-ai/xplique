import tensorflow as tf

from xplique.attributions import SmoothGrad
from ..utils import generate_data, generate_model, almost_equal


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 10)
        model = generate_model(input_shape, nb_labels)

        method = SmoothGrad(model, -2, nb_samples=100)
        smoothed_gradients = method.explain(x, y)

        assert x.shape == smoothed_gradients.shape


def test_noisy_mask():
    """Ensure the inputs generated are not the same as the original"""
    x = tf.zeros((1, 32, 32, 3), tf.float32)
    noise = tf.random.normal((4, *x.shape[1:]), dtype=tf.float32)

    x_noisy = SmoothGrad._apply_noise(x, noise)

    assert almost_equal(x_noisy, noise)
