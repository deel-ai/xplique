import numpy as np
import tensorflow as tf

from xplique.attributions import DeconvNet
from ..utils import generate_data, generate_model


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 100)
        model = generate_model(input_shape, nb_labels)

        method = DeconvNet(model, -2)
        explanations = method.explain(x, y)

        assert x.shape == explanations.shape


def test_deconv_mechanism():
    """Ensure we have a proper DeconvNet relu instead of relu"""

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(tf.nn.relu, input_shape=(4,)))
    model.add(tf.keras.layers.Lambda(lambda x: 10*x))

    x = tf.constant(np.expand_dims([5.0, 0.0, -5.0, 10.0], axis=0))

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    normal_grads = tape.gradient(y, x).numpy()[0]

    # dy / dx = { 10 for x > 0, else 0 }
    assert np.array_equal(normal_grads, np.array([10.0, 0.0, 0.0, 10.0]))

    deconvnet_model = DeconvNet(model).model

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = deconvnet_model(x)
    deconv_grads = tape.gradient(y, x).numpy()[0]

    # dy / dx = { 10 }
    assert np.array_equal(deconv_grads, np.array([10.0, 10.0, 10.0, 10.0]))

    # ensure we didn't change the mechanism of the original model
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    normal_grads_2 = tape.gradient(y, x).numpy()[0]

    assert np.array_equal(normal_grads, normal_grads_2)