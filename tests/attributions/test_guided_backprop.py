import numpy as np
import tensorflow as tf

from xplique.attributions import GuidedBackprop
from ..utils import generate_data, generate_model


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        x, y = generate_data(input_shape, nb_labels, 100)
        model = generate_model(input_shape, nb_labels)

        method = GuidedBackprop(model, -2)
        explanations = method.explain(x, y)

        assert x.shape == explanations.shape


def test_guided_mechanism():
    """Ensure we have a proper guided relu instead of relu"""

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Activation(tf.nn.relu, input_shape=(4,)))
    model.add(tf.keras.layers.Lambda(lambda x: -x ** 2))

    x = tf.constant(np.expand_dims([5.0, 0.0, -5.0, 10.0], axis=0))

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    normal_grads = tape.gradient(y, x).numpy()[0]

    # dy / dx = {-2x for x > 0, else 0 }
    assert np.array_equal(normal_grads, np.array([-10.0, 0.0, 0.0, -20.0]))

    guided_model = GuidedBackprop(model).model

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = guided_model(x)
    guided_grads = tape.gradient(y, x).numpy()[0]

    # dy / dx = { 0 }
    assert np.array_equal(guided_grads, np.array([0.0, 0.0, 0.0, 0.0]))

    # ensure we didn't change the mechanism of the original model
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x)
    normal_grads_2 = tape.gradient(y, x).numpy()[0]

    assert np.array_equal(normal_grads, normal_grads_2)
