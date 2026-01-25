import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D, Input

from xplique.attributions import FEM
from ..utils import generate_data, almost_equal


def _generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(3, 3),
                     activation='relu', name='conv_first'))
    model.add(Conv2D(4, kernel_size=(3, 3),
                     activation='relu', name='conv_second'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def test_output_shape():
    """The output shape must match the input spatial shape with a single channel."""
    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 5

    for input_shape in input_shapes:
        samples, labels = generate_data(input_shape, nb_labels, 12)
        model = _generate_model(input_shape, nb_labels)

        method = FEM(model)
        outputs = method.explain(samples, labels)

        assert samples.shape[:3] == outputs.shape[:3]
        assert outputs.shape[-1] == 1


def test_conv_layer():
    """We should target the right layer using either int, string or default procedure."""
    tf.keras.backend.clear_session()

    model = _generate_model()

    last_conv_layer = model.get_layer('conv_second')
    first_conv_layer = model.get_layer('conv_first')
    flatten_layer = model.get_layer('flatten')

    # default should target the last conv layer
    fem_default = FEM(model)
    assert fem_default.conv_layer == last_conv_layer

    # target the first conv layer
    fem_first_conv = FEM(model, conv_layer=0)
    assert fem_first_conv.conv_layer == first_conv_layer

    # target a random flatten layer
    fem_flatten = FEM(model, conv_layer='flatten')
    assert fem_flatten.conv_layer == flatten_layer


def test_weights_computation():
    """Ensure the FEM weights and masks follow the k-sigma rule."""
    feature_maps = np.array([[
        [[1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [1.0, 4.0]],
    ]], dtype=np.float32)  # shape (1, 2, 2, 2) with channels last

    weights, binary_mask = FEM._compute_weights(feature_maps, tf.constant(1.0, tf.float32))
    weights_np = weights.numpy().reshape(-1)

    assert almost_equal(weights_np, np.array([1.0, 1.0]))

    fem_map = FEM._apply_weights(weights, binary_mask).numpy()[0]
    expected = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
    ], dtype=np.float32)
    assert almost_equal(fem_map, expected)
