import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D, Input

from xplique.attributions import GradCAMPP
from ..utils import generate_data, almost_equal

def _generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', name='conv2d'))
    model.add(Conv2D(4, kernel_size=(2, 2),
                     activation='relu', name='conv2d_1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_shape))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    return model


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        samples, labels = generate_data(input_shape, nb_labels, 100)
        model = _generate_model(input_shape, nb_labels)

        method = GradCAMPP(model, -2)
        outputs = method.explain(samples, labels)   

        assert samples.shape[:3] == outputs.shape[:3]


def test_conv_layer():
    """We should target the right layer using either int, string or default procedure"""
    tf.keras.backend.clear_session()

    model = _generate_model()

    last_conv_layer = model.get_layer('conv2d_1')
    first_conv_layer = model.get_layer('conv2d')
    flatten_layer = model.get_layer('flatten')

    # default should target the last conv layer
    gc_default = GradCAMPP(model)
    assert gc_default.conv_layer == last_conv_layer

    # target the first conv layer
    gc_input_conv = GradCAMPP(model, conv_layer=0)
    assert gc_input_conv.conv_layer == first_conv_layer

    # target a random flatten layer
    gc_flatten = GradCAMPP(model, conv_layer='flatten')
    assert gc_flatten.conv_layer == flatten_layer

def test_weights_computation():
    """Ensure the grad-cam++ weights are correct"""
    activations = np.array([
        [[1.0, 1.0],
         [1.0, 1.0]],

        [[1.0, 1.0],
         [0.0, 0.0]],

        [[0.0, 0.0],
         [0.0, 0.0]],

        [[0.5, 0.0],
         [0.0, 0.0]],
    ], dtype=np.float32)[None, :, :, :]
    grads = np.array([
        [[1.0, 1.0],
         [1.0, 1.0]],

        [[1.0, 1.0],
         [0.0, 0.0]],

        [[0.0, 0.0],
         [0.0, 0.0]],

        [[0.5, 0.0],
         [0.0, 0.0]],
    ], dtype=np.float32)[None, :, :, :]

    # move so that the filters F are at the end [F, W, H] -> [W, H, F]
    activations = np.moveaxis(activations, 1, 3)
    grads = np.moveaxis(grads, 1, 3)

    weights = GradCAMPP._compute_weights(grads, activations)
    assert almost_equal(weights[0], [4.0 / (2.0 * 4.0 + 4.0) * (4.0 / 4.0),
                                     2.0 / (2.0 * 2.0  + 1.0) * (2.0 / 4.0),
                                     0.0,
                                     0.25/ (2.0 * 0.25 + 0.5**3*0.125) * 0.125], 1e-3)

    grad_cam_pp = GradCAMPP._apply_weights(weights, activations)
    assert almost_equal(grad_cam_pp, np.sum(activations * weights, -1)) # as we have no negative value