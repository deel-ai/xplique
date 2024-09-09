import pytest

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, Dropout, Flatten, MaxPooling2D, Input

from xplique.attributions import OptiCAM
from xplique.attributions.opti_cam import _normalization_dict, _loss_dict
from ..utils import generate_data, almost_equal


def _generate_model(input_shape=(32, 32, 3), output_shape=10):
    inputs = Input(shape=input_shape)
    x = Conv2D(4, kernel_size=(2, 2), activation='relu', name='conv2d')(inputs)
    x = Conv2D(4, kernel_size=(2, 2), activation='relu', name='conv2d_1')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(output_shape)(x)
    outputs = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')

    return model


def test_initiation():
    """Test the initiation of the OptiCAM class"""
    model = _generate_model()

    with pytest.raises(ValueError):
        OptiCAM(model, normalization='invalid')

    with pytest.raises(ValueError):
        OptiCAM(model, loss_type='invalid')

    with pytest.raises(AssertionError):
        OptiCAM(model, n_iters=0)


def test_output_shape():
    """The output shape must be the same as the input shape"""

    input_shapes = [(28, 28, 1), (32, 32, 3)]
    nb_labels = 10

    for input_shape in input_shapes:
        samples, labels = generate_data(input_shape, nb_labels, 100)
        model = _generate_model(input_shape, nb_labels)

        method = OptiCAM(model, -2)
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
    gc_default = OptiCAM(model)
    assert gc_default.conv_layer == last_conv_layer

    # target the first conv layer
    gc_input_conv = OptiCAM(model, conv_layer=1)
    assert gc_input_conv.conv_layer == first_conv_layer

    # target a random flatten layer
    gc_flatten = OptiCAM(model, conv_layer='flatten')
    assert gc_flatten.conv_layer == flatten_layer


def test_normalization_max_min():
    data = tf.constant([0.0, 2.0, 1.0, 3.0])
    normalized = _normalization_dict['max_min'](data)

    assert tf.reduce_all(almost_equal(normalized, [0.0, 0.6666667, 0.3333333, 1.0]))


def test_l1_loss_function():
    pred = tf.constant([2.0, 2.0, 2.0])
    true = tf.constant([1.0, 1.0, 1.0])
    loss_fn = _loss_dict['l1']
    loss = loss_fn(true, pred)

    assert tf.reduce_all(almost_equal(loss, 1.0))  # L1 = (|1| + |1| + |1|) / 3 = 1.0


def test_mse_loss_function():
    pred = tf.constant([2.0, 2.0, 2.0])
    true = tf.constant([1.0, 1.0, 1.0])
    loss_fn = _loss_dict['mse']
    loss = loss_fn(true, pred)

    assert tf.reduce_all(almost_equal(loss, 1.0))  # MSE = (1^2 + 1^2 + 1^2) / 3 = 1.0


def test_normalization_sigmoid():
    data = tf.constant([0.0, 2.0, 1.0, 3.0])
    normalized = _normalization_dict['sigmoid'](data)

    assert tf.reduce_all(almost_equal(normalized, tf.nn.sigmoid(data)))
