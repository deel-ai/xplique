import numpy as np
import tensorflow as tf

from xplique.commons import (
    deconv_relu_policy,
    guided_relu_policy,
    open_relu_policy,
    override_relu_gradient,
)

from ..utils import almost_equal


def test_guided_policy():
    """Ensure the forward pass is the same and the guided mechanism is correct"""

    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Activation(tf.nn.relu, input_shape=(4,)))
    model1.add(tf.keras.layers.Lambda(lambda x: -(x**2)))
    model1_guided = override_relu_gradient(model1, guided_relu_policy)

    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.ReLU(input_shape=(4,)))
    model2.add(tf.keras.layers.Lambda(lambda x: -(x**2)))
    model2_guided = override_relu_gradient(model2, guided_relu_policy)

    model3 = tf.keras.models.Sequential()
    model3.add(tf.keras.layers.ReLU(max_value=3, input_shape=(4,)))  # relu with params
    model3.add(tf.keras.layers.Lambda(lambda x: -(x**2)))
    model3_guided = override_relu_gradient(model3, guided_relu_policy)

    x = tf.constant(np.expand_dims([5.0, 0.0, -5.0, 10.0], axis=0))
    y1, y2, y3 = model1(x), model2(x), model3(x)
    y1g, y2g, y3g = model1_guided(x), model2_guided(x), model3_guided(x)

    # the policy should not impact the forward pass
    assert almost_equal(y1, y1g)
    assert almost_equal(y2, y2g)
    assert almost_equal(y3, y3g)

    # gradient of model 1, 2 and 3 should be the same
    with tf.GradientTape() as tape:
        tape.watch(x)
        y1g = model1_guided(x)
    guided_grads_1 = tape.gradient(y1g, x).numpy()[0]

    with tf.GradientTape() as tape:
        tape.watch(x)
        y2g = model2_guided(x)
    guided_grads_2 = tape.gradient(y2g, x).numpy()[0]

    with tf.GradientTape() as tape:
        tape.watch(x)
        y3g = model3_guided(x)
    guided_grads_3 = tape.gradient(y3g, x).numpy()[0]

    assert almost_equal(guided_grads_1, guided_grads_2)
    assert almost_equal(guided_grads_1, guided_grads_3)
    assert almost_equal(guided_grads_1, np.array([0.0, 0.0, 0.0, 0.0]))


def test_deconv_policy():
    """Ensure the forward pass is the same and the guided mechanism is correct"""

    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Activation(tf.nn.relu, input_shape=(4,)))
    model1.add(tf.keras.layers.Lambda(lambda x: x**2))
    model1_deconv = override_relu_gradient(model1, deconv_relu_policy)

    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.ReLU(input_shape=(4,)))
    model2.add(tf.keras.layers.Lambda(lambda x: x**2))
    model2_deconv = override_relu_gradient(model2, deconv_relu_policy)

    model3 = tf.keras.models.Sequential()
    model3.add(tf.keras.layers.ReLU(max_value=3, input_shape=(4,)))  # relu with params
    model3.add(tf.keras.layers.Lambda(lambda x: x**2))
    model3_deconv = override_relu_gradient(model3, deconv_relu_policy)

    x = tf.constant(np.expand_dims([5.0, 0.0, -5.0, 10.0], axis=0))
    y1, y2, y3 = model1(x), model2(x), model3(x)
    y1d, y2d, y3d = model1_deconv(x), model2_deconv(x), model3_deconv(x)

    # the policy should not impact the forward pass
    assert almost_equal(y1, y1d)
    assert almost_equal(y2, y2d)
    assert almost_equal(y3, y3d)

    # gradient of model 1, 2 should be the same
    with tf.GradientTape() as tape:
        tape.watch(x)
        y1d = model1_deconv(x)
    deconv_grads_1 = tape.gradient(y1d, x).numpy()[0]

    with tf.GradientTape() as tape:
        tape.watch(x)
        y2d = model2_deconv(x)
    deconv_grads_2 = tape.gradient(y2d, x).numpy()[0]

    assert almost_equal(deconv_grads_1, deconv_grads_2)
    assert almost_equal(deconv_grads_1, np.array([10.0, 0.0, 0.0, 20.0]))

    with tf.GradientTape() as tape:
        tape.watch(x)
        y3d = model3_deconv(x)
    deconv_grads_3 = tape.gradient(y3d, x).numpy()[0]

    assert almost_equal(deconv_grads_3, np.array([6.0, 0.0, 0.0, 6.0]))


def test_open_relu():
    """Ensure the backward pass let all the gradient flow"""

    model1 = tf.keras.models.Sequential()
    model1.add(tf.keras.layers.Lambda(lambda x: x**3, input_shape=(4,)))
    model1.add(tf.keras.layers.Activation(tf.nn.relu))
    model1_open = override_relu_gradient(model1, open_relu_policy)

    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Lambda(lambda x: x**3, input_shape=(4,)))
    model2.add(tf.keras.layers.ReLU())
    model2_open = override_relu_gradient(model2, open_relu_policy)

    model3 = tf.keras.models.Sequential()
    model3.add(tf.keras.layers.Lambda(lambda x: x**3, input_shape=(4,)))
    model3.add(
        tf.keras.layers.ReLU(
            max_value=3,
        )
    )  # relu with params
    model3_open = override_relu_gradient(model3, open_relu_policy)

    x = tf.constant(np.expand_dims([5.0, 0.0, -5.0, 10.0], axis=0))
    y1, y2, y3 = model1(x), model2(x), model3(x)
    y1d, y2d, y3d = model1_open(x), model2_open(x), model3_open(x)

    # the policy should not impact the forward pass
    assert almost_equal(y1, y1d)
    assert almost_equal(y2, y2d)
    assert almost_equal(y3, y3d)

    # gradient of model 1, 2 should be the same
    with tf.GradientTape() as tape:
        tape.watch(x)
        y1d = model1_open(x)
    open_grads_1 = tape.gradient(y1d, x).numpy()[0]

    with tf.GradientTape() as tape:
        tape.watch(x)
        y2d = model2_open(x)
    open_grads_2 = tape.gradient(y2d, x).numpy()[0]

    assert almost_equal(open_grads_1, open_grads_2)
    assert almost_equal(open_grads_1, 3.0 * x**2.0)  # dx**3/dx = 3x**2

    with tf.GradientTape() as tape:
        tape.watch(x)
        y3d = model3_open(x)
    open_grads_3 = tape.gradient(y3d, x).numpy()[0]

    assert almost_equal(open_grads_3, 3.0 * x**2.0)


def test_override_clones_lambda_without_deserialization():
    """Ensure Lambda layers are cloned without mutating the source model."""
    lambda_layer = tf.keras.layers.Lambda(
        lambda inputs, scale: inputs * scale,
        output_shape=(4,),
        arguments={"scale": 2.0},
        trainable=False,
        dtype=tf.float64,
    )
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input((4,)),
            tf.keras.layers.Activation(tf.nn.relu),
            lambda_layer,
        ]
    )

    cloned_model = override_relu_gradient(model, guided_relu_policy)
    cloned_lambda = cloned_model.layers[-1]

    assert cloned_lambda is not lambda_layer
    assert cloned_lambda.function is lambda_layer.function
    assert cloned_lambda.arguments == lambda_layer.arguments
    assert cloned_lambda.arguments is not lambda_layer.arguments
    assert cloned_lambda._output_shape == lambda_layer._output_shape
    assert cloned_lambda.trainable is False
    assert cloned_lambda.compute_dtype == "float64"
    assert model.layers[0].activation in [tf.nn.relu, tf.keras.activations.relu]
    assert cloned_model.layers[0].activation not in [tf.nn.relu, tf.keras.activations.relu]

    inputs = tf.constant([[1.0, -1.0, 2.0, -2.0]])
    assert almost_equal(model(inputs), cloned_model(inputs))
