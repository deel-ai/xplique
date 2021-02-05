import tensorflow as tf

from xplique.utils import guided_relu, override_relu_gradient


def test_override_commutation():
    """Ensure we commute correctly differents ReLU activations"""
    input_shape = (28, 28, 1)

    model_1 = tf.keras.models.Sequential()
    model_1.add(tf.keras.layers.Activation('relu'))
    model_1.build(input_shape)

    model_2 = tf.keras.models.Sequential()
    model_2.add(tf.keras.layers.Activation(tf.nn.relu))
    model_2.build(input_shape)

    for model in [model_1, model_2]:
        guided_model = override_relu_gradient(model, guided_relu)
        assert guided_model.layers[0].activation == guided_relu
