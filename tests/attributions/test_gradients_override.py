import tensorflow as tf

from xplique.utils import guided_relu, override_relu_gradient


def test_override_commutation():
    """Ensure we commute correctly differents ReLU activations"""

    model_1 = tf.keras.models.Sequential()
    model_1.add(tf.keras.layers.Activation('relu'))

    model_2 = tf.keras.models.Sequential()
    model_2.add(tf.keras.layers.ReLU())

    model_3 = tf.keras.models.Sequential()
    model_3.add(tf.keras.layers.Activation(tf.nn.relu))

    for model in [model_1, model_2, model_3]:
        guided_model = override_relu_gradient(model, guided_relu)
        assert guided_model.layers[0].activation == guided_relu
