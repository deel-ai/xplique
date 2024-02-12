import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Activation,
    Dropout,
    Flatten,
    MaxPooling2D,
    Input,
)

from xplique.example_based.projections import AttributionProjection
from xplique.example_based.projections import LatentSpaceProjection
from ..utils import generate_data, almost_equal


def _generate_model(input_shape=(32, 32, 3), output_shape=10):
    model = tf.keras.Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv2D(4, kernel_size=(2, 2), activation="relu", name="conv2d_1"))
    model.add(Conv2D(4, kernel_size=(2, 2), activation="relu", name="conv2d_2"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(output_shape, name="dense"))
    model.add(Activation("softmax", name="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    return model


def test_attribution_latent_layer():
    """We should target the right layer using either int, string or default procedure"""
    tf.keras.backend.clear_session()

    model = _generate_model()

    first_conv_layer = model.get_layer("conv2d_1")
    last_conv_layer = model.get_layer("conv2d_2")
    flatten_layer = model.get_layer("flatten")

    # default should not include model splitting
    projection_default = AttributionProjection(model)
    assert projection_default.latent_layer is None

    # last_conv should be recognized
    projection_default = AttributionProjection(model, latent_layer="last_conv")
    assert projection_default.latent_layer == last_conv_layer

    # target the first conv layer
    projection_default = AttributionProjection(model, latent_layer=0)
    assert projection_default.latent_layer == first_conv_layer

    # target a random flatten layer
    projection_default = AttributionProjection(model, latent_layer="flatten")
    assert projection_default.latent_layer == flatten_layer


def test_latent_space_latent_layer():
    """We should target the right layer using either int, string or default procedure"""
    tf.keras.backend.clear_session()

    model = _generate_model()

    first_conv_layer = model.get_layer("conv2d_1")
    last_conv_layer = model.get_layer("conv2d_2")
    flatten_layer = model.get_layer("flatten")
    last_layer = model.get_layer("softmax")

    # default should not include model spliting
    projection_default = LatentSpaceProjection(model)
    assert projection_default.latent_layer == last_layer

    # last_conv should be recognized
    projection_default = LatentSpaceProjection(model, latent_layer="last_conv")
    assert projection_default.latent_layer == last_conv_layer

    # target the first conv layer
    projection_default = LatentSpaceProjection(model, latent_layer=0)
    assert projection_default.latent_layer == first_conv_layer

    # target a random flatten layer
    projection_default = LatentSpaceProjection(model, latent_layer="flatten")
    assert projection_default.latent_layer == flatten_layer
