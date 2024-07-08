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

from xplique.attributions import Saliency
from xplique.example_based.projections import Projection, AttributionProjection, LatentSpaceProjection
from xplique.example_based.projections.commons import model_splitting


def get_setup(input_shape, nb_samples=10, nb_labels=2):
    """
    Generate data and model for SimilarExamples
    """
    # Data generation
    x_train = tf.stack(
        [i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)]
    )
    x_test = x_train[1:-1]
    y_train = tf.one_hot(tf.range(len(x_train)) % nb_labels, nb_labels)

    return x_train, x_test, y_train


def _generate_model(input_shape=(32, 32, 3), output_shape=2):
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


def test_model_splitting_latent_layer():
    """We should target the right layer using either int, string or default procedure"""
    tf.keras.backend.clear_session()

    model = _generate_model()

    first_conv_layer = model.get_layer("conv2d_1")
    last_conv_layer = model.get_layer("conv2d_2")
    flatten_layer = model.get_layer("flatten")

    # last_conv should be recognized
    _, _, latent_layer = model_splitting(model, latent_layer="last_conv", return_layer=True)
    assert latent_layer == last_conv_layer

    # target the first conv layer
    _, _, latent_layer = model_splitting(model, latent_layer=0, return_layer=True)
    assert latent_layer == first_conv_layer

    # target a random flatten layer
    _, _, latent_layer = model_splitting(model, latent_layer="flatten", return_layer=True)
    assert latent_layer == flatten_layer


def test_simple_projection_mapping():
    """
    Test if a simple projection can be mapped.
    """
    # Setup
    input_shape = (7, 7, 3)
    nb_samples = 10
    nb_labels = 2
    x_train, _, y_train = get_setup(input_shape, nb_samples=nb_samples, nb_labels=nb_labels)

    weights = tf.random.uniform((input_shape[0], input_shape[1], 1), minval=0, maxval=1)

    space_projection = lambda x, y=None: tf.nn.max_pool2d(x, ksize=3, strides=1, padding="SAME")

    projection = Projection(get_weights=weights, space_projection=space_projection)

    # Generate tf.data.Dataset from numpy
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(3)

    # Apply the projection by mapping the dataset
    projected_train_dataset = projection.project_dataset(train_dataset)


def test_latent_space_projection_mapping():
    """
    Test if the latent space projection can be mapped.
    """
    # Setup
    input_shape = (7, 7, 3)
    nb_samples = 10
    nb_labels = 2
    x_train, _, y_train = get_setup(input_shape, nb_samples=nb_samples, nb_labels=nb_labels)

    model = _generate_model(input_shape=input_shape, output_shape=nb_labels)

    projection = LatentSpaceProjection(model, "last_conv")

    # Generate tf.data.Dataset from numpy
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(3)

    # Apply the projection by mapping the dataset
    projected_train_dataset = projection.project_dataset(train_dataset)


def test_attribution_projection_mapping():
    """
    Test if the attribution projection can be mapped.
    """
    # Setup
    input_shape = (7, 7, 3)
    nb_samples = 10
    nb_labels = 2
    x_train, _, y_train = get_setup(input_shape, nb_samples=nb_samples, nb_labels=nb_labels)

    model = _generate_model(input_shape=input_shape, output_shape=nb_labels)

    projection = AttributionProjection(model, method=Saliency, latent_layer="last_conv")

    # Generate tf.data.Dataset from numpy
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(3)
    targets_dataset = tf.data.Dataset.from_tensor_slices(y_train).batch(3)

    # Apply the projection by mapping the dataset
    projected_train_dataset = projection.project_dataset(train_dataset, targets_dataset)