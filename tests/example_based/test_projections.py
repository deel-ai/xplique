import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling2D,
)

from xplique.attributions import Saliency
from xplique.commons.operators import predictions_operator
from xplique.example_based.projections import (
    AttributionProjection,
    HadamardProjection,
    LatentSpaceProjection,
    Projection,
)
from xplique.example_based.projections.commons import (
    model_splitting,
    target_free_classification_operator,
)

from ..utils import almost_equal


def get_setup(input_shape, nb_samples=10, nb_labels=2):
    """
    Generate data and model for SimilarExamples
    """
    # Data generation
    x_train = tf.stack([i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)])
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

    projection = Projection(get_weights=weights, space_projection=space_projection, mappable=True)

    # Generate tf.data.Dataset from numpy
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(3)
    targets_dataset = tf.data.Dataset.from_tensor_slices(y_train).batch(3)

    # Apply the projection by mapping the dataset
    projected_train_dataset = projection.project_dataset(train_dataset, targets_dataset)

    # Apply the projection by iterating over the dataset
    projection.mappable = False
    projected_train_dataset = projection.project_dataset(train_dataset, targets_dataset)


def test_model_splitting():
    """
    Test if projected samples have the expected values
    """
    x_train = np.reshape(np.arange(0, 100), (10, 10))

    model = tf.keras.Sequential()
    model.add(Input(shape=(10,)))
    model.add(Dense(10, name="dense1"))
    model.add(Dense(1, name="dense2"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    model.get_layer("dense1").set_weights(
        [np.eye(10) * np.sign(np.arange(-4.5, 5.5)), np.zeros(10)]
    )
    model.get_layer("dense2").set_weights([np.ones((10, 1)), np.zeros(1)])

    # Split the model
    _, _ = model_splitting(model, latent_layer=-1)
    _, _ = model_splitting(model, latent_layer="dense2")
    features_extractor, predictor = model_splitting(model, latent_layer="dense1")

    assert almost_equal(predictor(features_extractor(x_train)).numpy(), model(x_train))


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
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(3)
    targets_dataset = tf.data.Dataset.from_tensor_slices(y_train).batch(3)

    # Apply the projection by mapping the dataset
    projected_train_dataset = projection.project_dataset(train_dataset, targets_dataset)
    projected_train_dataset = projection._map_project_dataset(train_dataset, targets_dataset)
    projected_train_dataset = projection._loop_project_dataset(train_dataset, targets_dataset)


def test_hadamard_projection_mapping():
    """
    Test if the hadamard projection can be mapped.
    """
    # Setup
    input_shape = (7, 7, 3)
    nb_samples = 10
    nb_labels = 2
    x_train, _, y_train = get_setup(input_shape, nb_samples=nb_samples, nb_labels=nb_labels)

    model = _generate_model(input_shape=input_shape, output_shape=nb_labels)

    projection = HadamardProjection(model, "last_conv")

    # Generate tf.data.Dataset from numpy
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(3)
    targets_dataset = tf.data.Dataset.from_tensor_slices(y_train).batch(3)

    # Apply the projection by mapping the dataset
    projected_train_dataset = projection.project_dataset(train_dataset, targets_dataset)
    projected_train_dataset = projection._map_project_dataset(train_dataset, targets_dataset)
    projected_train_dataset = projection._loop_project_dataset(train_dataset, targets_dataset)


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

    projection = AttributionProjection(model, attribution_method=Saliency, latent_layer="last_conv")

    # Generate tf.data.Dataset from numpy
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(3)
    targets_dataset = tf.data.Dataset.from_tensor_slices(y_train).batch(3)

    # Apply the projection by mapping the dataset
    projected_train_dataset = projection.project_dataset(train_dataset, targets_dataset)


def test_from_splitted_model():
    """
    Test the other way of constructing the projection.
    """
    latent_width = 8
    nb_samples = 15
    input_features = 10
    output_features = 3
    x_train = np.reshape(np.arange(0, nb_samples * input_features), (nb_samples, input_features))
    tf_x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(3)

    model1 = tf.keras.Sequential()
    model1.add(Input(shape=(input_features,)))
    model1.add(Dense(latent_width, name="dense1"))
    model1.compile(loss="mean_absolute_error", optimizer="sgd")

    model2 = tf.keras.Sequential()
    model2.add(Input(shape=(latent_width,)))
    model2.add(Dense(output_features, name="dense2"))
    model2.compile(loss="categorical_crossentropy", optimizer="sgd")

    assert model1(x_train).shape == (nb_samples, latent_width)
    assert model2(model1(x_train)).shape == (nb_samples, output_features)

    # test LatentSpaceProjection from splitted model
    projection = LatentSpaceProjection(model=model1, latent_layer=None, mappable=True)
    projected_train_dataset = projection.project_dataset(train_dataset)

    # test HadamardProjection from splitted model
    projection = HadamardProjection(features_extractor=model1, predictor=model2, mappable=True)
    projected_train_dataset = projection.project_dataset(train_dataset)


def test_target_free_classification_operator():
    """
    Test if the target free classification operator works as expected.
    """
    nb_classes = 5
    x_train = np.reshape(np.arange(0, 100), (10, 10))

    model = tf.keras.Sequential()
    model.add(Input(shape=(10,)))
    model.add(Dense(10, name="dense1"))
    model.add(Dense(nb_classes, name="dense2"))
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    preds = model(x_train)
    targets = tf.one_hot(tf.argmax(preds, axis=1), nb_classes)

    scores1 = target_free_classification_operator(model, x_train)
    scores2 = predictions_operator(model, x_train, targets)

    assert almost_equal(scores1, scores2)
