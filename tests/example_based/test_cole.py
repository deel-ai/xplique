"""
Test Cole
"""
import os

import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from xplique.commons.operators_operations import gradients_predictions
from xplique.attributions import Occlusion, Saliency
from xplique.example_based import Cole, SimilarExamples
from xplique.example_based.projections import Projection

from tests.utils import (
    generate_data,
    generate_model,
    almost_equal,
    generate_timeseries_model,
)


def get_setup(input_shape, nb_samples=10, nb_labels=10):
    """
    Generate data and model for Cole
    """
    # Data generation
    x_train = tf.stack(
        [i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)]
    )
    x_test = x_train[1:-1]
    y_train = tf.one_hot(tf.range(len(x_train)) % nb_labels, depth=nb_labels)
    y_test = y_train[1:-1]

    # Model generation
    model = generate_model(input_shape, nb_labels)

    return model, x_train, x_test, y_train, y_test


def test_cole_attribution():
    """
    Test Cole attribution projection.
    It should be the same as a manual projection.
    Test that the distance has an impact.
    """
    # Setup
    nb_samples = 50
    input_shape = (5, 5)
    nb_labels = 10
    k = 3
    x_train = tf.random.uniform(
        (nb_samples,) + input_shape, minval=-1, maxval=1, seed=0
    )
    x_test = tf.random.uniform((nb_samples,) + input_shape, minval=-1, maxval=1, seed=2)
    labels = tf.one_hot(
        indices=tf.repeat(input=tf.range(nb_labels), repeats=[nb_samples // nb_labels]),
        depth=nb_labels,
    )
    y_train = labels
    y_test = tf.random.shuffle(labels, seed=1)

    # Model generation
    model = generate_timeseries_model(input_shape, nb_labels)

    # Cole with attribution method constructor
    method_constructor = Cole(
        cases_dataset=x_train,
        targets_dataset=y_train,
        k=k,
        batch_size=7,
        distance="euclidean",
        model=model,
        attribution_method=Saliency,
    )

    # Cole with attribution explain batch gradient is overwritten for test purpose, do not copy!
    explainer = Saliency(model)
    explainer.batch_gradient = \
    lambda model, inputs, targets, batch_size:\
        explainer.gradient(model, inputs, targets)
    projection = Projection(get_weights=explainer)

    euclidean_dist = lambda x, z: tf.sqrt(tf.reduce_sum(tf.square(x - z), axis=-1))
    method_call = SimilarExamples(
        cases_dataset=x_train,
        targets_dataset=y_train,
        k=k,
        distance=euclidean_dist,
        projection=projection,
    )

    method_different_distance = Cole(
        cases_dataset=x_train,
        targets_dataset=y_train,
        k=k,
        batch_size=2,
        distance=np.inf,  # infinity norm based distance
        model=model,
        attribution_method=Saliency,
    )

    # Generate explanation
    examples_constructor = method_constructor.explain(x_test, y_test)["examples"]
    examples_call = method_call.explain(x_test, y_test)["examples"]
    examples_different_distance = method_different_distance(x_test, y_test)["examples"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples_constructor.shape == (len(x_test), k) + input_shape
    assert examples_call.shape == (len(x_test), k) + input_shape
    assert examples_different_distance.shape == (len(x_test), k) + input_shape

    # both methods should be the same
    assert almost_equal(examples_constructor, examples_call)

    # a different distance should give different results
    assert not almost_equal(examples_constructor, examples_different_distance)

    # check weights are equal to the attribution directly on the input
    method_constructor.returns = ["weights", "include_inputs"]
    assert almost_equal(
        method_constructor.explain(x_test, y_test)["weights"][:, 0],
        Saliency(model)(x_test, y_test),
    )


def test_cole_hadamard():
    """
    Test Cole with Hadamard projection.
    It should be the same as a manual projection.
    """
    # Setup
    input_shape = (7, 7, 3)
    nb_samples = 10
    nb_labels = 2
    k = 3
    model, x_train, x_test, y_train, y_test =\
        get_setup(input_shape, nb_samples=nb_samples, nb_labels=nb_labels)

    # Cole with Hadamard projection constructor
    method_constructor = Cole(
        cases_dataset=x_train,
        targets_dataset=y_train,
        k=k,
        batch_size=7,
        distance="euclidean",
        model=model,
        projection_method="gradient",
    )

    # Cole with Hadamard projection explain batch gradient is overwritten for test purpose, do not copy!
    weights_extraction = lambda inputs, targets: gradients_predictions(model, inputs, targets)
    projection = Projection(get_weights=weights_extraction)

    euclidean_dist = lambda x, z: tf.sqrt(tf.reduce_sum(tf.square(x - z), axis=-1))
    method_call = SimilarExamples(
        cases_dataset=x_train,
        targets_dataset=y_train,
        k=k,
        distance=euclidean_dist,
        projection=projection,
    )

    # Generate explanation
    examples_constructor = method_constructor.explain(x_test, y_test)["examples"]
    examples_call = method_call.explain(x_test, y_test)["examples"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples_constructor.shape == (len(x_test), k) + input_shape
    assert examples_call.shape == (len(x_test), k) + input_shape

    # both methods should be the same
    assert almost_equal(examples_constructor, examples_call)


def test_cole_splitting():
    """
    Test Cole with a `latent_layer` provided.
    It should split the model.
    """
    # Setup
    nb_samples = 10
    input_shape = (6, 6, 3)
    nb_labels = 5
    k = 1
    x_train = tf.random.uniform((nb_samples,) + input_shape, minval=0, maxval=1)
    x_test = tf.random.uniform((nb_samples,) + input_shape, minval=0, maxval=1)
    labels = tf.one_hot(
        indices=tf.repeat(input=tf.range(nb_labels), repeats=[nb_samples // nb_labels]),
        depth=nb_labels,
    )
    y_train = labels
    y_test = tf.random.shuffle(labels)

    # Model generation
    model = generate_model(input_shape, nb_labels)

    # Cole with attribution method constructor
    method = Cole(
        cases_dataset=x_train,
        targets_dataset=y_train,
        k=k,
        case_returns=["examples", "weights", "include_inputs"],
        model=model,
        latent_layer="last_conv",
        attribution_method=Occlusion,
        patch_size=2,
        patch_stride=1,
    )

    # Generate explanation
    outputs = method.explain(x_test, y_test)
    examples, weights = outputs["examples"], outputs["weights"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    nb_samples_test = x_test.shape[0]
    assert examples.shape == (nb_samples_test, k + 1) + input_shape
    assert weights.shape[:-1] == (nb_samples_test, k + 1) + input_shape[:-1]


# test_cole_attribution()
# test_cole_splitting()
