"""
Test Cole
"""
import os

import sys

sys.path.append(os.getcwd())

from math import prod, sqrt

import numpy as np
import scipy
import tensorflow as tf

from xplique.attributions import Occlusion, Saliency

from xplique.example_based import Cole, SimilarExamples
from xplique.example_based.projections import CustomProjection
from xplique.example_based.search_methods import KNN
from xplique.types import Union

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

    # Model generation
    model = generate_model(input_shape, nb_labels)

    return model, x_train, x_test, y_train


def test_cole_attribution():
    """
    Test Cole attribution projection.
    It should be the same as a manual projection.
    Test that the distance has an impact.
    """
    # Setup
    nb_samples = 20
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

    # Cole with attribution explain
    projection = CustomProjection(weights=Saliency(model))

    euclidean_dist = lambda x, z: tf.sqrt(tf.reduce_sum(tf.square(x - z)))
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
    examples_constructor = method_constructor.explain(x_test, y_test)
    examples_call = method_call.explain(x_test, y_test)
    examples_different_distance = method_different_distance(x_test, y_test)

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
    method_constructor.set_returns(["weights", "include_inputs"])
    assert almost_equal(
        method_constructor.explain(x_test, y_test)[:, 0],
        Saliency(model)(x_test, y_test),
    )


def test_cole_spliting():
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
# test_cole_spliting()
