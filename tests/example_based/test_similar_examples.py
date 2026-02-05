"""
Test Cole
"""

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from tests.utils import almost_equal
from xplique.example_based import SimilarExamples
from xplique.example_based.projections import Projection


def get_setup(input_shape, nb_samples=10, nb_labels=10):
    """
    Generate data and model for SimilarExamples
    """
    # Data generation
    x_train = tf.stack([i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)])
    x_test = x_train[1:-1]
    y_train = tf.range(len(x_train), dtype=tf.float32) % nb_labels

    return x_train, x_test, y_train


def test_similar_examples_basic():
    """
    Test the SimilarExamples with an identity projection.
    """
    # Setup
    input_shape = (4, 4, 1)
    k = 3
    x_train, x_test, _ = get_setup(input_shape)

    identity_projection = Projection(space_projection=lambda inputs, targets=None: inputs)

    # Method initialization
    method = SimilarExamples(
        cases_dataset=x_train,
        projection=identity_projection,
        k=k,
        batch_size=3,
        distance="euclidean",
    )

    # Generate explanation
    examples = method.explain(x_test)["examples"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples.shape == (len(x_test), k) + input_shape

    for i in range(len(x_test)):
        # test examples:
        assert almost_equal(examples[i, 0], x_train[i + 1])
        assert almost_equal(examples[i, 1], x_train[i + 2]) or almost_equal(
            examples[i, 1], x_train[i]
        )
        assert almost_equal(examples[i, 2], x_train[i]) or almost_equal(
            examples[i, 2], x_train[i + 2]
        )


def test_similar_examples_return_multiple_elements():
    """
    Test the returns attribute.
    Test modifying k.
    """
    # Setup
    input_shape = (5, 5, 1)
    k = 3
    x_train, x_test, y_train = get_setup(input_shape)

    nb_samples_test = len(x_test)
    assert nb_samples_test + 2 == len(y_train)

    identity_projection = Projection(space_projection=lambda inputs, targets=None: inputs)

    # Method initialization
    method = SimilarExamples(
        cases_dataset=x_train,
        labels_dataset=y_train,
        projection=identity_projection,
        k=1,
        batch_size=3,
        distance="euclidean",
    )

    method.returns = "all"
    method.k = k

    # Generate explanation
    method_output = method.explain(x_test)

    assert isinstance(method_output, dict)

    examples = method_output["examples"]
    distances = method_output["distances"]
    labels = method_output["labels"]

    # test every outputs shape (with the include inputs)
    assert examples.shape == (nb_samples_test, k + 1) + input_shape
    # the inputs distance ae zero and indices do not exist
    assert distances.shape == (nb_samples_test, k)
    assert labels.shape == (nb_samples_test, k)

    for i in range(nb_samples_test):
        # test examples:
        assert almost_equal(examples[i, 0], x_test[i])
        assert almost_equal(examples[i, 1], x_train[i + 1])
        assert almost_equal(examples[i, 2], x_train[i + 2]) or almost_equal(
            examples[i, 2], x_train[i]
        )
        assert almost_equal(examples[i, 3], x_train[i]) or almost_equal(
            examples[i, 3], x_train[i + 2]
        )

        # test distances
        assert almost_equal(distances[i, 0], 0)
        assert almost_equal(distances[i, 1], np.sqrt(np.prod(input_shape)))
        assert almost_equal(distances[i, 2], np.sqrt(np.prod(input_shape)))

        # test labels
        assert almost_equal(labels[i, 0], y_train[i + 1])
        assert almost_equal(labels[i, 1], y_train[i]) or almost_equal(labels[i, 1], y_train[i + 2])
        assert almost_equal(labels[i, 2], y_train[i]) or almost_equal(labels[i, 2], y_train[i + 2])


def test_similar_examples_weighting():
    """
    Test the application of the projection weighting.
    """
    # Setup
    input_shape = (4, 4, 1)
    nb_samples = 10
    k = 3
    x_train, x_test, y_train = get_setup(input_shape, nb_samples)

    # Define the weighing function
    weights = np.zeros(x_train[0].shape)
    weights[1] = np.ones(weights[1].shape)

    # create huge noise on non interesting features
    noise = np.random.uniform(size=x_train.shape, low=-100, high=100)
    x_train = np.float32(weights * np.array(x_train) + (1 - weights) * noise)

    weighting_function = Projection(get_weights=weights)

    method = SimilarExamples(
        cases_dataset=x_train,
        labels_dataset=np.array(y_train),
        projection=weighting_function,
        k=k,
        batch_size=5,
        distance="euclidean",
    )

    # Generate explanation
    examples = method.explain(x_test)["examples"]

    # Verifications
    # Shape should be (n, k, h, w, c)
    nb_samples_test = x_test.shape[0]
    assert examples.shape == (nb_samples_test, k) + input_shape

    for i in range(nb_samples_test):
        # test examples:
        assert almost_equal(examples[i, 0], x_train[i + 1])
        assert almost_equal(examples[i, 1], x_train[i + 2]) or almost_equal(
            examples[i, 1], x_train[i]
        )
        assert almost_equal(examples[i, 2], x_train[i]) or almost_equal(
            examples[i, 2], x_train[i + 2]
        )
