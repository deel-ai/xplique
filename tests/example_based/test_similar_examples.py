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

from xplique.example_based import SimilarExamples
from xplique.example_based.projections import CustomProjection
from xplique.example_based.search_methods import SklearnKNN
from xplique.types import Union

from tests.utils import almost_equal


def get_setup(input_shape, nb_samples=10, nb_labels=10):
    """
    Generate data and model for SimilarExamples
    """
    # Data generation
    x_train = tf.stack([i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)])
    x_test = x_train[1:-1]
    y_train = tf.range(len(x_train)) % nb_labels

    return x_train, x_test, y_train



def test_similar_examples_basic():
    """
    Test the SimilarExamples with an identity projection.
    """
    # Setup
    input_shape = (4, 4, 1)
    k = 3
    x_train, x_test, _ = get_setup(input_shape)

    identity_projection = CustomProjection(space_projection=lambda inputs, targets=None: inputs)

    # Method initialization
    method = SimilarExamples(case_dataset=x_train,
                             projection=identity_projection,
                             search_method=SklearnKNN,
                             k=k,
                             distance="euclidean")

    # Generate explanation
    examples = method.explain(x_test)

    # Verifications
    # Shape should be (n, k, h, w, c)
    assert examples.shape == (len(x_test), k) + input_shape

    for i in range(len(x_test)):
        # test examples:
        assert almost_equal(examples[i, 0], x_train[i + 1])
        assert almost_equal(examples[i, 1], x_train[i + 2])\
            or almost_equal(examples[i, 1], x_train[i])
        assert almost_equal(examples[i, 2], x_train[i])\
            or almost_equal(examples[i, 2], x_train[i + 2])


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

    identity_projection = CustomProjection(space_projection=lambda inputs, targets=None: inputs)

    # Method initialization
    method = SimilarExamples(case_dataset=(x_train, y_train),
                             projection=identity_projection,
                             search_method=SklearnKNN,
                             k=1,
                             distance="euclidean")

    method.set_returns("all")
    
    method.set_k(k)

    # Generate explanation
    method_output = method.explain(x_test)

    assert isinstance(method_output, dict)

    examples = method_output["examples"]
    weights = method_output["weights"]
    distances = method_output["distances"]
    labels = method_output["labels"]

    # test every outputs shape (with the include inputs)
    assert examples.shape == (nb_samples_test, k + 1) + input_shape
    assert weights.shape == (nb_samples_test, k + 1) + input_shape
    # the inputs distance ae zero and indices do not exist
    assert distances.shape == (nb_samples_test, k)
    assert labels.shape == (nb_samples_test, k)

    for i in range(nb_samples_test):
        # test examples:
        assert almost_equal(examples[i, 0], x_test[i])
        assert almost_equal(examples[i, 1], x_train[i + 1])
        assert almost_equal(examples[i, 2], x_train[i + 2])\
            or almost_equal(examples[i, 2], x_train[i])
        assert almost_equal(examples[i, 3], x_train[i])\
            or almost_equal(examples[i, 3], x_train[i + 2])

        # test weights
        assert almost_equal(weights[i], tf.ones(weights[i].shape, dtype=tf.float32))

        # test distances
        assert almost_equal(distances[i, 0], 0)
        assert almost_equal(distances[i, 1], sqrt(prod(input_shape)))
        assert almost_equal(distances[i, 2], sqrt(prod(input_shape)))

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
    x_train = weights * np.array(x_train) +  (1 - weights) * noise

    weighting_function = CustomProjection(weights=weights)

    method = SimilarExamples(case_dataset=(x_train, y_train),
                             projection=weighting_function,
                             search_method=SklearnKNN,
                             k=k,
                             distance="euclidean")

    # Generate explanation
    examples = method.explain(x_test)

    # Verifications
    # Shape should be (n, k, h, w, c)
    nb_samples_test = x_test.shape[0]
    assert examples.shape == (nb_samples_test, k) + input_shape

    for i in range(nb_samples_test):
        # test examples:
        assert almost_equal(examples[i, 0], x_train[i + 1])
        assert almost_equal(examples[i, 1], x_train[i + 2])\
            or almost_equal(examples[i, 1], x_train[i])
        assert almost_equal(examples[i, 2], x_train[i])\
            or almost_equal(examples[i, 2], x_train[i + 2])
