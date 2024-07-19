"""
Test Cole
"""
import os
import sys

sys.path.append(os.getcwd())

from math import prod, sqrt
import unittest

import numpy as np
import tensorflow as tf

from xplique.commons import are_dataset_first_elems_equal

from xplique.example_based import SimilarExamples
from xplique.example_based.projections import Projection

from tests.utils import almost_equal


def get_setup(input_shape, nb_samples=10, nb_labels=10):
    """
    Generate data and model for SimilarExamples
    """
    # Data generation
    x_train = tf.stack(
        [i * tf.ones(input_shape, tf.float32) for i in range(nb_samples)]
    )
    x_test = x_train[1:-1]
    y_train = tf.range(len(x_train), dtype=tf.float32) % nb_labels

    return x_train, x_test, y_train


def test_similar_examples_input_datasets_management():
    """
    Test management of dataset init inputs
    """
    proj = Projection(space_projection=lambda inputs, targets=None: inputs)

    tf_tensor = tf.reshape(tf.range(90, dtype=tf.float32), (10, 3, 3))
    np_array = np.array(tf_tensor)
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_tensor)
    too_short_np_array = np_array[:3]
    too_long_tf_dataset = tf_dataset.concatenate(tf_dataset)

    tf_dataset_b3 = tf_dataset.batch(3)
    tf_dataset_b5 = tf_dataset.batch(5)
    too_long_tf_dataset_b5 = too_long_tf_dataset.batch(5)
    too_long_tf_dataset_b10 = too_long_tf_dataset.batch(10)

    tf_shuffled = tf_dataset.shuffle(32, 0).batch(4)
    tf_one_shuffle = tf_dataset.shuffle(32, 0, reshuffle_each_iteration=False).batch(4)

    # Method initialization that should work
    method = SimilarExamples(tf_dataset_b3, None, np_array, projection=proj)
    assert are_dataset_first_elems_equal(method.cases_dataset, tf_dataset_b3)
    assert are_dataset_first_elems_equal(method.labels_dataset, None)
    assert are_dataset_first_elems_equal(method.targets_dataset, tf_dataset_b3)

    method = SimilarExamples(np_array, tf_tensor, None, batch_size=5, projection=proj)
    assert are_dataset_first_elems_equal(method.cases_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(method.labels_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(method.targets_dataset, None)

    method = SimilarExamples(
        tf.data.Dataset.zip((tf_dataset_b5, tf_dataset_b5)),
        None,
        np_array,
        projection=proj,
    )
    assert are_dataset_first_elems_equal(method.cases_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(method.labels_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(method.targets_dataset, tf_dataset_b5)

    method = SimilarExamples(
        tf.data.Dataset.zip((tf_one_shuffle, tf_one_shuffle)), projection=proj
    )
    assert are_dataset_first_elems_equal(method.cases_dataset, tf_one_shuffle)
    assert are_dataset_first_elems_equal(method.labels_dataset, tf_one_shuffle)
    assert are_dataset_first_elems_equal(method.targets_dataset, None)

    method = SimilarExamples(tf_one_shuffle, projection=proj)
    assert are_dataset_first_elems_equal(method.cases_dataset, tf_one_shuffle)
    assert are_dataset_first_elems_equal(method.labels_dataset, None)
    assert are_dataset_first_elems_equal(method.targets_dataset, None)

    # Method initialization that should not work
    test_raise_assertion_error = unittest.TestCase().assertRaises
    test_raise_assertion_error(TypeError, SimilarExamples)
    test_raise_assertion_error(AssertionError, SimilarExamples, tf_tensor)
    test_raise_assertion_error(
        AssertionError, SimilarExamples, tf_shuffled, projection=proj
    )
    test_raise_assertion_error(
        AssertionError, SimilarExamples, tf_dataset, tf_tensor, projection=proj
    )
    test_raise_assertion_error(
        AssertionError, SimilarExamples, tf_dataset_b3, tf_dataset_b5, projection=proj
    )
    test_raise_assertion_error(
        AssertionError,
        SimilarExamples,
        tf.data.Dataset.zip((tf_dataset_b5, tf_dataset_b5)),
        np_array,
        projection=proj,
    )
    test_raise_assertion_error(
        AssertionError, SimilarExamples, tf_dataset_b3, too_short_np_array
    )
    test_raise_assertion_error(
        AssertionError, SimilarExamples, tf_dataset, None, too_long_tf_dataset
    )
    test_raise_assertion_error(
        AssertionError,
        SimilarExamples,
        tf_dataset_b5,
        too_long_tf_dataset_b5,
        projection=proj,
    )
    test_raise_assertion_error(
        AssertionError,
        SimilarExamples,
        too_long_tf_dataset_b10,
        tf_dataset_b5,
        projection=proj,
    )


def test_similar_examples_basic():
    """
    Test the SimilarExamples with an identity projection.
    """
    # Setup
    input_shape = (4, 4, 1)
    k = 3
    x_train, x_test, _ = get_setup(input_shape)

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

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

    identity_projection = Projection(
        space_projection=lambda inputs, targets=None: inputs
    )

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
        assert almost_equal(distances[i, 1], sqrt(prod(input_shape)))
        assert almost_equal(distances[i, 2], sqrt(prod(input_shape)))

        # test labels
        assert almost_equal(labels[i, 0], y_train[i + 1])
        assert almost_equal(labels[i, 1], y_train[i]) or almost_equal(
            labels[i, 1], y_train[i + 2]
        )
        assert almost_equal(labels[i, 2], y_train[i]) or almost_equal(
            labels[i, 2], y_train[i + 2]
        )


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
        labels_dataset=y_train,
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
        print(i)
        print(examples[i, 0])
        print(x_train[i + 1])
        # test examples:
        assert almost_equal(examples[i, 0], x_train[i + 1])
        assert almost_equal(examples[i, 1], x_train[i + 2]) or almost_equal(
            examples[i, 1], x_train[i]
        )
        assert almost_equal(examples[i, 2], x_train[i]) or almost_equal(
            examples[i, 2], x_train[i + 2]
        )
