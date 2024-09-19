"""
Test operations on tf datasets
"""
import os
import sys

sys.path.append(os.getcwd())

import unittest

import numpy as np
import tensorflow as tf


from xplique.example_based.datasets_operations.tf_dataset_operations import *
from xplique.example_based.datasets_operations.tf_dataset_operations import _almost_equal


def test_are_dataset_first_elems_equal():
    """
    Verify that the function is able to compare the first element of datasets
    """
    tf_dataset_up = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(90), (10, 3, 3))
    )
    tf_dataset_up_small = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(45), (5, 3, 3))
    )
    tf_dataset_down = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(90, 0, -1), (10, 3, 3))
    )

    zipped = tf.data.Dataset.zip((tf_dataset_up, tf_dataset_up))
    zipped_batched_in = tf.data.Dataset.zip(
        (tf_dataset_up.batch(3), tf_dataset_up.batch(3))
    )

    assert are_dataset_first_elems_equal(tf_dataset_up, tf_dataset_up)
    assert are_dataset_first_elems_equal(tf_dataset_up.batch(3), tf_dataset_up.batch(3))
    assert are_dataset_first_elems_equal(tf_dataset_up, tf_dataset_up_small)
    assert are_dataset_first_elems_equal(
        tf_dataset_up.batch(3), tf_dataset_up_small.batch(3)
    )
    assert are_dataset_first_elems_equal(zipped, zipped)
    assert are_dataset_first_elems_equal(zipped.batch(3), zipped.batch(3))
    assert are_dataset_first_elems_equal(zipped_batched_in, zipped_batched_in)
    assert not are_dataset_first_elems_equal(tf_dataset_up, zipped)
    assert not are_dataset_first_elems_equal(tf_dataset_up.batch(3), zipped.batch(3))
    assert not are_dataset_first_elems_equal(tf_dataset_up.batch(3), zipped_batched_in)
    assert not are_dataset_first_elems_equal(tf_dataset_up, tf_dataset_down)
    assert not are_dataset_first_elems_equal(
        tf_dataset_up.batch(3), tf_dataset_down.batch(3)
    )


def test_is_not_shuffled():
    """
    Verify the function is able to detect dataset that do not provide stable order of elements
    """
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(90), (10, 3, 3))
    )
    tf_shuffled_once = tf_dataset.shuffle(3, reshuffle_each_iteration=False)
    zipped = tf.data.Dataset.zip((tf_dataset, tf_dataset))

    assert is_not_shuffled(tf_dataset)
    assert is_not_shuffled(tf_dataset.batch(3))
    assert is_not_shuffled(tf_shuffled_once)
    assert is_not_shuffled(tf_shuffled_once.batch(3))
    assert is_not_shuffled(zipped)
    assert is_not_shuffled(zipped.batch(3))


def test_batch_size_matches():
    """
    Test that the function is able to detect incoherence between dataset and batch_size
    """
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(90), (10, 3, 3))
    )
    tf_dataset_b1 = tf_dataset.batch(1)
    tf_dataset_b2 = tf_dataset.batch(2)
    tf_dataset_b5 = tf_dataset.batch(5)
    tf_dataset_b25 = tf_dataset_b5.batch(2)
    tf_dataset_b52 = tf_dataset_b2.batch(5)
    tf_dataset_b32 = tf_dataset.batch(32)

    assert batch_size_matches(tf_dataset_b1, 1)
    assert batch_size_matches(tf_dataset_b2, 2)
    assert batch_size_matches(tf_dataset_b5, 5)
    assert batch_size_matches(tf_dataset_b25, 2)
    assert batch_size_matches(tf_dataset_b52, 5)
    assert batch_size_matches(tf_dataset_b32, 10)


def test_sanitize_dataset():
    """
    Test that verifies that the function harmonize inputs into datasets
    """
    tf_tensor = tf.reshape(tf.range(90), (10, 3, 3))
    np_array = np.array(tf_tensor)
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_tensor)
    tf_dataset_b4 = tf_dataset.batch(4)

    # test convertion
    assert sanitize_dataset(None, 1) is None
    assert are_dataset_first_elems_equal(tf_dataset, tf_dataset)
    assert are_dataset_first_elems_equal(tf_dataset_b4, tf_dataset_b4)
    assert are_dataset_first_elems_equal(
        sanitize_dataset(tf_tensor, 4, 3), tf_dataset_b4
    )
    assert are_dataset_first_elems_equal(
        sanitize_dataset(np_array, 4, 3), tf_dataset_b4
    )

    # test catch assertion errors
    test_raise_assertion_error = unittest.TestCase().assertRaises
    test_raise_assertion_error(
        AssertionError, sanitize_dataset, tf_dataset.shuffle(2).batch(4), 4
    )
    test_raise_assertion_error(AssertionError, sanitize_dataset, tf_dataset_b4, 3)
    test_raise_assertion_error(AssertionError, sanitize_dataset, tf_dataset_b4, 4, 4)
    test_raise_assertion_error(AssertionError, sanitize_dataset, np_array[:6], 4, 4)


def test_dataset_gather():
    """
    Test dataset gather function
    """
    # (5, 2, 3, 3)
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(90), (10, 3, 3))
    ).batch(2)

    indices_1 = np.array([[[0, 0], [1, 1]], [[2, 1], [0, 0]]])
    # (2, 2, 3, 3)
    results_1 = dataset_gather(tf_dataset, indices_1)
    assert np.all(tf.shape(results_1).numpy() == np.array([2, 2, 3, 3]))
    assert _almost_equal(results_1[0, 0], results_1[1, 1])

    indices_2 = tf.constant([[[1, 1]]])
    # (1, 1, 3, 3)
    results_2 = dataset_gather(tf_dataset, indices_2)
    assert np.all(tf.shape(results_2).numpy() == np.array([1, 1, 3, 3]))
    assert _almost_equal(results_1[0, 1], results_2[0, 0])
