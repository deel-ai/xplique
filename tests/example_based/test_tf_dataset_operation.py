"""
Test operations on tf datasets
"""

import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pytest
import tensorflow as tf

from xplique.example_based.datasets_operations.tf_dataset_operations import *
from xplique.example_based.datasets_operations.tf_dataset_operations import _almost_equal


def datasets_are_equal(dataset_1, dataset_2):
    """
    Iterate over the datasets and compare the elements
    """
    for elem_1, elem_2 in zip(dataset_1, dataset_2):
        if not _almost_equal(elem_1, elem_2):
            return False
    return True


def test_are_dataset_first_elems_equal():
    """
    Verify that the function is able to compare the first element of datasets
    """
    tf_dataset_up = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(90), (10, 3, 3)))
    tf_dataset_up_small = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(45), (5, 3, 3)))
    tf_dataset_down = tf.data.Dataset.from_tensor_slices(
        tf.reshape(tf.range(90, 0, -1), (10, 3, 3))
    )

    zipped = tf.data.Dataset.zip((tf_dataset_up, tf_dataset_up))
    zipped_batched_in = tf.data.Dataset.zip((tf_dataset_up.batch(3), tf_dataset_up.batch(3)))

    assert are_dataset_first_elems_equal(tf_dataset_up, tf_dataset_up)
    assert are_dataset_first_elems_equal(tf_dataset_up.batch(3), tf_dataset_up.batch(3))
    assert are_dataset_first_elems_equal(tf_dataset_up, tf_dataset_up_small)
    assert are_dataset_first_elems_equal(tf_dataset_up.batch(3), tf_dataset_up_small.batch(3))
    assert are_dataset_first_elems_equal(zipped, zipped)
    assert are_dataset_first_elems_equal(zipped.batch(3), zipped.batch(3))
    assert are_dataset_first_elems_equal(zipped_batched_in, zipped_batched_in)
    assert not are_dataset_first_elems_equal(tf_dataset_up, zipped)
    assert not are_dataset_first_elems_equal(tf_dataset_up.batch(3), zipped.batch(3))
    assert not are_dataset_first_elems_equal(tf_dataset_up.batch(3), zipped_batched_in)
    assert not are_dataset_first_elems_equal(tf_dataset_up, tf_dataset_down)
    assert not are_dataset_first_elems_equal(tf_dataset_up.batch(3), tf_dataset_down.batch(3))


def test_is_shuffled():
    """
    Verify the function is able to detect dataset that do not provide stable order of elements
    """
    # test with non-shuffled datasets
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(900), (100, 3, 3)))
    zipped = tf.data.Dataset.zip((tf_dataset, tf_dataset))
    tf_mapped = tf_dataset.map(lambda x: x)

    assert not is_shuffled(tf_dataset)
    assert not is_shuffled(tf_dataset.batch(3))
    assert not is_shuffled(zipped)
    assert not is_shuffled(zipped.batch(3))
    assert not is_shuffled(tf_mapped)

    # test with shuffled datasets
    tf_shuffled_once = tf_dataset.shuffle(3, reshuffle_each_iteration=False)
    tf_shuffled_once_zipped = tf.data.Dataset.zip((tf_shuffled_once, tf_shuffled_once))
    tf_shuffled_once_mapped = tf_shuffled_once.map(lambda x: x)

    assert not is_shuffled(tf_shuffled_once)
    assert not is_shuffled(tf_shuffled_once.batch(3))
    assert not is_shuffled(tf_shuffled_once_zipped)
    assert not is_shuffled(tf_shuffled_once_zipped.batch(3))
    assert not is_shuffled(tf_shuffled_once_mapped)

    # test with reshuffled datasets
    tf_reshuffled = tf_dataset.shuffle(3, reshuffle_each_iteration=True)
    tf_reshuffled_zipped = tf.data.Dataset.zip((tf_reshuffled, tf_reshuffled))
    tf_reshuffled_mapped = tf_reshuffled.map(lambda x: x)

    assert is_shuffled(tf_reshuffled)
    assert is_shuffled(tf_reshuffled.batch(3))
    assert is_shuffled(tf_reshuffled_zipped)
    assert is_shuffled(tf_reshuffled_zipped.batch(3))
    assert is_shuffled(tf_reshuffled_mapped)


def test_batch_size_matches():
    """
    Test that the function is able to detect incoherence between dataset and batch_size
    """
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(90), (10, 3, 3)))
    tf_b1 = tf_dataset.batch(1)
    tf_b2 = tf_dataset.batch(2)
    tf_b5 = tf_dataset.batch(5)
    tf_b25 = tf_b5.batch(2)
    tf_b52 = tf_b2.batch(5)
    tf_b32 = tf_dataset.batch(32)

    tf_b5_shuffled = tf_b5.shuffle(3)
    tf_b5_zipped = tf.data.Dataset.zip((tf_b5, tf_b5))
    tf_b5_mapped = tf_b5.map(lambda x: x)

    assert batch_size_matches(tf_b1, 1)
    assert batch_size_matches(tf_b2, 2)
    assert batch_size_matches(tf_b5, 5)
    assert batch_size_matches(tf_b25, 2)
    assert batch_size_matches(tf_b52, 5)
    assert batch_size_matches(tf_b32, 10)
    assert batch_size_matches(tf_b5_shuffled, 5)
    assert batch_size_matches(tf_b5_zipped, 5)
    assert batch_size_matches(tf_b5_mapped, 5)

    assert not batch_size_matches(tf_b1, 2)
    assert not batch_size_matches(tf_b2, 1)
    assert not batch_size_matches(tf_b5, 2)
    assert not batch_size_matches(tf_b25, 5)
    assert not batch_size_matches(tf_b52, 2)
    assert not batch_size_matches(tf_b32, 5)
    assert not batch_size_matches(tf_b5_shuffled, 2)
    assert not batch_size_matches(tf_b5_zipped, 2)
    assert not batch_size_matches(tf_b5_mapped, 2)


def test_sanitize_dataset():
    """
    Test that verifies that the function harmonize inputs into datasets
    """
    tf_tensor = tf.reshape(tf.range(90), (10, 3, 3))
    np_array = np.array(tf_tensor)
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_tensor)
    tf_dataset_b4 = tf_dataset.batch(4)
    tf_dataset_b4_mapped = tf_dataset_b4.map(lambda x: x).prefetch(2)

    # test sanitize_dataset do not destroy the dataset
    assert sanitize_dataset(None, 1) is None
    assert datasets_are_equal(sanitize_dataset(tf_dataset_b4, 4), tf_dataset_b4)
    assert datasets_are_equal(sanitize_dataset(tf_dataset_b4_mapped, 4), tf_dataset_b4)

    # test convertion to tf dataset
    assert datasets_are_equal(sanitize_dataset(np_array, 4), tf_dataset_b4)
    assert datasets_are_equal(sanitize_dataset(tf_tensor, 4), tf_dataset_b4)
    assert datasets_are_equal(sanitize_dataset(tf_dataset, 4), tf_dataset_b4)

    # test catch assertion errors
    with pytest.raises(AssertionError):
        sanitize_dataset(tf_dataset.shuffle(2).batch(4), 4)
    with pytest.raises(AssertionError):
        sanitize_dataset(tf_dataset_b4, 3)
    with pytest.raises(AssertionError):
        sanitize_dataset(tf_dataset_b4, 4, 4)
    with pytest.raises(AssertionError):
        sanitize_dataset(np_array[:6], 4, 4)


def test_dataset_gather():
    """
    Test dataset gather function
    """
    # (5, 2, 3, 3)
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.range(90), (10, 3, 3))).batch(2)

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
