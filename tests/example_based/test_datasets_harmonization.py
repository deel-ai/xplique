import pytest
import unittest
import tensorflow as tf
import numpy as np


from xplique.example_based.datasets_operations.tf_dataset_operations import are_dataset_first_elems_equal
from xplique.example_based.datasets_operations.harmonize import split_tf_dataset, harmonize_datasets


def generate_tf_dataset(n_samples=100, n_features=10, n_labels=1, n_targets=None, batch_size=None):
    """
    Utility function to generate TensorFlow datasets for testing.
    """
    cases = np.random.random((n_samples, n_features, n_features)).astype(np.float32)
    labels = np.random.randint(0, 2, size=(n_samples, n_labels)).astype(np.int64)
    
    if n_targets is not None:
        targets = np.random.random((n_samples, n_targets)).astype(np.float32)
        dataset = tf.data.Dataset.from_tensor_slices((cases, labels, targets))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((cases, labels))
    
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    
    return dataset


def test_split_tf_dataset_two_columns():
    dataset = generate_tf_dataset(n_samples=100, n_features=5, n_labels=2, batch_size=8)

    cases, labels, targets = split_tf_dataset(dataset)
    
    assert labels is not None, "Labels dataset should not be None for a 2-column dataset."
    assert targets is None, "Targets dataset should be None for a 2-column dataset."
    
    for case_h, label_h, (case, label) in zip(cases, labels, dataset):
        assert len(case.shape) == 3
        assert len(label.shape) == 2
        assert np.allclose(case_h, case), "Cases should match the original dataset."
        assert np.allclose(label_h, label), "Labels should match the original dataset."


def test_split_tf_dataset_three_columns():
    dataset = generate_tf_dataset(n_samples=100, n_features=5, n_labels=2, n_targets=2, batch_size=8)
    
    cases, labels, targets = split_tf_dataset(dataset)
    
    assert labels is not None, "Labels dataset should not be None for a 3-column dataset."
    assert targets is not None, "Targets dataset should not be None for a 3-column dataset."

    for case_h, label_h, target_h, (case, label, target) in zip(cases, labels, targets, dataset):
        assert len(case.shape) == 3
        assert len(label.shape) == 2
        assert len(target.shape) == 2
        assert np.allclose(case_h, case), "Cases should match the original dataset."
        assert np.allclose(label_h, label), "Labels should match the original dataset."
        assert np.allclose(target_h, target), "Targets should match the original dataset."


def test_harmonize_datasets_with_tf_dataset():
    dataset = generate_tf_dataset(n_samples=100, n_features=5, n_labels=3)
    batch_size = 10

    cases, labels, targets, batch_size_out = harmonize_datasets(dataset, batch_size=batch_size)

    assert cases is not None, "Cases dataset should not be None."
    assert labels is not None, "Labels dataset should not be None."
    assert targets is None, "Targets dataset should be None for a 2-column input dataset."
    assert batch_size_out == batch_size, "Output batch size should match the input batch size."


def test_harmonize_datasets_with_tf_dataset_three_columns():
    batch_size = 10
    dataset = generate_tf_dataset(n_samples=100, n_features=10, n_labels=1, n_targets=1, batch_size=batch_size)
    
    cases, labels, targets, batch_size_out = harmonize_datasets(dataset, batch_size=batch_size)
    
    assert cases is not None, "Cases dataset should not be None."
    assert labels is not None, "Labels dataset should not be None."
    assert targets is not None, "Targets dataset should not be None for a 3-column input dataset."
    assert batch_size_out == batch_size, "Output batch size should match the input batch size."


def test_harmonize_datasets_with_numpy():
    cases = np.random.random((100, 10)).astype(np.float32)
    labels = np.random.randint(0, 2, size=(100, 1)).astype(np.int64)
    batch_size = 10
    
    cases_out, labels_out, targets_out, batch_size_out = harmonize_datasets(cases, labels, batch_size=batch_size)
    
    assert targets_out is None, "Targets should be None when not provided."
    assert batch_size_out == batch_size, "Output batch size should match the input batch size."

    for case, label in zip(cases_out, labels_out):
        assert case.shape == (batch_size, cases.shape[1]), "Each case should have the same shape as the input cases."
        assert label.shape == (batch_size, labels.shape[1]), "Each label should have the same shape as the input labels."
        break


def test_inputs_combinations():
    """
    Test management of dataset init inputs
    """

    tf_tensor = tf.reshape(tf.range(90, dtype=tf.float32), (10, 3, 3))
    np_array = np.array(tf_tensor)
    tf_dataset = tf.data.Dataset.from_tensor_slices(tf_tensor)

    tf_dataset_b3 = tf_dataset.batch(3)
    tf_dataset_b5 = tf_dataset.batch(5)

    tf_one_shuffle = tf_dataset.shuffle(32, 0, reshuffle_each_iteration=False).batch(4)

    # Method initialization that should work
    cases_dataset, labels_dataset, targets_dataset, batch_size = harmonize_datasets(tf_dataset_b3, None, tf_dataset_b3)
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b3)
    assert are_dataset_first_elems_equal(labels_dataset, None)
    assert are_dataset_first_elems_equal(targets_dataset, tf_dataset_b3)
    assert batch_size == 3

    cases_dataset, labels_dataset, targets_dataset, batch_size = harmonize_datasets(tf_tensor, tf_tensor, None, batch_size=5)
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(labels_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(targets_dataset, None)
    assert batch_size == 5

    cases_dataset, labels_dataset, targets_dataset, batch_size =\
        harmonize_datasets(tf.data.Dataset.zip((tf_dataset_b5, tf_dataset_b5)), None, tf_dataset_b5)
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(labels_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(targets_dataset, tf_dataset_b5)
    assert batch_size == 5

    cases_dataset, labels_dataset, targets_dataset, batch_size =\
        harmonize_datasets(tf.data.Dataset.zip((tf_dataset_b5, tf_dataset_b5, tf_dataset_b5)))
    assert are_dataset_first_elems_equal(cases_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(labels_dataset, tf_dataset_b5)
    assert are_dataset_first_elems_equal(targets_dataset, tf_dataset_b5)
    assert batch_size == 5

    cases_dataset, labels_dataset, targets_dataset, batch_size =\
        harmonize_datasets(tf.data.Dataset.zip((tf_one_shuffle, tf_one_shuffle)))
    assert are_dataset_first_elems_equal(cases_dataset, tf_one_shuffle)
    assert are_dataset_first_elems_equal(labels_dataset, tf_one_shuffle)
    assert are_dataset_first_elems_equal(targets_dataset, None)
    assert batch_size == 4

    cases_dataset, labels_dataset, targets_dataset, batch_size = harmonize_datasets(tf_one_shuffle)
    assert are_dataset_first_elems_equal(cases_dataset, tf_one_shuffle)
    assert are_dataset_first_elems_equal(labels_dataset, None)
    assert are_dataset_first_elems_equal(targets_dataset, None)
    assert batch_size == 4



def test_error_raising():
    """
    Test management of dataset init inputs
    """

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

    # Method initialization that should not work
    test_raise_assertion_error = unittest.TestCase().assertRaises

    # not input
    test_raise_assertion_error(TypeError, harmonize_datasets)

    # shuffled
    test_raise_assertion_error(AssertionError, harmonize_datasets, tf_shuffled,)

    # mismatching types
    test_raise_assertion_error(AssertionError, harmonize_datasets, tf_dataset, tf_tensor,)
    test_raise_assertion_error(
        AssertionError,
        harmonize_datasets,
        tf.data.Dataset.zip((tf_dataset_b5, tf_dataset_b5)),
        np_array,
    )
    test_raise_assertion_error(
        AssertionError, harmonize_datasets, tf_dataset_b3, too_short_np_array
    )
    test_raise_assertion_error(
        AssertionError, harmonize_datasets, tf_dataset, None, too_long_tf_dataset
    )

    # not batched and no batch size provided
    test_raise_assertion_error(
        AssertionError,
        harmonize_datasets,
        tf.data.Dataset.from_tensor_slices((tf_tensor, tf_tensor)),
        tf_dataset,
    )

    # not matching batch sizes
    test_raise_assertion_error(
        AssertionError, harmonize_datasets, tf_dataset_b3, tf_dataset_b5,
    )
    test_raise_assertion_error(
        AssertionError,
        harmonize_datasets,
        too_long_tf_dataset_b10,
        tf_dataset_b5,
    )

    # mismatching cardinality
    test_raise_assertion_error(
        AssertionError,
        harmonize_datasets,
        tf_dataset_b5,
        too_long_tf_dataset_b5,
    )

    # multiple datasets for labels or targets
    test_raise_assertion_error(
        AssertionError,
        harmonize_datasets,
        tf.data.Dataset.zip((tf_dataset_b5, tf_dataset_b5)),
        tf_dataset_b5,
    )
    test_raise_assertion_error(
        AssertionError,
        harmonize_datasets,
        tf.data.Dataset.zip((tf_dataset_b5, tf_dataset_b5, tf_dataset_b5)),
        None,
        tf_dataset_b5,
    )
