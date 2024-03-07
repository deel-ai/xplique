"""
Tests for the contrastive methods.
"""
import pytest

import tensorflow as tf
import numpy as np

from xplique.example_based import NaiveSemiFactuals, PredictedLabelAwareSemiFactuals

def test_naive_semi_factuals():
    """
    """
    cases = tf.constant([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]], dtype=tf.float32)
    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)
    semi_factuals = NaiveSemiFactuals(cases_dataset, cases_targets_dataset, k=2, case_returns=["examples", "indices", "distances", "include_inputs"], batch_size=2)

    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)

    mask = semi_factuals.filter_fn(inputs, cases, targets, cases_targets)
    assert mask.shape == (inputs.shape[0], cases.shape[0])

    expected_mask = tf.constant([
        [True, False, False, True, False],
        [False, True, True, False, True],
        [False, True, True, False, True]], dtype=tf.bool)
    assert tf.reduce_all(tf.equal(mask, expected_mask))

    return_dict = semi_factuals(inputs, targets)
    assert set(return_dict.keys()) == set(["examples", "indices", "distances"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]
    indices = return_dict["indices"]

    assert examples.shape == (3, 3, 2) # (n, k+1, W)
    assert distances.shape == (3, 2) # (n, k)
    assert indices.shape == (3, 2, 2) # (n, k, 2)

    expected_examples = tf.constant([
        [[1.5, 2.5], [4., 5.], [1., 2.]],
        [[2.5, 3.5], [5., 6.], [2., 3.]],
        [[4.5, 5.5], [2., 3.], [3., 4.]]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(examples, expected_examples))

    expected_distances = tf.constant([[np.sqrt(2*2.5**2), np.sqrt(0.5)], [np.sqrt(2*2.5**2), np.sqrt(0.5)], [np.sqrt(2*2.5**2), np.sqrt(2*1.5**2)]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)

    expected_indices = tf.constant([[[1, 1], [0, 0]],[[2, 0], [0, 1]],[[0, 1], [1, 0]]], dtype=tf.int32)
    assert tf.reduce_all(tf.equal(indices, expected_indices))

def test_labelaware_semifactuals():
    """
    """
    cases = tf.constant([[1., 2.], [2., 3.], [3., 4.], [4., 5.], [5., 6.]], dtype=tf.float32)
    cases_targets = tf.constant([[0, 1], [1, 0], [1, 0], [0, 1], [1, 0]], dtype=tf.float32)

    cases_dataset = tf.data.Dataset.from_tensor_slices(cases).batch(2)
    cases_targets_dataset = tf.data.Dataset.from_tensor_slices(cases_targets).batch(2)

    inputs = tf.constant([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    targets = tf.constant([[0, 1], [1, 0], [1, 0]], dtype=tf.float32)

    semi_factuals = PredictedLabelAwareSemiFactuals(cases_dataset, cases_targets_dataset, target_label=0, k=2, batch_size=2, case_returns=["examples", "distances", "include_inputs"])
    # assert the filtering on the right label went right

    combined_dataset = tf.data.Dataset.zip((cases_dataset.unbatch(), cases_targets_dataset.unbatch()))
    for elem, label in combined_dataset:
        print(f"elem: {elem}, label: {label}")
        print(f"lambda_fn: {tf.equal(tf.argmax(label, axis=-1),0)}")
    combined_dataset = combined_dataset.filter(lambda x, y: tf.equal(tf.argmax(y, axis=-1),0))

    for elem, label in combined_dataset:
        print(f"elem: {elem}, label: {label}")

    filter_cases = semi_factuals.cases_dataset
    filter_targets = semi_factuals.targets_dataset

    # for elem in filter_cases:
    #     print(elem)
    # for elem in filter_targets:
    #     print(elem)

    expected_filter_cases = tf.constant([[2., 3.], [3., 4.], [5., 6.]], dtype=tf.float32)
    expected_filter_targets = tf.constant([[1, 0], [1, 0], [1, 0]], dtype=tf.float32)

    tensor_filter_cases = []
    for elem in filter_cases.unbatch():
        tensor_filter_cases.append(elem)
    tensor_filter_cases = tf.stack(tensor_filter_cases)
    assert tf.reduce_all(tf.equal(tensor_filter_cases, expected_filter_cases))

    tensor_filter_targets = []
    for elem in filter_targets.unbatch():
        tensor_filter_targets.append(elem)
    tensor_filter_targets = tf.stack(tensor_filter_targets)
    assert tf.reduce_all(tf.equal(tensor_filter_targets, expected_filter_targets))
    
    # check the call method
    filter_inputs = tf.constant([[2.5, 3.5], [4.5, 5.5]], dtype=tf.float32)
    filter_targets = tf.constant([[1, 0], [1, 0]], dtype=tf.float32)

    return_dict = semi_factuals(filter_inputs, filter_targets)
    assert set(return_dict.keys()) == set(["examples", "distances"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]

    assert examples.shape == (2, 3, 2) # (n_label0, k+1, W)
    assert distances.shape == (2, 2) # (n_label0, k)

    expected_examples = tf.constant([
        [[2.5, 3.5], [5., 6.], [2., 3.]],
        [[4.5, 5.5], [2., 3.], [3., 4.]]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(examples, expected_examples))

    expected_distances = tf.constant([[np.sqrt(2*2.5**2), np.sqrt(0.5)], [np.sqrt(2*2.5**2), np.sqrt(2*1.5**2)]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)


    # check an error is raised when a target does not match the target label
    with pytest.raises(AssertionError):
        semi_factuals(inputs, targets)
    
    # same but with the other label
    semi_factuals = PredictedLabelAwareSemiFactuals(cases_dataset, cases_targets_dataset, target_label=1, k=2, batch_size=2, case_returns=["examples", "distances", "include_inputs"])
    filter_cases = semi_factuals.cases_dataset
    filter_targets = semi_factuals.targets_dataset

    expected_filter_cases = tf.constant([[1., 2.], [4., 5.]], dtype=tf.float32)
    expected_filter_targets = tf.constant([[0, 1], [0, 1]], dtype=tf.float32)

    tensor_filter_cases = []
    for elem in filter_cases.unbatch():
        tensor_filter_cases.append(elem)
    tensor_filter_cases = tf.stack(tensor_filter_cases)
    assert tf.reduce_all(tf.equal(tensor_filter_cases, expected_filter_cases))

    tensor_filter_targets = []
    for elem in filter_targets.unbatch():
        tensor_filter_targets.append(elem)
    tensor_filter_targets = tf.stack(tensor_filter_targets)
    assert tf.reduce_all(tf.equal(tensor_filter_targets, expected_filter_targets))
    
    # check the call method
    filter_inputs = tf.constant([[1.5, 2.5]], dtype=tf.float32)
    filter_targets = tf.constant([[0, 1]], dtype=tf.float32)

    return_dict = semi_factuals(filter_inputs, filter_targets)
    assert set(return_dict.keys()) == set(["examples", "distances"])

    examples = return_dict["examples"]
    distances = return_dict["distances"]

    assert examples.shape == (1, 3, 2) # (n_label1, k+1, W)
    assert distances.shape == (1, 2) # (n_label1, k)

    expected_examples = tf.constant([
        [[1.5, 2.5], [4., 5.], [1., 2.]]], dtype=tf.float32)
    assert tf.reduce_all(tf.equal(examples, expected_examples))

    expected_distances = tf.constant([[np.sqrt(2*2.5**2), np.sqrt(0.5)]], dtype=tf.float32)
    assert tf.reduce_all(tf.abs(distances - expected_distances) < 1e-5)
