"""
Tests for the contrastive methods.
"""
import tensorflow as tf
import numpy as np

from xplique.example_based import NaiveSemiFactuals
from xplique.example_based.projections import Projection

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
