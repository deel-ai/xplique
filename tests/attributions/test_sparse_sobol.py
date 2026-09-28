"""Design and estimator contracts for support-restricted sparse Sobol attribution.

Shared execution contracts are in test_concept_channel_common.py.
"""

from numbers import Integral

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import SparseSobol
from xplique.attributions.global_sensitivity_analysis import JansenEstimator
from xplique.attributions.global_sensitivity_analysis.replicated_designs import ReplicatedSampler


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def _reference_ab(seed, input_index, nb_design, dimension):
    folded = tf.random.experimental.stateless_fold_in(
        tf.constant([seed, 0], tf.int64), tf.cast(input_index, tf.int64)
    )
    seed_a = tf.random.experimental.stateless_fold_in(folded, tf.constant(0, tf.int64))
    seed_b = tf.random.experimental.stateless_fold_in(folded, tf.constant(1, tf.int64))
    shape = [nb_design, dimension]
    return (
        tf.random.stateless_uniform(shape, seed_a, dtype=tf.float32),
        tf.random.stateless_uniform(shape, seed_b, dtype=tf.float32),
    )


@pytest.mark.parametrize("nb_design", [2, 3, 7, np.int64(9)])
@pytest.mark.parametrize("distribution", ["uniform", "bernoulli"])
def test_valid_parameters_include_non_power_of_two(nb_design, distribution):
    """Any integral design size of at least two is accepted."""
    explainer = SparseSobol(
        _sum, operator=_operator, nb_design=nb_design, mask_distribution=distribution
    )
    assert explainer.nb_design == int(nb_design)
    assert isinstance(explainer.nb_design, Integral)


@pytest.mark.parametrize(
    "name,value",
    [("nb_design", value) for value in [0, 1, -2, 2.0, True, np.bool_(False), None]]
    + [("mask_distribution", value) for value in ["Uniform", "binary", "", None, 1]],
)
def test_invalid_sparse_sobol_parameters(name, value):
    """Design size and distribution use strict public validation."""
    with pytest.raises(ValueError):
        SparseSobol(_sum, operator=_operator, **{name: value})


@pytest.mark.parametrize("distribution", ["uniform", "bernoulli"])
def test_exact_stateless_ab_c_design(distribution):
    """Masks are ordered A, B, then dimension-major A-with-one-B-column blocks."""
    nb_design, dimension = 5, 3
    explainer = SparseSobol(
        _sum, operator=_operator, nb_design=nb_design, mask_distribution=distribution, seed=-19
    )
    actual = explainer._sample_masks(dimension, 4)
    expected_a, expected_b = _reference_ab(-19, 4, nb_design, dimension)
    if distribution == "bernoulli":
        expected_a = tf.cast(expected_a < 0.5, tf.float32)
        expected_b = tf.cast(expected_b < 0.5, tf.float32)
    expected_c = ReplicatedSampler.build_replicated_design(expected_a, expected_b)
    expected = tf.concat([expected_a, expected_b, expected_c], axis=0)

    assert actual.dtype == tf.float32
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[:nb_design], expected_a)
    np.testing.assert_array_equal(actual[nb_design : 2 * nb_design], expected_b)
    for column in range(dimension):
        block = actual[(2 + column) * nb_design : (3 + column) * nb_design].numpy()
        expected_block = expected_a.numpy().copy()
        expected_block[:, column] = expected_b.numpy()[:, column]
        np.testing.assert_array_equal(block, expected_block)
    if distribution == "uniform":
        assert bool(tf.reduce_all((actual >= 0) & (actual < 1)))
    else:
        assert set(np.unique(actual.numpy())) <= {0.0, 1.0}


@pytest.mark.parametrize("seed", [-(2**63), -1, 2**63 - 1, np.int64(7)])
def test_signed64_seed_endpoints(seed):
    """The complete signed 64-bit seed domain follows the same reference folding."""
    explainer = SparseSobol(_sum, operator=_operator, nb_design=3, seed=seed)
    actual = explainer._sample_masks(2, 9)
    sampling_a, sampling_b = _reference_ab(int(seed), 9, 3, 2)
    expected_c = ReplicatedSampler.build_replicated_design(sampling_a, sampling_b)
    np.testing.assert_array_equal(actual, tf.concat([sampling_a, sampling_b, expected_c], 0))


def test_estimate_is_direct_jansen_evaluation():
    """SparseSobol delegates its sampled outputs unchanged to Jansen total indices."""
    explainer = SparseSobol(_sum, operator=_operator, nb_design=7, seed=31)
    masks = explainer._sample_masks(4, 2)
    outputs = tf.cast(tf.range(masks.shape[0]), tf.float64) ** 2 + 0.25
    expected = JansenEstimator()(masks, outputs, explainer.nb_design)
    np.testing.assert_array_equal(explainer._estimate(masks, outputs), expected)


def test_constant_scores_are_exactly_zero():
    """Jansen's protected zero-variance case returns exact zero indices."""

    def constant(inputs):
        return tf.fill([tf.shape(inputs)[0]], 13.0)

    result = SparseSobol(constant, operator=_operator, nb_design=7)([[1.0, -2.0, 0.0]], [[1.0]])
    np.testing.assert_array_equal(result, np.zeros((1, 3), np.float32))


def test_bernoulli_xor_has_positive_total_indices():
    """Unlike signed main effects, total indices detect a pure XOR interaction."""

    def xor(inputs):
        return tf.cast(tf.not_equal(inputs[:, 0], inputs[:, 1]), tf.float32)

    result = SparseSobol(
        xor, operator=_operator, nb_design=256, mask_distribution="bernoulli", seed=17
    )([[1.0, 1.0, 0.0]], [[1.0]])
    assert np.all(result.numpy()[0, :2] > 0)
    assert result.numpy()[0, 2] == 0
