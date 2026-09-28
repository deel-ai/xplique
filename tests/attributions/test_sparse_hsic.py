"""Design and estimator contracts for support-restricted sparse HSIC attribution.

Shared execution contracts are in test_concept_channel_common.py.
"""

from numbers import Integral

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import SparseHSIC
from xplique.attributions.concept_attributions.hsic import (
    _median_positive_pairwise_distance,
)


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def _reference_masks(seed, input_index, nb_samples, dimension):
    input_seed = tf.random.experimental.stateless_fold_in(
        tf.constant([seed, 0], tf.int64), tf.cast(input_index, tf.int64)
    )
    uniform = tf.random.stateless_uniform(
        [nb_samples, dimension], seed=input_seed, dtype=tf.float32
    )
    return tf.cast(uniform < 0.5, tf.float32)


def _reference_hsic(masks, outputs):
    masks = np.asarray(masks, dtype=np.float64)
    outputs = np.asarray(outputs, dtype=np.float64).reshape(-1)
    size = len(outputs)
    distances = np.abs(outputs[:, None] - outputs[None, :])
    positive = distances[distances > 0]
    bandwidth = np.median(positive) if positive.size else 1.0
    output_gram = np.exp(-(distances**2) / (2.0 * bandwidth**2))
    centering = np.eye(size) - np.ones((size, size)) / size
    centered_output_gram = centering @ output_gram @ centering

    scores = []
    for column in masks.T:
        mask_gram = 0.5 - (column[:, None] - column[None, :]) ** 2
        centered_mask_gram = centering @ mask_gram @ centering
        scores.append(np.trace(centered_mask_gram @ centered_output_gram) / size**2)
    return np.maximum(scores, 0).astype(np.float32)


@pytest.mark.parametrize("nb_samples", [2, 3, 5, np.int64(9)])
def test_valid_sample_counts_include_odd_integrals(nb_samples):
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=nb_samples)
    assert explainer.nb_samples == int(nb_samples)
    assert isinstance(explainer.nb_samples, Integral)


@pytest.mark.parametrize("value", [0, 1, -3, 2.0, True, np.bool_(False), None])
def test_invalid_sample_counts(value):
    with pytest.raises(ValueError):
        SparseHSIC(_sum, operator=_operator, nb_samples=value)


def test_masks_match_exact_iid_stateless_reference_without_pairing_or_enumeration():
    """Sampling is one Bernoulli draw per requested row and active channel."""
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=9, seed=-19)
    actual = explainer._sample_masks(7, 4)
    expected = _reference_masks(-19, 4, 9, 7)

    assert actual.shape == (9, 7)
    assert actual.dtype == tf.float32
    np.testing.assert_array_equal(actual, expected)
    assert not np.array_equal(actual[:4], 1 - actual[4:8])
    enumeration = (np.arange(8)[:, None] >> np.arange(3)) & 1
    assert not np.array_equal(explainer._sample_masks(3, 4)[:8], enumeration)


@pytest.mark.parametrize(
    "outputs, expected",
    [
        ([0.0, 0.0, 2.0], 2.0),
        ([0.0, 1.0, 4.0, 10.0], 5.0),
        ([1e12, 1e12 + 1.0, 1e12 + 4.0, 1e12 + 10.0], 5.0),
        ([7.0, 7.0, 7.0], 1.0),
        ([7.0], 1.0),
    ],
)
def test_positive_pairwise_distance_median(outputs, expected):
    """Zeros are excluded, even medians interpolate, and constants fall back to one."""
    actual = _median_positive_pairwise_distance(tf.constant(outputs, tf.float64))
    assert actual.dtype == tf.float64
    assert float(actual) == expected


def test_estimate_matches_full_numpy_centered_gram_hsic_with_ties():
    """The shortcut equals biased HSIC using centered binary and output Gram matrices."""
    masks = np.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.float32,
    )
    outputs = np.array([0.0, 0.0, 1.0, 2.0, 2.0, 4.0], dtype=np.float64)
    expected = _reference_hsic(masks, outputs)
    actual = SparseHSIC(_sum, operator=_operator, nb_samples=6)._estimate(masks, outputs)

    assert actual.dtype == tf.float32
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
    assert actual.numpy()[1] == 0


def test_two_sample_normalization_sentinel():
    """The biased normalization is n squared and the binary Gram contributes factor two."""
    masks = tf.constant([[0.0], [1.0]])
    outputs = tf.constant([0.0, 1.0], tf.float64)
    actual = SparseHSIC(_sum, operator=_operator, nb_samples=2)._estimate(masks, outputs)
    expected = np.float32((1.0 - np.exp(-0.5)) / 4.0)
    np.testing.assert_allclose(actual, [expected], rtol=0, atol=1e-8)


@pytest.mark.parametrize("magnitude", [8e307, 1e308])
def test_extreme_finite_scores_do_not_overflow_pairwise_distances(magnitude):
    """Extreme finite distances and their median remain numerically usable."""
    masks = tf.constant([[0.0], [1.0]])
    outputs = tf.constant([-magnitude, magnitude], tf.float64)
    actual = SparseHSIC(_sum, operator=_operator, nb_samples=2)._estimate(masks, outputs)
    expected = np.float32((1.0 - np.exp(-0.5)) / 4.0)
    np.testing.assert_allclose(actual, [expected], rtol=0, atol=1e-8)
    assert bool(tf.reduce_all(tf.math.is_finite(actual)))


def test_extreme_outlier_does_not_erase_small_distances():
    masks = tf.constant([[0.0], [0.0], [1.0], [1.0], [1.0]])
    outputs = tf.constant([1e308, 0.0, 1e-100, 2e-100, 3e-100], tf.float64)
    actual = SparseHSIC(_sum, operator=_operator, nb_samples=5)._estimate(masks, outputs)

    output_gram = np.zeros((5, 5), dtype=np.float64)
    output_gram[0, 0] = 1.0
    small_outputs = np.arange(4, dtype=np.float64)
    small_distances = small_outputs[:, None] - small_outputs[None, :]
    output_gram[1:, 1:] = np.exp(-0.5 * np.square(small_distances / 2.5))
    centered_masks = masks.numpy().astype(np.float64)
    centered_masks -= np.mean(centered_masks, axis=0, keepdims=True)
    expected = 2.0 * np.sum(centered_masks * (output_gram @ centered_masks), axis=0) / 25.0

    np.testing.assert_allclose(actual, expected.astype(np.float32), rtol=0, atol=1e-8)


def test_constant_outputs_are_exactly_zero():
    masks = _reference_masks(3, 0, 7, 4)
    outputs = tf.fill([7], tf.constant(1e100, tf.float64))
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=7)
    np.testing.assert_array_equal(explainer._estimate(masks, outputs), np.zeros(4, np.float32))

    def constant(inputs):
        return tf.fill([tf.shape(inputs)[0]], tf.constant(1e100, tf.float64))

    result = SparseHSIC(constant, operator=_operator, nb_samples=7)([[1.0, -2.0, 0.0]], [[1.0]])
    np.testing.assert_array_equal(result, np.zeros((1, 3), np.float32))


def test_balanced_xor_cancels_at_estimator_level():
    masks = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], tf.float32)
    outputs = tf.constant([0, 1, 1, 0], tf.float64)
    actual = SparseHSIC(_sum, operator=_operator, nb_samples=4)._estimate(masks, outputs)
    np.testing.assert_array_equal(actual, np.zeros(2, np.float32))


def test_model_score_translation_invariance():
    """Large exactly representable score offsets do not alter pairwise output geometry."""

    def shifted(inputs, offset):
        scores = tf.cast(_sum(inputs), tf.float64)
        return scores + tf.constant(offset, tf.float64)

    inputs = np.array([[1.0, 2.0, 4.0, 8.0]], np.float32)
    targets = np.ones((1, 1), np.float32)
    plain = SparseHSIC(
        lambda values: shifted(values, 0.0), operator=_operator, nb_samples=31, seed=7
    )(inputs, targets)
    translated = SparseHSIC(
        lambda values: shifted(values, 2**40), operator=_operator, nb_samples=31, seed=7
    )(inputs, targets)
    np.testing.assert_array_equal(translated, plain)
