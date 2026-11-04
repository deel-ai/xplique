"""Estimator contracts for signed, support-restricted Banzhaf attribution.

Shared execution contracts are in test_concept_channel_common.py.
"""

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import Banzhaf


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


@pytest.mark.parametrize("interaction", [0.0, 3.0, -4.0])
@pytest.mark.parametrize("column", [False, True])
def test_exact_additive_and_pair_effects(interaction, column):
    """Conditional contrasts retain signs and split a pair interaction equally."""

    def model(inputs):
        scores = 7.0 + 2.0 * inputs[:, 0] - inputs[:, 1] + interaction * inputs[:, 0] * inputs[:, 1]
        return scores[:, None] if column else scores

    inputs = np.array([[2.0, 3.0, 0.0], [-1.0, 2.0, 5.0]], np.float32)
    expected = np.column_stack((2 * inputs[:, 0], -inputs[:, 1], np.zeros(2)))
    expected[:, :2] += (interaction * inputs[:, 0] * inputs[:, 1] / 2)[:, None]
    result = Banzhaf(model, operator=_operator, nb_samples=8)(inputs, np.ones((2, 1)))
    assert isinstance(result, tf.Tensor)
    assert result.dtype == tf.float32
    np.testing.assert_allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("game", ["constant", "xor"])
def test_exact_zero_effects(game):
    """Constants and a balanced two-player XOR have exactly zero effects."""

    def model(inputs):
        if game == "constant":
            return tf.fill([tf.shape(inputs)[0]], 13.0)
        return tf.cast(tf.not_equal(inputs[:, 0], inputs[:, 1]), tf.float32)

    result = Banzhaf(model, operator=_operator, nb_samples=4)([[1.0, 1.0, 0.0]], [[1.0]])
    np.testing.assert_array_equal(result, np.zeros((1, 3), np.float32))


@pytest.mark.parametrize("value", [1e-12, -1e-12, 2.5, -7.0])
@pytest.mark.parametrize("channels", [1, 3])
def test_no_support_threshold_or_magnitude_normalization(value, channels):
    """Every exactly nonzero coefficient participates, regardless of magnitude."""
    inputs = np.zeros((1, channels), np.float32)
    inputs[0, channels // 2] = value
    result = Banzhaf(_sum, operator=_operator, nb_samples=2)(inputs, [[1.0]])
    np.testing.assert_array_equal(result, inputs)


@pytest.mark.parametrize(
    "name, value",
    [("nb_samples", value) for value in [0, -2, 1, 3, 2.0, True, np.bool_(False), None]],
)
def test_invalid_parameters(name, value):
    """Integer parameters reject booleans, invalid ranges, and fractional values."""
    with pytest.raises(ValueError):
        Banzhaf(_sum, operator=_operator, **{name: value})


@pytest.mark.parametrize("nb_active", [1, 2, 3])
def test_exhaustive_mask_order(nb_active):
    """Exact designs enumerate integers with channel zero as the least significant bit."""
    explainer = Banzhaf(_sum, operator=_operator, nb_samples=8)
    masks = explainer._sample_masks(nb_active, 17)
    expected = (np.arange(2**nb_active)[:, None] >> np.arange(nb_active)) & 1
    assert masks.dtype == tf.float32
    np.testing.assert_array_equal(masks, expected)


def test_antithetic_masks_are_stateless_balanced_and_indexed():
    """Monte Carlo designs append complements and depend on seed and input index."""
    explainer = Banzhaf(_sum, operator=_operator, nb_samples=30, seed=91)
    masks = explainer._sample_masks(12, 0).numpy()
    assert masks.dtype == np.float32
    assert np.all((masks == 0) | (masks == 1))
    np.testing.assert_array_equal(masks[15:], 1 - masks[:15])
    np.testing.assert_array_equal(masks.sum(axis=0), np.full(12, 15))
    tf.random.uniform((100,))
    np.testing.assert_array_equal(masks, explainer._sample_masks(12, 0))


def test_sampled_effects_counts_and_batch_invariance():
    """Sampled estimates match conditional means and use exactly the requested rows."""
    inputs = np.arange(1, 25, dtype=np.float32).reshape(2, 12)
    results = []
    for batch_size in [1, 7, 32, None]:
        calls = []

        def operator(model, perturbed, targets):
            calls.append(int(perturbed.shape[0]))
            return model(perturbed)

        explainer = Banzhaf(_sum, operator=operator, nb_samples=30, batch_size=batch_size, seed=91)
        result = explainer(inputs, np.ones(2)).numpy()
        expected = []
        for index, row in enumerate(inputs):
            masks = explainer._sample_masks(12, index).numpy()
            scores = (masks * row).sum(axis=1)
            expected.append(
                [
                    scores[masks[:, j] == 1].mean() - scores[masks[:, j] == 0].mean()
                    for j in range(12)
                ]
            )
        np.testing.assert_allclose(result, expected, atol=2e-5)
        assert sum(calls) == 60
        assert max(calls) <= (30 if batch_size is None else batch_size)
        results.append(result)
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], atol=2e-5)


def test_eight_active_channels_in_384_use_exact_design():
    """Enumeration cost depends on active support, not the ambient channel count."""
    inputs = np.zeros((1, 384), np.float32)
    active = np.array([0, 3, 17, 80, 160, 255, 300, 383])
    inputs[0, active] = np.arange(1, 9) * np.array([1, -1] * 4)
    calls = []

    def operator(model, perturbed, targets):
        assert perturbed.shape[1:] == (384,)
        calls.append(int(perturbed.shape[0]))
        return model(perturbed)

    result = Banzhaf(_sum, operator=operator, nb_samples=256, batch_size=19)(inputs, [[1.0]])
    np.testing.assert_array_equal(result, inputs)
    assert sum(calls) == 256
    assert max(calls) <= 19


def test_float64_scores_preserve_small_effects_with_large_offsets():
    """Do not round valid operator scores to the attribution dtype before centering."""

    def model(inputs):
        return tf.cast(inputs[:, 0], tf.float64) + tf.constant(1e8, tf.float64)

    result = Banzhaf(model, operator=_operator, nb_samples=2)([[1.0]], [[1.0]])
    assert result.dtype == tf.float32
    np.testing.assert_array_equal(result, [[1.0]])
