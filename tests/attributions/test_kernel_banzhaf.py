"""Contracts for unregularized, support-restricted Kernel Banzhaf regression.

Shared execution contracts are in test_concept_channel_common.py.
"""

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import Banzhaf, KernelBanzhaf


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def test_inherited_constructor():
    """KernelBanzhaf shares the Banzhaf constructor and designs."""
    assert issubclass(KernelBanzhaf, Banzhaf)
    assert KernelBanzhaf.__init__ is Banzhaf.__init__


@pytest.mark.parametrize("interaction", [0.0, 3.0, -4.0])
@pytest.mark.parametrize("column", [False, True])
def test_exact_additive_and_pair_effects(interaction, column):
    """Exact regression agrees with analytic effects and conditional contrasts."""

    def model(inputs):
        scores = 7 + 2 * inputs[:, 0] - inputs[:, 1]
        scores += interaction * inputs[:, 0] * inputs[:, 1]
        return scores[:, None] if column else scores

    inputs = np.array([[2, 3, 0], [-1, 2, 5]], np.float32)
    targets = np.ones((2, 1), np.float32)
    expected = np.column_stack((2 * inputs[:, 0], -inputs[:, 1], np.zeros(2)))
    expected[:, :2] += (interaction * inputs[:, 0] * inputs[:, 1] / 2)[:, None]
    result = KernelBanzhaf(model, operator=_operator, nb_samples=8)(inputs, targets)
    contrast = Banzhaf(model, operator=_operator, nb_samples=8)(inputs, targets)
    assert isinstance(result, tf.Tensor)
    assert result.dtype == tf.float32
    np.testing.assert_allclose(result, expected, atol=1e-6)
    np.testing.assert_allclose(result, contrast, atol=1e-6)


@pytest.mark.parametrize("batch_size", [1, 7, None])
def test_sampled_nonlinear_lstsq_and_parent_seed(batch_size):
    """Sampled effects solve the actual seeded design, independent of batching."""
    inputs = np.arange(1, 17, dtype=np.float32).reshape(2, 8) / 4
    targets = np.array([[2.0], [-1.0]], np.float32)
    calls = []

    def model(values):
        values = tf.cast(values, tf.float64)
        return (
            tf.reduce_sum(values, axis=1)
            + 3 * values[:, 0] * values[:, 1]
            - 2 * values[:, 2] * values[:, 3] * values[:, 4]
        )

    def operator(model, perturbed, repeated):
        calls.append(len(perturbed))
        return model(perturbed) * tf.cast(repeated[:, 0], tf.float64)

    explainer = KernelBanzhaf(
        model, operator=operator, nb_samples=64, batch_size=batch_size, seed=91
    )
    parent = Banzhaf(model, operator=operator, nb_samples=64, seed=91)
    expected = []
    designs = []
    for index, row in enumerate(inputs):
        masks = parent._sample_masks(8, index).numpy().astype(np.float64)
        np.testing.assert_array_equal(explainer._sample_masks(8, index), masks)
        np.testing.assert_array_equal(masks[32:], 1 - masks[:32])
        values = masks * row
        scores = (
            values.sum(axis=1)
            + 3 * values[:, 0] * values[:, 1]
            - 2 * values[:, 2] * values[:, 3] * values[:, 4]
        ) * targets[index, 0]
        design = masks - masks.mean(axis=0)
        effects, _, rank, _ = np.linalg.lstsq(design, scores - scores.mean(), rcond=None)
        assert rank == 8
        expected.append(effects)
        designs.append(masks)
    assert not np.array_equal(designs[0], designs[1])
    result = explainer(inputs, targets)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-5)
    assert result.dtype == tf.float32
    assert sum(calls) == 128
    assert max(calls) <= (64 if batch_size is None else batch_size)
    tf.random.uniform((100,))
    np.testing.assert_array_equal(explainer._sample_masks(8, 0), designs[0])
    np.testing.assert_allclose(explainer(inputs, targets), result, atol=1e-6)
    unbatched = KernelBanzhaf(model, operator=operator, nb_samples=64, batch_size=None, seed=91)(
        inputs, targets
    )
    np.testing.assert_allclose(result, unbatched, rtol=0, atol=1e-6)


def test_impossible_sampled_budget_rejected_before_sampling_or_inference(monkeypatch):
    """Antithetic Q=4 cannot identify three effects, even before drawing masks."""

    def forbidden(model, inputs, targets):
        pytest.fail("An impossible budget must fail before sampling or inference")

    monkeypatch.setattr(Banzhaf, "_sample_masks", forbidden)
    explainer = KernelBanzhaf(forbidden, operator=forbidden, nb_samples=4)
    with pytest.raises(ValueError) as error:
        explainer(np.ones((1, 3)), [[1.0]])
    assert "3" in str(error.value)
    assert "4" in str(error.value)


@pytest.mark.parametrize("full_rank", [False, True])
def test_rank_check_and_full_rank_budget_boundary(monkeypatch, full_rank):
    """Reject a rank-one draw without retries, but accept full rank at Q=2d."""
    half = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], np.float32)
    if not full_rank:
        half[:] = 0
    masks = tf.constant(np.concatenate((half, 1 - half)))
    sampled = []
    calls = []

    def sample(self, nb_active, input_index):
        sampled.append((nb_active, input_index))
        return masks

    def operator(model, inputs, targets):
        calls.append(len(inputs))
        return model(inputs)

    monkeypatch.setattr(Banzhaf, "_sample_masks", sample)
    explainer = KernelBanzhaf(_sum, operator=operator, nb_samples=6)
    inputs = np.array([[2.0, -3.0, 4.0]], np.float32)
    if full_rank:
        np.testing.assert_allclose(explainer(inputs, [[1.0]]), inputs, atol=1e-6)
        assert sum(calls) == 6
    else:
        with pytest.raises(ValueError) as error:
            explainer(inputs, [[1.0]])
        message = str(error.value).lower()
        assert "rank" in message
        assert "3" in message and "6" in message
        assert calls == []
    assert sampled == [(3, 0)]


@pytest.mark.parametrize("game", ["constant", "xor"])
def test_exact_zero_effects(game):
    """An intercept and balanced XOR leave no linear effect up to SVD roundoff."""

    def model(inputs):
        if game == "constant":
            return tf.fill([tf.shape(inputs)[0]], 13.0)
        return tf.cast(tf.not_equal(inputs[:, 0], inputs[:, 1]), tf.float32)

    result = KernelBanzhaf(model, operator=_operator, nb_samples=4)([[1.0, 1.0, 0.0]], [[1.0]])
    np.testing.assert_allclose(result, 0, atol=1e-6)
    np.testing.assert_array_equal(result.numpy()[:, 2], 0)


@pytest.mark.parametrize("nb_active, budget", [(2, 4), (8, 64)])
def test_float64_scores_preserve_small_effects_with_large_offset(nb_active, budget):
    """Center scores before any float32 conversion destroys their small effects."""

    def model(inputs):
        values = tf.cast(inputs, tf.float64)
        return tf.constant(1e12, tf.float64) + values[:, 0] - 2 * values[:, 1]

    inputs = np.arange(1, nb_active + 1, dtype=np.float32)[None]
    result = KernelBanzhaf(model, operator=_operator, nb_samples=budget, seed=91)(inputs, [[1.0]])
    expected = np.zeros_like(inputs)
    expected[0, :2] = [1.0, -4.0]
    assert result.dtype == tf.float32
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_exact_arbitrary_game():
    """Exhaustive regression agrees with contrasts beyond pairwise games."""
    scores = tf.constant([3.0, -2.0, 9.0, 1.0, 4.0, -7.0, 0.0, 11.0], tf.float64)

    def model(inputs):
        ids = tf.reduce_sum(tf.cast(inputs, tf.int32) * [1, 2, 4], axis=1)
        return tf.gather(scores, ids)

    inputs, targets = np.ones((1, 3)), [[1.0]]
    reference = Banzhaf(model, operator=_operator, nb_samples=8)(inputs, targets)
    result = KernelBanzhaf(model, operator=_operator, nb_samples=8)(inputs, targets)
    np.testing.assert_allclose(result, reference, atol=1e-6)


def test_rank_failure_is_per_input():
    """A later invalid support fails before its own inference, not earlier inputs'."""
    calls = []

    def operator(model, inputs, targets):
        calls.append(len(inputs))
        return model(inputs)

    explainer = KernelBanzhaf(_sum, operator=operator, nb_samples=4)
    with pytest.raises(ValueError, match="rank bound"):
        explainer([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [[1.0], [1.0]])
    assert calls == [2]
    np.testing.assert_allclose(explainer([[2.0, 0.0, 0.0]], [[1.0]]), [[2.0, 0.0, 0.0]])
