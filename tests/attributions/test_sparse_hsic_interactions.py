"""Mathematical contracts for indexed SparseHSIC interactions.

Shared interaction execution contracts are in test_concept_channel_common.py.
"""

from itertools import product

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import SparseHSIC


def _sum(inputs):
    return tf.reduce_sum(inputs, axis=tf.range(1, tf.rank(inputs)))


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def _grams(outputs):
    outputs = np.asarray(outputs, np.float64)
    n = len(outputs)
    distances = np.abs(outputs[:, None] - outputs[None, :])
    positive = distances[distances > 0]
    bandwidth = np.median(positive) if positive.size else 1.0
    gram = np.exp(-0.5 * np.square(distances / bandwidth))
    centering = np.eye(n) - np.ones((n, n)) / n
    return gram, centering


def _reference_pair(masks, outputs, left, right):
    gram, centering = _grams(outputs)
    signs = 2 * np.asarray(masks, np.float64) - 1
    kernel_left = 0.5 * np.outer(signs[:, left], signs[:, left])
    kernel_right = 0.5 * np.outer(signs[:, right], signs[:, right])
    interaction_kernel = kernel_left * kernel_right
    n = len(signs)
    direct = np.trace(centering @ interaction_kernel @ centering @ gram) / n**2
    joint = (1 + kernel_left) * (1 + kernel_right)
    joint_minus_singletons = (
        np.trace(centering @ joint @ centering @ gram)
        - np.trace(centering @ kernel_left @ centering @ gram)
        - np.trace(centering @ kernel_right @ centering @ gram)
    ) / n**2
    np.testing.assert_allclose(direct, joint_minus_singletons, rtol=0, atol=1e-16)
    return direct


def _reference_singletons(masks, outputs):
    gram, centering = _grams(outputs)
    n = len(outputs)
    return [
        np.trace(centering @ (column[:, None] == column[None, :]) @ centering @ gram) / n**2
        for column in np.asarray(masks).T
    ]


def _fixed_explainer(masks, outputs):
    """Evaluate an exhaustive mask design with a known scalar score per row."""
    masks = np.asarray(masks, np.float32)
    values = {tuple(row): float(score) for row, score in zip(masks, outputs)}

    def model(inputs):
        return tf.constant([values[tuple(row)] for row in inputs.numpy()], tf.float64)

    explainer = SparseHSIC(model, operator=_operator, nb_samples=len(masks), batch_size=2)
    explainer._sample_masks = lambda nb_active, input_index: tf.constant(masks)
    return explainer


def test_pair_matches_centered_gram_and_joint_minus_singletons_on_unbalanced_design():
    masks = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    outputs = np.array([0.0, 1.0, 1.0, 3.0, 5.0])
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=len(masks))
    _, state = explainer._prepare_interactions(masks, tf.constant(outputs), None)
    signed = tf.cast(2 * masks - 1, tf.float64)
    pairs = tf.constant([[0, 1], [1, 2], [0, 2], [1, 0]], tf.int64)
    actual = explainer._estimate_pair_chunk(state, pairs)
    expected = [_reference_pair(masks, outputs, *pair) for pair in pairs.numpy()]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-8)
    np.testing.assert_allclose(
        explainer._estimate(masks, outputs), _reference_singletons(masks, outputs), atol=2e-8
    )
    assert actual[0] == actual[-1]
    # Centering the individual signs before multiplying is incorrect here.
    centered_signs = signed.numpy() - signed.numpy().mean(axis=0)
    wrong = centered_signs[:, 0] * centered_signs[:, 1]
    assert not np.allclose(
        wrong - wrong.mean(),
        (signed.numpy()[:, 0] * signed.numpy()[:, 1])
        - (signed.numpy()[:, 0] * signed.numpy()[:, 1]).mean(),
    )


@pytest.mark.parametrize("n_bits", [2, 3])
def test_exhaustive_parity_is_purely_pairwise_only_for_two_bits(n_bits):
    masks = np.array(list(product((0, 1), repeat=n_bits)), np.float32)
    outputs = (masks.sum(axis=1) % 2).astype(np.float64)
    result = _fixed_explainer(masks, outputs).explain_interactions(
        np.ones((1, n_bits), np.float32), [[1.0]]
    )[0]
    np.testing.assert_array_equal(result.main_effects, np.zeros(n_bits, np.float32))
    if n_bits == 2:
        expected = (1 - np.exp(-0.5)) / 8
        np.testing.assert_allclose(result.interaction_scores, [expected], atol=1e-8)
    else:
        np.testing.assert_array_equal(result.interaction_scores, np.zeros(3, np.float32))


def test_exhaustive_single_variable_output_has_no_irrelevant_pair():
    masks = np.array(list(product((0, 1), repeat=2)), np.float32)
    result = _fixed_explainer(masks, masks[:, 0]).explain_interactions([[1.0, 1.0]], [[1.0]])[0]
    assert result.main_effects[0] > 0
    assert result.main_effects[1] == 0
    assert result.interaction_scores[0] == 0


def test_extreme_output_bandwidth_remains_finite_for_pairs():
    masks = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]], tf.float32)
    outputs = tf.constant([-1e308, -1e308, 1e308, 1e308], tf.float64)
    explainer = SparseHSIC(_sum, operator=_operator, nb_samples=4)
    _, state = explainer._prepare_interactions(masks, outputs, None)
    scores = explainer._estimate_pair_chunk(state, tf.constant([[0, 1]]))
    assert np.isfinite(scores.numpy()).all()


def test_constant_scores_are_exactly_zero():
    def constant(inputs):
        return tf.fill([len(inputs)], tf.constant(1e100, tf.float64))

    zeros = SparseHSIC(constant, operator=_operator, nb_samples=9).explain_interactions(
        [[1.0, 2.0]], [[1.0]]
    )[0]
    np.testing.assert_array_equal(zeros.main_effects, [0.0, 0.0])
    np.testing.assert_array_equal(zeros.interaction_scores, [0.0])
