"""Signed Banzhaf pair interactions on exact active concept channels."""

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import Banzhaf


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def test_enumerated_mixed_differences_on_arbitrary_game():
    coefficients = np.array([1.5, -2.0, 3.0], np.float32)

    def model(values):
        left, middle, right = (values[:, j] / coefficients[j] for j in range(3))
        return 17 + 2 * left - 3 * middle + 5 * left * middle - 7 * left * middle * right

    result = Banzhaf(model, operator=_operator, nb_samples=8).explain_interactions(
        coefficients[None], [[1.0]]
    )[0]
    np.testing.assert_array_equal(result.pair_indices, [[0, 1], [0, 2], [1, 2]])
    np.testing.assert_allclose(result.interaction_scores, [1.5, -3.5, -3.5], atol=1e-6)


@pytest.mark.parametrize("game,expected", [("additive", 0), ("and", -3), ("xor", -2)])
def test_pair_sign_and_zero(game, expected):
    def model(values):
        if game == "additive":
            return 8 + values[:, 0] - 2 * values[:, 1]
        if game == "and":
            return -3 * values[:, 0] * values[:, 1]
        return tf.cast(tf.not_equal(values[:, 0], values[:, 1]), tf.float32)

    result = Banzhaf(model, operator=_operator, nb_samples=4).explain_interactions(
        [[1.0, 1.0]], [[1.0]]
    )[0]
    np.testing.assert_allclose(result.interaction_scores, [expected], atol=1e-6)


def test_sampled_pairs_match_independent_group_covariance():
    def model(values):
        return tf.cast(1e12, tf.float64) + tf.cast(
            values[:, 0] * values[:, 1] - values[:, 2] * values[:, 3], tf.float64
        )

    explainer = Banzhaf(model, operator=_operator, nb_samples=12, seed=17)
    masks = explainer._sample_masks(5, 0).numpy().astype(np.float64)
    scores = model(tf.constant(masks, tf.float32)).numpy()
    half = len(masks) // 2
    averaged = (scores[:half] + scores[half:]) / 2
    signs = 2 * masks[:half] - 1
    features = signs[:, 0] * signs[:, 1]
    expected = 4 * np.cov(averaged, features, ddof=1)[0, 1]
    result = explainer.explain_interactions(np.ones((1, 5)), [[1.0]], pairs=[[0, 1]])[0]
    np.testing.assert_allclose(result.interaction_scores, [expected], atol=1e-6)


def test_three_way_parity_has_zero_enumerated_pairs():
    def parity(values):
        return tf.math.floormod(tf.reduce_sum(values, axis=1), 2)

    result = Banzhaf(parity, operator=_operator, nb_samples=8).explain_interactions(
        np.ones((1, 3)), [[1.0]]
    )[0]
    np.testing.assert_array_equal(result.interaction_scores, np.zeros(3))


def test_small_sampled_budget_rejected_only_for_valid_pairs():
    def forbidden(model, inputs, targets):
        pytest.fail("Budget validation should precede inference")

    explainer = Banzhaf(forbidden, operator=forbidden, nb_samples=2)
    with pytest.raises(ValueError, match="nb_samples"):
        explainer.explain_interactions(np.ones((1, 3)), [[1.0]])
    allowed = Banzhaf(
        lambda values: tf.reduce_sum(values, axis=1), operator=_operator, nb_samples=2
    )
    result = allowed.explain_interactions(np.ones((1, 3)), [[1.0]], pairs=np.empty((0, 2), int))
    assert result[0].interaction_scores.shape == (0,)
