"""Quadratic, full-rank Banzhaf interaction regression."""

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import Banzhaf, KernelBanzhaf


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def test_exhaustive_matches_banzhaf_even_for_higher_order_game():
    def model(values):
        return (
            9
            + values[:, 0]
            - 3 * values[:, 1] * values[:, 2]
            + (6 * values[:, 0] * values[:, 1] * values[:, 2])
        )

    inputs = np.ones((1, 3), np.float32)
    expected = Banzhaf(model, operator=_operator, nb_samples=8).explain_interactions(
        inputs, [[1.0]]
    )[0]
    actual = KernelBanzhaf(model, operator=_operator, nb_samples=8).explain_interactions(
        inputs, [[1.0]], pairs=[[1, 2], [0, 2]], pair_batch_size=1
    )[0]
    np.testing.assert_allclose(
        actual.interaction_scores, expected.interaction_scores.numpy()[[2, 1]], atol=1e-6
    )
    np.testing.assert_allclose(actual.main_effects, expected.main_effects, atol=1e-6)


def test_sampled_quadratic_reference_and_pair_selection():
    def model(values):
        values = tf.cast(values, tf.float64)
        return (
            tf.reduce_sum(values, axis=1)
            + 5 * values[:, 0] * values[:, 1]
            - 2 * values[:, 2] * values[:, 3]
            + values[:, 0] * values[:, 1] * values[:, 3]
        )

    inputs = np.ones((1, 4), np.float32)
    # Choose a reproducible full-rank antithetic design; deficient draws must raise.
    for seed in range(100):
        explainer = KernelBanzhaf(model, operator=_operator, nb_samples=14, seed=seed)
        masks = (
            Banzhaf(model, operator=_operator, nb_samples=14, seed=seed)
            ._sample_masks(4, 0)
            .numpy()
            .astype(np.float64)
        )
        pair_ids = np.stack(np.triu_indices(4, 1), axis=1)
        features = np.prod((masks[:7] - 0.5)[:, pair_ids], axis=2)
        features -= features.mean(axis=0)
        if np.linalg.matrix_rank(features) == 6 and np.linalg.matrix_rank(masks - 0.5) == 4:
            break
    else:
        pytest.fail("No full-rank seeded pair design found")

    scores = model(tf.constant(masks, tf.float32)).numpy()
    response = (scores[:7] + scores[7:]) / 2
    expected = np.linalg.lstsq(features, response - response.mean(), rcond=None)[0]
    all_pairs = explainer.explain_interactions(inputs, [[1.0]])[0]
    np.testing.assert_allclose(all_pairs.interaction_scores, expected, atol=1e-6)
    chosen = explainer.explain_interactions(inputs, [[1.0]], pairs=[[2, 3], [0, 1]])[0]
    np.testing.assert_allclose(chosen.interaction_scores, expected[[5, 0]], atol=1e-6)
    np.testing.assert_array_equal(all_pairs.main_effects, explainer.explain(inputs, [[1.0]])[0])


def test_insufficient_pair_rank_rejected_before_inference():
    def forbidden(model, inputs, targets):
        pytest.fail("Quadratic rank preflight must precede inference")

    explainer = KernelBanzhaf(forbidden, operator=forbidden, nb_samples=6)
    with pytest.raises(ValueError, match="Quadratic pair design"):
        explainer.explain_interactions(np.ones((1, 3)), [[1.0]])


def test_no_requested_active_pairs_skips_quadratic_rank_requirement():
    def model(values):
        return tf.reduce_sum(values, axis=1)

    explainer = KernelBanzhaf(model, operator=_operator, nb_samples=6)
    result = explainer.explain_interactions(np.ones((1, 3)), [[1.0]], pairs=np.empty((0, 2), int))[
        0
    ]
    assert result.interaction_scores.shape == (0,)
    np.testing.assert_allclose(result.main_effects, [1, 1, 1], atol=1e-6)


def test_realized_pair_rank_checked_before_inference(monkeypatch):
    half = np.array(
        [
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        np.float32,
    )
    masks = tf.constant(np.concatenate((half, 1 - half)))
    assert np.linalg.matrix_rank(half - 0.5) == 4

    def sample(self, dimension, index):
        assert (dimension, index) == (4, 0)
        return masks

    def forbidden(model, inputs, targets):
        pytest.fail("A deficient quadratic design must fail before inference")

    monkeypatch.setattr(Banzhaf, "_sample_masks", sample)
    explainer = KernelBanzhaf(forbidden, operator=forbidden, nb_samples=14)
    with pytest.raises(ValueError, match="quadratic pair design has rank"):
        explainer.explain_interactions(np.ones((1, 4)), [[1.0]])


def test_pair_budget_rejected_before_any_input_is_evaluated():
    def forbidden(model, inputs, targets):
        pytest.fail("Every input's pair budget must be checked before inference")

    explainer = KernelBanzhaf(forbidden, operator=forbidden, nb_samples=6)
    with pytest.raises(ValueError, match="Quadratic pair design"):
        explainer.explain_interactions([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], [[1.0], [1.0]])


def test_sampled_quadratic_game_recovered_exactly():
    """Complement averaging cancels linear terms, so pair coefficients are exact."""
    nb_active = 14
    rng = np.random.default_rng(0)
    linear = rng.normal(size=nb_active)
    quadratic = np.triu(rng.normal(size=(nb_active, nb_active)), k=1)

    def model(values):
        values = tf.cast(values, tf.float64)
        pairwise = tf.reduce_sum(tf.linalg.matmul(values, quadratic) * values, axis=1)
        return tf.linalg.matvec(values, linear) + pairwise

    result = KernelBanzhaf(model, operator=_operator, nb_samples=512).explain_interactions(
        np.ones((1, nb_active), np.float32), [[1.0]]
    )[0]
    np.testing.assert_allclose(
        result.interaction_scores, quadratic[np.triu_indices(nb_active, 1)], atol=1e-6
    )
