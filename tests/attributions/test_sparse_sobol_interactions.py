"""Second-order and total-pair Sobol indices on concept channels."""

from itertools import product

import numpy as np
import pytest
import tensorflow as tf

from xplique.attributions import Banzhaf, SparseSobol
from xplique.attributions.global_sensitivity_analysis.replicated_designs import ReplicatedSampler


def _operator(model, inputs, targets):
    del targets
    return model(inputs)


def _exact_binary_design(explainer, dimension):
    """Enumerate every (A, B) pair for a deterministic population reference."""
    combinations = np.array(list(product((0, 1), repeat=dimension)), np.float32)
    a = tf.constant(np.repeat(combinations, len(combinations), axis=0))
    b = tf.constant(np.tile(combinations, (len(combinations), 1)))
    explainer.nb_design = len(a)
    explainer._sample_masks = lambda active, index: tf.concat(
        [a, b, ReplicatedSampler.build_replicated_design(a, b)], axis=0
    )


@pytest.mark.parametrize(
    "game,pure,total",
    [
        ("additive", 0.0, 0.0),
        ("and", 1 / 3, 1 / 3),
        ("xor", 1.0, 1.0),
        ("parity3", 0.0, 1.0),
    ],
)
def test_binary_population_games(game, pure, total):
    dimension = 3 if game == "parity3" else 2

    def model(values):
        if game == "additive":
            return values[:, 0] - 2 * values[:, 1]
        if game == "and":
            return values[:, 0] * values[:, 1]
        return tf.math.floormod(tf.reduce_sum(values, axis=1), 2)

    for kind, expected in (("pure", pure), ("total", total)):
        explainer = SparseSobol(model, operator=_operator, interaction_kind=kind)
        _exact_binary_design(explainer, dimension)
        result = explainer.explain_interactions(np.ones((1, dimension)), [[1.0]])[0]
        # The mixed-difference numerator uses 1/n; reference sample variance uses 1/(n-1).
        factor = (explainer.nb_design - 1) / explainer.nb_design if kind == "total" else 1
        np.testing.assert_allclose(result.interaction_scores, expected * factor, atol=1e-6)
        np.testing.assert_array_equal(
            result.main_effects,
            explainer.explain(np.ones((1, dimension)), [[1.0]])[0],
        )


@pytest.mark.parametrize("kind", ["pure", "total"])
def test_seeded_pair_hybrids_match_independent_numpy_formula(kind):
    def model(values):
        return values[:, 0] + values[:, 1] * values[:, 2]

    explainer = SparseSobol(
        model, operator=_operator, nb_design=23, interaction_kind=kind, seed=11, batch_size=7
    )
    masks = explainer._sample_masks(3, 0).numpy()
    a, b = masks[:23], masks[23:46]
    scores_a = model(tf.constant(a)).numpy().astype(np.float64)
    scores_b = model(tf.constant(b)).numpy().astype(np.float64)
    scores_c = np.stack(
        [model(tf.constant(block)).numpy() for block in np.split(masks[46:], 3)]
    ).astype(np.float64)
    variance = max(np.var(scores_a, ddof=1), 1e-12)
    expected = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        hybrid = a.copy()
        hybrid[:, [i, j]] = b[:, [i, j]]
        scores_pair = model(tf.constant(hybrid)).numpy().astype(np.float64)
        if kind == "pure":
            value = (
                np.cov(scores_b, scores_pair)[0, 1]
                - np.cov(scores_b, scores_c[i])[0, 1]
                - np.cov(scores_b, scores_c[j])[0, 1]
            ) / variance
        else:
            value = np.mean((scores_a - scores_c[i] - scores_c[j] + scores_pair) ** 2)
            value /= 4 * variance
        expected.append(value)
    result = explainer.explain_interactions(np.ones((1, 3)), [[1.0]], pair_batch_size=1)[0]
    np.testing.assert_allclose(result.interaction_scores, expected, atol=1e-6)


def test_two_channel_hybrid_reuses_b_and_empty_pair_request_uses_singleton_budget():
    calls = []

    def operator(model, values, targets):
        calls.append(len(values))
        return model(values)

    explainer = SparseSobol(
        lambda values: values[:, 0] * values[:, 1], operator=operator, nb_design=6, batch_size=4
    )
    explainer.explain_interactions([[1.0, 1.0]], [[1.0]])
    assert sum(calls) == 6 * (2 + 2)
    calls.clear()
    explainer.explain_interactions([[1.0, 1.0, 1.0]], [[1.0]], pairs=np.empty((0, 2), int))
    assert sum(calls) == 6 * (3 + 2)


def test_invalid_kind_rejected():
    with pytest.raises(ValueError, match="interaction_kind"):
        SparseSobol(lambda values: values, interaction_kind="joint")


@pytest.mark.parametrize("kind", ["pure", "total"])
def test_uniform_product_converges_to_one_seventh(kind):
    def product_model(values):
        return values[:, 0] * values[:, 1]

    result = SparseSobol(
        product_model,
        operator=_operator,
        nb_design=4096,
        batch_size=None,
        interaction_kind=kind,
        seed=43,
    ).explain_interactions([[1.0, 1.0]], [[1.0]])[0]
    np.testing.assert_allclose(result.interaction_scores, [1 / 7], atol=0.035)


@pytest.mark.parametrize("kind", ["pure", "total"])
def test_constant_scores_give_exact_zeros(kind):
    def constant(values):
        return tf.fill([len(values)], tf.constant(1e12, tf.float64))

    result = SparseSobol(
        constant, operator=_operator, nb_design=8, interaction_kind=kind
    ).explain_interactions([[1.0, 2.0, 3.0]], [[1.0]])[0]
    np.testing.assert_array_equal(result.main_effects, np.zeros(3))
    np.testing.assert_array_equal(result.interaction_scores, np.zeros(3))


def test_binary_pair_variance_matches_squared_banzhaf_interaction():
    def model(values):
        return (
            values[:, 0] * values[:, 1]
            + 2 * values[:, 0] * values[:, 1] * values[:, 2]
            - values[:, 1] * values[:, 2]
        )

    combinations = np.asarray(list(product((0, 1), repeat=3)), np.float32)
    variance = np.var(model(tf.constant(combinations)).numpy())
    banzhaf = Banzhaf(model, operator=_operator, nb_samples=8).explain_interactions(
        np.ones((1, 3)), [[1.0]]
    )[0]
    sobol = SparseSobol(model, operator=_operator, mask_distribution="bernoulli")
    _exact_binary_design(sobol, 3)
    pairs = sobol.explain_interactions(np.ones((1, 3)), [[1.0]])[0]
    np.testing.assert_allclose(
        pairs.interaction_scores,
        banzhaf.interaction_scores.numpy() ** 2 / (16 * variance),
        atol=1e-6,
    )


@pytest.mark.parametrize("kind", ["pure", "total"])
def test_pair_selection_and_hybrid_chunking_preserve_scores(kind):
    def model(values):
        return values[:, 0] * values[:, 1] + values[:, 2] * values[:, 3]

    explainer = SparseSobol(
        model, operator=_operator, nb_design=13, interaction_kind=kind, seed=17, batch_size=5
    )
    automatic = explainer.explain_interactions(np.ones((1, 4)), [[1.0]], pair_batch_size=1)[0]
    chosen = explainer.explain_interactions(
        np.ones((1, 4)), [[1.0]], pairs=[[2, 3], [0, 1]], pair_batch_size=6
    )[0]
    np.testing.assert_array_equal(
        chosen.interaction_scores, automatic.interaction_scores.numpy()[[5, 0]]
    )
    np.testing.assert_array_equal(chosen.main_effects, automatic.main_effects)
